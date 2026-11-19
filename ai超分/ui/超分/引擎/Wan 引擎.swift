import Foundation
import Metal
import MetalPerformanceShaders

/// Wan 5B 主干（RelayVSR 的参考端点生成器）超参。
nonisolated enum Wan形状 {
    static let 维度 = 3072
    static let 层数 = 30
    static let 头数 = 24
    static let 头维 = 128
    /// V 直取：MPS 侧把 B 按 [内维][列] 解释，省去每帧 720 次 head_transpose。编译期常量，无运行时开关。
    static let 直取值矩阵 = true
    static let 文本令牌 = 512
    static let 时间频率 = 256
    static let 逐头缩放 = 1.0 / sqrt(Float(头维))
    static let theta: Float = 10000
    static let 归一化eps: Float = 1e-6
}

nonisolated struct WanHeadParams {
    var tokens: Int32
    var dim: Int32
    var srcStride: Int32
    var dstStride: Int32
    var srcOffset: Int32
    var dstOffset: Int32
}

/// Wan 主干推理引擎：latent 噪声 + 低清帧 → 预测 latent，结果按 `潜变量 - 预测` 即得超分 latent。
///
/// 权重以 fp16 常驻 GPU（约 5.03 GB）。载入时整份 bf16 源数据经 GPU 就地转换，
/// 避免 CPU 逐元素转换 10 GB 带来的内存峰值。
nonisolated final class Wan引擎 {

    enum 错误: LocalizedError {
        case 初始化失败(String)
        case 尺寸不符(String)
        case 执行失败(String)

        var errorDescription: String? {
            switch self {
            case .初始化失败(let 说明): "Wan 引擎初始化失败：\(说明)"
            case .尺寸不符(let 说明): "Wan 张量尺寸不符：\(说明)"
            case .执行失败(let 说明): "Wan 执行失败：\(说明)"
            }
        }
    }

    struct 线性 {
        let 权重: MTLBuffer
        let 偏置: MTLBuffer?
        let 输出维: Int
        let 输入维: Int
    }

    struct 块 {
        let 查询: 线性
        let 键: 线性
        let 值: 线性
        let 输出: 线性
        let 查询归一化: MTLBuffer
        let 键归一化: MTLBuffer
        let 交叉查询: 线性
        let 交叉键: 线性
        let 交叉值: 线性
        let 交叉输出: 线性
        let 交叉查询归一化: MTLBuffer
        let 交叉键归一化: MTLBuffer
        let 交叉归一化w: MTLBuffer
        let 交叉归一化b: MTLBuffer
        let 前馈入口: 线性
        let 前馈出口: 线性
        let 调制: MTLBuffer
        let 交叉K: MTLBuffer
        let 交叉V: MTLBuffer
    }

    let 设备: MTLDevice
    private let 队列: MTLCommandQueue
    private let 库: MTLLibrary
    private var 管线: [String: MTLComputePipelineState] = [:]

    private var 补丁权重: 线性!
    private var 低清卷积1: 线性!
    private var 低清卷积2: 线性!
    private var 低清输出: 线性!
    private var 低清归一化1: MTLBuffer!
    private var 低清归一化2: MTLBuffer!
    private var 输出投影: 线性!
    private var 输出调制: MTLBuffer!
    private var 块表: [块] = []

    private(set) var 令牌高 = 0
    private(set) var 令牌宽 = 0
    private(set) var 潜高 = 0
    private(set) var 潜宽 = 0

    private var 临时池: [MTLBuffer] = []
    private var 临时游标 = 0

    var 诊断阶段 = 0
    /// 跨层合并提交：单批命令缓冲的 GPU 时长预算（毫秒）。
    private let 单批GPU预算毫秒 = 1200.0
    /// 实测单层 GPU 耗时（毫秒），用于自适应决定每批层数；首帧未知时按 2 层保守起步。
    private var 单层毫秒估计 = 0.0
    /// 对拍用：非空时把每层输出按 token×channel 主序导出到该目录
    var 导出目录: String? = nil
    var 诊断采集 = false
    var 诊断回调: ((String, [Int], [Float]) -> Void)?

    // MARK: - 初始化

    init(设备: MTLDevice, 权重路径: String, 库 指定库: MTLLibrary? = nil) throws {
        guard let 队列 = 设备.makeCommandQueue() else {
            throw 错误.初始化失败("无法创建命令队列")
        }
        self.设备 = 设备
        self.队列 = 队列
        if let 有库 = 指定库 {
            self.库 = 有库
        } else if let 默认库 = 设备.makeDefaultLibrary() {
            self.库 = 默认库
        } else {
            throw 错误.初始化失败("找不到 Wan 算子的 Metal 库")
        }

        let 权重库 = try SafeTensors权重库(路径: 权重路径)
        let 补丁尺寸 = try 权重库.头("patch_embedding.weight")
        guard 补丁尺寸.形状 == [3072, 48, 1, 2, 2] else {
            throw 错误.尺寸不符("patch_embedding.weight 形状 \(补丁尺寸.形状)")
        }

        try 建管线()
        let 池 = try 搬权重(路径: 权重路径, 权重库: 权重库)
        权重池存储 = 池
        print("DBG 搬权重完成")
        try 装载权重(池: 池, 权重库: 权重库)
        print("DBG 装载权重完成")
        print("DBG 预计算完成")
    }

    private func 建管线() throws {
        let 名单 = ["wan_bf16_to_half", "wan_lq_prepare", "wan_bias_act", "wan_im2col3",
                  "wan_patch_unfold", "wan_layernorm", "wan_rmsnorm_row", "wan_modulate",
                  "wan_gate_add", "wan_rope", "wan_mask_softmax", "wan_softmax_dense",
                  "wan_gelu", "wan_unpatch", "wan_latent_step", "wan_copy_half",
                  "wan_head_transpose", "wan_gemm", "wan_gemm_v4", "wan_gemm_v5"]
        for 名 in 名单 {
            guard let 函数 = 库.makeFunction(name: 名) else {
                throw 错误.初始化失败("缺少 Metal 函数 \(名)")
            }
            管线[名] = try 设备.makeComputePipelineState(function: 函数)
        }
    }

    private func 管(_ 名: String) throws -> MTLComputePipelineState {
        guard let 状态 = 管线[名] else { throw 错误.初始化失败("缺少管线 \(名)") }
        return 状态
    }

    // MARK: - 权重搬运

    private var 权重池存储: 权重池?

    private struct 权重池 {
        let 缓冲表: [String: MTLBuffer]
    }

    /// 整份 bf16 → fp16：每张量各分配一块 fp16 缓冲，偏移恒为 0，
    /// 既规避 Metal setBuffer 的 256 字节对齐限制，也让 kernel 无需处理权重基址。
    /// bf16 → fp16 转换结果磁盘缓存：首次启动转换后落盘，后续启动直接 mmap 零拷贝挂载，
    /// 把 3.9 s 的整份转换 + 10 GB 读盘成本降到接近 0。缓存头记录源文件 size/mtime 与张量表，
    /// 权重文件或转换版本一变即判定失效并重转。
    private static let 缓存版本 = 1
    static let 缓存对齐 = 16384
    private var 缓存映射: Data?

    private struct 缓存张量: Codable { let 名: String; let 元素: Int; let 偏移: Int }
    private struct 缓存头: Codable {
        let 版本: Int; let 源大小: Int; let 源修改: Double; let 张量: [缓存张量]
    }

    private func 缓存路径(源: String) -> String {
        let 目录 = (源 as NSString).deletingLastPathComponent
        let 名 = (源 as NSString).lastPathComponent
        return 目录 + "/." + 名 + ".fp16v\(Self.缓存版本).cache"
    }

    /// 缓存命中则按张量逐个拷入真实 MTLBuffer（GPU 直接可用，占用计入进程 footprint）。
    private func 尝试载入缓存(源: String, 权重库: SafeTensors权重库) -> 权重池? {
        let 路径 = 缓存路径(源: 源)
        guard let 属性 = try? FileManager.default.attributesOfItem(atPath: 源),
              let 源大小 = 属性[.size] as? Int,
              let 源修改 = (属性[.modificationDate] as? Date)?.timeIntervalSince1970 else { return nil }
        guard let 数据 = try? Data(contentsOf: URL(fileURLWithPath: 路径), options: [.mappedIfSafe]),
              数据.count > 32 else { return nil }
        let 头长 = 数据.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 0, as: UInt64.self).littleEndian) }
        guard 头长 > 0, 8 + 头长 <= 数据.count else { return nil }
        let 头段 = 数据.subdata(in: 8..<(8 + 头长))
        guard let 头 = try? JSONDecoder().decode(缓存头.self, from: 头段) else { return nil }
        guard 头.版本 == Self.缓存版本, 头.源大小 == 源大小,
              abs(头.源修改 - 源修改) < 1.0 else { return nil }

        // 与权重库逐张量核对（名字 + 元素数），任何不一致即视作缓存失效
        var 元素表: [String: Int] = [:]
        for 名 in 权重库.张量名 {
            guard let h = try? 权重库.头(名), h.类型 == .bf16 || h.类型 == .f16, h.元素数 > 0 else { continue }
            元素表[名] = h.元素数
        }
        guard 元素表.count == 头.张量.count else { return nil }

        let 基址 = UnsafeMutableRawPointer(mutating: (数据 as NSData).bytes)
        guard Int(bitPattern: 基址) % Self.缓存对齐 == 0 else { return nil }
        var 缓冲表: [String: MTLBuffer] = [:]
        缓冲表.reserveCapacity(头.张量.count)
        for t in 头.张量 {
            guard 元素表[t.名] == t.元素, t.偏移 % Self.缓存对齐 == 0,
                  t.偏移 >= 0, t.偏移 + t.元素 * 2 <= 数据.count else { return nil }
            guard let 缓冲 = 设备.makeBuffer(bytes: 基址.advanced(by: t.偏移),
                                          length: t.元素 * 2, options: .storageModeShared) else { return nil }
            缓冲.label = t.名
            缓冲表[t.名] = 缓冲
        }
        return 权重池(缓冲表: 缓冲表)
    }

    /// 转换完成后顺序落盘缓存（后台低优先级，不与出帧抢主线程）。
    private func 写出缓存(源: String, 缓冲表: [String: MTLBuffer]) {
        let 路径 = 缓存路径(源: 源)
        guard let 属性 = try? FileManager.default.attributesOfItem(atPath: 源),
              let 源大小 = 属性[.size] as? Int,
              let 源修改 = (属性[.modificationDate] as? Date)?.timeIntervalSince1970 else { return }
        // 按名字排序保证布局稳定；偏移以「数据起点」为基准，而数据起点取决于头部长度，
        // 头部 JSON 又含偏移数字，故迭代收敛到 16 KB 桶稳定为止。
        let 名单 = 缓冲表.keys.sorted()
        var 表: [缓存张量] = []
        var 头数据 = Data()
        var 头占位 = 8
        var 数据起点 = Self.缓存对齐
        for _ in 0..<8 {
            var 偏移 = 数据起点
            表 = []
            for 名 in 名单 {
                guard let 缓冲 = 缓冲表[名] else { continue }
                let 元素 = 缓冲.length / 2
                if 偏移 % Self.缓存对齐 != 0 { 偏移 += Self.缓存对齐 - 偏移 % Self.缓存对齐 }
                表.append(缓存张量(名: 名, 元素: 元素, 偏移: 偏移))
                偏移 += 元素 * 2
            }
            let 头 = 缓存头(版本: Self.缓存版本, 源大小: 源大小, 源修改: 源修改, 张量: 表)
            guard let 编码 = try? JSONEncoder().encode(头) else { return }
            头数据 = 编码
            头占位 = 8 + 编码.count
            let 新起点 = (头占位 + Self.缓存对齐 - 1) / Self.缓存对齐 * Self.缓存对齐
            if 新起点 == 数据起点 { break }
            数据起点 = 新起点
        }

        let 表快照 = 缓冲表
        let 路径快照 = 路径
        let 数据起点快照 = 数据起点
        DispatchQueue.global(qos: .utility).async {
            let 临时 = 路径快照 + ".tmp"
            FileManager.default.createFile(atPath: 临时, contents: nil)
            guard let 句柄 = FileHandle(forWritingAtPath: 临时) else { return }
            defer { try? 句柄.close() }
            var 长 = UInt64(头数据.count).littleEndian
            withUnsafeBytes(of: &长) { 句柄.write(Data($0)) }
            句柄.write(头数据)
            句柄.write(Data(count: 数据起点快照 - 头占位))
            var 已写 = 数据起点快照
            for t in 表 {
                guard let 缓冲 = 表快照[t.名] else { return }
                // 与头部偏移严格对齐：每张量起点按 16 KB 对齐，缺的补零，保证偏移与文件布局一致
                if t.偏移 > 已写 { 句柄.write(Data(count: t.偏移 - 已写)) }
                句柄.write(Data(bytes: 缓冲.contents(), count: t.元素 * 2))
                已写 = t.偏移 + t.元素 * 2
            }
            try? 句柄.synchronize()
            try? FileManager.default.removeItem(atPath: 路径快照)
            try? FileManager.default.moveItem(atPath: 临时, toPath: 路径快照)
        }
    }

    private func 搬权重(路径: String, 权重库: SafeTensors权重库) throws -> 权重池 {
        if let 缓存池 = 尝试载入缓存(源: 路径, 权重库: 权重库) { return 缓存池 }
        let 数据 = try Data(contentsOf: URL(fileURLWithPath: 路径), options: [.mappedIfSafe])
        let 头长度 = 数据.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self).littleEndian }
        let 数据段起点 = 8 + Int(头长度)

        var 缓冲表: [String: MTLBuffer] = [:]
        缓冲表.reserveCapacity(权重库.张量名.count)

        guard let 源 = 设备.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: (数据 as NSData).bytes),
                                    length: 数据.count,
                                    options: .storageModeShared,
                                    deallocator: nil) else {
            throw 错误.初始化失败("无法映射权重文件")
        }

        guard let 命令 = 队列.makeCommandBuffer(),
              let 编码器 = 命令.makeComputeCommandEncoder() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_bf16_to_half"))
        编码器.setBuffer(源, offset: 0, index: 0)

        var 已搬元素 = 0
        for 名 in 权重库.张量名 {
            let 头 = try 权重库.头(名)
            guard 头.类型 == .bf16 || 头.类型 == .f16, 头.元素数 > 0 else { continue }
            guard let 目的 = 设备.makeBuffer(length: 头.元素数 * 2, options: .storageModeShared) else {
                throw 错误.初始化失败("无法分配 \(名) 的 fp16 缓冲")
            }
            目的.label = 名
            缓冲表[名] = 目的
            var 参数 = WanCopyParams(n: UInt32(头.元素数), pad0: 0,
                                    srcOff: UInt64((数据段起点 + 头.数据起点) / 2),
                                    dstOff: 0)
            编码器.setBuffer(目的, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<WanCopyParams>.size, index: 2)
            let 线程 = 256
            编码器.dispatchThreadgroups(MTLSize(width: (头.元素数 + 线程 - 1) / 线程, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 线程, height: 1, depth: 1))
            已搬元素 += 头.元素数
        }
        编码器.endEncoding()
        命令.commit()
        命令.waitUntilCompleted()
        if let e = 命令.error { throw 错误.执行失败("权重转换：\(e.localizedDescription)") }
        print("DBG 搬运张量", 缓冲表.count, "元素", 已搬元素)
        写出缓存(源: 路径, 缓冲表: 缓冲表)
        return 权重池(缓冲表: 缓冲表)
    }

    private func 取张量(_ 池: 权重池, _ 名: String, _ 元素数: Int) throws -> MTLBuffer {
        guard let 缓冲 = 池.缓冲表[名] else { throw 错误.尺寸不符("缺少张量 \(名)") }
        guard 缓冲.length == 元素数 * 2 else {
            throw 错误.尺寸不符("\(名) 期望 \(元素数) 实际 \(缓冲.length / 2)")
        }
        return 缓冲
    }

    private func 取线性(_ 池: 权重池, _ 权名: String, _ 偏名: String,
                     _ 输出维: Int, _ 输入维: Int) throws -> 线性 {
        _ = try 取张量(池, 权名, 输出维 * 输入维)
        let 偏 = 池.缓冲表[偏名] != nil ? try 取张量(池, 偏名, 输出维) : nil
        return 线性(权重: try 取张量(池, 权名, 输出维 * 输入维), 偏置: 偏,
                  输出维: 输出维, 输入维: 输入维)
    }

    private func 装载权重(池: 权重池, 权重库: SafeTensors权重库) throws {
        print("DBG 装载A")
        低清卷积1 = try 取线性(池, "lq_proj.conv1.weight", "lq_proj.conv1.bias", 768, 768)
        低清卷积2 = try 取线性(池, "lq_proj.conv2.weight", "lq_proj.conv2.bias", 3072, 768 * 9)
        低清输出 = try 取线性(池, "lq_proj.out_proj.weight", "lq_proj.out_proj.bias", 3072, 3072)
        低清归一化1 = try 取张量(池, "lq_proj.norm1.weight", 768)
        低清归一化2 = try 取张量(池, "lq_proj.norm2.weight", 3072)

        补丁权重 = 线性(权重: try 取张量(池, "patch_embedding.weight", 3072 * 192),
                    偏置: try 取张量(池, "patch_embedding.bias", 3072),
                    输出维: 3072, 输入维: 192)
        输出投影 = try 取线性(池, "proj_out.weight", "proj_out.bias", 192, 3072)
        输出调制 = 新建浮点缓冲(2 * 3072, "输出调制")
        print("DBG 装载B")

        let 文本一 = try 取线性(池, "condition_embedder.text_embedder.linear_1.weight",
                            "condition_embedder.text_embedder.linear_1.bias", 3072, 4096)
        let 文本二 = try 取线性(池, "condition_embedder.text_embedder.linear_2.weight",
                            "condition_embedder.text_embedder.linear_2.bias", 3072, 3072)
        let 时间一 = try 取线性(池, "condition_embedder.time_embedder.linear_1.weight",
                            "condition_embedder.time_embedder.linear_1.bias", 3072, 256)
        let 时间二 = try 取线性(池, "condition_embedder.time_embedder.linear_2.weight",
                            "condition_embedder.time_embedder.linear_2.bias", 3072, 3072)
        let 时间投影 = try 取线性(池, "condition_embedder.time_proj.weight",
                              "condition_embedder.time_proj.bias", 3072 * 6, 3072)
        _ = try 取张量(池, "prompt_embeds", 512 * 4096)
        _ = try 取张量(池, "scale_shift_table", 2 * 3072)

        print("DBG 装载C")
        var 表: [块] = []
        表.reserveCapacity(Wan形状.层数)
        for i in 0..<Wan形状.层数 {
            let p = "blocks.\(i)"
            表.append(块(
                查询: try 取线性(池, "\(p).attn1.to_q.weight", "\(p).attn1.to_q.bias", 3072, 3072),
                键: try 取线性(池, "\(p).attn1.to_k.weight", "\(p).attn1.to_k.bias", 3072, 3072),
                值: try 取线性(池, "\(p).attn1.to_v.weight", "\(p).attn1.to_v.bias", 3072, 3072),
                输出: try 取线性(池, "\(p).attn1.to_out.0.weight", "\(p).attn1.to_out.0.bias", 3072, 3072),
                查询归一化: try 取张量(池, "\(p).attn1.norm_q.weight", 3072),
                键归一化: try 取张量(池, "\(p).attn1.norm_k.weight", 3072),
                交叉查询: try 取线性(池, "\(p).attn2.to_q.weight", "\(p).attn2.to_q.bias", 3072, 3072),
                交叉键: try 取线性(池, "\(p).attn2.to_k.weight", "\(p).attn2.to_k.bias", 3072, 3072),
                交叉值: try 取线性(池, "\(p).attn2.to_v.weight", "\(p).attn2.to_v.bias", 3072, 3072),
                交叉输出: try 取线性(池, "\(p).attn2.to_out.0.weight", "\(p).attn2.to_out.0.bias", 3072, 3072),
                交叉查询归一化: try 取张量(池, "\(p).attn2.norm_q.weight", 3072),
                交叉键归一化: try 取张量(池, "\(p).attn2.norm_k.weight", 3072),
                交叉归一化w: try 取张量(池, "\(p).norm2.weight", 3072),
                交叉归一化b: try 取张量(池, "\(p).norm2.bias", 3072),
                前馈入口: try 取线性(池, "\(p).ffn.net.0.proj.weight", "\(p).ffn.net.0.proj.bias", 14336, 3072),
                前馈出口: try 取线性(池, "\(p).ffn.net.2.weight", "\(p).ffn.net.2.bias", 3072, 14336),
                调制: 新建浮点缓冲(6 * 3072, "块\(i)调制"),
                交叉K: 新建半精度缓冲(512 * 3072, "块\(i)交叉K"),
                交叉V: 新建半精度缓冲(512 * 3072, "块\(i)交叉V")
            ))
        }
        print("DBG 装载D 块数", 表.count)
        块表 = 表
        try 预计算条件(池: 池, 文本一: 文本一, 文本二: 文本二,
                  时间一: 时间一, 时间二: 时间二, 时间投影: 时间投影)
    }

    // MARK: - 缓冲

    private func 新建半精度缓冲(_ 元素数: Int, _ 名称: String) -> MTLBuffer {
        guard let 缓冲 = 设备.makeBuffer(length: max(元素数, 1) * 2, options: .storageModeShared) else {
            fatalError("Wan 引擎分配缓冲失败：\(名称)")
        }
        缓冲.label = 名称
        return 缓冲
    }

    private func 新建浮点缓冲(_ 元素数: Int, _ 名称: String) -> MTLBuffer {
        guard let 缓冲 = 设备.makeBuffer(length: max(元素数, 1) * 4, options: .storageModeShared) else {
            fatalError("Wan 引擎分配缓冲失败：\(名称)")
        }
        缓冲.label = 名称
        return 缓冲
    }

    private func 借(_ 元素数: Int, _ 名称: String = "临时") -> MTLBuffer {
        while 临时池.count <= 临时游标 {
            临时池.append(新建半精度缓冲(1 << 12, 名称))
        }
        var 缓冲 = 临时池[临时游标]
        if 缓冲.length < 元素数 * 2 {
            guard let 替 = 设备.makeBuffer(length: 元素数 * 2, options: .storageModeShared) else {
                fatalError("Wan 引擎扩容失败：\(名称)")
            }
            替.label = 名称
            临时池[临时游标] = 替
            缓冲 = 替
        }
        临时游标 += 1
        return 缓冲
    }

    /// 调试用：打印缓冲内 fp16 数值统计。
    private func 统计(_ 标签: String, _ 缓冲: MTLBuffer, _ 元素数: Int) {
        let 指针 = 缓冲.contents().assumingMemoryBound(to: Float16.self)
        var mn = Float.infinity, mx = -Float.infinity, 坏 = 0, 异 = 0
        for i in 0..<元素数 {
            let v = Float(指针[i])
            if v.isNaN { 坏 += 1 } else if v.isInfinite { 异 += 1 } else { mn = min(mn, v); mx = max(mx, v) }
        }
        let 头 = (0..<min(6, 元素数)).map { String(format: "%.4f", Float(指针[$0])) }.joined(separator: ",")
        var 首坏 = -1
        if 坏 > 0 { for i in 0..<元素数 where 指针[i].isNaN { 首坏 = i; break } }
        print("STAT", 标签, "min", mn, "max", mx, "nan", 坏, "inf", 异, "head", 头, "首NaN", 首坏, "行", 首坏 >= 0 ? 首坏 / (元素数 / max(1, 元素数 / 14336 == 0 ? 1 : 1)) : -1)
    }

    /// 调试用：读取指定张量前若干值（已完成 bf16 → fp16）。
    func 探针(_ 名称: String, _ 数量: Int) throws -> [Float] {
        guard let 池 = 权重池存储, let 缓冲 = 池.缓冲表[名称] else { return [] }
        let 指针 = 缓冲.contents().assumingMemoryBound(to: Float16.self)
        let 可用 = min(数量, 缓冲.length / 2)
        return (0..<可用).map { Float(指针[$0]) }
    }

    private func 读半精度(_ 池: 权重池, _ 名: String, _ 元素数: Int) throws -> [Float] {
        let 缓冲 = try 取张量(池, 名, 元素数)
        let 指针 = 缓冲.contents().assumingMemoryBound(to: Float16.self)
        var 输出 = [Float](repeating: 0, count: 元素数)
        for i in 0..<元素数 { 输出[i] = Float(指针[i]) }
        return 输出
    }

    private func 读半精度缓冲(_ 缓冲: MTLBuffer, _ 元素数: Int) -> [Float] {
        let 指针 = 缓冲.contents().assumingMemoryBound(to: Float16.self)
        var 输出 = [Float](repeating: 0, count: 元素数)
        for i in 0..<元素数 { 输出[i] = Float(指针[i]) }
        return 输出
    }

    // MARK: - 基础算子

    private var gemm缓存: [String: MPSMatrixMultiplication] = [:]

    /// MPS 硬件 GEMM 的对齐前提：half 下偏移为偶数，行跨满足 32 字节对齐。
    @inline(__always)
    private func 可走MPS(行: Int, 列: Int, 内维: Int,
                     aOffset: Int, bOffset: Int, cOffset: Int,
                     aRowBytes: Int, bRowBytes: Int, cRowBytes: Int) -> Bool {
        guard 行 > 0, 列 > 0, 内维 > 0 else { return false }
        guard aOffset % 2 == 0, bOffset % 2 == 0, cOffset % 2 == 0 else { return false }
        // MPS half 仅要求行跨 16 字节（8 half）对齐；32 字节是更严的保守值，会误拒多数窗口键数
        return aRowBytes % 16 == 0 && bRowBytes % 16 == 0 && cRowBytes % 16 == 0
    }

    private func 矩阵乘(_ 编码器: inout MTLComputeCommandEncoder, 命令: MTLCommandBuffer,
                    A: MTLBuffer, aOffset: Int, aRowBytes: Int, 行: Int, 内维: Int,
                    B: MTLBuffer, bOffset: Int, bRowBytes: Int, 列: Int,
                    C: MTLBuffer, cOffset: Int, cRowBytes: Int, 转置B: Bool = true) throws {
        if 可走MPS(行: 行, 列: 列, 内维: 内维,
                  aOffset: aOffset, bOffset: bOffset, cOffset: cOffset,
                  aRowBytes: aRowBytes, bRowBytes: bRowBytes, cRowBytes: cRowBytes) {
            let 键 = "\(行)_\(列)_\(内维)_\(转置B ? "t" : "n")"
            let gemm: MPSMatrixMultiplication
            if let 已 = gemm缓存[键] {
                gemm = 已
            } else {
                gemm = MPSMatrixMultiplication(device: 设备, transposeLeft: false, transposeRight: 转置B,
                                               resultRows: 行, resultColumns: 列, interiorColumns: 内维,
                                               alpha: 1, beta: 0)
                gemm缓存[键] = gemm
            }
            let 甲 = MPSMatrix(buffer: A, offset: aOffset,
                             descriptor: MPSMatrixDescriptor(rows: 行, columns: 内维,
                                                             rowBytes: aRowBytes, dataType: .float16))
            let 乙 = MPSMatrix(buffer: B, offset: bOffset,
                             descriptor: MPSMatrixDescriptor(rows: 转置B ? 列 : 内维,
                                                             columns: 转置B ? 内维 : 列,
                                                             rowBytes: bRowBytes, dataType: .float16))
            let 丙 = MPSMatrix(buffer: C, offset: cOffset,
                             descriptor: MPSMatrixDescriptor(rows: 行, columns: 列,
                                                             rowBytes: cRowBytes, dataType: .float16))
            // MPS 编码器与计算编码器不能混用同一个 encoder：先收尾当前编码器，
            // 让 MPS 成为命令缓冲里的后序 encoder，再重开计算编码器供后续算子继续编码，
            // 从而保持命令顺序（Metal 串行 encoder 自动同步数据依赖）。
            编码器.endEncoding()
            gemm.encode(commandBuffer: 命令, leftMatrix: 甲, rightMatrix: 乙, resultMatrix: 丙)
            guard let 新编码器 = 命令.makeComputeCommandEncoder() else {
                throw 错误.初始化失败("GEMM 后重建命令编码器失败")
            }
            编码器 = 新编码器
            return
        }
        guard 转置B else { throw 错误.初始化失败("B 直取仅支持 MPS 路径") }
        // C[M, N] = A[M, K] · B[N, K]ᵀ；B 为行主序 [N][K]，故 QKᵀ 可直取多头切片、无需转置。
        var 参数 = WanGemmParams(M: Int32(行), N: Int32(列), K: Int32(内维),
                              aRowStride: Int32(aRowBytes / 2), bRowStride: Int32(bRowBytes / 2),
                              cRowStride: Int32(cRowBytes / 2))
        编码器.memoryBarrier(scope: .buffers)
        // half4 向量化变体要求：切片偏移与行跨均为 8 字节对齐、内维为 4 的倍数；
        // 不满足时回退逐元素版本，保证任意形状下数值一致。
        let 可向量 = aOffset % 8 == 0 && bOffset % 8 == 0 && aRowBytes % 8 == 0
            && bRowBytes % 8 == 0 && 内维 % 4 == 0
        // 大 M 形状走 v5（128x64 瓦片 / 256 线程 / K 维寄存器双缓冲），小形状仍用 v4 避免瓦片空转。
        let 用大瓦片 = 可向量 && 行 >= 512 && 列 >= 64
        编码器.setComputePipelineState(try 管(可向量 ? (用大瓦片 ? "wan_gemm_v5" : "wan_gemm_v4") : "wan_gemm"))
        编码器.setBuffer(A, offset: aOffset, index: 0)
        编码器.setBuffer(B, offset: bOffset, index: 1)
        编码器.setBuffer(C, offset: cOffset, index: 2)
        编码器.setBytes(&参数, length: MemoryLayout<WanGemmParams>.stride, index: 3)
        if 用大瓦片 {
            编码器.dispatchThreadgroups(MTLSize(width: (列 + 63) / 64, height: (行 + 127) / 128, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            编码器.dispatchThreadgroups(MTLSize(width: (列 + 63) / 64, height: (行 + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
    }

    private func 偏置激活(_ 编码器: MTLComputeCommandEncoder, 缓冲: MTLBuffer,
                     偏置: MTLBuffer?, 行: Int, 列: Int, 激活: Int32) throws {
        var 参数 = WanBiasParams(cols: Int32(列), hasBias: 偏置 != nil ? 1 : 0, act: 激活, rows: Int32(行))
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_bias_act"))
        编码器.setBuffer(缓冲, offset: 0, index: 0)
        编码器.setBuffer(偏置 ?? 缓冲, offset: 0, index: 1)
        编码器.setBytes(&参数, length: MemoryLayout<WanBiasParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: (列 + 15) / 16, height: (行 + 15) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
    }

    private func 线性前进(_ 编码器: inout MTLComputeCommandEncoder, 命令: MTLCommandBuffer, _ 层: 线性,
                       x: MTLBuffer, xRowBytes: Int, 行: Int,
                       out: MTLBuffer, outRowBytes: Int, 激活: Int32) throws {
        try 矩阵乘(&编码器, 命令: 命令, A: x, aOffset: 0, aRowBytes: xRowBytes, 行: 行, 内维: 层.输入维,
            B: 层.权重, bOffset: 0, bRowBytes: 层.输入维 * 2, 列: 层.输出维,
            C: out, cOffset: 0, cRowBytes: outRowBytes)
        try 偏置激活(编码器, 缓冲: out, 偏置: 层.偏置, 行: 行, 列: 层.输出维, 激活: 激活)
    }

    private func 归一化行(_ 编码器: MTLComputeCommandEncoder, x: MTLBuffer, 行: Int, 列: Int,
                       out: MTLBuffer, 权重: MTLBuffer, eps: Float) throws {
        var 参数 = WanNormParams(rows: Int32(行), cols: Int32(列), eps: eps, hasWeight: 1, hasBias: 0)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_rmsnorm_row"))
        编码器.setBuffer(x, offset: 0, index: 0)
        编码器.setBuffer(权重, offset: 0, index: 1)
        编码器.setBuffer(out, offset: 0, index: 2)
        编码器.setBytes(&参数, length: MemoryLayout<WanNormParams>.stride, index: 3)
        编码器.dispatchThreadgroups(MTLSize(width: 行, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func 层归一化(_ 编码器: MTLComputeCommandEncoder, x: MTLBuffer, 行: Int, 列: Int,
                       out: MTLBuffer, 权重: MTLBuffer?, 偏置: MTLBuffer?, eps: Float) throws {
        var 参数 = WanNormParams(rows: Int32(行), cols: Int32(列), eps: eps,
                                hasWeight: 权重 != nil ? 1 : 0, hasBias: 偏置 != nil ? 1 : 0)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_layernorm"))
        编码器.setBuffer(x, offset: 0, index: 0)
        编码器.setBuffer(权重 ?? x, offset: 0, index: 1)
        编码器.setBuffer(偏置 ?? x, offset: 0, index: 2)
        编码器.setBuffer(out, offset: 0, index: 3)
        编码器.setBytes(&参数, length: MemoryLayout<WanNormParams>.stride, index: 4)
        编码器.dispatchThreadgroups(MTLSize(width: 行, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    private func 调制应用(_ 编码器: MTLComputeCommandEncoder, 输入: MTLBuffer, 调制表: MTLBuffer,
                      out: MTLBuffer, 行: Int, 列: Int, shift索引: Int32, scale索引: Int32) throws {
        var 参数 = WanModParams(count: Int32(行 * 列), cols: Int32(列),
                               shiftIdx: shift索引, scaleIdx: scale索引)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_modulate"))
        编码器.setBuffer(输入, offset: 0, index: 0)
        编码器.setBuffer(调制表, offset: 0, index: 1)
        编码器.setBuffer(out, offset: 0, index: 2)
        编码器.setBytes(&参数, length: MemoryLayout<WanModParams>.stride, index: 3)
        编码器.dispatchThreadgroups(MTLSize(width: (列 + 15) / 16, height: (行 + 15) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
    }

    private func 门控相加(_ 编码器: MTLComputeCommandEncoder, 输出: MTLBuffer, 增项: MTLBuffer,
                     调制表: MTLBuffer?, 行: Int, 列: Int, gate索引: Int32) throws {
        var 参数 = WanGateParams(rows: Int32(行), cols: Int32(列), idx: gate索引)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_gate_add"))
        编码器.setBuffer(输出, offset: 0, index: 0)
        编码器.setBuffer(增项, offset: 0, index: 1)
        编码器.setBuffer(调制表 ?? 增项, offset: 0, index: 2)
        编码器.setBytes(&参数, length: MemoryLayout<WanGateParams>.stride, index: 3)
        编码器.dispatchThreadgroups(MTLSize(width: (列 + 15) / 16, height: (行 + 15) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
    }

    private func 旋转位置(_ 编码器: MTLComputeCommandEncoder, x: MTLBuffer, 令牌数: Int,
                      planeW: Int, planeH: Int, 帧号: Int) throws {
        var 参数 = WanRopeParams(tokens: Int32(令牌数), heads: Int32(Wan形状.头数),
                                dim: Int32(Wan形状.头维), planeW: Int32(planeW), planeH: Int32(planeH),
                                frameIndex: Int32(帧号), theta: Wan形状.theta, pad: 0)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_rope"))
        编码器.setBuffer(x, offset: 0, index: 0)
        编码器.setBytes(&参数, length: MemoryLayout<WanRopeParams>.stride, index: 1)
        编码器.dispatchThreadgroups(MTLSize(width: (Wan形状.头维 / 2 + 7) / 8,
                                         height: (令牌数 + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }

    private func 拷贝(_ 编码器: MTLComputeCommandEncoder, 源: MTLBuffer, 目的: MTLBuffer, 元素数: Int) throws {
        var n = UInt32(元素数)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_copy_half"))
        编码器.setBuffer(源, offset: 0, index: 0)
        编码器.setBuffer(目的, offset: 0, index: 1)
        编码器.setBytes(&n, length: 4, index: 2)
        let 线程 = 256
        编码器.dispatchThreadgroups(MTLSize(width: (元素数 + 线程 - 1) / 线程, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 线程, height: 1, depth: 1))
    }

    private func 转置头(_ 编码器: MTLComputeCommandEncoder, 源: MTLBuffer, 源偏移: Int,
                    目的: MTLBuffer, 目的偏移: Int, 令牌数: Int, 源跨距: Int, 目的跨距: Int) throws {
        var 参数 = WanHeadParams(tokens: Int32(令牌数), dim: Int32(Wan形状.头维),
                              srcStride: Int32(源跨距), dstStride: Int32(目的跨距),
                              srcOffset: Int32(源偏移 / 2), dstOffset: Int32(目的偏移 / 2))
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_head_transpose"))
        编码器.setBuffer(源, offset: 0, index: 0)
        编码器.setBuffer(目的, offset: 0, index: 1)
        编码器.setBytes(&参数, length: MemoryLayout<WanHeadParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: (Wan形状.头维 + 7) / 8, height: (令牌数 + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }

    // MARK: - 条件预计算

    private func 预计算条件(池: 权重池, 文本一: 线性, 文本二: 线性,
                      时间一: 线性, 时间二: 线性, 时间投影: 线性) throws {
        临时池.removeAll()
        临时游标 = 0
        guard let 命令 = 队列.makeCommandBuffer(),
              var 编码器 = 命令.makeComputeCommandEncoder() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }

        let 文本令牌 = Wan形状.文本令牌
        let 文本中间 = 借(文本令牌 * 3072)
        let 文本输出 = 借(文本令牌 * 3072)
        let 提示嵌入 = try 取张量(池, "prompt_embeds", 文本令牌 * 4096)

        try 矩阵乘(&编码器, 命令: 命令, A: 提示嵌入, aOffset: 0, aRowBytes: 4096 * 2,
            行: 文本令牌, 内维: 4096,
            B: 文本一.权重, bOffset: 0, bRowBytes: 4096 * 2, 列: 3072,
            C: 文本中间, cOffset: 0, cRowBytes: 3072 * 2)
        try 偏置激活(编码器, 缓冲: 文本中间, 偏置: 文本一.偏置, 行: 文本令牌, 列: 3072, 激活: 0)
        var 文本元素 = UInt32(文本令牌 * 3072)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_gelu"))
        编码器.setBuffer(文本中间, offset: 0, index: 0)
        编码器.setBytes(&文本元素, length: 4, index: 1)
        编码器.dispatchThreadgroups(MTLSize(width: (文本令牌 * 3072 + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        try 矩阵乘(&编码器, 命令: 命令, A: 文本中间, aOffset: 0, aRowBytes: 3072 * 2,
            行: 文本令牌, 内维: 3072,
            B: 文本二.权重, bOffset: 0, bRowBytes: 3072 * 2, 列: 3072,
            C: 文本输出, cOffset: 0, cRowBytes: 3072 * 2)
        try 偏置激活(编码器, 缓冲: 文本输出, 偏置: 文本二.偏置, 行: 文本令牌, 列: 3072, 激活: 0)

        let 键缓冲 = 借(文本令牌 * 3072)
        let 值缓冲 = 借(文本令牌 * 3072)
        for 块 in 块表 {
            try 矩阵乘(&编码器, 命令: 命令, A: 文本输出, aOffset: 0, aRowBytes: 3072 * 2,
                行: 文本令牌, 内维: 3072,
                B: 块.交叉键.权重, bOffset: 0, bRowBytes: 3072 * 2, 列: 3072,
                C: 键缓冲, cOffset: 0, cRowBytes: 3072 * 2)
            try 偏置激活(编码器, 缓冲: 键缓冲, 偏置: 块.交叉键.偏置, 行: 文本令牌, 列: 3072, 激活: 0)
            try 矩阵乘(&编码器, 命令: 命令, A: 文本输出, aOffset: 0, aRowBytes: 3072 * 2,
                行: 文本令牌, 内维: 3072,
                B: 块.交叉值.权重, bOffset: 0, bRowBytes: 3072 * 2, 列: 3072,
                C: 值缓冲, cOffset: 0, cRowBytes: 3072 * 2)
            try 偏置激活(编码器, 缓冲: 值缓冲, 偏置: 块.交叉值.偏置, 行: 文本令牌, 列: 3072, 激活: 0)
            try 归一化行(编码器, x: 键缓冲, 行: 文本令牌, 列: 3072, out: 块.交叉K,
                     权重: 块.交叉键归一化, eps: Wan形状.归一化eps)
            try 拷贝(编码器, 源: 值缓冲, 目的: 块.交叉V, 元素数: 文本令牌 * 3072)
        }

        // timestep = 1000 → 正弦嵌入 → temb → timestep_proj
        var 正弦 = [Float](repeating: 0, count: Wan形状.时间频率)
        let 半频 = Wan形状.时间频率 / 2
        for i in 0..<半频 {
            let 角 = exp(-log(Float(10000)) * Float(i) / Float(半频))
            let v = 1000.0 * 角
            // flip_sin_to_cos=True：cos 在前、sin 在后
            正弦[i] = cos(v)
            正弦[半频 + i] = sin(v)
        }
        let 正弦半 = 借(Wan形状.时间频率)
        let 正弦指针 = 正弦半.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<Wan形状.时间频率 { 正弦指针[i] = Float16(正弦[i]) }

        let 时间中间 = 借(3072)
        let 时间输出 = 借(3072)
        try 矩阵乘(&编码器, 命令: 命令, A: 正弦半, aOffset: 0, aRowBytes: Wan形状.时间频率 * 2,
            行: 1, 内维: Wan形状.时间频率,
            B: 时间一.权重, bOffset: 0, bRowBytes: Wan形状.时间频率 * 2, 列: 3072,
            C: 时间中间, cOffset: 0, cRowBytes: 3072 * 2)
        try 偏置激活(编码器, 缓冲: 时间中间, 偏置: 时间一.偏置, 行: 1, 列: 3072, 激活: 1)
        try 矩阵乘(&编码器, 命令: 命令, A: 时间中间, aOffset: 0, aRowBytes: 3072 * 2,
            行: 1, 内维: 3072,
            B: 时间二.权重, bOffset: 0, bRowBytes: 3072 * 2, 列: 3072,
            C: 时间输出, cOffset: 0, cRowBytes: 3072 * 2)
        try 偏置激活(编码器, 缓冲: 时间输出, 偏置: 时间二.偏置, 行: 1, 列: 3072, 激活: 0)

        let 激励 = 借(3072)
        try 拷贝(编码器, 源: 时间输出, 目的: 激励, 元素数: 3072)
        try 偏置激活(编码器, 缓冲: 激励, 偏置: nil, 行: 1, 列: 3072, 激活: 1)
        let 时间调 = 借(3072 * 6)
        try 矩阵乘(&编码器, 命令: 命令, A: 激励, aOffset: 0, aRowBytes: 3072 * 2,
            行: 1, 内维: 3072,
            B: 时间投影.权重, bOffset: 0, bRowBytes: 3072 * 2, 列: 3072 * 6,
            C: 时间调, cOffset: 0, cRowBytes: 3072 * 6 * 2)
        try 偏置激活(编码器, 缓冲: 时间调, 偏置: 时间投影.偏置, 行: 1, 列: 3072 * 6, 激活: 0)

        编码器.endEncoding()
        命令.commit()
        命令.waitUntilCompleted()
        if let e = 命令.error { throw 错误.执行失败("条件预计算：\(e.localizedDescription)") }

        let 时间嵌入 = 读半精度缓冲(时间输出, 3072)
        let 时间调制 = 读半精度缓冲(时间调, 3072 * 6)
        let 全局表 = try 读半精度(池, "scale_shift_table", 2 * 3072)
        let 全局指针 = 输出调制.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<3072 {
            全局指针[i] = 全局表[i] + 时间嵌入[i]
            全局指针[3072 + i] = 全局表[3072 + i] + 时间嵌入[i]
        }
        for (索引, 块) in 块表.enumerated() {
            let 表 = try 读半精度(池, "blocks.\(索引).scale_shift_table", 6 * 3072)
            let 指针 = 块.调制.contents().assumingMemoryBound(to: Float.self)
            for k in 0..<(6 * 3072) { 指针[k] = 表[k] + 时间调制[k] }
        }
        临时池.removeAll()
        临时游标 = 0
    }

    // MARK: - 前向

    /// 单帧前向。`潜变量` half [48, 潜高, 潜宽]；`低清` planar RGB uchar [3, 源高, 源宽]。
    func 前向(潜变量: MTLBuffer, 低清: MTLBuffer, 源高: Int, 源宽: Int) throws -> MTLBuffer {
        // 官方口径（models/wan/lq_proj.py + runtime/streaming.py）：
        //   低清 native --F.interpolate(scale=4)--> LQ4x(4×低清，等于输出分辨率)
        //   LQ4x --PixelUnshuffle(16)--> latent = LQ4x/16 = 源/4
        //   latent --conv2(k3,s2,p1)--> token = latent/2 = 源/8
        // 权重佐证：lq_proj.conv1.weight [768,768,1,1] 的输入 768 = 3×16×16（16× unshuffle），
        // patch_embedding.weight [3072,48,1,2,2] 的 48/2×2 给出 latent 通道与 patch 尺寸。
        令牌高 = 源高 / 8
        令牌宽 = 源宽 / 8
        潜高 = 源高 / 4
        潜宽 = 源宽 / 4
        let 令牌数 = 令牌高 * 令牌宽

        临时游标 = 0
        guard let 首命令 = 队列.makeCommandBuffer(),
              let 首编码器 = 首命令.makeComputeCommandEncoder() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }
        var 命令 = 首命令
        var 编码器 = 首编码器

        // 低清分支：lq_prepare 输出按 latent 像素行主序（潜高 * 潜宽 行，768 列）
        let 潜像素数 = 潜高 * 潜宽
        let 低清特征 = 借(潜像素数 * 768)
        // dstH/dstW = 4× 上采样目标尺寸（LQ4x），内核据此做 align_corners=False 双线性取样
        var lq参数 = WanLQParams(srcH: Int32(源高), srcW: Int32(源宽),
                               dstH: Int32(源高 * 4), dstW: Int32(源宽 * 4),
                               outW: Int32(潜宽), outH: Int32(潜高), pad0: 0, pad1: 0)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_lq_prepare"))
        编码器.setBuffer(低清, offset: 0, index: 0)
        编码器.setBuffer(低清特征, offset: 0, index: 1)
        编码器.setBytes(&lq参数, length: MemoryLayout<WanLQParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: (潜宽 + 3) / 4, height: (潜高 + 3) / 4, depth: 768),
                                 threadsPerThreadgroup: MTLSize(width: 4, height: 4, depth: 1))

        if 诊断阶段 == 1 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            统计("低清特征", 低清特征, 潜像素数 * 768)
            return 低清特征
        }
        let 卷积1输出 = 借(潜像素数 * 768)
        try 线性前进(&编码器, 命令: 命令, 低清卷积1, x: 低清特征, xRowBytes: 768 * 2,
                 行: 潜像素数, out: 卷积1输出, outRowBytes: 768 * 2, 激活: 0)
        let 归一1 = 借(潜像素数 * 768)
        try 归一化行(编码器, x: 卷积1输出, 行: 潜像素数, 列: 768, out: 归一1,
                 权重: 低清归一化1, eps: Wan形状.归一化eps)
        try 偏置激活(编码器, 缓冲: 归一1, 偏置: nil, 行: 潜像素数, 列: 768, 激活: 1)

        let 展开 = 借(令牌数 * 768 * 9)
        var col参数 = WanColParams(cols: 768, kOut: 768 * 9,
                                 H: Int32(潜高), W: Int32(潜宽),
                                 OH: Int32(令牌高), OW: Int32(令牌宽),
                                 stride: 2, pad: 1)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_im2col3"))
        编码器.setBuffer(归一1, offset: 0, index: 0)
        编码器.setBuffer(展开, offset: 0, index: 1)
        编码器.setBytes(&col参数, length: MemoryLayout<WanColParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: (768 * 9 + 255) / 256, height: (令牌数 + 3) / 4, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 4, depth: 1))

        let 卷积2输出 = 借(令牌数 * 3072)
        try 线性前进(&编码器, 命令: 命令, 低清卷积2, x: 展开, xRowBytes: 768 * 9 * 2,
                 行: 令牌数, out: 卷积2输出, outRowBytes: 3072 * 2, 激活: 0)
        let 归一2 = 借(令牌数 * 3072)
        try 归一化行(编码器, x: 卷积2输出, 行: 令牌数, 列: 3072, out: 归一2,
                 权重: 低清归一化2, eps: Wan形状.归一化eps)
        try 偏置激活(编码器, 缓冲: 归一2, 偏置: nil, 行: 令牌数, 列: 3072, 激活: 1)
        let 低清令牌 = 借(令牌数 * 3072)
        try 线性前进(&编码器, 命令: 命令, 低清输出, x: 归一2, xRowBytes: 3072 * 2,
                 行: 令牌数, out: 低清令牌, outRowBytes: 3072 * 2, 激活: 0)

        if 诊断阶段 == 2 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            统计("归一2", 归一2, 令牌数 * 3072)
            统计("低清令牌", 低清令牌, 令牌数 * 3072)
            if let 目录 = 导出目录 {
                try? Data(bytes: 低清令牌.contents(), count: 令牌数 * 3072 * 2)
                    .write(to: URL(fileURLWithPath: 目录 + "/lqtoken2.bin"))
            }
            return 低清令牌
        }
        // patch 分支
        let 展开补丁 = 借(令牌数 * 192)
        var 补丁参数 = WanPatchUnfoldParams(patchW: Int32(令牌宽), patchH: Int32(令牌高),
                                       C: 48, H: Int32(潜高), W: Int32(潜宽))
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_patch_unfold"))
        编码器.setBuffer(潜变量, offset: 0, index: 0)
        编码器.setBuffer(展开补丁, offset: 0, index: 1)
        编码器.setBytes(&补丁参数, length: MemoryLayout<WanPatchUnfoldParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: 2, height: (令牌数 + 7) / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 96, height: 8, depth: 1))

        let 隐藏 = 借(令牌数 * 3072)
        try 线性前进(&编码器, 命令: 命令, 补丁权重, x: 展开补丁, xRowBytes: 192 * 2,
                 行: 令牌数, out: 隐藏, outRowBytes: 3072 * 2, 激活: 0)
        if 诊断阶段 == 98 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            if let 目录 = 导出目录 {
                try? Data(bytes: 隐藏.contents(), count: 令牌数 * 3072 * 2)
                    .write(to: URL(fileURLWithPath: 目录 + "/patch_sw.bin"))
            }
        }
        try 门控相加(编码器, 输出: 隐藏, 增项: 低清令牌, 调制表: nil, 行: 令牌数, 列: 3072, gate索引: -1)

        if 诊断阶段 == 3 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            统计("展开补丁", 展开补丁, 令牌数 * 192)
            统计("隐藏(嵌入)", 隐藏, 令牌数 * 3072)
            if let 目录 = 导出目录 {
                try? Data(bytes: 隐藏.contents(), count: 令牌数 * 3072 * 2)
                    .write(to: URL(fileURLWithPath: 目录 + "/embed.bin"))
                    try? Data(bytes: 低清令牌.contents(), count: 令牌数 * 3072 * 2)
                    .write(to: URL(fileURLWithPath: 目录 + "/lqtoken.bin"))
            }
            return 隐藏
        }
        // 缓冲区
        let 归一缓冲 = 借(令牌数 * 3072)
        let 调制缓冲 = 借(令牌数 * 3072)
        let 查询 = 借(令牌数 * 3072)
        let 键 = 借(令牌数 * 3072)
        let 值 = 借(令牌数 * 3072)
        let 注意力 = 借(令牌数 * 3072)
        let 头转置 = 借(令牌数 * 128 * Wan形状.头数)
        let 合并 = 借(令牌数 * 3072)
        let 前馈中间 = 借(令牌数 * 14336)
        let 前馈输出 = 借(令牌数 * 3072)
        let 交叉查询 = 借(令牌数 * 3072)
        let 交叉输出 = 借(令牌数 * 3072)

        let 键数 = 令牌数
        let 键跨距 = ((键数 + 7) / 8) * 8
        let 分数 = 借(令牌数 * 键跨距)
        let 概率 = 借(令牌数 * 键跨距)

        let 交叉键数 = Wan形状.文本令牌
        let 交叉分数 = 借(令牌数 * 交叉键数)
        let 交叉概率 = 借(令牌数 * 交叉键数)

        let 空间窗高 = min(22, 令牌高)
        let 空间窗宽 = min(40, 令牌宽)

        // 合并提交状态：每批 = 预算/单层耗时 层；提交后不立即等待，
        // 保留最多 2 个在飞命令缓冲，令「CPU 编码下一批」与「GPU 执行上一批」重叠。
        var 批内层数 = 0
        var 在飞: [(缓冲: MTLCommandBuffer, 层数: Int)] = []
        let 在飞上限 = 2
        let 本批层数上限 = max(1, min(块表.count,
                                  单层毫秒估计 > 0 ? Int((单批GPU预算毫秒 / 单层毫秒估计).rounded()) : 3))
        for (层序, 层) in 块表.enumerated() {
            // 自注意力
            try 层归一化(编码器, x: 隐藏, 行: 令牌数, 列: 3072, out: 归一缓冲,
                    权重: nil, 偏置: nil, eps: Wan形状.归一化eps)
            try 调制应用(编码器, 输入: 归一缓冲, 调制表: 层.调制, out: 调制缓冲,
                    行: 令牌数, 列: 3072, shift索引: 0, scale索引: 1)
            if 层序 == 0 && 诊断阶段 == 4 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层0归一", 归一缓冲, 令牌数 * 3072)
                统计("层0调制", 调制缓冲, 令牌数 * 3072)
                if let d = 导出目录 {
                    try? Data(bytes: 归一缓冲.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_norm1.bin"))
                    try? Data(bytes: 调制缓冲.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_mod1.bin"))
                }
                return 调制缓冲
            }
            try 线性前进(&编码器, 命令: 命令, 层.查询, x: 调制缓冲, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 查询, outRowBytes: 3072 * 2, 激活: 0)
            try 线性前进(&编码器, 命令: 命令, 层.键, x: 调制缓冲, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 键, outRowBytes: 3072 * 2, 激活: 0)
            try 线性前进(&编码器, 命令: 命令, 层.值, x: 调制缓冲, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 值, outRowBytes: 3072 * 2, 激活: 0)
            try 归一化行(编码器, x: 查询, 行: 令牌数, 列: 3072, out: 查询,
                     权重: 层.查询归一化, eps: Wan形状.归一化eps)
            try 归一化行(编码器, x: 键, 行: 令牌数, 列: 3072, out: 键,
                     权重: 层.键归一化, eps: Wan形状.归一化eps)
            try 旋转位置(编码器, x: 查询, 令牌数: 令牌数, planeW: 令牌宽, planeH: 令牌高, 帧号: 0)
            try 旋转位置(编码器, x: 键, 令牌数: 令牌数, planeW: 令牌宽, planeH: 令牌高, 帧号: 0)

            if 层序 == 0 && 诊断阶段 == 5 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层0查询", 查询, 令牌数 * 3072)
                统计("层0键", 键, 令牌数 * 3072)
                统计("层0值", 值, 令牌数 * 3072)
                if let d = 导出目录 {
                    try? Data(bytes: 查询.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_q.bin"))
                    try? Data(bytes: 键.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_k.bin"))
                    try? Data(bytes: 值.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_v.bin"))
                }
                return 查询
            }
            for 头 in 0..<Wan形状.头数 {
                let 偏移 = 头 * Wan形状.头维 * 2
                let 目标 = 头 * 令牌数 * Wan形状.头维 * 2
                try 矩阵乘(&编码器, 命令: 命令, A: 查询, aOffset: 偏移, aRowBytes: 3072 * 2,
                        行: 令牌数, 内维: Wan形状.头维,
                        B: 键, bOffset: 偏移, bRowBytes: 3072 * 2, 列: 令牌数,
                        C: 分数, cOffset: 0, cRowBytes: 键跨距 * 2)
                // 官方 wan.safetensors config：spatial_window_size=[22,40]、spatial_attention_mode=local，
                // 即 query 只与中心对齐的 22×40 矩形窗口内 key 做注意力（models/wan/spatial_attention.py）。
                // 时间轴单帧时 window=1，key 帧范围 [qt, qt]。
                var 掩码 = WanMaskParams(span: Int32(令牌数), H: Int32(令牌高), W: Int32(令牌宽),
                                       wh: Int32(空间窗高), ww: Int32(空间窗宽), window: 1,
                                       qOffset: 0, kv: Int32(键数), kvStride: Int32(键跨距),
                                       scale: Wan形状.逐头缩放, pad0: 0, pad1: 0)
                编码器.memoryBarrier(scope: .buffers)
                编码器.setComputePipelineState(try 管("wan_mask_softmax"))
                编码器.setBuffer(分数, offset: 0, index: 0)
                编码器.setBuffer(概率, offset: 0, index: 1)
                编码器.setBytes(&掩码, length: MemoryLayout<WanMaskParams>.stride, index: 2)
                编码器.dispatchThreadgroups(MTLSize(width: 令牌数, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                if Wan形状.直取值矩阵 {
                    // V 已是 [令牌][3072] 行主序，MPS 直接按 B=[K][N] 取，省掉逐头转置。
                    try 矩阵乘(&编码器, 命令: 命令, A: 概率, aOffset: 0, aRowBytes: 键跨距 * 2,
                        行: 令牌数, 内维: 键数,
                        B: 值, bOffset: 偏移, bRowBytes: 3072 * 2, 列: 128,
                        C: 合并, cOffset: 偏移, cRowBytes: 3072 * 2, 转置B: false)
                } else {
                    try 转置头(编码器, 源: 值, 源偏移: 偏移, 目的: 头转置, 目的偏移: 目标,
                            令牌数: 令牌数, 源跨距: 3072, 目的跨距: 令牌数)
                    try 矩阵乘(&编码器, 命令: 命令, A: 概率, aOffset: 0, aRowBytes: 键跨距 * 2,
                        行: 令牌数, 内维: 键数,
                        B: 头转置, bOffset: 目标, bRowBytes: 令牌数 * 2, 列: 128,
                        C: 合并, cOffset: 偏移, cRowBytes: 3072 * 2)
                }
            }
            try 线性前进(&编码器, 命令: 命令, 层.输出, x: 合并, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 注意力, outRowBytes: 3072 * 2, 激活: 0)
            try 门控相加(编码器, 输出: 隐藏, 增项: 注意力, 调制表: 层.调制,
                     行: 令牌数, 列: 3072, gate索引: 2)

            if 层序 == 0 && 诊断阶段 == 6 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层0分数", 分数, 令牌数 * 键跨距)
                统计("层0概率", 概率, 令牌数 * 键跨距)
                统计("层0合并", 合并, 令牌数 * 3072)
                统计("层0注意力", 注意力, 令牌数 * 3072)
                if let d = 导出目录 {
                    try? Data(bytes: 合并.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_merge.bin"))
                    try? Data(bytes: 注意力.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_attnout.bin"))
                }
                return 注意力
            }
            // 交叉注意力
            try 层归一化(编码器, x: 隐藏, 行: 令牌数, 列: 3072, out: 归一缓冲,
                    权重: 层.交叉归一化w, 偏置: 层.交叉归一化b, eps: Wan形状.归一化eps)
            try 线性前进(&编码器, 命令: 命令, 层.交叉查询, x: 归一缓冲, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 交叉查询, outRowBytes: 3072 * 2, 激活: 0)
            try 归一化行(编码器, x: 交叉查询, 行: 令牌数, 列: 3072, out: 交叉查询,
                     权重: 层.交叉查询归一化, eps: Wan形状.归一化eps)
            if 层序 == 0 && 诊断阶段 == 77 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                if let d = 导出目录 {
                    try? Data(bytes: 交叉查询.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_cq.bin"))
                    try? Data(bytes: 层.交叉K.contents(), count: 交叉键数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_ck.bin"))
                    try? Data(bytes: 隐藏.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_hidden.bin"))
                    try? Data(bytes: 归一缓冲.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_n2.bin"))
                    try? Data(bytes: 层.调制.contents(), count: 6 * 3072 * 4).write(to: URL(fileURLWithPath: d + "/s_mod.全.bin"))
                }
                return 交叉输出
            }
            for 头 in 0..<Wan形状.头数 {
                let 偏移 = 头 * Wan形状.头维 * 2
                let 目标 = 头 * 交叉键数 * 128 * 2
                try 矩阵乘(&编码器, 命令: 命令, A: 交叉查询, aOffset: 偏移, aRowBytes: 3072 * 2,
                        行: 令牌数, 内维: 128,
                        B: 层.交叉K, bOffset: 偏移, bRowBytes: 3072 * 2, 列: 交叉键数,
                        C: 交叉分数, cOffset: 0, cRowBytes: 交叉键数 * 2)
                var 掩码 = WanMaskParams(span: Int32(令牌数), H: Int32(令牌高), W: Int32(令牌宽),
                                       wh: 0, ww: 0, window: 0, qOffset: 0,
                                       kv: Int32(交叉键数), kvStride: Int32(交叉键数),
                                       scale: Wan形状.逐头缩放, pad0: 0, pad1: 0)
                编码器.memoryBarrier(scope: .buffers)
                编码器.setComputePipelineState(try 管("wan_softmax_dense"))
                编码器.setBuffer(交叉分数, offset: 0, index: 0)
                编码器.setBuffer(交叉概率, offset: 0, index: 1)
                编码器.setBytes(&掩码, length: MemoryLayout<WanMaskParams>.stride, index: 2)
                编码器.dispatchThreadgroups(MTLSize(width: 令牌数, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                if Wan形状.直取值矩阵 {
                    try 矩阵乘(&编码器, 命令: 命令, A: 交叉概率, aOffset: 0, aRowBytes: 交叉键数 * 2,
                        行: 令牌数, 内维: 交叉键数,
                        B: 层.交叉V, bOffset: 偏移, bRowBytes: 3072 * 2, 列: 128,
                        C: 合并, cOffset: 偏移, cRowBytes: 3072 * 2, 转置B: false)
                } else {
                    try 转置头(编码器, 源: 层.交叉V, 源偏移: 偏移, 目的: 头转置, 目的偏移: 目标,
                            令牌数: 交叉键数, 源跨距: 3072, 目的跨距: 交叉键数)
                    try 矩阵乘(&编码器, 命令: 命令, A: 交叉概率, aOffset: 0, aRowBytes: 交叉键数 * 2,
                        行: 令牌数, 内维: 交叉键数,
                        B: 头转置, bOffset: 目标, bRowBytes: 交叉键数 * 2, 列: 128,
                        C: 合并, cOffset: 偏移, cRowBytes: 3072 * 2)
                }
            }
            try 线性前进(&编码器, 命令: 命令, 层.交叉输出, x: 合并, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 交叉输出, outRowBytes: 3072 * 2, 激活: 0)
            try 门控相加(编码器, 输出: 隐藏, 增项: 交叉输出, 调制表: nil,
                     行: 令牌数, 列: 3072, gate索引: -1)

            if 层序 == 0 && 诊断阶段 == 7 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层0交叉分数", 交叉分数, 令牌数 * 交叉键数)
                统计("层0交叉概率", 交叉概率, 令牌数 * 交叉键数)
                统计("层0交叉输出", 交叉输出, 令牌数 * 3072)
                if let d = 导出目录 {
                    try? Data(bytes: 交叉输出.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_cross.bin"))
                }
                return 交叉输出
            }
            // 前馈
            try 层归一化(编码器, x: 隐藏, 行: 令牌数, 列: 3072, out: 归一缓冲,
                    权重: nil, 偏置: nil, eps: Wan形状.归一化eps)
            try 调制应用(编码器, 输入: 归一缓冲, 调制表: 层.调制, out: 调制缓冲,
                    行: 令牌数, 列: 3072, shift索引: 3, scale索引: 4)
            try 线性前进(&编码器, 命令: 命令, 层.前馈入口, x: 调制缓冲, xRowBytes: 3072 * 2,
                     行: 令牌数, out: 前馈中间, outRowBytes: 14336 * 2, 激活: 0)
            if 层序 == 1 && 诊断阶段 == 206 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1前馈入口GEMM原始", 前馈中间, 令牌数 * 14336)
                return 前馈中间
            }
            if 层序 == 1 && 诊断阶段 == 207 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                let 半 = 前馈中间.contents().bindMemory(to: Float16.self, capacity: 令牌数 * 14336)
                var 非有限 = 0
                var 记录: [Int] = []
                for i in 0..<(令牌数 * 14336) {
                    let v = Float(半[i])
                    if !v.isFinite { 非有限 += 1; if 记录.count < 5 { 记录.append(i) } }
                }
                print("GELU输入非有限 \(非有限) 样例 \(记录)")
                if let d = 导出目录 {
                    try? Data(bytes: 前馈中间.contents(), count: 令牌数 * 14336 * 2).write(to: URL(fileURLWithPath: d + "/s_ffpre.bin"))
                }
                return 前馈中间
            }
            var n = UInt32(令牌数 * 14336)
            编码器.memoryBarrier(scope: .buffers)
            编码器.setComputePipelineState(try 管("wan_gelu"))
            编码器.setBuffer(前馈中间, offset: 0, index: 0)
            编码器.setBytes(&n, length: 4, index: 1)
            编码器.dispatchThreadgroups(MTLSize(width: (Int(n) + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if 层序 == 1 && 诊断阶段 == 208 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                if let d = 导出目录 {
                    try? Data(bytes: 前馈中间.contents(), count: 令牌数 * 14336 * 2).write(to: URL(fileURLWithPath: d + "/s_ffgelu2.bin"))
                }
                return 前馈中间
            }
            try 线性前进(&编码器, 命令: 命令, 层.前馈出口, x: 前馈中间, xRowBytes: 14336 * 2,
                     行: 令牌数, out: 前馈输出, outRowBytes: 3072 * 2, 激活: 0)
            try 门控相加(编码器, 输出: 隐藏, 增项: 前馈输出, 调制表: 层.调制,
                     行: 令牌数, 列: 3072, gate索引: 5)

            if 层序 == 1 && 诊断阶段 == 200 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1归一", 归一缓冲, 令牌数 * 3072)
                统计("层1调制", 调制缓冲, 令牌数 * 3072)
                return 调制缓冲
            }
            if 层序 == 1 && 诊断阶段 == 202 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1查询", 查询, 令牌数 * 3072)
                统计("层1概率", 概率, 令牌数 * 键跨距)
                统计("层1注意力", 注意力, 令牌数 * 3072)
                return 注意力
            }
            if 层序 == 1 && 诊断阶段 == 203 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1交叉输出", 交叉输出, 令牌数 * 3072)
                return 交叉输出
            }
            if 层序 == 1 && 诊断阶段 == 205 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1调制全", 调制缓冲, 令牌数 * 3072)
                统计("层1前馈入口权重", 层.前馈入口.权重, 14336 * 3072)
                return 前馈中间
            }
            if 层序 == 1 && 诊断阶段 == 204 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层1前馈中间", 前馈中间, 令牌数 * 14336)
                统计("层1前馈输出", 前馈输出, 令牌数 * 3072)
                if let d = 导出目录 {
                    try? Data(bytes: 前馈中间.contents(), count: 令牌数 * 14336 * 2).write(to: URL(fileURLWithPath: d + "/s_ffgelu.bin"))
                    try? Data(bytes: 前馈输出.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/s_ffout.bin"))
                }
                return 前馈输出
            }
            if 诊断阶段 >= 100 && 层序 == 诊断阶段 - 100 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层\(层序)隐藏", 隐藏, 令牌数 * 3072)
                if let d = 导出目录 { try? Data(bytes: 隐藏.contents(), count: 令牌数 * 3072 * 2).write(to: URL(fileURLWithPath: d + "/h_\(层序).bin")) }
                if let 目录 = 导出目录 {
                    let 字节 = Data(bytes: 隐藏.contents(), count: 令牌数 * 3072 * 2)
                    try? 字节.write(to: URL(fileURLWithPath: 目录 + "/blk_\(层序).bin"))
                }
                return 隐藏
            }
            if 层序 == 0 && 诊断阶段 == 8 {
                编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
                统计("层0前馈中间", 前馈中间, 令牌数 * 14336)
                统计("层0前馈输出", 前馈输出, 令牌数 * 3072)
                统计("层0隐藏", 隐藏, 令牌数 * 3072)
                return 前馈输出
            }

            // 跨层合并提交：原先每层 commit+wait，30 处同步且 CPU 编码与 GPU 执行完全串行。
            // 现在按预算凑批提交，且提交后不等待，最多 2 个命令缓冲在飞：
            // CPU 编码下一批的同时 GPU 在执行上一批，单个命令缓冲的 GPU 时长仍在预算内。
            批内层数 += 1
            if 批内层数 >= 本批层数上限 || 层序 == 块表.count - 1 {
                编码器.endEncoding()
                命令.commit()
                在飞.append((命令, 批内层数))
                批内层数 = 0
                if 在飞.count > 在飞上限 {
                    let 旧 = 在飞.removeFirst()

                    if let e = 旧.缓冲.error { throw 错误.执行失败(e.localizedDescription) }
                    let 本批毫秒 = (旧.缓冲.gpuEndTime - 旧.缓冲.gpuStartTime) * 1000
                    if 本批毫秒 > 0, 旧.层数 > 0 { 单层毫秒估计 = 本批毫秒 / Double(旧.层数) }
                }
                guard let 下个命令 = 队列.makeCommandBuffer(),
                      let 下个编码器 = 下个命令.makeComputeCommandEncoder() else {
                    throw 错误.初始化失败("命令缓冲创建失败")
                }
                命令 = 下个命令
                编码器 = 下个编码器
            } else {
                // 同批内跨层：沿用同一编码器继续编码，跨层写读之间补显式屏障
                编码器.memoryBarrier(scope: .buffers)
            }
        }
        // 层循环结束：等待所有在飞命令缓冲完成，后续尾部算子读取其输出
        for 项 in 在飞 {
            if let e = 项.缓冲.error { throw 错误.执行失败(e.localizedDescription) }
            let 本批毫秒 = (项.缓冲.gpuEndTime - 项.缓冲.gpuStartTime) * 1000
            if 本批毫秒 > 0, 项.层数 > 0 { 单层毫秒估计 = 本批毫秒 / Double(项.层数) }
        }
        在飞.removeAll()

        if 诊断阶段 == 9 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            统计("末层隐藏", 隐藏, 令牌数 * 3072)
            return 隐藏
        }

        // 输出
        let 输出归一 = 借(令牌数 * 3072)
        let 输出调制后 = 借(令牌数 * 3072)
        let 未分块 = 借(令牌数 * 192)
        try 层归一化(编码器, x: 隐藏, 行: 令牌数, 列: 3072, out: 输出归一,
                权重: nil, 偏置: nil, eps: Wan形状.归一化eps)
        try 调制应用(编码器, 输入: 输出归一, 调制表: 输出调制, out: 输出调制后,
                行: 令牌数, 列: 3072, shift索引: 0, scale索引: 1)
        try 线性前进(&编码器, 命令: 命令, 输出投影, x: 输出调制后, xRowBytes: 3072 * 2,
                 行: 令牌数, out: 未分块, outRowBytes: 192 * 2, 激活: 0)

        if 诊断阶段 == 10 {
            编码器.endEncoding(); 命令.commit(); 命令.waitUntilCompleted()
            统计("输出调制后", 输出调制后, 令牌数 * 3072)
            统计("未分块", 未分块, 令牌数 * 192)
            return 未分块
        }
        let 预测 = 新建半精度缓冲(48 * 潜高 * 潜宽, "预测")
        var 反补丁 = WanUnpatchParams(ph: Int32(令牌高), pw: Int32(令牌宽), C: 48,
                                 planeW: Int32(潜宽), planeH: Int32(潜高), pad0: 0, pad1: 0, pad2: 0)
        编码器.memoryBarrier(scope: .buffers)
        编码器.setComputePipelineState(try 管("wan_unpatch"))
        编码器.setBuffer(未分块, offset: 0, index: 0)
        编码器.setBuffer(预测, offset: 0, index: 1)
        编码器.setBytes(&反补丁, length: MemoryLayout<WanUnpatchParams>.stride, index: 2)
        编码器.dispatchThreadgroups(MTLSize(width: (潜宽 + 7) / 8, height: (潜高 + 7) / 8, depth: 48),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))

        编码器.endEncoding()
        命令.commit()
        命令.waitUntilCompleted()
        if let e = 命令.error { throw 错误.执行失败(e.localizedDescription) }

        if 诊断采集, let 回调 = 诊断回调 {
            回调("预测", [48, 潜高, 潜宽], 读半精度缓冲(预测, 48 * 潜高 * 潜宽))
        }
        return 预测
    }
}
