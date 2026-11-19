import Foundation
import Metal
import MetalPerformanceShaders

/// Flash 条件主干的配置。
///
/// 超参来自 flash.safetensors 的元数据 `config`，另有变体表（这里只支持 S 档）。
nonisolated struct Flash配置: Decodable {

    struct 空间注意力: Decodable {
        let 视频窗: [Int]
        let 起始条件窗: [Int]
        let 结束条件窗: [Int]
        let 条件自窗: [Int]

        enum CodingKeys: String, CodingKey {
            case 视频窗 = "video_window_size"
            case 起始条件窗 = "start_condition_window_size"
            case 结束条件窗 = "end_condition_window_size"
            case 条件自窗 = "condition_self_window_size"
        }
    }

    let 条件通道数: Int
    let 编码通道: [Int]
    let 归一化Eps: Double
    let RoPETheta: Double
    let 空间头维度: Int
    let 放大倍数: Int
    let 窗口帧数: Int
    let 空间注意力: 空间注意力

    enum CodingKeys: String, CodingKey {
        case 条件通道数 = "condition_channels"
        case 编码通道 = "encoder_channels"
        case 归一化Eps = "norm_eps"
        case RoPETheta = "rope_theta"
        case 空间头维度 = "spatial_head_dim"
        case 放大倍数 = "upscale_factor"
        case 窗口帧数 = "window_size"
        case 空间注意力 = "spatial_attention"
    }
}

/// S 档变体：depth 10 / dim 512 / 8 头 / 2 组 KV
nonisolated struct Flash主干形状 {
    let 深度 = 10
    let 维度 = 512
    let 头数 = 8
    let KV组数 = 2
    let 头维 = 64
    let 隐藏维 = 1408          // dim * mlp_ratio(2.666) 取整
    let 块分辨率 = 4           // encoder_downsample
    let 重排倍数 = 16          // pixel_shuffle_factor
}

nonisolated final class Flash引擎 {

    // MARK: - 错误

    enum 错误: LocalizedError {
        case 初始化失败(String)
        case 尺寸不符(String)

        var errorDescription: String? {
            switch self {
            case .初始化失败(let 说明): "Flash 引擎初始化失败：\(说明)"
            case .尺寸不符(let 说明): "Flash 引擎张量尺寸不符：\(说明)"
            }
        }
    }

    // MARK: - 权重容器

    struct 线性层 {
        let 权重: MTLBuffer
        let 偏置: MTLBuffer?
        let 输出维: Int
        let 输入维: Int
        /// 权重在内存中以 fp16 常驻、GEMM 走 MPS half 路径（f32 仅用于必须的累加与归一化）
        /// 必填，避免 f32 权重被误当 fp16 读取
        let 半精度: Bool
    }

    struct 注意力权重 {
        let 查询: 线性层
        let 键: 线性层
        let 值: 线性层
        let 条件键: 线性层
        let 条件值: 线性层
        let 键归一化: MTLBuffer
        let 值归一化: MTLBuffer
        let 输出: 线性层
    }

    struct 块权重 {
        let 注意力: 注意力权重
        let 注意力归一化: MTLBuffer
        let 前馈门: 线性层     // dim -> 2*hidden
        let 前馈降维: 线性层   // hidden -> dim
        let 前馈归一化: MTLBuffer
    }

    private var 设备: MTLDevice!
    private var 队列: MTLCommandQueue!
    private var 配置: Flash配置!
    let 形状 = Flash主干形状()
    private var 权重文件: SafeTensors权重库!

    private var 库: MTLLibrary!
    private let 上下文: MTLCommandBuffer? = nil

    // 管线
    private var 管线im2col: MTLComputePipelineState!
    private var 管线偏置激活: MTLComputePipelineState!
    private var 管线偏置激活V4: MTLComputePipelineState!
    private var 管线SwiGLUV4: MTLComputePipelineState!
    private var 管线归一化: MTLComputePipelineState!
    private var 管线RoPE: MTLComputePipelineState!
    /// 流式 KV 半精度路径（官方 KV cache 即 bf16，f32 常驻会多占一倍显存）
    private var 管线RoPE半: MTLComputePipelineState!
    private var 管线注意力打包半: MTLComputePipelineState!
    private var 管线转换半: MTLComputePipelineState!
    /// fp16 → f32 回写（fp16 GEMM 输出接 f32 逐元素算子）
    private var 管线转换单: MTLComputePipelineState!
    /// 唯一注意力内核：64 维头、simdgroup MMA 版（每组 16 查询、线程组 64 线程）
    private var 管线注意力打包: MTLComputePipelineState!
    private var 管线SwiGLU: MTLComputePipelineState!
    private var 管线SwiGLU偏置: MTLComputePipelineState!
    private var 管线相加: MTLComputePipelineState!
    private var 管线原地相加: MTLComputePipelineState!
    private var 管线加行向量: MTLComputePipelineState!
    private var 管线拷贝: MTLComputePipelineState!
    private var 管线重排: MTLComputePipelineState!
    private var 管线双三次: MTLComputePipelineState!

    // 激活全链 fp16：逐元素算子的半精度变体（读 half 转 f32 计算、写回 half，数值路径不变）
    private var 管线im2col半: MTLComputePipelineState!
    private var 管线im2col半输入单: MTLComputePipelineState!
    private var 管线偏置激活半: MTLComputePipelineState!
    private var 管线偏置激活V4半: MTLComputePipelineState!
    private var 管线归一化半: MTLComputePipelineState!
    private var 管线SwiGLU半: MTLComputePipelineState!
    private var 管线SwiGLU偏置半: MTLComputePipelineState!
    private var 管线相加半: MTLComputePipelineState!
    private var 管线原地相加半: MTLComputePipelineState!
    private var 管线加行向量半: MTLComputePipelineState!
    private var 管线拷贝半: MTLComputePipelineState!
    private var 管线重排半: MTLComputePipelineState!
    private var 管线双三次半: MTLComputePipelineState!

    /// GEMM 形状参数缓存：每帧每层形状固定，避免重复创建 MPS 描述符与乘法算子
    private var 乘法缓存: [String: MPSMatrixMultiplication] = [:]
    private var 矩阵描述缓存: [String: MPSMatrixDescriptor] = [:]

    // 权重
    private var 编码器层: [(线性层, Int, Int, Int)]!   // (层权重, 卷积核, 步长, 填充)
    private var 编码器归一化: MTLBuffer!
    private var 条件投影: 线性层!
    private var 条件归一化: MTLBuffer!
    private var 视频类型嵌入: MTLBuffer!
    private var 条件类型嵌入: MTLBuffer!
    private var 块: [块权重]!
    private var 空间头: 线性层!
    private var 残差投影: 线性层!

    // MARK: - 初始化

    init(设备: MTLDevice, 权重路径: String, 库: MTLLibrary? = nil) throws {
        guard let 队列 = 设备.makeCommandQueue() else {
            throw 错误.初始化失败("无法创建命令队列")
        }
        self.设备 = 设备
        self.队列 = 队列

        let 权重库 = try SafeTensors权重库(路径: 权重路径)
        self.权重文件 = 权重库
        let 配置 = try 权重库.配置(Flash配置.self)
        self.配置 = 配置

        if let 指定库 = 库 {
            self.库 = 指定库
        } else if let 默认库 = try? 设备.makeDefaultLibrary(bundle: .main) {
            self.库 = 默认库
        } else if let 默认库 = 设备.makeDefaultLibrary() {
            self.库 = 默认库
        } else {
            throw 错误.初始化失败("找不到 Flash 算子的 Metal 库")
        }

        func 建管线(_ 名称: String) throws -> MTLComputePipelineState {
            guard let 函数 = self.库.makeFunction(name: 名称) else {
                throw 错误.初始化失败("缺少 Metal 函数 \(名称)")
            }
            return try 设备.makeComputePipelineState(function: 函数)
        }

        管线im2col = try 建管线("flash_im2col")
        管线偏置激活 = try 建管线("flash_bias_act")
        管线偏置激活V4 = try 建管线("flash_bias_act_v4")
        管线归一化 = try 建管线("flash_rmsnorm")
        管线RoPE = try 建管线("flash_rope")
        管线RoPE半 = try 建管线("flash_rope_half")
        管线注意力打包半 = try 建管线("flash_attention_mma_h_32_16")
        管线转换半 = try 建管线("flash_cast_f2h")
        管线转换单 = try 建管线("flash_cast_h2f")
        管线注意力打包 = try 建管线("flash_attention_mma_32_16")
        管线SwiGLU = try 建管线("flash_swiglu")
        管线SwiGLU偏置 = try 建管线("flash_swiglu_bias_v4")
        管线SwiGLUV4 = try 建管线("flash_swiglu_v4")
        管线相加 = try 建管线("flash_add")
        管线原地相加 = try 建管线("flash_add_inplace")
        管线加行向量 = try 建管线("flash_add_row_vector")
        管线拷贝 = try 建管线("flash_copy")
        管线重排 = try 建管线("flash_pixel_shuffle")
        管线双三次 = try 建管线("flash_bicubic_add")
        管线im2col半 = try 建管线("flash_im2col_h")
        管线im2col半输入单 = try 建管线("flash_im2col_f2h")
        管线偏置激活半 = try 建管线("flash_bias_act_h")
        管线偏置激活V4半 = try 建管线("flash_bias_act_v4_h")
        管线归一化半 = try 建管线("flash_rmsnorm_h")
        管线SwiGLU半 = try 建管线("flash_swiglu_h")
        管线SwiGLU偏置半 = try 建管线("flash_swiglu_bias_v4_h")
        管线相加半 = try 建管线("flash_add_h")
        管线原地相加半 = try 建管线("flash_add_inplace_h")
        管线加行向量半 = try 建管线("flash_add_row_vector_h")
        管线拷贝半 = try 建管线("flash_copy_h")
        管线重排半 = try 建管线("flash_pixel_shuffle_h")
        管线双三次半 = try 建管线("flash_bicubic_add_h")

        // MARK: 权重装载

        func 取张量(_ 名: String) throws -> (形状: [Int], 数值: [Float]) {
            try 权重库.Float32张量(名)
        }

        func 上传(_ 名: String, 形状 期望形状: [Int]) throws -> MTLBuffer {
            let (形状, 数值) = try 取张量(名)
            guard 形状 == 期望形状 else {
                throw 错误.尺寸不符("\(名) 期望 \(期望形状) 实际 \(形状)")
            }
            guard let 缓冲 = 设备.makeBuffer(bytes: 数值,
                                            length: 数值.count * MemoryLayout<Float>.stride,
                                            options: .storageModeShared) else {
                throw 错误.初始化失败("上传 \(名) 失败")
            }
            return 缓冲
        }

        func 上传压平(_ 名: String, 期望元素数: Int) throws -> MTLBuffer {
            let (实际形状, 数值) = try 取张量(名)
            guard 实际形状.reduce(1, *) == 期望元素数 else {
                throw 错误.尺寸不符("\(名) 期望 \(期望元素数) 个元素 实际 \(实际形状)")
            }
            guard let 缓冲 = 设备.makeBuffer(bytes: 数值,
                                           length: 数值.count * MemoryLayout<Float>.stride,
                                           options: .storageModeShared) else {
                throw 错误.初始化失败("上传 \(名) 失败")
            }
            return 缓冲
        }

        func 上传半(_ 名: String, 形状 期望形状: [Int]) throws -> MTLBuffer {
            let (形状, 数值) = try 取张量(名)
            guard 形状 == 期望形状 else {
                throw 错误.尺寸不符("\(名) 期望 \(期望形状) 实际 \(形状)")
            }
            return try 建半缓冲(名, 数值)
        }

        func 上传半压平(_ 名: String, 期望元素数: Int) throws -> MTLBuffer {
            let (实际形状, 数值) = try 取张量(名)
            guard 实际形状.reduce(1, *) == 期望元素数 else {
                throw 错误.尺寸不符("\(名) 期望 \(期望元素数) 个元素 实际 \(实际形状)")
            }
            return try 建半缓冲(名, 数值)
        }

        func 建半缓冲(_ 名: String, _ 数值: [Float]) throws -> MTLBuffer {
            let 半值 = 数值.map { Float16($0) }
            guard let 缓冲 = 设备.makeBuffer(bytes: 半值,
                                            length: 半值.count * MemoryLayout<Float16>.stride,
                                            options: .storageModeShared) else {
                throw 错误.初始化失败("上传 \(名) 失败")
            }
            return 缓冲
        }

        func 可选上传(_ 名: String, 期望形状: [Int]) throws -> MTLBuffer? {
            guard 权重库.是否包含(名) else { return nil }
            return try 上传(名, 形状: 期望形状)
        }

        func 线性(_ 前缀: String, 输出维: Int, 输入维: Int) throws -> 线性层 {
            // fp16 GEMM 要求行字节 8 字节对齐，维度非 4 的倍数时退回 f32 路径
            let 可半精度 = (输出维 % 4 == 0 && 输入维 % 4 == 0)
            let 权重 = 可半精度
                ? try 上传半压平("\(前缀).weight", 期望元素数: 输出维 * 输入维)
                : try 上传压平("\(前缀).weight", 期望元素数: 输出维 * 输入维)
            let 偏置 = try 可选上传("\(前缀).bias", 期望形状: [输出维])
            return 线性层(权重: 权重, 偏置: 偏置, 输出维: 输出维, 输入维: 输入维,
                       半精度: 可半精度)
        }

        func 卷积(_ 前缀: String, 输出维: Int, 输入维: Int, 核: Int) throws -> 线性层 {
            let 内维 = 输入维 * 核 * 核
            let 可半精度 = (输出维 % 4 == 0 && 内维 % 4 == 0)
            let 权重 = 可半精度
                ? try 上传半("\(前缀).weight", 形状: [输出维, 输入维, 核, 核])
                : try 上传("\(前缀).weight", 形状: [输出维, 输入维, 核, 核])
            let 偏置 = try 可选上传("\(前缀).bias", 期望形状: [输出维])
            // 展平后的内维是 通道数 × 核面积，与 im2col 的列数一致
            return 线性层(权重: 权重, 偏置: 偏置, 输出维: 输出维, 输入维: 内维,
                       半精度: 可半精度)
        }

        let 维度 = 形状.维度
        let 编码通道 = 配置.编码通道
        guard 编码通道.count == 3 else {
            throw 错误.尺寸不符("encoder_channels 必须是 3 个通道")
        }
        var 编码层: [(线性层, Int, Int, Int)] = []

        // 第一层卷积的展开列数 3*3*3 = 27 不是 4 的倍数，MPS 要求行字节 16 对齐，
        // 因此把权重补零到 28 列，im2col 同步补到 28 列。
        do {
            let (形状, 数值) = try 取张量("lr_encoder.layers.0.weight")
            guard 形状 == [编码通道[0], 3, 3, 3] else {
                throw 错误.尺寸不符("lr_encoder.layers.0.weight 形状 \(形状) 异常")
            }
            let 原始列数 = 3 * 3 * 3
            let 补齐列数 = 28
            var 补齐 = [Float16](repeating: 0, count: 编码通道[0] * 补齐列数)
            for 行 in 0..<编码通道[0] {
                for 列 in 0..<原始列数 {
                    补齐[行 * 补齐列数 + 列] = Float16(数值[行 * 原始列数 + 列])
                }
            }
            guard let 权重缓冲 = 设备.makeBuffer(bytes: 补齐,
                                               length: 补齐.count * MemoryLayout<Float16>.stride,
                                               options: .storageModeShared) else {
                throw 错误.初始化失败("上传 lr_encoder.layers.0.weight 失败")
            }
            let 偏置 = try 上传("lr_encoder.layers.0.bias", 形状: [编码通道[0]])
            编码层.append((线性层(权重: 权重缓冲, 偏置: 偏置,
                              输出维: 编码通道[0], 输入维: 补齐列数,
                              半精度: (编码通道[0] % 4 == 0 && 补齐列数 % 4 == 0)), 3, 2, 1))
        }

        编码层.append((try 卷积("lr_encoder.layers.2", 输出维: 编码通道[1], 输入维: 编码通道[0], 核: 3), 3, 2, 1))
        编码层.append((try 卷积("lr_encoder.layers.4", 输出维: 编码通道[2], 输入维: 编码通道[1], 核: 3), 3, 1, 1))
        编码层.append((try 卷积("lr_encoder.layers.6", 输出维: 维度, 输入维: 编码通道[2], 核: 1), 1, 1, 0))
        编码器层 = 编码层
        编码器归一化 = try 上传("lr_encoder.norm.weight", 形状: [维度])

        条件投影 = try 线性("condition_encoder.projection", 输出维: 维度, 输入维: 配置.条件通道数)
        条件归一化 = try 上传("condition_encoder.norm.weight", 形状: [维度])
        视频类型嵌入 = try 上传("video_type_embedding", 形状: [1, 1, 维度])
        条件类型嵌入 = try 上传("condition_type_embedding", 形状: [1, 1, 维度])

        var 块列表: [块权重] = []
        let 键值维 = 形状.KV组数 * 形状.头维
        for index in 0..<形状.深度 {
            let 前缀 = "backbone.blocks.\(index)"
            块列表.append(块权重(
                注意力: 注意力权重(
                    查询: try 线性("\(前缀).attention.query", 输出维: 维度, 输入维: 维度),
                    键: try 线性("\(前缀).attention.key", 输出维: 键值维, 输入维: 维度),
                    值: try 线性("\(前缀).attention.value", 输出维: 键值维, 输入维: 维度),
                    条件键: try 线性("\(前缀).attention.condition_key", 输出维: 键值维, 输入维: 维度),
                    条件值: try 线性("\(前缀).attention.condition_value", 输出维: 键值维, 输入维: 维度),
                    键归一化: try 上传("\(前缀).attention.key_norm.weight", 形状: [形状.头维]),
                    值归一化: try 上传("\(前缀).attention.value_norm.weight", 形状: [形状.头维]),
                    输出: try 线性("\(前缀).attention.output", 输出维: 维度, 输入维: 维度)
                ),
                注意力归一化: try 上传("\(前缀).attention_norm.weight", 形状: [维度]),
                前馈门: try 线性("\(前缀).feed_forward.gate_up_proj", 输出维: 2 * 形状.隐藏维, 输入维: 维度),
                前馈降维: try 线性("\(前缀).feed_forward.down", 输出维: 维度, 输入维: 形状.隐藏维),
                前馈归一化: try 上传("\(前缀).feed_forward_norm.weight", 形状: [维度])
            ))
        }
        块 = 块列表

        空间头 = try 线性("spatial_head.0", 输出维: 配置.空间头维度, 输入维: 维度)
        残差投影 = try 线性("residual_projection", 输出维: 3 * 形状.重排倍数 * 形状.重排倍数, 输入维: 配置.空间头维度)
    }

    // MARK: - 缓冲

    var 层数: Int { 块.count }

    private func 新建(_ 元素数: Int, _ 名称: String) -> MTLBuffer {
        let 长度 = max(元素数, 1) * MemoryLayout<Float>.stride
        guard let 缓冲 = 设备.makeBuffer(length: 长度, options: .storageModeShared) else {
            fatalError("Flash 引擎分配缓冲失败：\(名称)")
        }
        缓冲.label = 名称
        return 缓冲
    }

    /// 半精度缓冲（2 字节/元素）：流式 KV 常驻用这个，显存直接省一半
    private func 新建半(_ 元素数: Int, _ 名称: String) -> MTLBuffer {
        let 长度 = max(元素数, 1) * MemoryLayout<UInt16>.stride
        guard let 缓冲 = 设备.makeBuffer(length: 长度, options: .storageModeShared) else {
            fatalError("Flash 引擎分配半精度缓冲失败：\(名称)")
        }
        缓冲.label = 名称
        return 缓冲
    }

    /// f32 → 半精度（计算仍全 f32，只有常驻 KV 降精度，与官方 bf16 KV cache 同义）
    private func 转为半(_ 命令: MTLCommandBuffer,
                     _ 源: MTLBuffer,
                     _ 元素数: Int,
                     _ 名称: String,
                     目标 复用目标: MTLBuffer? = nil) -> MTLBuffer {
        let 需要 = max(元素数, 1) * MemoryLayout<UInt16>.stride
        let 输出: MTLBuffer
        if let 复用 = 复用目标, 复用.length >= 需要 {
            输出 = 复用
        } else {
            输出 = 新建半(元素数, 名称)
        }
        var 参数 = CopyParams(count: Int32(元素数))
        派发(命令, 管线转换半, MTLSize(width: 元素数, height: 1, depth: 1),
            MTLSize(width: 256, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(源, offset: 0, index: 0)
            编码器.setBuffer(输出, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<CopyParams>.stride, index: 2)
        }
        return 输出
    }

    /// 线性 GEMM 的 fp16 输入/输出暂存：同一时刻只有一路线性在编码，各一块按最大尺寸复用即可
    private var 半输入暂存: MTLBuffer?
    private var 半输出暂存: MTLBuffer?

    private func 半暂存(_ 元素数: Int, 输入侧: Bool) -> MTLBuffer {
        let 需要 = max(元素数, 1) * MemoryLayout<UInt16>.stride
        if 输入侧 {
            if let 现有 = 半输入暂存, 现有.length >= 需要 { return 现有 }
            let 新建 = 新建半(元素数, "线性输入 fp16 暂存")
            半输入暂存 = 新建
            return 新建
        }
        if let 现有 = 半输出暂存, 现有.length >= 需要 { return 现有 }
        let 新建 = 新建半(元素数, "线性输出 fp16 暂存")
        半输出暂存 = 新建
        return 新建
    }

    /// fp16 → f32：让半精度 GEMM 的输出直接接现有 f32 逐元素算子（偏置/SiLU/残差等）
    private func 转回单(_ 命令: MTLCommandBuffer,
                     _ 源: MTLBuffer,
                     _ 元素数: Int,
                     _ 目标: MTLBuffer) {
        var 参数 = CopyParams(count: Int32(元素数))
        派发(命令, 管线转换单, MTLSize(width: 元素数, height: 1, depth: 1),
            MTLSize(width: 256, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(源, offset: 0, index: 0)
            编码器.setBuffer(目标, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<CopyParams>.stride, index: 2)
        }
    }

    /// f32 源 → 半精度目标的指定位置（当前帧 KV 落进滚动缓存）
    private func 写半(_ 命令: MTLCommandBuffer,
                   _ 源: MTLBuffer,
                   _ 目标: MTLBuffer,
                   _ 目标偏移: Int,
                   _ 元素数: Int) {
        var 参数 = CopyParams(count: Int32(元素数))
        派发(命令, 管线转换半, MTLSize(width: 元素数, height: 1, depth: 1),
            MTLSize(width: 256, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(源, offset: 0, index: 0)
            编码器.setBuffer(目标, offset: 目标偏移, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<CopyParams>.stride, index: 2)
        }
    }

    /// 一次性（非流式）重建输出缓冲：逐帧复用，避免每帧新建百 MB 缓冲
    private var 一次性输出缓冲: MTLBuffer?

    /// 显存与借池占用概览（诊断用）：区分「权重+KV 常驻」与「临时缓冲」两类占用。
    func 占用概览(_ 状态: 流式状态?) -> String {
        var 层数 = 0
        var KV字节 = 0
        if let 状态 {
            层数 = 状态.视频键.count
            for 缓冲 in 状态.视频键 + 状态.视频值 { KV字节 += 缓冲.length }
            KV字节 += (状态.拼接键?.length ?? 0) + (状态.拼接值?.length ?? 0)
            KV字节 += (状态.条件键 + 状态.条件值).reduce(0) { $0 + $1.length }
            KV字节 += 状态.输出缓冲?.length ?? 0
        }
        let 池字节 = 临时缓冲.reduce(0) { $0 + $1.length }
        let MB = 1_048_576
        let 前几 = 临时缓冲.map { $0.length / MB }.sorted(by: >).prefix(6).map(String.init).joined(separator: ",")
        return "显存 device=\(设备.currentAllocatedSize / MB)MB 会话(KV+输出)=\(KV字节 / MB)MB(\(层数)层) 借池=\(池字节 / MB)MB(\(临时缓冲.count)块) 最大几块=\(前几)"
    }

    private var 临时缓冲: [MTLBuffer] = []
    private var 在用临时: Set<Int> = []

    /// 归还全部临时缓冲（同一命令缓冲内跨层复用）：池规模收敛到「单层并发峰值 × 最大尺寸」。
    private func 释放临时() { 在用临时.removeAll(keepingCapacity: true) }

    /// 借一块至少容纳 元素数 个**半精度**激活的临时缓冲（激活全链 fp16 后池的默认元素宽度）。
    /// 按「容量最省的最优匹配」复用，避免尺寸逐帧增长时按位置游标不断新开缓冲导致显存膨胀。
    /// 池内按字节容量匹配，fp32 需求走 借单，不会误取到只够半精度的小块。
    private func 借(_ 元素数: Int) -> MTLBuffer {
        借用(元素数, 每元素字节: MemoryLayout<UInt16>.stride, 名称: "flash 临时")
    }

    /// 借一块 fp32 临时缓冲：只给维度非 4 倍数的回退线性层用
    private func 借单(_ 元素数: Int) -> MTLBuffer {
        借用(元素数, 每元素字节: MemoryLayout<Float>.stride, 名称: "flash 临时 fp32")
    }

    private func 借用(_ 元素数: Int, 每元素字节: Int, 名称: String) -> MTLBuffer {
        let 需字节 = max(元素数, 1) * 每元素字节
        var 最优 = -1
        for 索引 in 临时缓冲.indices where !在用临时.contains(索引) && 临时缓冲[索引].length >= 需字节 {
            if 最优 < 0 || 临时缓冲[索引].length < 临时缓冲[最优].length { 最优 = 索引 }
        }
        if 最优 >= 0 {
            在用临时.insert(最优)
            return 临时缓冲[最优]
        }
        guard let 缓冲 = 设备.makeBuffer(length: 需字节, options: .storageModeShared) else {
            fatalError("Flash 引擎分配缓冲失败：\(名称)")
        }
        缓冲.label = "\(名称) \(临时缓冲.count)"
        临时缓冲.append(缓冲)
        在用临时.insert(临时缓冲.count - 1)
        return 缓冲
    }

    /// 等待命令缓冲完成：完成回调 + 信号量超时，避免 GPU 异常时线程永久挂住。
    private func 等待完成(_ 目标: MTLCommandBuffer, 超时秒: Double = 300) throws {
        let 起始 = DispatchTime.now().uptimeNanoseconds
        while 目标.status != .completed && 目标.status != .error {
            if Double(DispatchTime.now().uptimeNanoseconds - 起始) / 1e9 > 超时秒 {
                throw 错误.初始化失败("Metal 命令缓冲超过 \(Int(超时秒)) 秒未完成（GPU 未返回）")
            }
            usleep(3_000)
        }
    }
    // MARK: - 基础算子

    private func 编码<T>(_ 值: T) -> MTLBuffer {
        var 拷贝 = 值
        return 设备.makeBuffer(bytes: &拷贝,
                              length: MemoryLayout<T>.stride,
                              options: .storageModeShared)!
    }

    private func 派发(_ 命令: MTLCommandBuffer,
                     _ 管线: MTLComputePipelineState,
                     _ 网格: MTLSize,
                     _ 线程组: MTLSize,
                     _ 绑定: (MTLComputeCommandEncoder) -> Void) {
        guard let 编码器 = 命令.makeComputeCommandEncoder() else { return }
        编码器.setComputePipelineState(管线)
        绑定(编码器)
        编码器.dispatchThreads(网格, threadsPerThreadgroup: 线程组)
        编码器.endEncoding()
    }

    private func 派发组(_ 命令: MTLCommandBuffer,
                      _ 管线: MTLComputePipelineState,
                      _ 组数: MTLSize,
                      _ 线程组: MTLSize,
                      _ 绑定: (MTLComputeCommandEncoder) -> Void) {
        guard let 编码器 = 命令.makeComputeCommandEncoder() else { return }
        编码器.setComputePipelineState(管线)
        绑定(编码器)
        编码器.dispatchThreadgroups(组数, threadsPerThreadgroup: 线程组)
        编码器.endEncoding()
    }

    /// C[行数, 输出维] = A[行数, 内维] @ W[输出维, 内维]^T
    /// 半 = true 时 A/W/C 均为 fp16（MPS half 路径，峰值算力为 f32 的两倍、带宽减半）
    private func 矩阵乘法(_ 命令: MTLCommandBuffer,
                       _ 左侧: MTLBuffer,
                       _ 行数: Int,
                       _ 内维: Int,
                       _ 权重: MTLBuffer,
                       _ 输出维: Int,
                       _ 结果: MTLBuffer,
                       _ 半: Bool = false) {
        let 左矩阵 = MPSMatrix(buffer: 左侧, descriptor: 矩阵描述(行数, 内维, 半))
        let 右矩阵 = MPSMatrix(buffer: 权重, descriptor: 矩阵描述(输出维, 内维, 半))
        let 果矩阵 = MPSMatrix(buffer: 结果, descriptor: 矩阵描述(行数, 输出维, 半))
        let 乘键 = "M\(半 ? "h" : "f")\(行数)_\(内维)_\(输出维)"
        let 乘法: MPSMatrixMultiplication
        if let 命中 = 乘法缓存[乘键] {
            乘法 = 命中
        } else {
            let 新建 = MPSMatrixMultiplication(device: 设备,
                                              transposeLeft: false,
                                              transposeRight: true,
                                              resultRows: 行数,
                                              resultColumns: 输出维,
                                              interiorColumns: 内维,
                                              alpha: 1,
                                              beta: 0)
            乘法缓存[乘键] = 新建
            乘法 = 新建
        }
        乘法.encode(commandBuffer: 命令,
                   leftMatrix: 左矩阵,
                   rightMatrix: 右矩阵,
                   resultMatrix: 果矩阵)
    }

    /// 矩阵描述按形状与精度缓存：每帧每层形状固定，避免重复分配 MPS 对象
    private func 矩阵描述(_ 行数: Int, _ 列数: Int, _ 半: Bool) -> MPSMatrixDescriptor {
        let 键 = "\(半 ? "h" : "f")\(行数)_\(列数)"
        if let 命中 = 矩阵描述缓存[键] { return 命中 }
        let 新建 = MPSMatrixDescriptor(rows: 行数,
                                      columns: 列数,
                                      rowBytes: 列数 * (半 ? MemoryLayout<UInt16>.stride
                                                          : MemoryLayout<Float>.stride),
                                      dataType: 半 ? .float16 : .float32)
        矩阵描述缓存[键] = 新建
        return 新建
    }

    /// Y[行数, 输出维] = X @ W^T + b（可选 SiLU）
    private func 线性(_ 命令: MTLCommandBuffer,
                    _ 输入: MTLBuffer,
                    _ 行数: Int,
                    _ 层: 线性层,
                    _ 结果: MTLBuffer,
                    _ 激活: Bool,
                    仅矩阵乘: Bool = false) {
        if 层.半精度 {
            // 激活全链 fp16：输入输出本身都是半精度缓冲，GEMM 直接读写，两端不再有 cast pass
            矩阵乘法(命令, 输入, 行数, 层.输入维, 层.权重, 层.输出维, 结果, true)
        } else {
            // 维度非 4 倍数的回退层：就地补两次转换，代价只落在极少数层上
            let 输入元素 = 行数 * 层.输入维
            let 输入单 = 借单(输入元素)
            转回单(命令, 输入, 输入元素, 输入单)
            let 输出元素 = 行数 * 层.输出维
            let 结果单 = 借单(输出元素)
            矩阵乘法(命令, 输入单, 行数, 层.输入维, 层.权重, 层.输出维, 结果单)
            转为半(命令, 结果单, 输出元素, "线性输出 fp16", 目标: 结果)
        }
        if 仅矩阵乘 { return }
        var 参数 = BiasParams(rows: Int32(行数), cols: Int32(层.输出维),
                              hasBias: 层.偏置 == nil ? 0 : 1, act: 激活 ? 1 : 0)
        if 层.输出维 % 4 == 0 {
            // 向量化路径：gid.x 沿列方向（float4），线程组 64×4=256 线程，合并访存
            派发(命令, 管线偏置激活V4半, MTLSize(width: 层.输出维 / 4, height: 行数, depth: 1),
                MTLSize(width: 64, height: 4, depth: 1)) { 编码器 in
                编码器.setBuffer(结果, offset: 0, index: 0)
                编码器.setBuffer(层.偏置 ?? 结果, offset: 0, index: 1)
                编码器.setBytes(&参数, length: MemoryLayout<BiasParams>.stride, index: 2)
            }
        } else {
            派发(命令, 管线偏置激活半, MTLSize(width: 行数, height: 层.输出维, depth: 1),
                MTLSize(width: 1, height: 16, depth: 1)) { 编码器 in
                编码器.setBuffer(结果, offset: 0, index: 0)
                编码器.setBuffer(层.偏置 ?? 结果, offset: 0, index: 1)
                编码器.setBytes(&参数, length: MemoryLayout<BiasParams>.stride, index: 2)
            }
        }
    }

    private func 归一化(_ 命令: MTLCommandBuffer,
                     _ 输入: MTLBuffer,
                     _ 权重: MTLBuffer,
                     _ 行数: Int,
                     _ 输出: MTLBuffer) {
        var 参数 = NormParams(rows: Int32(行数), cols: Int32(形状.维度),
                              eps: Float(配置.归一化Eps))
        派发组(命令, 管线归一化半, MTLSize(width: 行数, height: 1, depth: 1),
             MTLSize(width: 128, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(输入, offset: 0, index: 0)
            编码器.setBuffer(权重, offset: 0, index: 1)
            编码器.setBuffer(输出, offset: 0, index: 2)
            编码器.setBytes(&参数, length: MemoryLayout<NormParams>.stride, index: 3)
        }
    }

    /// 按 head_dim 做 RMSNorm（K/V 的归一化）
    private func 分组归一化(_ 命令: MTLCommandBuffer,
                       _ 缓冲: MTLBuffer,
                       _ 权重: MTLBuffer,
                       _ 行数: Int,
                       _ 列数: Int) {
        var 参数 = NormParams(rows: Int32(行数), cols: Int32(列数),
                              eps: Float(配置.归一化Eps))
        派发组(命令, 管线归一化半, MTLSize(width: 行数, height: 1, depth: 1),
             MTLSize(width: 64, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(缓冲, offset: 0, index: 0)
            编码器.setBuffer(权重, offset: 0, index: 1)
            编码器.setBuffer(缓冲, offset: 0, index: 2)
            编码器.setBytes(&参数, length: MemoryLayout<NormParams>.stride, index: 3)
        }
    }

    private func 施加RoPE(_ 命令: MTLCommandBuffer,
                       _ 缓冲: MTLBuffer,
                       _ 令牌数: Int,
                       _ 头数: Int,
                       _ 宽: Int,
                       _ 平面令牌: Int = 0,
                       时间位置: Int32 = 0,
                       平面步长: Int32 = 0,
                       缓冲偏移: Int = 0,
                       半精度: Bool = true) {
        var 参数 = RoPEParams(tokens: Int32(令牌数), heads: Int32(头数), width: Int32(宽),
                              temporalPosition: 时间位置,
                              theta: Float(配置.RoPETheta),
                              temporalDim: 16, heightDim: 24, widthDim: 24,
                              headDim: Int32(形状.头维), planeTokens: Int32(平面令牌),
                              planeStep: 平面步长)
        派发(命令, 半精度 ? 管线RoPE半 : 管线RoPE, MTLSize(width: 令牌数, height: 头数, depth: 1),
            MTLSize(width: 8, height: 8, depth: 1)) { 编码器 in
            编码器.setBuffer(缓冲, offset: 缓冲偏移, index: 0)
            编码器.setBytes(&参数, length: MemoryLayout<RoPEParams>.stride, index: 1)
        }
    }

    private func 拷贝(_ 命令: MTLCommandBuffer,
                    _ 源: MTLBuffer,
                    _ 源偏移: Int,
                    _ 目标: MTLBuffer,
                    _ 目标偏移: Int,
                    _ 字节数: Int) {
        guard let 传输 = 命令.makeBlitCommandEncoder() else { return }
        传输.copy(from: 源, sourceOffset: 源偏移,
                 to: 目标, destinationOffset: 目标偏移, size: 字节数)
        传输.endEncoding()
    }

    private func 窗口缓冲(_ 窗口: [(高: Int, 宽: Int)]) -> MTLBuffer {
        var 值 = 窗口.map { SIMD2<Int32>(Int32($0.高), Int32($0.宽)) }
        return 设备.makeBuffer(bytes: &值,
                             length: 值.count * MemoryLayout<SIMD2<Int32>>.stride,
                             options: .storageModeShared)!
    }

    private func 加行向量(_ 命令: MTLCommandBuffer,
                       _ 缓冲: MTLBuffer,
                       _ 向量: MTLBuffer,
                       _ 行数: Int) {
        var 参数 = BiasParams(rows: Int32(行数), cols: Int32(形状.维度), hasBias: 1, act: 0)
        派发(命令, 管线加行向量半, MTLSize(width: 行数, height: 形状.维度, depth: 1),
            MTLSize(width: 1, height: 32, depth: 1)) { 编码器 in
            编码器.setBuffer(缓冲, offset: 0, index: 0)
            编码器.setBuffer(向量, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<BiasParams>.stride, index: 2)
        }
    }

    private func 原地相加(_ 命令: MTLCommandBuffer,
                       _ 目标: MTLBuffer,
                       _ 增量: MTLBuffer,
                       _ 元素数: Int) {
        var 参数 = CopyParams(count: Int32(元素数))
        派发(命令, 管线原地相加半, MTLSize(width: 元素数, height: 1, depth: 1),
            MTLSize(width: 256, height: 1, depth: 1)) { 编码器 in
            编码器.setBuffer(目标, offset: 0, index: 0)
            编码器.setBuffer(增量, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<CopyParams>.stride, index: 2)
        }
    }

    private func SwiGLU(_ 命令: MTLCommandBuffer,
                       _ 门控: MTLBuffer,
                       _ 输出: MTLBuffer,
                       _ 行数: Int,
                       _ 偏置: MTLBuffer? = nil) {
        var 参数 = BiasParams(rows: Int32(行数), cols: Int32(形状.隐藏维),
                              hasBias: 偏置 == nil ? 0 : 1, act: 0)
        if 形状.隐藏维 % 4 == 0 {
            派发(命令, 管线SwiGLU偏置半, MTLSize(width: 形状.隐藏维 / 4, height: 行数, depth: 1),
                MTLSize(width: 64, height: 4, depth: 1)) { 编码器 in
                编码器.setBuffer(门控, offset: 0, index: 0)
                编码器.setBuffer(偏置 ?? 门控, offset: 0, index: 1)
                编码器.setBuffer(输出, offset: 0, index: 2)
                编码器.setBytes(&参数, length: MemoryLayout<BiasParams>.stride, index: 3)
            }
        } else {
            派发(命令, 管线SwiGLU半, MTLSize(width: 行数, height: 形状.隐藏维, depth: 1),
                MTLSize(width: 1, height: 32, depth: 1)) { 编码器 in
                编码器.setBuffer(门控, offset: 0, index: 0)
                编码器.setBuffer(输出, offset: 0, index: 1)
                编码器.setBytes(&参数, length: MemoryLayout<BiasParams>.stride, index: 2)
            }
        }
    }

    private func 窗口注意力(_ 命令: MTLCommandBuffer,
                         _ 查询: MTLBuffer,
                         _ 键: MTLBuffer,
                         _ 值: MTLBuffer,
                         _ 输出: MTLBuffer,
                         批次数: Int,
                         每批令牌: Int,
                         令牌高: Int,
                         令牌宽: Int,
                         平面数: Int,
                         条件平面数: Int,
                         窗口: [(高: Int, 宽: Int)],
                         时间窗: Int = 0,
                         半精度: Bool = false) {
        let 窗口值 = 窗口缓冲(窗口)
        var 参数 = AttnParams(batchTokens: Int32(每批令牌),
                              batches: Int32(批次数),
                              height: Int32(令牌高),
                              width: Int32(令牌宽),
                              heads: Int32(形状.头数),
                              kvGroups: Int32(形状.KV组数),
                              headDim: Int32(形状.头维),
                              planes: Int32(平面数),
                              condPlanes: Int32(条件平面数),
                              temporalWindow: Int32(时间窗),
                              hasPlaneMask: 0)
        guard let 编码器 = 命令.makeComputeCommandEncoder() else { return }
        编码器.setBuffer(查询, offset: 0, index: 0)
        编码器.setBuffer(键, offset: 0, index: 1)
        编码器.setBuffer(值, offset: 0, index: 2)
        编码器.setBuffer(输出, offset: 0, index: 3)
        编码器.setBuffer(窗口值, offset: 0, index: 4)
        编码器.setBuffer(窗口值, offset: 0, index: 5)   // 无平面掩码时占位
        编码器.setBytes(&参数, length: MemoryLayout<AttnParams>.stride, index: 6)

        // 唯一派发路径：MMA 内核 + 单遍在线 softmax（每组查询数 × 4 线程，K/V 在组内查询间复用）
        precondition(形状.头维 == 64, "注意力内核仅支持 64 维头")
        let 每组查询数 = 32
        let 列组数 = (令牌宽 + 每组查询数 - 1) / 每组查询数
        let 组数 = 批次数 * 形状.头数 * 令牌高 * 列组数
        precondition(半精度 || true, "激活全链 fp16 后 Q/K/V 均为半精度，固定走 half 内核")
        编码器.setComputePipelineState(管线注意力打包半)
        编码器.dispatchThreadgroups(MTLSize(width: 组数, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 每组查询数 * 4, height: 1, depth: 1))
        编码器.endEncoding()
    }

    // MARK: - 前向

    /// 帧输入 —— 类型契约（务必遵守：历史上 uint8 低清 / fp16 条件混入曾导致整帧 100% NaN）
    ///  - 低清：fp32 NCHW [3, 高, 宽]，值域 [-1, 1]（uint8 → /127.5 - 1，与官方 prepare_lr 一致）
    ///  - 起始/结束条件：fp32 NCHW [条件通道数, 高/4, 宽/4]，来源是 Wan 输出的 fp16 latent，必须先转 fp32
    ///  - 所有 buffer 的字节数必须 ≥ 表内期望值，前向 会在派发 GPU 前校验，不符直接抛错
    struct 帧输入 {
        /// [3, 高, 宽] fp32 NCHW，值域 [-1, 1]
        let 低清: MTLBuffer
        let 高: Int
        let 宽: Int
        /// [条件通道数, 高/4, 宽/4] fp32 NCHW
        let 起始条件: MTLBuffer
        let 结束条件: MTLBuffer
    }

    struct 帧输出 {
        let 像素: MTLBuffer
        let 高: Int
        let 宽: Int
    }

    /// 总帧数：本次前向所代表的协作窗口帧数，决定条件端点 RoPE 的平面步长（官方 model.py:989-998 的
    /// 端点位置取 (0, total_frames-1)，streaming.py:173 把窗口帧数抬到 max(2, len(window))）。
    /// 缺省 1 表示单帧无时间跨度的旧口径（平面步长 0，端点同相位）。
    func 前向(_ 输入: 帧输入, 总帧数: Int = 1) throws -> 帧输出 {
        // 类型契约硬校验：宁可抛错，也不要静默产出 NaN 帧
        try 校验类型契约(输入)
        释放临时()
        guard let 命令 = 队列.makeCommandBuffer() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }
        let 令牌高 = 输入.高 / 形状.块分辨率
        let 令牌宽 = 输入.宽 / 形状.块分辨率
        let 令牌数 = 令牌高 * 令牌宽

        let 低清令牌 = 编码低清帧(命令, 低清: 输入.低清, 高: 输入.高, 宽: 输入.宽)
        let 起始令牌 = 编码条件(命令, 输入.起始条件, 令牌高, 令牌宽, "start")
        let 结束令牌 = 编码条件(命令, 输入.结束条件, 令牌高, 令牌宽, "end")

        let 端点 = 新建半(2 * 令牌数 * 形状.维度, "端点令牌")
        let 行字节 = 令牌数 * 形状.维度 * MemoryLayout<UInt16>.stride
        拷贝(命令, 起始令牌, 0, 端点, 0, 行字节)
        拷贝(命令, 结束令牌, 0, 端点, 行字节, 行字节)

        let (条件键, 条件值) = 预热(命令,
                                 端点: 端点,
                                 令牌数: 令牌数,
                                 令牌高: 令牌高,
                                 令牌宽: 令牌宽,
                                 半精度: true)

        let 隐藏 = 视频前向(命令,
                          低清令牌: 低清令牌,
                          条件键: 条件键,
                          条件值: 条件值,
                          令牌数: 令牌数,
                          令牌高: 令牌高,
                          令牌宽: 令牌宽,
                          总帧数: 总帧数)

        let 像素 = 重建(命令,
                       隐藏: 隐藏,
                       低清: 输入.低清,
                       令牌数: 令牌数,
                       令牌高: 令牌高,
                       令牌宽: 令牌宽,
                       高: 输入.高,
                       宽: 输入.宽,
                       输出槽: 一次性输出缓冲)

        命令.commit()
        try 等待完成(命令)
        if let 执行错误 = 命令.error {
            throw Flash引擎.错误.初始化失败("Metal 执行失败：\(执行错误.localizedDescription)")
        }
        一次性输出缓冲 = 像素
        return 帧输出(像素: 像素, 高: 令牌高 * 形状.重排倍数, 宽: 令牌宽 * 形状.重排倍数)
    }

    /// 校验帧输入类型契约：全 fp32 + 字节数精确匹配 + 尺寸对齐块分辨率。
    /// 任一条不满足立即抛错（错误信息直接指出正确类型与形状），避免 fp16/uint8 混入后整帧 NaN。
    private func 校验类型契约(_ 输入: 帧输入) throws {
        let 元素 = MemoryLayout<Float>.stride
        let 块 = 形状.块分辨率
        guard 输入.高 > 0, 输入.宽 > 0, 输入.高 % 块 == 0, 输入.宽 % 块 == 0 else {
            throw 错误.初始化失败("低清尺寸 \(输入.宽)x\(输入.高) 必须是块分辨率 \(块) 的正整数倍")
        }
        let 低清应有 = 3 * 输入.高 * 输入.宽 * 元素
        guard 输入.低清.length >= 低清应有 else {
            throw 错误.初始化失败("低清 buffer \(输入.低清.length) 字节 < 期望 \(低清应有)：需 fp32 NCHW [3,\(输入.高),\(输入.宽)]、值域 [-1,1]（传 uint8/fp16 会导致 NaN）")
        }
        let 令牌高 = 输入.高 / 块
        let 令牌宽 = 输入.宽 / 块
        let 条件应有 = 配置.条件通道数 * 令牌高 * 令牌宽 * 元素
        for (标签, 条件) in [("起始", 输入.起始条件), ("结束", 输入.结束条件)] {
            guard 条件.length >= 条件应有 else {
                throw 错误.初始化失败("\(标签)条件 buffer \(条件.length) 字节 < 期望 \(条件应有)：需 fp32 NCHW [\(配置.条件通道数),\(令牌高),\(令牌宽)]（Wan 的 fp16 输出必须先转 fp32）")
            }
        }
    }

    private func 编码低清帧(_ 命令: MTLCommandBuffer,
                        低清: MTLBuffer,
                        高 输入高: Int,
                        宽 输入宽: Int) -> MTLBuffer {
        var 高 = 输入高
        var 宽 = 输入宽
        var 通道 = 3
        var 当前 = 低清
        var 行数 = 0

        for (层索引, 单元) in 编码器层.enumerated() {
            let (层, 核, 步长, 填充) = 单元
            let 出高 = (高 + 2 * 填充 - 核) / 步长 + 1
            let 出宽 = (宽 + 2 * 填充 - 核) / 步长 + 1
            let 原始列数 = 通道 * 核 * 核
            let 列数 = ((原始列数 + 3) / 4) * 4
            行数 = 出高 * 出宽

            let 展开 = 借(行数 * 列数)
            var 参数 = Im2ColParams(inH: Int32(高), inW: Int32(宽), inC: Int32(通道),
                                    kernelSize: Int32(核), stride: Int32(步长), pad: Int32(填充),
                                    outH: Int32(出高), outW: Int32(出宽),
                                    outCols: Int32(列数), realCols: Int32(原始列数),
                                    channelsLast: 层索引 == 0 ? 0 : 1)
            派发(命令, 层索引 == 0 ? 管线im2col半输入单 : 管线im2col半, MTLSize(width: 行数, height: 列数, depth: 1),
                MTLSize(width: 16, height: 16, depth: 1)) { 编码器 in
                编码器.setBuffer(当前, offset: 0, index: 0)
                编码器.setBuffer(展开, offset: 0, index: 1)
                编码器.setBytes(&参数, length: MemoryLayout<Im2ColParams>.stride, index: 2)
            }

            let 输出 = 借(行数 * 层.输出维)
            // 只有前三个 3x3 卷积后接 SiLU，最后的 1x1 投影不激活
            线性(命令, 展开, 行数, 层, 输出, 层索引 < 编码器层.count - 1)

            当前 = 输出
            通道 = 层.输出维
            高 = 出高
            宽 = 出宽
        }

        let 归一 = 借(行数 * 形状.维度)
        归一化(命令, 当前, 编码器归一化, 行数, 归一)
        加行向量(命令, 归一, 视频类型嵌入, 行数)
        return 归一
    }

    /// 条件 latent 编码：[48,令牌高,令牌宽] → [N, 512]
    private func 编码条件(_ 命令: MTLCommandBuffer,
                       _ latent: MTLBuffer,
                       _ 令牌高: Int,
                       _ 令牌宽: Int,
                       _ 标签: String) -> MTLBuffer {
        let 行数 = 令牌高 * 令牌宽
        let 通道 = 配置.条件通道数
        let 展开 = 借(行数 * 通道)
        var 参数 = Im2ColParams(inH: Int32(令牌高), inW: Int32(令牌宽), inC: Int32(通道),
                                kernelSize: 1, stride: 1, pad: 0,
                                outH: Int32(令牌高), outW: Int32(令牌宽),
                                outCols: Int32(通道), realCols: Int32(通道),
                                channelsLast: 0)
        派发(命令, 管线im2col半输入单, MTLSize(width: 行数, height: 通道, depth: 1),
            MTLSize(width: 16, height: 16, depth: 1)) { 编码器 in
            编码器.setBuffer(latent, offset: 0, index: 0)
            编码器.setBuffer(展开, offset: 0, index: 1)
            编码器.setBytes(&参数, length: MemoryLayout<Im2ColParams>.stride, index: 2)
        }

        let 投影 = 借(行数 * 形状.维度)
        线性(命令, 展开, 行数, 条件投影, 投影, false)

        let 归一 = 借(行数 * 形状.维度)
        归一化(命令, 投影, 条件归一化, 行数, 归一)
        加行向量(命令, 归一, 条件类型嵌入, 行数)
        return 归一
    }

    /// 端点条件预热，返回每层的条件 K/V（[2N, 128]）
    /// 预热端点条件 KV。
    /// - 半精度：true 时**逐层**把条件 KV 直接写成半精度（f32 中间量当层即释放），
    ///   避免「先把所有层 f32 条件 KV 攒齐再整体转半」造成的多 GB 瞬时峰值。
    /// - 键目标/值目标：可选的半精度复用缓冲，跨窗口复用同一批显存，免去每窗重建。
    private func 预热(_ 命令: MTLCommandBuffer,
                   端点: MTLBuffer,
                   令牌数: Int,
                   令牌高: Int,
                   令牌宽: Int,
                   半精度: Bool,
                   键目标: [MTLBuffer]? = nil,
                   值目标: [MTLBuffer]? = nil) -> ([MTLBuffer], [MTLBuffer]) {
        let 行数 = 2 * 令牌数
        let 维度 = 形状.维度
        let 键值维 = 形状.KV组数 * 形状.头维
        let 窗口 = [(高: 配置.空间注意力.条件自窗[0], 宽: 配置.空间注意力.条件自窗[1])]

        let 隐藏 = 端点
        var 键列表: [MTLBuffer] = []
        var 值列表: [MTLBuffer] = []

        for (层索引, 层) in 块.enumerated() {
            let 归一 = 借(行数 * 维度)
            归一化(命令, 隐藏, 层.注意力归一化, 行数, 归一)

            let 查询 = 借(行数 * 维度)
            线性(命令, 归一, 行数, 层.注意力.查询, 查询, false)
            施加RoPE(命令, 查询, 行数, 形状.头数, 令牌宽, 令牌数)

            // 缓存的条件 K 保持**未旋转**：官方语义下条件时间位置（0 与 total-1）在每次
            // 解码时由调用方重新施加，缓存本身不含 RoPE。
            let 键 = 新建半(行数 * 键值维, "条件 K")
            线性(命令, 归一, 行数, 层.注意力.条件键, 键, false)
            分组归一化(命令, 键, 层.注意力.键归一化, 行数 * 形状.KV组数, 形状.头维)
            let 旋转键 = 借(行数 * 键值维)
            拷贝(命令, 键, 0, 旋转键, 0, 行数 * 键值维 * MemoryLayout<UInt16>.stride)
            施加RoPE(命令, 旋转键, 行数, 形状.KV组数, 令牌宽, 令牌数)

            let 值 = 新建半(行数 * 键值维, "条件 V")
            线性(命令, 归一, 行数, 层.注意力.条件值, 值, false)
            分组归一化(命令, 值, 层.注意力.值归一化, 行数 * 形状.KV组数, 形状.头维)

            let 复用键 = (键目标?.count ?? 0) > 层索引 ? 键目标?[层索引] : nil
            let 复用值 = (值目标?.count ?? 0) > 层索引 ? 值目标?[层索引] : nil
            // 条件 KV 本身已是半精度：只需在有跨窗复用目标时 blit 过去，无目标时直接持有
            let 条件字节 = 行数 * 键值维 * MemoryLayout<UInt16>.stride
            let 键缓存: MTLBuffer
            if 半精度, let 复用 = 复用键, 复用.length >= 条件字节 {
                拷贝(命令, 键, 0, 复用, 0, 条件字节)
                键缓存 = 复用
            } else { 键缓存 = 键 }
            let 值缓存: MTLBuffer
            if 半精度, let 复用 = 复用值, 复用.length >= 条件字节 {
                拷贝(命令, 值, 0, 复用, 0, 条件字节)
                值缓存 = 复用
            } else { 值缓存 = 值 }

            let 注意力输出 = 借(行数 * 维度)
            窗口注意力(命令, 查询, 旋转键, 值, 注意力输出,
                    批次数: 2, 每批令牌: 令牌数,
                    令牌高: 令牌高, 令牌宽: 令牌宽,
                    平面数: 1, 条件平面数: 1, 窗口: 窗口)

            let 投影 = 借(行数 * 维度)
            线性(命令, 注意力输出, 行数, 层.注意力.输出, 投影, false)
            原地相加(命令, 隐藏, 投影, 行数 * 维度)

            let 前馈归一 = 借(行数 * 维度)
            归一化(命令, 隐藏, 层.前馈归一化, 行数, 前馈归一)
            let 门控 = 借(行数 * 2 * 形状.隐藏维)
            线性(命令, 前馈归一, 行数, 层.前馈门, 门控, false, 仅矩阵乘: true)
            let 激活 = 借(行数 * 形状.隐藏维)
            SwiGLU(命令, 门控, 激活, 行数, 层.前馈门.偏置)
            let 降维 = 借(行数 * 维度)
            线性(命令, 激活, 行数, 层.前馈降维, 降维, false)
            原地相加(命令, 隐藏, 降维, 行数 * 维度)

            键列表.append(键缓存)
            值列表.append(值缓存)
        }
        return (键列表, 值列表)
    }

    /// 当前帧主干前向：[N,512] → [N,512]
    private func 视频前向(_ 命令: MTLCommandBuffer,
                      低清令牌: MTLBuffer,
                      条件键: [MTLBuffer],
                      条件值: [MTLBuffer],
                      令牌数: Int,
                      令牌高: Int,
                      令牌宽: Int,
                      总帧数: Int = 1) -> MTLBuffer {
        let 维度 = 形状.维度
        let 键值维 = 形状.KV组数 * 形状.头维
        let 条件行数 = 2 * 令牌数
        let 窗口 = [
            (高: 配置.空间注意力.起始条件窗[0], 宽: 配置.空间注意力.起始条件窗[1]),
            (高: 配置.空间注意力.结束条件窗[0], 宽: 配置.空间注意力.结束条件窗[1]),
            (高: 配置.空间注意力.视频窗[0], 宽: 配置.空间注意力.视频窗[1])
        ]

        let 隐藏 = 低清令牌
        // 层内中间缓冲跨层不保留，每层归还后复用：池规模 = 单层峰值，而非「层数 × 单层峰值」

        for (索引, 层) in 块.enumerated() {
            释放临时()
            let 归一 = 借(令牌数 * 维度)
            归一化(命令, 隐藏, 层.注意力归一化, 令牌数, 归一)

            let 查询 = 借(令牌数 * 维度)
            线性(命令, 归一, 令牌数, 层.注意力.查询, 查询, false)
            施加RoPE(命令, 查询, 令牌数, 形状.头数, 令牌宽)

            let 视频键 = 借(令牌数 * 键值维)
            线性(命令, 归一, 令牌数, 层.注意力.键, 视频键, false)
            分组归一化(命令, 视频键, 层.注意力.键归一化, 令牌数 * 形状.KV组数, 形状.头维)
            施加RoPE(命令, 视频键, 令牌数, 形状.KV组数, 令牌宽)

            let 视频值 = 借(令牌数 * 键值维)
            线性(命令, 归一, 令牌数, 层.注意力.值, 视频值, false)
            分组归一化(命令, 视频值, 层.注意力.值归一化, 令牌数 * 形状.KV组数, 形状.头维)

            let 全部键 = 借((条件行数 + 令牌数) * 键值维)
            let 全部值 = 借((条件行数 + 令牌数) * 键值维)
            let 条件字节 = 条件行数 * 键值维 * MemoryLayout<UInt16>.stride
            let 视频字节 = 令牌数 * 键值维 * MemoryLayout<UInt16>.stride
            拷贝(命令, 条件键[索引], 0, 全部键, 0, 条件字节)
            拷贝(命令, 视频键, 0, 全部键, 条件字节, 视频字节)
            拷贝(命令, 条件值[索引], 0, 全部值, 0, 条件字节)
            拷贝(命令, 视频值, 0, 全部值, 条件字节, 视频字节)
            // 条件 K 缓存不含 RoPE，这里按 (0, 总帧数-1) 旋转；总帧数<2 时步长退化为 0，起止端点同相位
            施加RoPE(命令, 全部键, 条件行数, 形状.KV组数, 令牌宽, 令牌数,
                    时间位置: 0, 平面步长: Int32(max(总帧数, 1) - 1))

            if 索引 == 0 {
            }
            let 注意力输出 = 借(令牌数 * 维度)
            窗口注意力(命令, 查询, 全部键, 全部值, 注意力输出,
                    批次数: 1, 每批令牌: 令牌数,
                    令牌高: 令牌高, 令牌宽: 令牌宽,
                    平面数: 3, 条件平面数: 2, 窗口: 窗口)

            let 投影 = 借(令牌数 * 维度)
            线性(命令, 注意力输出, 令牌数, 层.注意力.输出, 投影, false)
            原地相加(命令, 隐藏, 投影, 令牌数 * 维度)

            let 前馈归一 = 借(令牌数 * 维度)
            归一化(命令, 隐藏, 层.前馈归一化, 令牌数, 前馈归一)
            let 门控 = 借(令牌数 * 2 * 形状.隐藏维)
            线性(命令, 前馈归一, 令牌数, 层.前馈门, 门控, false, 仅矩阵乘: true)
            let 激活 = 借(令牌数 * 形状.隐藏维)
            SwiGLU(命令, 门控, 激活, 令牌数, 层.前馈门.偏置)
            let 降维 = 借(令牌数 * 维度)
            线性(命令, 激活, 令牌数, 层.前馈降维, 降维, false)
            原地相加(命令, 隐藏, 降维, 令牌数 * 维度)
        }
        return 隐藏
    }

    // MARK: - 流式超分（对齐官方 begin_stream / decode_step）

    /// 流式状态：每层滚动视频 KV + 常驻端点条件 KV。
    /// 语义与官方一致——视频 K 以**未施加 RoPE** 的形式缓存，每步按相对时间位置重新旋转；
    /// 端点条件 K 的时间位置恒为 0，一次旋转后常驻复用。
    struct 流式状态 {
        var 视频键: [MTLBuffer] = []
        var 视频值: [MTLBuffer] = []
        var 条件键: [MTLBuffer] = []
        var 条件值: [MTLBuffer] = []
        /// 每层复用的 KV 拼接缓冲（流式下拼接后可达百 MB，不能每层各分配一份）
        var 拼接键: MTLBuffer?
        var 拼接值: MTLBuffer?
        var 已出帧数 = 0
        var 令牌数 = 0
        var 令牌高 = 0
        var 令牌宽 = 0
        var 键值维 = 0
        var 维度 = 0
        var 最大历史帧 = 0
        var 帧字节 = 0
        /// KV 是否按半精度存储（true=官方 bf16 语义，显存省一半；false=f32 对拍基线）
        var 半精度KV = true
        /// 超分输出缓冲（3×出高×出宽 fp32）：逐帧复用，避免每帧新建百 MB 缓冲
        var 输出缓冲: MTLBuffer?
        /// 整个流的帧数：结束条件的时间位置恒为「总帧数 - 1」（官方 condition_temporal_positions）
        var 总帧数 = 1
    }

    /// 注意力真正可见的视频帧数（当前帧 + 其前驱）：核内按 relative <= -时间窗 裁剪平面。
    /// 环容量与拼接缓冲都按它推导——比它更老的帧永远不会被读到，缓存与拷贝留着纯属浪费。
    private var 视频时间窗帧数: Int { 2 }

    /// 建立流：编码两端点条件 latent 并预热各层条件 KV，同时分配滚动视频 KV。
    /// - 参数条件形状：[条件通道数, 高/4, 宽/4] fp32 NCHW（Wan fp16 输出必须先转 fp32）
    func 建立流(起始条件: MTLBuffer,
              结束条件: MTLBuffer,
              高: Int,
              宽: Int,
              总帧数: Int,
              半精度KV: Bool = true) throws -> 流式状态 {
        let 块尺寸 = 形状.块分辨率
        guard 高 > 0, 宽 > 0, 高 % 块尺寸 == 0, 宽 % 块尺寸 == 0 else {
            throw 错误.初始化失败("尺寸 \(宽)x\(高) 必须是块分辨率 \(块尺寸) 的正整数倍")
        }
        let 令牌高 = 高 / 块尺寸
        let 令牌宽 = 宽 / 块尺寸
        let 令牌数 = 令牌高 * 令牌宽
        let 元素 = MemoryLayout<Float>.stride
        // 条件 latent 是 [条件通道数, 高/4, 宽/4] fp32，按本次任务的高宽推导精确容量；
        // 不能用注意力令牌数或 Flash 隐藏维，否则换素材尺寸后必然对不上。
        let 条件应有 = 配置.条件通道数 * 令牌高 * 令牌宽 * 元素
        for (标签, 条件) in [("起始", 起始条件), ("结束", 结束条件)] {
            guard 条件.length >= 条件应有 else {
                throw 错误.初始化失败("\(标签)条件 buffer \(条件.length) 字节 < 期望 \(条件应有)：需 fp32 NCHW [\(配置.条件通道数),\(令牌高),\(令牌宽)]（Wan 的 fp16 输出必须先转 fp32）")
            }
        }

        let 键值维 = 形状.KV组数 * 形状.头维
        // 滚动 KV 环容量 = 时间窗帧数 + 1：核内只保留 relative >= -时间窗 的平面，更老的帧读不到，
        // 因此环只需容纳「时间窗内前驱 + 当前帧」。原文按窗口帧数（16）分配时，多出的 13 个槽位
        // 每帧都要写一次、却永不参与注意力，纯属浪费带宽与显存，故收窄到实际参与深度（等价省略）。
        let 最大历史帧 = max(视频时间窗帧数 + 1, 1)

        释放临时()
        guard let 命令 = 队列.makeCommandBuffer() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }

        let 起始令牌 = 编码条件(命令, 起始条件, 令牌高, 令牌宽, "start")
        let 结束令牌 = 编码条件(命令, 结束条件, 令牌高, 令牌宽, "end")
        let 端点 = 新建半(2 * 令牌数 * 形状.维度, "端点令牌")
        let 行字节 = 令牌数 * 形状.维度 * MemoryLayout<UInt16>.stride
        拷贝(命令, 起始令牌, 0, 端点, 0, 行字节)
        拷贝(命令, 结束令牌, 0, 端点, 行字节, 行字节)

        // 条件 KV 常驻整个流；半精度时由 预热 逐层直出半精度（f32 中间量当层释放）
        let (条件键半, 条件值半) = 预热(命令,
                                 端点: 端点,
                                 令牌数: 令牌数,
                                 令牌高: 令牌高,
                                 令牌宽: 令牌宽,
                                 半精度: true)

        命令.commit()
        try 等待完成(命令)
        if let 执行错误 = 命令.error {
            throw 错误.初始化失败("Metal 执行失败：\(执行错误.localizedDescription)")
        }

        var 状态 = 流式状态()
        状态.令牌数 = 令牌数
        状态.令牌高 = 令牌高
        状态.令牌宽 = 令牌宽
        状态.键值维 = 键值维
        状态.维度 = 形状.维度
        状态.最大历史帧 = 最大历史帧
        状态.总帧数 = max(总帧数, 1)
        // 激活全链 fp16 后 KV 缓存一律半精度常驻，f32 KV 分支不再可达
        _ = 半精度KV
        let 键值元素字节 = MemoryLayout<UInt16>.stride
        状态.帧字节 = 令牌数 * 键值维 * 键值元素字节
        状态.半精度KV = true
        状态.条件键 = 条件键半
        状态.条件值 = 条件值半
        状态.视频键 = self.块.map { _ in 新建半(最大历史帧 * 令牌数 * 键值维, "视频 K 缓存") }
        状态.视频值 = self.块.map { _ in 新建半(最大历史帧 * 令牌数 * 键值维, "视频 V 缓存") }
        let 拼接元素 = (2 * 令牌数 + (最大历史帧 + 1) * 令牌数) * 键值维
        状态.拼接键 = 新建半(拼接元素, "KV 拼接 K")
        状态.拼接值 = 新建半(拼接元素, "KV 拼接 V")
        // 超分输出缓冲（每帧约百 MB）随会话常驻复用，不再每帧新建
        let 出高 = 令牌高 * 形状.重排倍数
        let 出宽 = 令牌宽 * 形状.重排倍数
        状态.输出缓冲 = 新建(3 * 出高 * 出宽, "超分输出（复用）")
        return 状态
    }

    /// 把已有流状态原地重置为新一窗（跨窗复用会话）。
    /// 滚动视频 KV、拼接缓冲、超分输出缓冲全部原地保留；只按新端点重算条件 KV，
    /// 半精度条件缓冲走 预热 的复用目标，不产生新分配。尺寸/精度不一致时抛错，由调用方回退 建立流。
    /// - 保留历史 = false：滚动视频 KV 视为全新起点，已出帧数归零（窗首帧历史帧数 0）。
    /// - 保留历史 = true：已出帧数续接上一窗，滚动槽位与历史深度自然接续。同一段视频的相邻窗口
    ///   用它，窗首帧就不再出现「历史归零」这种与窗内普通帧不同的生成条件。
    ///   条件 KV 仍按本窗端点重新预热，与历史延续互不影响。
    func 重置流(状态: inout 流式状态,
              起始条件: MTLBuffer,
              结束条件: MTLBuffer,
              总帧数: Int,
              保留历史: Bool = false) throws {
        guard 状态.令牌数 > 0, 状态.令牌高 > 0, 状态.令牌宽 > 0, 状态.键值维 > 0, 状态.维度 > 0,
              状态.条件键.count == 块.count, 状态.条件值.count == 块.count,
              状态.视频键.count == 块.count, 状态.视频值.count == 块.count,
              状态.输出缓冲 != nil else {
            throw 错误.初始化失败("流式状态不完整，无法复用（需重新建立流）")
        }
        let 令牌数 = 状态.令牌数
        let 元素 = MemoryLayout<Float>.stride
        // 端点条件是由 Wan 主干输出的 latent，形状恒为 [条件通道数, 令牌高, 令牌宽] fp32（每次前向一行）；
        // 「[2, 令牌数, 维度]」只是 编码条件 阶段临时拼出来的端点令牌，与本函数入参无关。
        // 因此容量必须按「条件通道数 × 本次令牌数 × 元素」推导，不能用 状态.维度（Flash 隐藏维）。
        let 端点应有 = 配置.条件通道数 * 令牌数 * 元素
        for (标签, 条件) in [("起始", 起始条件), ("结束", 结束条件)] {
            guard 条件.length >= 端点应有 else {
                throw 错误.初始化失败("\(标签)条件 buffer \(条件.length) 字节 < 期望 \(端点应有)")
            }
        }
        释放临时()
        guard let 命令 = 队列.makeCommandBuffer() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }
        let 起始令牌 = 编码条件(命令, 起始条件, 状态.令牌高, 状态.令牌宽, "start")
        let 结束令牌 = 编码条件(命令, 结束条件, 状态.令牌高, 状态.令牌宽, "end")
        let 端点 = 新建半(2 * 令牌数 * 状态.维度, "端点令牌")
        let 行字节 = 令牌数 * 状态.维度 * MemoryLayout<UInt16>.stride
        拷贝(命令, 起始令牌, 0, 端点, 0, 行字节)
        拷贝(命令, 结束令牌, 0, 端点, 行字节, 行字节)
        let (条件键, 条件值) = 预热(命令,
                                端点: 端点,
                                令牌数: 令牌数,
                                令牌高: 状态.令牌高,
                                令牌宽: 状态.令牌宽,
                                半精度: true,
                                键目标: 状态.条件键,
                                值目标: 状态.条件值)
        命令.commit()
        try 等待完成(命令)
        if let 执行错误 = 命令.error {
            throw 错误.初始化失败("Metal 执行失败：\(执行错误.localizedDescription)")
        }
        状态.条件键 = 条件键
        状态.条件值 = 条件值
        // 保留历史时不动已出帧数：滚动槽位 = 已出帧数 % 最大历史帧 才能与上一窗写完的槽位对齐，
        // 历史帧数 = min(已出帧数, 最大历史帧) 也才能接续到满历史，窗首帧与窗内帧条件一致。
        if !保留历史 {
            状态.已出帧数 = 0
        }
        状态.总帧数 = max(1, 总帧数)
    }

    /// 流式解码一帧：条件 KV 复用 + 最近「窗口帧数 - 1」帧视频 KV 滚动参与注意力。
    /// - 低清：fp32 NCHW [3, 高, 宽]，值域 [-1, 1]
    func 流式前向(低清: MTLBuffer, 状态: inout 流式状态) throws -> 帧输出 {
        let 令牌数 = 状态.令牌数
        let 令牌高 = 状态.令牌高
        let 令牌宽 = 状态.令牌宽
        let 键值维 = 状态.键值维
        let 维度 = 状态.维度
        let 最大历史帧 = max(状态.最大历史帧, 1)
        let 帧字节 = 状态.帧字节
        let 状态总帧数 = 状态.总帧数
        let 已出帧数 = 状态.已出帧数

        // 历史深度上限 = 环容量 − 1（当前帧自己占一格），保证历史帧不与当前帧抢同一槽位。
        let 历史帧数 = min(已出帧数, max(最大历史帧 - 1, 0))
        let 总帧数 = 历史帧数 + 1
        // 注意力真正可见的视频平面数：核内按 relative <= -时间窗 裁剪，
        // 比时间窗更老的帧既不会被读到，组装/拷贝/RoPE 也就不该产出（等价省略）。
        let 时间窗帧数 = 视频时间窗帧数
        let 参与帧数 = min(总帧数, 时间窗帧数)
        let 条件行数 = 2 * 令牌数
        let 视频键缓存 = 状态.视频键
        let 视频值缓存 = 状态.视频值
        let 条件键 = 状态.条件键
        let 条件值 = 状态.条件值
        guard let 拼接键 = 状态.拼接键, let 拼接值 = 状态.拼接值 else {
            throw 错误.初始化失败("流式状态未初始化，请先调用 建立流")
        }
        let 元素 = MemoryLayout<Float>.stride
        let 像素高 = 令牌高 * 形状.块分辨率
        let 像素宽 = 令牌宽 * 形状.块分辨率
        guard 低清.length >= 3 * 像素高 * 像素宽 * 元素 else {
            throw 错误.初始化失败("低清 buffer \(低清.length) 字节 < 期望 \(3 * 像素高 * 像素宽 * 元素)：需 fp32 NCHW [3,\(像素高),\(像素宽)]、值域 [-1,1]")
        }

        释放临时()
        guard let 命令 = 队列.makeCommandBuffer() else {
            throw 错误.初始化失败("命令缓冲创建失败")
        }

        let 槽 = 已出帧数 % 最大历史帧
        let 低清令牌 = 编码低清帧(命令, 低清: 低清, 高: 像素高, 宽: 像素宽)

        let 视频窗 = (高: 配置.空间注意力.视频窗[0], 宽: 配置.空间注意力.视频窗[1])
        var 窗口 = [
            (高: 配置.空间注意力.起始条件窗[0], 宽: 配置.空间注意力.起始条件窗[1]),
            (高: 配置.空间注意力.结束条件窗[0], 宽: 配置.空间注意力.结束条件窗[1])
        ]
        窗口.append(contentsOf: Array(repeating: 视频窗, count: 参与帧数))

        let 隐藏 = 低清令牌
        // 单块（注意力 + 前馈）编码抽成局部闭包，便于按命令缓冲分段提交。
        // 同一帧内 arena 每次 借() 都取新缓冲、不复用，分两段提交不会产生别名竞争。
        func 编码块(_ 命令: MTLCommandBuffer, _ 索引: Int, _ 层: 块权重) {
            let 归一 = 借(令牌数 * 维度)
            归一化(命令, 隐藏, 层.注意力归一化, 令牌数, 归一)

            let 查询 = 借(令牌数 * 维度)
            线性(命令, 归一, 令牌数, 层.注意力.查询, 查询, false)
            施加RoPE(命令, 查询, 令牌数, 形状.头数, 令牌宽)

            // 视频 K 不旋转入缓存（官方语义：缓存 key 每步按相对时间位置重新旋转）
            let 当前键 = 借(令牌数 * 键值维)
            线性(命令, 归一, 令牌数, 层.注意力.键, 当前键, false)
            分组归一化(命令, 当前键, 层.注意力.键归一化, 令牌数 * 形状.KV组数, 形状.头维)

            let 当前值 = 借(令牌数 * 键值维)
            线性(命令, 归一, 令牌数, 层.注意力.值, 当前值, false)
            分组归一化(命令, 当前值, 层.注意力.值归一化, 令牌数 * 形状.KV组数, 形状.头维)

            // 当前帧 K/V 已是半精度，直接 blit 进滚动缓存
            拷贝(命令, 当前键, 0, 视频键缓存[索引], 槽 * 帧字节, 帧字节)
            拷贝(命令, 当前值, 0, 视频值缓存[索引], 槽 * 帧字节, 帧字节)

            // 组装：[起始条件, 结束条件] + 最近 总帧数 帧视频（按时间升序，末帧为当前帧）
            let 条件字节 = 条件行数 * 键值维 * MemoryLayout<UInt16>.stride
            拷贝(命令, 条件键[索引], 0, 拼接键, 0, 条件字节)
            拷贝(命令, 条件值[索引], 0, 拼接值, 0, 条件字节)
            let 首帧序 = 已出帧数 - (参与帧数 - 1)
            for j in 0..<参与帧数 {
                let 源槽 = (首帧序 + j) % 最大历史帧
                let 目标偏移 = 条件字节 + j * 帧字节
                拷贝(命令, 视频键缓存[索引], 源槽 * 帧字节, 拼接键, 目标偏移, 帧字节)
                拷贝(命令, 视频值缓存[索引], 源槽 * 帧字节, 拼接值, 目标偏移, 帧字节)
            }
            // 条件区按官方 condition_temporal_positions=(0, 总帧数-1) 旋转：
            // 起始条件 t=0，结束条件 t=总帧数-1（与当前帧索引无关）
            施加RoPE(命令, 拼接键, 条件行数, 形状.KV组数, 令牌宽, 令牌数,
                    时间位置: 0,
                    平面步长: Int32(max(状态总帧数, 1) - 1),
                    半精度: true)
            // 视频区逐平面施加 RoPE：最老帧 t=-(参与帧数-1)，逐平面 +1，当前帧 t=0
            施加RoPE(命令, 拼接键, 参与帧数 * 令牌数, 形状.KV组数, 令牌宽, 令牌数,
                    时间位置: Int32(-(参与帧数 - 1)),
                    平面步长: 1,
                    缓冲偏移: 条件字节,
                    半精度: true)

            let 注意力输出 = 借(令牌数 * 维度)
            窗口注意力(命令, 查询, 拼接键, 拼接值, 注意力输出,
                    批次数: 1, 每批令牌: 令牌数,
                    令牌高: 令牌高, 令牌宽: 令牌宽,
                    平面数: 2 + 参与帧数, 条件平面数: 2, 窗口: 窗口, 时间窗: 时间窗帧数,
                    半精度: true)

            let 投影 = 借(令牌数 * 维度)
            线性(命令, 注意力输出, 令牌数, 层.注意力.输出, 投影, false)
            原地相加(命令, 隐藏, 投影, 令牌数 * 维度)

            let 前馈归一 = 借(令牌数 * 维度)
            归一化(命令, 隐藏, 层.前馈归一化, 令牌数, 前馈归一)
            let 门控 = 借(令牌数 * 2 * 形状.隐藏维)
            线性(命令, 前馈归一, 令牌数, 层.前馈门, 门控, false, 仅矩阵乘: true)
            let 激活 = 借(令牌数 * 形状.隐藏维)
            SwiGLU(命令, 门控, 激活, 令牌数, 层.前馈门.偏置)
            let 降维 = 借(令牌数 * 维度)
            线性(命令, 激活, 令牌数, 层.前馈降维, 降维, false)
            原地相加(命令, 隐藏, 降维, 令牌数 * 维度)
        }

        // 分段提交：前半段一编码完就 commit，GPU 立即开跑，CPU 继续编码后半段，
        // 避免一个超大命令缓冲把 CPU 阻塞到整帧编码结束才让 GPU 启动。
        let 总块数 = 块.count
        let 中点 = max((总块数 + 1) / 2, 1)
        for (索引, 层) in 块.enumerated() where 索引 < 中点 { 编码块(命令, 索引, 层) }
        命令.commit()
        guard let 后半命令 = 队列.makeCommandBuffer() else {
            throw 错误.初始化失败("命令缓冲创建失败（后半段）")
        }
        for (索引, 层) in 块.enumerated() where 索引 >= 中点 { 编码块(后半命令, 索引, 层) }

        let 像素 = 重建(后半命令,
                       隐藏: 隐藏,
                       低清: 低清,
                       令牌数: 令牌数,
                       令牌高: 令牌高,
                       令牌宽: 令牌宽,
                       高: 像素高,
                       宽: 像素宽,
                       输出槽: 状态.输出缓冲)

        后半命令.commit()
        try 等待完成(后半命令)
        if let 执行错误 = 命令.error ?? 后半命令.error {
            let 已完成 = 命令.status == .completed && 后半命令.status == .completed
            let 诊断 = "显存 device=\(设备.currentAllocatedSize / 1_048_576)MB 上限=\(设备.recommendedMaxWorkingSetSize / 1_048_576)MB 借池=\(临时缓冲.reduce(0) { $0 + $1.length } / 1_048_576)MB/\(临时缓冲.count)块 完成=\(已完成)"
            throw 错误.初始化失败("Metal 执行失败：\(执行错误.localizedDescription)｜\(诊断)")
        }
        状态.已出帧数 += 1
        return 帧输出(像素: 像素, 高: 令牌高 * 形状.重排倍数, 宽: 令牌宽 * 形状.重排倍数)
    }

    /// 重建像素：[N,512] → [3, 4高, 4宽]
    private func 重建(_ 命令: MTLCommandBuffer,
                   隐藏: MTLBuffer,
                   低清: MTLBuffer,
                   令牌数: Int,
                   令牌高: Int,
                   令牌宽: Int,
                   高: Int,
                   宽: Int,
                   输出槽: MTLBuffer? = nil) -> MTLBuffer {
        let 头特征 = 借(令牌数 * 配置.空间头维度)
        线性(命令, 隐藏, 令牌数, 空间头, 头特征, true)

        let 残差维 = 3 * 形状.重排倍数 * 形状.重排倍数
        let 残差 = 借(令牌数 * 残差维)
        线性(命令, 头特征, 令牌数, 残差投影, 残差, false)

        let 出高 = 令牌高 * 形状.重排倍数
        let 出宽 = 令牌宽 * 形状.重排倍数
        let 残差图 = 借(3 * 出高 * 出宽)
        var 重排参数 = PixelShuffleParams(tokens: Int32(令牌数),
                                         tokenWidth: Int32(令牌宽),
                                         scale: Int32(形状.重排倍数),
                                         channels: 3,
                                         outWidth: Int32(出宽),
                                         outHeight: Int32(出高))
        派发(命令, 管线重排半, MTLSize(width: 出宽, height: 出高, depth: 1),
            MTLSize(width: 16, height: 16, depth: 1)) { 编码器 in
            编码器.setBuffer(残差, offset: 0, index: 0)
            编码器.setBuffer(残差图, offset: 0, index: 1)
            编码器.setBytes(&重排参数, length: MemoryLayout<PixelShuffleParams>.stride, index: 2)
        }

        let 需要字节 = 3 * 出高 * 出宽 * MemoryLayout<Float>.stride
        let 输出: MTLBuffer
        if let 槽 = 输出槽, 槽.length >= 需要字节 {
            输出 = 槽
        } else {
            输出 = 新建(3 * 出高 * 出宽, "超分输出")
        }
        var 上采样参数 = BicubicParams(inH: Int32(高), inW: Int32(宽),
                                      outH: Int32(出高), outW: Int32(出宽),
                                      scale: Int32(配置.放大倍数), channels: 3)
        派发(命令, 管线双三次半, MTLSize(width: 出宽, height: 出高, depth: 1),
            MTLSize(width: 16, height: 16, depth: 1)) { 编码器 in
            编码器.setBuffer(低清, offset: 0, index: 0)
            编码器.setBuffer(残差图, offset: 0, index: 1)
            编码器.setBuffer(输出, offset: 0, index: 2)
            编码器.setBytes(&上采样参数, length: MemoryLayout<BicubicParams>.stride, index: 3)
        }
        return 输出
    }
}
