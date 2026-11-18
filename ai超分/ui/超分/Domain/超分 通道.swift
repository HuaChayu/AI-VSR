import SwiftUI
import Metal
import CoreGraphics
import ImageIO
import QuartzCore

/// 性能探针：只累计各阶段真实耗时（毫秒），供整段运行结束后写基线摘要。
/// 只做加法与快照，不参与任何计算路径，也不影响任何缓冲/管线状态。
final class 性能探针: @unchecked Sendable {
    static let 共享 = 性能探针()
    private let 锁 = NSLock()
    private var 分段合计: [String: Double] = [:]

    func 重置() {
        锁.lock(); 分段合计 = [:]; 锁.unlock()
    }

    func 累加(_ 分段: [String: Double]) {
        锁.lock()
        for (键, 值) in 分段 { 分段合计[键, default: 0] += 值 }
        锁.unlock()
    }

    func 快照() -> [String: Double] {
        锁.lock(); defer { 锁.unlock() }; return 分段合计
    }
}

/// 超分阶段的错误。
nonisolated struct 超分错误: Error, CustomStringConvertible {
    let 说明: String
    var description: String { 说明 }
}

/// 单帧超分结果：单帧预览与整段视频逐帧共用同一份产物描述。
nonisolated struct 超分单帧结果 {
    var 图: CGImage
    /// 输入素材本身的像素尺寸（图片为原图解全分辨率、视频为源帧显示尺寸），供对比层与文案如实展示。
    var 输入宽: Int
    var 输入高: Int
    /// 送模型的低清尺寸（短边封顶后对齐 8），与 输入宽高 是两回事，不可混用。
    var 低清宽: Int
    var 低清高: Int
    var 输出宽: Int
    var 输出高: Int
    var flash毫秒: Double
    var wan毫秒: Double?
    var 分段: [String: Double]
}

/// Flash 引擎常驻宿主：权重只加载一次，推理串行执行。
nonisolated final class Flash宿主: @unchecked Sendable {
    static let 共享 = Flash宿主()
    static let 权重路径 = 文档目录.模型路径("flash.safetensors")
    /// 低清帧 → 条件潜变量（[48, 高/4, 宽/4]）全零：单帧推理不启用端点条件。
    static let 条件通道 = 48

    private let 锁 = NSLock()
    private var 引擎: Flash引擎?
    private var 设备: MTLDevice?
    /// 流式窗口会话缓存：跨窗复用滚动 KV / 拼接 / 输出缓冲，尺寸变化时整份重建
    private var 窗口会话: 流式会话?
    /// 单帧（图片）专用会话：与视频窗口会话完全隔离，避免图片预览打断正在出片的视频历史
    private var 单帧会话: 流式会话?

    static var 权重就绪: Bool {
        FileManager.default.fileExists(atPath: 权重路径)
    }

    struct 输出 {
        var 像素: [Float]     // NCHW，值域 [-1, 1]
        var 宽: Int
        var 高: Int
    }

    /// 单帧 4x 超分（单帧流式窗口）。字节为 RGBA8（sRGB，行 0 = 图像顶部）。
    /// `条件` 为 Wan 主干给出的端点 latent（[48, 高/4, 宽/4]，planar）；传 nil 则退化为零条件。
    /// 总帧数 决定条件端点 RoPE 的平面步长（端点位置 = (0, 总帧数-1)），窗口总帧数取 max(2, 总帧数)。
    /// 与整段视频共用同一条流式前向，并持有独立会话缓存，不触碰正在出片的视频窗口会话。
    func 单帧流式超分(字节: [UInt8], 宽: Int, 高: Int,
               条件: [Float]? = nil,
               总帧数: Int = 1) throws -> (输出: 输出, 耗时毫秒: Double) {
        锁.lock()
        defer { 锁.unlock() }

        guard 宽 > 0, 高 > 0, 宽 % 4 == 0, 高 % 4 == 0 else {
            throw 超分错误(说明: "输入尺寸 \(宽)×\(高) 不是 4 的倍数")
        }
        guard 字节.count >= 宽 * 高 * 4 else {
            throw 超分错误(说明: "像素缓冲长度不足")
        }
        guard Self.权重就绪 else {
            throw 超分错误(说明: "找不到权重：\(Self.权重路径)")
        }

        let 设备: MTLDevice
        if let 现有 = self.设备 {
            设备 = 现有
        } else {
            guard let 找到 = MTLCreateSystemDefaultDevice() else {
                throw 超分错误(说明: "找不到 Metal 设备")
            }
            设备 = 找到
            self.设备 = 找到
        }

        let 引擎: Flash引擎
        if let 现有 = self.引擎 {
            引擎 = 现有
        } else {
            引擎 = try Flash引擎(设备: 设备, 权重路径: Self.权重路径)
            self.引擎 = 引擎
        }

        // RGBA8 → NCHW float [-1, 1]
        let 平面 = 宽 * 高
        var 值 = [Float](repeating: 0, count: 3 * 平面)
        for i in 0..<平面 {
            let 基 = i * 4
            值[i] = Float(字节[基]) * (2.0 / 255.0) - 1.0
            值[平面 + i] = Float(字节[基 + 1]) * (2.0 / 255.0) - 1.0
            值[2 * 平面 + i] = Float(字节[基 + 2]) * (2.0 / 255.0) - 1.0
        }

        let 低清长度 = 3 * 平面 * MemoryLayout<Float>.stride
        guard let 低清 = 设备.makeBuffer(length: 低清长度, options: .storageModeShared) else {
            throw 超分错误(说明: "无法分配低清缓冲")
        }
        值.withUnsafeBytes { raw in
            if let 源 = raw.baseAddress {
                memcpy(低清.contents(), 源, 低清长度)
            }
        }

        let 条件高 = 高 / 4
        let 条件宽 = 宽 / 4
        let 条件个数 = Self.条件通道 * 条件高 * 条件宽
        let 条件值: [Float]
        if let 传入 = 条件, 传入.count == 条件个数 {
            条件值 = 传入
        } else {
            条件值 = [Float](repeating: 0, count: 条件个数)
        }
        let 起 = try 端点缓冲(条件值, 宽: 宽, 高: 高, 设备: 设备)
        let 止 = try 端点缓冲(条件值, 宽: 宽, 高: 高, 设备: 设备)

        // 单帧 = 只出一帧的流式窗口（历史深度 0）：条件端点按官方 (0, 总帧数-1) 旋转，
        // 与逐帧视频完全同一条流式前向，不再走会在 16px 块边界留下网格的一次性前向。
        let 窗口总帧数 = max(2, 总帧数)
        var 状态: Flash引擎.流式状态
        if var 现有 = 单帧会话, 现有.宽 == 宽, 现有.高 == 高 {
            try 引擎.重置流(状态: &现有.状态, 起始条件: 起, 结束条件: 止,
                          总帧数: 窗口总帧数, 保留历史: false)
            单帧会话 = 现有
            状态 = 现有.状态
        } else {
            状态 = try 引擎.建立流(起始条件: 起, 结束条件: 止, 高: 高, 宽: 宽,
                                 总帧数: 窗口总帧数)
            单帧会话 = 流式会话(状态: 状态, 宽: 宽, 高: 高)
        }

        let 开始 = Date()
        let 结果 = try 引擎.流式前向(低清: 低清, 状态: &状态)
        let 耗时 = Date().timeIntervalSince(开始) * 1000

        guard 结果.宽 > 0, 结果.高 > 0 else {
            throw 超分错误(说明: "引擎输出尺寸异常")
        }
        let 输出平面 = 结果.宽 * 结果.高
        let 指针 = 结果.像素.contents().bindMemory(to: Float.self, capacity: 3 * 输出平面)
        let 像素 = Array(UnsafeBufferPointer(start: 指针, count: 3 * 输出平面))

        return (输出(像素: 像素, 宽: 结果.宽, 高: 结果.高), 耗时)
    }

    // MARK: - 窗口流式（对齐官方 streaming.py：一窗只跑两端点 Wan，中间帧走滚动 KV）

    struct 流式会话 {
        var 状态: Flash引擎.流式状态
        var 宽: Int
        var 高: Int
    }

    private func 资源() throws -> (设备: MTLDevice, 引擎: Flash引擎) {
        let 设备: MTLDevice
        if let 现有 = self.设备 {
            设备 = 现有
        } else {
            guard let 找到 = MTLCreateSystemDefaultDevice() else {
                throw 超分错误(说明: "找不到 Metal 设备")
            }
            设备 = 找到
            self.设备 = 找到
        }
        let 引擎: Flash引擎
        if let 现有 = self.引擎 {
            引擎 = 现有
        } else {
            引擎 = try Flash引擎(设备: 设备, 权重路径: Self.权重路径)
            self.引擎 = 引擎
        }
        return (设备, 引擎)
    }

    private func 端点缓冲(_ 值: [Float], 宽: Int, 高: Int, 设备: MTLDevice) throws -> MTLBuffer {
        let 条件高 = 高 / 4
        let 条件宽 = 宽 / 4
        let 个数 = Self.条件通道 * 条件高 * 条件宽
        let 长度 = 个数 * MemoryLayout<Float>.stride
        guard 值.count == 个数 else {
            throw 超分错误(说明: "端点条件元素数不符：\(值.count) ≠ \(个数)")
        }
        guard let 缓冲 = 设备.makeBuffer(length: 长度, options: .storageModeShared) else {
            throw 超分错误(说明: "无法分配条件缓冲")
        }
        值.withUnsafeBytes { raw in
            if let 源 = raw.baseAddress {
                memcpy(缓冲.contents(), 源, 长度)
            }
        }
        return 缓冲
    }

    private func 像素缓冲(_ 字节: [UInt8], 宽: Int, 高: Int, 设备: MTLDevice) throws -> MTLBuffer {
        guard 宽 > 0, 高 > 0, 宽 % 4 == 0, 高 % 4 == 0 else {
            throw 超分错误(说明: "输入尺寸 \(宽)×\(高) 不是 4 的倍数")
        }
        guard 字节.count >= 宽 * 高 * 4 else {
            throw 超分错误(说明: "像素缓冲长度不足")
        }
        let 平面 = 宽 * 高
        var 值 = [Float](repeating: 0, count: 3 * 平面)
        for i in 0..<平面 {
            let 基 = i * 4
            值[i] = Float(字节[基]) * (2.0 / 255.0) - 1.0
            值[平面 + i] = Float(字节[基 + 1]) * (2.0 / 255.0) - 1.0
            值[2 * 平面 + i] = Float(字节[基 + 2]) * (2.0 / 255.0) - 1.0
        }
        let 长度 = 3 * 平面 * MemoryLayout<Float>.stride
        guard let 低清 = 设备.makeBuffer(length: 长度, options: .storageModeShared) else {
            throw 超分错误(说明: "无法分配低清缓冲")
        }
        值.withUnsafeBytes { raw in
            if let 源 = raw.baseAddress {
                memcpy(低清.contents(), 源, 长度)
            }
        }
        return 低清
    }

    /// 建立/重置一窗的流式会话。
    /// - 延续历史 = true 表示这一窗接着上一窗继续（同一段视频的相邻窗口）：
    ///   滚动视频 KV 不清零，窗首帧与窗内普通帧拥有同样深度的历史，边界不再出现「历史深度突变」。
    func 建立窗口(起始条件: [Float], 结束条件: [Float], 宽: Int, 高: Int,
               总帧数: Int, 延续历史: Bool = false) throws -> 流式会话 {
        锁.lock()
        defer { 锁.unlock() }

        guard Self.权重就绪 else {
            throw 超分错误(说明: "找不到权重：\(Self.权重路径)")
        }
        let (设备, 引擎) = try 资源()
        let 起 = try 端点缓冲(起始条件, 宽: 宽, 高: 高, 设备: 设备)
        let 止 = try 端点缓冲(结束条件, 宽: 宽, 高: 高, 设备: 设备)
        // 跨窗复用：尺寸不变时只重置端点条件，滚动 KV / KV 拼接 / 超分输出缓冲原地保留；
        // 尺寸或精度变化则整份重建，任意素材尺寸都走同一条通用路径。
        if var 现有 = 窗口会话, 现有.宽 == 宽, 现有.高 == 高 {
            do {
                try 引擎.重置流(状态: &现有.状态,
                              起始条件: 起,
                              结束条件: 止,
                              总帧数: max(1, 总帧数),
                              保留历史: 延续历史)
                窗口会话 = 现有
                return 现有
            } catch {
                窗口会话 = nil
            }
        }
        let 状态 = try 引擎.建立流(起始条件: 起, 结束条件: 止, 高: 高, 宽: 宽, 总帧数: max(1, 总帧数))
        let 会话 = 流式会话(状态: 状态, 宽: 宽, 高: 高)
        窗口会话 = 会话
        return 会话
    }

    /// 显存/借池占用概览（诊断用，透传引擎统计）。
    func 占用概览(_ 会话: 流式会话?) -> String {
        guard let 引擎有 = 引擎 else { return "引擎未就绪" }
        return 引擎有.占用概览(会话?.状态)
    }

    func 流式超分(字节: [UInt8], 宽: Int, 高: Int, 会话: inout 流式会话) throws -> (输出: 输出, 耗时毫秒: Double) {
        锁.lock()
        defer { 锁.unlock() }
        // 逐帧自动释放池：CGImage / CIImage / CVPixelBuffer 等 Objective-C 对象每帧排空，
        // 避免长期驻留的推理线程（无 RunLoop 自动释放池）把它们攒到进程结束。
        return try autoreleasepool { () throws -> (输出: 输出, 耗时毫秒: Double) in
            guard 会话.宽 == 宽, 会话.高 == 高 else {
                throw 超分错误(说明: "流式会话尺寸不符：\(会话.宽)×\(会话.高) ≠ \(宽)×\(高)")
            }
            let (设备, 引擎) = try 资源()
            let 低清 = try 像素缓冲(字节, 宽: 宽, 高: 高, 设备: 设备)

            let 开始 = Date()
            let 结果 = try 引擎.流式前向(低清: 低清, 状态: &会话.状态)
            let 耗时 = Date().timeIntervalSince(开始) * 1000

            guard 结果.宽 > 0, 结果.高 > 0 else {
                throw 超分错误(说明: "引擎输出尺寸异常")
            }
            let 输出平面 = 结果.宽 * 结果.高
            let 指针 = 结果.像素.contents().bindMemory(to: Float.self, capacity: 3 * 输出平面)
            let 像素 = Array(UnsafeBufferPointer(start: 指针, count: 3 * 输出平面))
            return (输出(像素: 像素, 宽: 结果.宽, 高: 结果.高), 耗时)
        }
    }
}

/// 超分通道：把当前素材的预览帧送进 Flash 引擎，做 4x 出图。
@MainActor
@Observable
final class 超分通道 {
    enum 进度: Equatable {
        case 空闲
        case 运行
        case 完成
        case 失败
    }

    struct 超分产物 {
        var 输出图: NSImage
        var 输出CG: CGImage
        var 输入图: NSImage
        var 耗时毫秒: Double
        /// 输入素材本身尺寸（图片=原图解全分辨率，视频=源帧显示尺寸）：对比层右侧展示的就是这份像素。
        var 输入宽: Int
        var 输入高: Int
        /// 送模型推理的低清尺寸（短边封顶后对齐 8），只用于说明推理口径，不代表右侧对比内容。
        var 低清宽: Int
        var 低清高: Int
        var 输出宽: Int
        var 输出高: Int
        /// 色彩通路分段耗时（毫秒）：前置解码/线性缩放、Wan、Flash、回归线性/调色/输出。
        var 分段毫秒: [String: Double] = [:]
        var 调色: 色彩通路.调色参数 = 色彩通路.调色参数()

        var 输入尺寸文本: String { "\(输入宽) × \(输入高)" }
        var 低清尺寸文本: String { "\(低清宽) × \(低清高)" }
        var 输出尺寸文本: String { "\(输出宽) × \(输出高)" }
        var 耗时文本: String { String(format: "%.0f ms", 耗时毫秒) }
    }

    /// 整段超分成片。
    struct 视频超分成片 {
        var 输出URL: URL
        var 帧数: Int
        /// 源视频显示尺寸（转正后）。
        var 输入宽: Int
        var 输入高: Int
        /// 送模型推理的低清尺寸（短边封顶后对齐 8）。
        var 低清宽: Int
        var 低清高: Int
        var 输出宽: Int
        var 输出高: Int
        var 耗时秒: Double
        var 含音轨: Bool
        var 体积字节: Int64

        var 尺寸文本: String { "\(输入宽) × \(输入高) · 低清 \(低清宽) × \(低清高) → \(输出宽) × \(输出高)" }
        var 耗时文本: String { String(format: "%d 帧 · %.1f s", 帧数, 耗时秒) }
        var 体积文本: String { ByteCountFormatter.string(fromByteCount: 体积字节, countStyle: .file) }
    }

    /// 低清侧短边上限（自适应：横屏/竖屏/方屏共用同一短边，输出 4 倍后短边恒定）。
    /// 只缩不放：源短边已小于上限时按原尺寸输入，避免放大零信息增量。
    var 输入短边 = 540
    var 阶段: 进度 = .空闲
    var 提示: String?
    /// 当前产物来自磁盘缓存（切走素材再切回时按落盘索引重建），用于文案区分「刚跑完」与「已超分（缓存）」。
    var 缓存命中 = false
    var 产物: 超分产物?
    /// 单帧超分结果落地的 PNG 路径（图片素材自动写入，视频不产生）。
    var 单帧文件: URL?
    /// 整段视频超分成片（逐帧超分 + 编码落盘的结果）。
    var 视频产物: 视频超分成片?
    /// 整段是否正在跑（用来把状态文案切到「帧进度」口径）。
    var 视频运行 = false
    /// 当前整段产物是否为动图（GIF 多帧）通路：只用于文案口径，进度/取消/落盘复用视频那一套。
    var 动图运行 = false
    var 视频已处理 = 0
    var 视频总数 = 0

    /// 整段进度比例（0...1），无进度时返回 nil。
    var 视频进度比例: Double? {
        guard 视频运行, 视频总数 > 0 else { return nil }
        return min(1, Double(视频已处理) / Double(视频总数))
    }
    /// 预览区显示超分结果还是原帧
    var 显示超分 = true

    /// 线性光调色参数（曝光/对比/饱和，全部作用在峰值线性光上）。
    var 调色参数 = 色彩通路.调色参数()
    /// 输出 sRGB 8bit 时是否启用 TPDF 抖动（关闭仅用于对比测试，产品路径默认开）。
    var 输出抖动 = true

    /// 整段超分的取消开关（跨线程读取）。
    private let 取消开关 = 视频超分.取消开关()

    /// 任务代次：每次 清空() 自增。在途的异步任务回写结果前比对代次，
    /// 保证「换素材 / 清空素材箱」之后旧素材的结果不会复活到界面上。
    private var 任务代次 = 0

    /// 素材栏「批量超分」的队列状态。挂在通道上而不是视图 @State 上，
    /// 这样视图身份变化（切 Tab、面板重建）不会丢失正在跑的批量进度。
    let 批量队列 = 批量超分队列()

    /// 是否使用 Wan 5B 主干生成端点条件（关闭则退化为零条件 Flash，画质显著下降）。
    var 启用Wan = true
    /// 上一次运行里 Wan 主干耗时，便于观察两段各自开销。
    var wan耗时毫秒: Double?

    var 运行中: Bool { 阶段 == .运行 }
    var 可运行: Bool { Flash宿主.权重就绪 && (!启用Wan || Wan宿主.权重就绪) }
    var 缺权重提示: String? {
        if !Flash宿主.权重就绪 { return "找不到 Flash 权重：\(Flash宿主.权重路径)" }
        if 启用Wan && !Wan宿主.权重就绪 { return "找不到 Wan 权重：\(Wan宿主.权重路径)" }
        return nil
    }

    var 按钮标题: String {
        if 视频运行 { return 动图运行 ? "动图超分中…" : "整段超分中…" }
        if 运行中 { return "超分中…" }
        if 视频产物 != nil {
            let 主体 = 动图运行 ? "动图" : "整段"
            return 缓存命中 ? "重新超分\(主体)（缓存）" : "重新超分\(主体)"
        }
        if 产物 != nil { return 缓存命中 ? "重新超分（缓存）" : "重新超分" }
        return "开始超分"
    }

    var 状态文本: String {
        if 视频运行 {
            let 总 = max(视频总数, 视频已处理)
            let 进度 = 总 > 0 ? "\(视频已处理)/\(总) 帧" : "首帧准备中"
            let 主体 = 动图运行 ? "动图超分中" : "整段超分中"
            return (启用Wan ? 主体 + " Wan + Flash · " : 主体 + " Flash · ") + 进度
        }
        switch 阶段 {
        case .空闲: return 可运行 ? (启用Wan ? "Wan + Flash 4× 就绪" : "Flash 4× 就绪（零条件）") : "缺少权重"
        case .运行: return 启用Wan ? "Wan + Flash 4× 推理中…" : "Flash 4× 推理中…"
        case .完成:
            if let 视频产物 {
                let 主体 = 动图运行 ? "动图超分完成" : "整段超分完成"
                return (缓存命中 ? 主体 + "（缓存） " : 主体 + " ") + 视频产物.耗时文本
            }
            if 缓存命中 {
                let 尺寸 = 产物?.输出尺寸文本 ?? "已落盘"
                return "已超分（缓存）· 输出 " + 尺寸
            }
            return (启用Wan ? "Wan + Flash 4× 完成 " : "Flash 4× 完成 ") + (产物?.耗时文本 ?? "")
        case .失败: return "超分失败"
        }
    }

    // MARK: 运行

    /// 统一入口：按素材自身的类型分流——视频走整段逐帧超分，静态图片走单帧，
    /// 动图（GIF 等多帧图片，类型仍是 .图片）走逐帧超分并回写同格式 GIF。
    /// 三条通路共用 超分单帧 核心，因此图片单帧、视频逐帧、动图逐帧的像素口径完全一致。
    func 开始(素材 条目: 素材条目, 预览帧: NSImage?) {
        switch 条目.类型 {
        case .视频:
            开始视频(源: 条目.url)
        case .图片:
            // 动图不能只解首帧：帧数 > 1 时走多帧通路，逐帧超分后写回同格式 GIF。
            if 条目.是动图 {
                开始动图(源: 条目.url)
            } else {
                开始图片(url: 条目.url, 备用帧: 预览帧, 名称: 条目.名称)
            }
        }
    }

    /// 静态图片：按原文件解码取全分辨率位图后走单帧通道，结果自动落成 PNG。
    /// 目标尺寸仍由 输入短边 上限统一决定，不按素材名、扩展名或分辨率做任何特判。
    func 开始图片(url: URL, 备用帧: NSImage?, 名称: String) {
        guard !运行中 else { return }
        guard let 帧 = Self.原图(url: url, 备用: 备用帧) else {
            阶段 = .失败
            提示 = "读不到图片像素：" + url.lastPathComponent
            return
        }
        开始(帧: 帧, 落盘名: 名称 + "_超分4x", 缓存键: url)
    }

    /// 优先用 CGImageSource 直读原文件（像素尺寸与位深保真），失败再退回内存里的预览帧。
    nonisolated static func 原图(url: URL, 备用: NSImage?) -> CGImage? {
        if let 源 = CGImageSourceCreateWithURL(url as CFURL, nil),
           let 图 = CGImageSourceCreateImageAtIndex(源, 0, nil) {
            return 图
        }
        guard let 备用 else { return nil }
        var 区域 = CGRect(origin: .zero, size: 备用.size)
        return 备用.cgImage(forProposedRect: &区域, context: nil, hints: nil)
    }

    func 开始(帧: CGImage, 落盘名: String? = nil, 缓存键: URL? = nil) {
        guard !运行中 else { return }
        guard 可运行 else {
            阶段 = .失败
            提示 = 缺权重提示
            return
        }

        阶段 = .运行
        视频运行 = false
        提示 = nil
        视频产物 = nil
        单帧文件 = nil
        视频已处理 = 0
        视频总数 = 0

        let 短边上限 = min(max(输入短边, 64), 1536)
        let 原帧图 = NSImage(cgImage: 帧, size: NSSize(width: 帧.width, height: 帧.height))
        let 用Wan = 启用Wan
        let 调色值 = 调色参数
        let 抖动值 = 输出抖动
        let 代次 = 任务代次

        Task.detached(priority: .userInitiated) {
            do {
                let 单帧 = try Self.超分单帧(帧,
                                          短边上限: 短边上限,
                                          用Wan: 用Wan,
                                          调色: 调色值,
                                          抖动: 抖动值,
                                          总帧数: Self.单帧窗口总帧数)
                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.产物 = 超分产物(输出图: NSImage(cgImage: 单帧.图,
                                                    size: NSSize(width: 单帧.输出宽, height: 单帧.输出高)),
                                  输出CG: 单帧.图,
                                  输入图: 原帧图,
                                  耗时毫秒: 单帧.flash毫秒,
                                  输入宽: 单帧.输入宽,
                                  输入高: 单帧.输入高,
                                  低清宽: 单帧.低清宽,
                                  低清高: 单帧.低清高,
                                  输出宽: 单帧.输出宽,
                                  输出高: 单帧.输出高,
                                  分段毫秒: 单帧.分段,
                                  调色: 调色值)
                    self.wan耗时毫秒 = 单帧.wan毫秒
                    self.显示超分 = true
                    self.提示 = nil
                    self.缓存命中 = false
                    if let 落盘名 {
                        let 落盘 = Self.写单帧(单帧.图, 名称: 落盘名)
                        self.单帧文件 = 落盘
                        // 产物登记进落盘索引：切走素材再切回时靠它从 PNG 重建对比层。
                        if let 落盘, let 缓存键 {
                            超分缓存索引.共享.记录(超分缓存条目(源路径: 超分缓存索引.键(缓存键),
                                                                 类型: "图片",
                                                                 产物路径: 落盘.path,
                                                                 输入宽: 单帧.输入宽,
                                                                 输入高: 单帧.输入高,
                                                                 低清宽: 单帧.低清宽,
                                                                 低清高: 单帧.低清高,
                                                                 输出宽: 单帧.输出宽,
                                                                 输出高: 单帧.输出高,
                                                                 耗时毫秒: 单帧.flash毫秒))
                        }
                    }
                    self.阶段 = .完成
                }
            } catch {
                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.提示 = "\(error)"
                    self.阶段 = .失败
                }
            }
        }
    }

    /// 静态图片一次性前向的总帧数口径：官方 streaming.py:173 对窗口帧数取 max(2, len(window))，
    /// model.py:827 直接拒绝 total_frames < 2——单帧窗口也必须抬到 2，端点位置才按
    /// model.py:989-998 的 (0, total_frames-1) 分离；否则平面步长退化为 0，起止端点 RoPE 同相位。
    nonisolated static let 单帧窗口帧数 = 1
    nonisolated static var 单帧窗口总帧数: Int { max(2, 单帧窗口帧数) }

    /// 单帧超分核心：线性化解码 + 缩放 → Wan 出端点条件 → Flash 4× → 回归线性光 → 调色 → sRGB 编码。
    /// 单帧预览与整段视频逐帧共用这一条路径，保证预览与成片像素结果完全一致。
    nonisolated static func 超分单帧(_ 帧: CGImage,
                                 短边上限: Int,
                                 用Wan: Bool,
                                 调色: 色彩通路.调色参数,
                                 抖动: Bool,
                                 输出编码: 色彩通路.传递编码 = .sRGB,
                                 输入解码: 色彩通路.传递编码 = .sRGB,
                                 总帧数: Int = 1) throws -> 超分单帧结果 {
        guard let 目标 = 目标尺寸(帧, 短边上限: 短边上限) else {
            throw 超分错误(说明: "预览帧读取失败")
        }

        var 分段: [String: Double] = [:]

        // ① 解码 + 缩放：8bit 源码值按源声明的传递函数 → 线性光 fp16，缩放也在线性光域完成，无 8bit 中间缓冲
        let 前置起 = CACurrentMediaTime()
        let (设备, 队) = try 色彩通路.上下文()
        let 线性全 = try 色彩通路.线性图(from: 帧, 设备: 设备, 队: 队, 解码: 输入解码)
        let 线性低 = try 色彩通路.线性缩放(线性全, 到: 目标.宽, 高: 目标.高, 队: 队)
        // ② 模型域桥接：Flash/Wan 训练域为 sRGB gamma 域，进模型前编码回 sRGB（不抖动）
        let 模型域 = try 色彩通路.输出RGBA8(from: 线性低, 队: 队, 抖动: false)
        分段["前置"] = (CACurrentMediaTime() - 前置起) * 1000

        var 条件: [Float]?
        var wan耗时: Double?
        if 用Wan {
            // 唯一路径，与官方 streaming.py 同口径：每帧都用本帧自身跑一次 Wan 单步前向取端点条件，
            // 不做任何跨帧条件近似（不复用、不隔帧保持），质量零损失。
            let 端点 = try Wan宿主.共享.条件(低清: 模型域, 宽: 目标.宽, 高: 目标.高)
            条件 = 端点.潜变量
            wan耗时 = 端点.耗时毫秒
        }
        // 与整段视频逐帧共用同一条流式前向：单帧 = 只出 1 帧的流式窗口（历史深度 0），
        // 保证预览与成片像素结果同源；端点条件按 (0, 总帧数-1) 旋转，总帧数≥2 时端点相位分离。
        let 结果 = try Flash宿主.共享.单帧流式超分(字节: 模型域, 宽: 目标.宽, 高: 目标.高,
                                           条件: 条件,
                                           总帧数: 总帧数)
        guard 结果.输出.宽 > 0, 结果.输出.高 > 0 else {
            throw 超分错误(说明: "引擎输出尺寸异常")
        }

        // ③ 回归线性光 → 调色 → 输出编码：8bit 只在最后一步出现
        let 回归起 = CACurrentMediaTime()
        let 平面缓冲 = try 色彩通路.平面缓冲(结果.输出.像素, 设备: 设备)
        let 线性超 = try 色彩通路.线性纹理(from平面: 平面缓冲,
                                      宽: 结果.输出.宽,
                                      高: 结果.输出.高,
                                      设备: 设备,
                                      队: 队)
        let 线性成 = try 色彩通路.调色(线性超, 参数: 调色, 队: 队)
        let 输出字节 = try 色彩通路.输出RGBA8(from: 线性成, 队: 队, 抖动: 抖动, 传递: 输出编码)
        guard let 图 = 色彩通路.CG图(fromRGBA8: 输出字节, 宽: 结果.输出.宽, 高: 结果.输出.高,
                                  色彩: 输出编码.色彩空间) else {
            throw 超分错误(说明: "输出像素转图失败")
        }
        分段["回归"] = (CACurrentMediaTime() - 回归起) * 1000

        性能探针.共享.累加(分段)
        return 超分单帧结果(图: 图,
                        输入宽: 帧.width,
                        输入高: 帧.height,
                        低清宽: 目标.宽,
                        低清高: 目标.高,
                        输出宽: 结果.输出.宽,
                        输出高: 结果.输出.高,
                        flash毫秒: 结果.耗时毫秒,
                        wan毫秒: wan耗时,
                        分段: 分段)
    }

    /// 单帧端点潜变量：与逐帧超分同一套「转正 → 短边封顶 → 线性光 → 模型域」前置，
    /// 只跑一次 Wan 编码。整段视频用它取真实首帧/真实末帧两个端点，全程只跑两次。
    nonisolated static func 端点潜变量(_ 帧: CGImage,
                                  短边上限: Int,
                                  输入解码: 色彩通路.传递编码 = .sRGB) throws -> (潜变量: [Float], 耗时毫秒: Double) {
        let (设备, 队) = try 色彩通路.上下文()
        return try autoreleasepool {
            guard let 目标 = 目标尺寸(帧, 短边上限: 短边上限) else {
                throw 超分错误(说明: "端点帧读取失败")
            }
            let 线性全 = try 色彩通路.线性图(from: 帧, 设备: 设备, 队: 队, 解码: 输入解码)
            let 线性低 = try 色彩通路.线性缩放(线性全, 到: 目标.宽, 高: 目标.高, 队: 队)
            let 模型域 = try 色彩通路.输出RGBA8(from: 线性低, 队: 队, 抖动: false)
            let 端点 = try Wan宿主.共享.条件(低清: 模型域, 宽: 目标.宽, 高: 目标.高)
            return (端点.潜变量, 端点.耗时毫秒)
        }
    }

    /// 连续流式单帧：整段视频共用一条滚动 KV，只有第一帧建立窗口，之后每帧接续上一帧的历史。
    /// 端点条件由调用方给定（整段的真实首帧与真实末帧），窗口内不再逐窗重算，
    /// 因此任何源帧的生成条件都是「满历史 + 同一对端点」，不存在固定周期的边界跳变。
    nonisolated static func 超分流式帧(_ 帧: CGImage,
                                  短边上限: Int,
                                  调色: 色彩通路.调色参数,
                                  抖动: Bool,
                                  输出编码: 色彩通路.传递编码 = .sRGB,
                                  输入解码: 色彩通路.传递编码 = .sRGB,
                                  起始条件: [Float],
                                  结束条件: [Float],
                                  窗口总帧数: Int,
                                  会话: inout Flash宿主.流式会话?) throws -> 超分单帧结果 {
        let (设备, 队) = try 色彩通路.上下文()
        return try autoreleasepool { () throws -> 超分单帧结果 in
            guard let 目标 = 目标尺寸(帧, 短边上限: 短边上限) else {
                throw 超分错误(说明: "预览帧读取失败")
            }
            var 分段: [String: Double] = [:]
            let 前置起 = CACurrentMediaTime()
            let 线性全 = try 色彩通路.线性图(from: 帧, 设备: 设备, 队: 队, 解码: 输入解码)
            let 线性低 = try 色彩通路.线性缩放(线性全, 到: 目标.宽, 高: 目标.高, 队: 队)
            let 模型域 = try 色彩通路.输出RGBA8(from: 线性低, 队: 队, 抖动: false)
            分段["前置"] = (CACurrentMediaTime() - 前置起) * 1000
            分段["Wan"] = 0

            if 会话 == nil {
                会话 = try Flash宿主.共享.建立窗口(起始条件: 起始条件, 结束条件: 结束条件,
                                              宽: 目标.宽, 高: 目标.高,
                                              总帧数: max(1, 窗口总帧数),
                                              延续历史: false)
            }
            guard var 活 = 会话 else { throw 超分错误(说明: "流式会话未就绪") }
            let flash结果 = try Flash宿主.共享.流式超分(字节: 模型域, 宽: 目标.宽, 高: 目标.高, 会话: &活)
            会话 = 活
            分段["Flash"] = flash结果.耗时毫秒

            let 回归起 = CACurrentMediaTime()
            let 上传起 = CACurrentMediaTime()
            let 平面缓冲 = try 色彩通路.平面缓冲(flash结果.输出.像素, 设备: 设备)
            let 线性超 = try 色彩通路.线性纹理(from平面: 平面缓冲,
                                          宽: flash结果.输出.宽,
                                          高: flash结果.输出.高,
                                          设备: 设备,
                                          队: 队)
            分段["回归上传"] = (CACurrentMediaTime() - 上传起) * 1000
            let 调色起 = CACurrentMediaTime()
            let 线性成 = try 色彩通路.调色(线性超, 参数: 调色, 队: 队)
            分段["回归调色"] = (CACurrentMediaTime() - 调色起) * 1000
            let 输出起 = CACurrentMediaTime()
            let 输出字节 = try 色彩通路.输出RGBA8(from: 线性成, 队: 队, 抖动: 抖动, 传递: 输出编码)
            guard let 图 = 色彩通路.CG图(fromRGBA8: 输出字节, 宽: flash结果.输出.宽, 高: flash结果.输出.高,
                                      色彩: 输出编码.色彩空间) else {
                throw 超分错误(说明: "输出像素转图失败")
            }
            分段["回归输出"] = (CACurrentMediaTime() - 输出起) * 1000
            分段["回归"] = (CACurrentMediaTime() - 回归起) * 1000

            let 单帧 = 超分单帧结果(图: 图,
                                输入宽: 帧.width,
                                输入高: 帧.height,
                                低清宽: 目标.宽,
                                低清高: 目标.高,
                                输出宽: flash结果.输出.宽,
                                输出高: flash结果.输出.高,
                                flash毫秒: flash结果.耗时毫秒,
                                wan毫秒: nil,
                                分段: 分段)
            return 单帧
        }
    }

    // MARK: 整段视频

    /// 整段视频逐帧超分 + 编码落盘：AVAssetReader 取帧 → 与单帧预览同一条超分核心 → AVAssetWriter 写 HEVC。
    func 开始视频(源: URL) {
        guard !运行中 else { return }
        guard 可运行 else {
            阶段 = .失败
            提示 = 缺权重提示
            return
        }

        阶段 = .运行
        视频运行 = true
        提示 = nil
        产物 = nil
        视频产物 = nil
        视频已处理 = 0
        视频总数 = 0
        取消开关.重置()

        let 短边上限 = min(max(输入短边, 64), 1536)
        let 用Wan = 启用Wan
        let 调色值 = 调色参数
        let 抖动值 = 输出抖动
        let 输出地址 = 视频超分.输出地址(源: 源)
        let 开关 = 取消开关
        let 代次 = 任务代次

        Task.detached(priority: .userInitiated) {
            do {
                let 结果 = try await 视频超分.运行(源: 源,
                                              输出: 输出地址,
                                              短边上限: 短边上限,
                                              用Wan: 用Wan,
                                              调色: 调色值,
                                              抖动: 抖动值,
                                              取消: { 开关.值 },
                                              进度: { 已处理, 总数 in
                                                  Task { @MainActor in
                                                      self.视频已处理 = 已处理
                                                      self.视频总数 = 总数
                                                  }
                                              })

                // 成片抽一帧同刻源帧，供 1:1 像素对比层继续可用
                let 对比 = await Self.对比帧(源: 源, 输出: 结果.输出URL)
                let 平均帧毫秒 = 结果.帧数 > 0 ? 结果.耗时秒 * 1000 / Double(结果.帧数) : 0

                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.视频产物 = 视频超分成片(输出URL: 结果.输出URL,
                                        帧数: 结果.帧数,
                                        输入宽: 结果.输入宽,
                                        输入高: 结果.输入高,
                                        低清宽: 结果.低清宽,
                                        低清高: 结果.低清高,
                                        输出宽: 结果.输出宽,
                                        输出高: 结果.输出高,
                                        耗时秒: 结果.耗时秒,
                                        含音轨: 结果.含音轨,
                                        体积字节: 结果.输出字节)
                    if let 对比 {
                        self.产物 = 超分产物(
                            输出图: NSImage(cgImage: 对比.成片,
                                        size: NSSize(width: 对比.成片.width, height: 对比.成片.height)),
                            输出CG: 对比.成片,
                            输入图: NSImage(cgImage: 对比.原图,
                                        size: NSSize(width: 对比.原图.width, height: 对比.原图.height)),
                            耗时毫秒: 平均帧毫秒,
                            输入宽: 对比.原图.width,
                            输入高: 对比.原图.height,
                            低清宽: 结果.低清宽,
                            低清高: 结果.低清高,
                            输出宽: 对比.成片.width,
                            输出高: 对比.成片.height,
                            分段毫秒: [:],
                            调色: 调色值)
                    }
                    self.显示超分 = true
                    self.缓存命中 = false
                    // 成片登记进落盘索引：切走素材再切回时靠它恢复 视频产物 并重抽对比帧。
                    超分缓存索引.共享.记录(超分缓存条目(源路径: 超分缓存索引.键(源),
                                                         类型: "视频",
                                                         产物路径: 结果.输出URL.path,
                                                         输入宽: 结果.输入宽,
                                                         输入高: 结果.输入高,
                                                         低清宽: 结果.低清宽,
                                                         低清高: 结果.低清高,
                                                         输出宽: 结果.输出宽,
                                                         输出高: 结果.输出高,
                                                         耗时毫秒: 平均帧毫秒,
                                                         帧数: 结果.帧数,
                                                         耗时秒: 结果.耗时秒,
                                                         含音轨: 结果.含音轨,
                                                         体积字节: 结果.输出字节))
                    self.视频运行 = false
                    self.提示 = nil
                    self.阶段 = .完成
                }
            } catch {
                // 失败原因同时落到 stderr，便于自动化验收与非交互场景定位
                FileHandle.standardError.write("整段超分失败：\(error)\n".data(using: .utf8)!)
                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.视频运行 = false
                    self.提示 = "\(error)"
                    self.阶段 = .失败
                }
            }
        }
    }

    // MARK: 动图（GIF 多帧）

    /// 动图逐帧超分 + 编码落盘：CGImageSource 逐帧解全分辨率 → 与单帧/整段同一条超分核心
    /// → ImageIO 写 GIF 到 超分输出目录（源名_超分4x.gif，重名递增，保持源格式）。
    /// 进度、取消、落盘索引全部复用整段视频那一套，不额外引入状态。
    func 开始动图(源: URL) {
        guard !运行中 else { return }
        guard 可运行 else {
            阶段 = .失败
            提示 = 缺权重提示
            return
        }

        阶段 = .运行
        视频运行 = true
        动图运行 = true
        提示 = nil
        产物 = nil
        视频产物 = nil
        单帧文件 = nil
        视频已处理 = 0
        视频总数 = 0
        取消开关.重置()

        let 短边上限 = min(max(输入短边, 64), 1536)
        let 用Wan = 启用Wan
        let 调色值 = 调色参数
        let 抖动值 = 输出抖动
        // 命名与整段视频同规则，但扩展名跟源格式走：源名 + _超分4x.gif，重名自动加序号。
        let 输出地址 = 动图超分.输出地址(源: 源)
        let 开关 = 取消开关
        let 代次 = 任务代次

        Task.detached(priority: .userInitiated) {
            do {
                let 结果 = try await 动图超分.运行(源: 源,
                                              输出: 输出地址,
                                              短边上限: 短边上限,
                                              用Wan: 用Wan,
                                              调色: 调色值,
                                              抖动: 抖动值,
                                              取消: { 开关.值 },
                                              进度: { 已处理, 总数 in
                                                  Task { @MainActor in
                                                      self.视频已处理 = 已处理
                                                      self.视频总数 = 总数
                                                  }
                                              })

                // 对比层取动图首帧与成片首帧：GIF 成片同样用 CGImageSource 解帧，两侧同一帧内容 1:1 对比。
                let 对比 = await Self.对比帧(动图源: 源, 输出: 结果.输出URL)
                let 平均帧毫秒 = 结果.帧数 > 0 ? 结果.耗时秒 * 1000 / Double(结果.帧数) : 0

                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.视频产物 = 视频超分成片(输出URL: 结果.输出URL,
                                        帧数: 结果.帧数,
                                        输入宽: 结果.输入宽,
                                        输入高: 结果.输入高,
                                        低清宽: 结果.低清宽,
                                        低清高: 结果.低清高,
                                        输出宽: 结果.输出宽,
                                        输出高: 结果.输出高,
                                        耗时秒: 结果.耗时秒,
                                        含音轨: 结果.含音轨,
                                        体积字节: 结果.输出字节)
                    if let 对比 {
                        self.产物 = 超分产物(
                            输出图: NSImage(cgImage: 对比.成片,
                                        size: NSSize(width: 对比.成片.width, height: 对比.成片.height)),
                            输出CG: 对比.成片,
                            输入图: NSImage(cgImage: 对比.原图,
                                        size: NSSize(width: 对比.原图.width, height: 对比.原图.height)),
                            耗时毫秒: 平均帧毫秒,
                            输入宽: 对比.原图.width,
                            输入高: 对比.原图.height,
                            低清宽: 结果.低清宽,
                            低清高: 结果.低清高,
                            输出宽: 对比.成片.width,
                            输出高: 对比.成片.height,
                            分段毫秒: [:],
                            调色: 调色值)
                    }
                    self.显示超分 = true
                    self.缓存命中 = false
                    // 动图产物是 GIF，按「动图」类登记；切回时仍由 是动图 判定走动图那一支重建对比层。
                    超分缓存索引.共享.记录(超分缓存条目(源路径: 超分缓存索引.键(源),
                                                         类型: "动图",
                                                         产物路径: 结果.输出URL.path,
                                                         输入宽: 结果.输入宽,
                                                         输入高: 结果.输入高,
                                                         低清宽: 结果.低清宽,
                                                         低清高: 结果.低清高,
                                                         输出宽: 结果.输出宽,
                                                         输出高: 结果.输出高,
                                                         耗时毫秒: 平均帧毫秒,
                                                         帧数: 结果.帧数,
                                                         耗时秒: 结果.耗时秒,
                                                         含音轨: 结果.含音轨,
                                                         体积字节: 结果.输出字节))
                    self.视频运行 = false
                    self.提示 = nil
                    self.阶段 = .完成
                }
            } catch {
                FileHandle.standardError.write("动图超分失败：\(error)\n".data(using: .utf8)!)
                await MainActor.run {
                    guard self.任务代次 == 代次 else { return }
                    self.视频运行 = false
                    self.提示 = "\(error)"
                    self.阶段 = .失败
                }
            }
        }
    }

    /// 取消整段超分（当前帧跑完即停，已写入的成片会被丢弃）。
    func 取消视频() {
        guard 视频运行 else { return }
        取消开关.取消()
        提示 = "正在取消…"
    }

    /// 成片与源片在同一时刻各抽一帧，供 1:1 像素对比。
    nonisolated static func 对比帧(源: URL,
                                输出: URL,
                                时间: Double = 0.5) async -> (成片: CGImage, 原图: CGImage)? {
        guard let 成片 = try? await 视频解码.抽帧(url: 输出, 时间: 时间, 最大边长: 0) else { return nil }
        guard let 原图 = try? await 视频解码.抽帧(url: 源, 时间: 时间, 最大边长: 0) else { return nil }
        return (成片, 原图)
    }

    /// 动图成片与动图源帧的对比帧：成片和源都从动图容器里取同一序号的全分辨率帧。
    /// GIF 既不是视频资源、也不能被 视频解码.抽帧 读取，所以两侧统一走 CGImageSource 解帧。
    nonisolated static func 对比帧(动图源 源: URL,
                                输出: URL,
                                帧序: Int = 0) async -> (成片: CGImage, 原图: CGImage)? {
        func 取帧(_ url: URL) -> CGImage? {
            guard let 图源 = CGImageSourceCreateWithURL(url as CFURL, nil),
                  CGImageSourceGetCount(图源) > 帧序 else { return nil }
            return CGImageSourceCreateImageAtIndex(图源, 帧序, nil)
        }
        guard let 成片 = 取帧(输出), let 原图 = 取帧(源) else { return nil }
        return (成片, 原图)
    }

    /// PNG 编码后台队列：4K 级单帧编码约百毫秒量级，交给后台串行队列，避免卡住主线程与出帧。
    private nonisolated static let 编码队列 = DispatchQueue(label: "relayvsr.png", qos: .userInitiated)

    /// 保存当前超分结果到指定目录，返回落地路径（PNG 编码在后台完成，不阻塞调用方）。
    @discardableResult
    func 保存(目录: URL? = nil) -> URL? {
        guard let 产物 else { return nil }
        let 时间 = DateFormatter()
        时间.dateFormat = "HHmmss"
        let 基名 = "超分_\(产物.输出宽)x\(产物.输出高)_\(时间.string(from: Date()))"
        return Self.写单帧(产物.输出CG, 名称: 基名, 目录: 目录)
    }

    /// 把一张单帧图写成 PNG（默认落 文档目录.超分输出目录，重名自动编号），返回落地路径；编码在后台队列完成。
    nonisolated static func 写单帧(_ 图: CGImage, 名称: String, 目录: URL? = nil) -> URL? {
        // 未指定目录时统一收口到 ~/Documents/AI超分/超分输出，不再散落到桌面
        let 目标目录 = 目录 ?? 文档目录.超分输出目录
        try? FileManager.default.createDirectory(at: 目标目录, withIntermediateDirectories: true)
        var 路径 = 目标目录.appendingPathComponent(名称 + ".png")
        var 序号 = 2
        while FileManager.default.fileExists(atPath: 路径.path) {
            路径 = 目标目录.appendingPathComponent("\(名称)_\(序号).png")
            序号 += 1
        }
        编码队列.async {
            guard let 目标 = CGImageDestinationCreateWithURL(路径 as CFURL, "public.png" as CFString, 1, nil) else {
                return
            }
            CGImageDestinationAddImage(目标, 图, nil)
            _ = CGImageDestinationFinalize(目标)
        }
        return 路径
    }

    /// 清空一次运行的全部状态（换素材时调用）。
    /// 同时推进任务代次：在途的单帧与整段结果回写前会发现代次已变而丢弃，
    /// 因此清空之后不会再有「旧素材的结果复活」。
    func 清空() {
        任务代次 += 1
        产物 = nil
        单帧文件 = nil
        视频产物 = nil
        视频运行 = false
        动图运行 = false
        视频已处理 = 0
        视频总数 = 0
        阶段 = .空闲
        提示 = nil
        显示超分 = true
        缓存命中 = false
    }

    /// 中止一切在跑的活并清空状态：单帧结果作废、整段超分被取消开关打断。
    /// 清空素材箱时用；换素材仍走 清空()，不打断正在进行的长任务。
    func 中止全部() {
        清空()
        取消开关.取消()
    }

    // MARK: 落盘缓存恢复

    /// 切换素材后按落盘索引重建对比层：图片从磁盘 PNG 读回产物，视频读回成片并重抽对比帧。
    /// 只有通道空闲、索引命中且产物文件仍在时生效；恢复后 阶段 = .完成、缓存命中 = true，
    /// 详情页的「可对比」判定随即成立，滑动对比不用重新跑推理。
    /// - Parameters:
    ///   - 条目: 当前选中的素材
    ///   - 等待: 索引刚登记而 PNG 可能仍在后台编码，最多等这么多秒（默认 3s）
    func 恢复缓存(素材 条目: 素材条目, 等待: Double = 3) async {
        guard !运行中, !视频运行 else { return }
        let 代次 = 任务代次
        guard let 登记 = await 等待缓存登记(源: 条目.url, 等待: 等待) else { return }
        guard 任务代次 == 代次 else { return }
        let 产物地址 = URL(fileURLWithPath: 登记.产物路径)

        // 动图成片是 GIF：产物本身即动图，解首帧与源首帧重建对比层；只有静态图片才从 PNG 读回。
        switch (条目.类型, 条目.是动图) {
        case (.图片, true):
            guard let 对比 = await Self.对比帧(动图源: 条目.url, 输出: 产物地址) else { return }
            guard 任务代次 == 代次 else { return }
            视频产物 = 视频超分成片(输出URL: 产物地址,
                                   帧数: 登记.帧数,
                                   输入宽: 登记.输入宽,
                                   输入高: 登记.输入高,
                                   低清宽: 登记.低清宽,
                                   低清高: 登记.低清高,
                                   输出宽: 登记.输出宽,
                                   输出高: 登记.输出高,
                                   耗时秒: 登记.耗时秒,
                                   含音轨: 登记.含音轨,
                                   体积字节: 登记.体积字节)
            产物 = 超分产物(输出图: NSImage(cgImage: 对比.成片,
                                       size: NSSize(width: 对比.成片.width, height: 对比.成片.height)),
                          输出CG: 对比.成片,
                          输入图: NSImage(cgImage: 对比.原图,
                                       size: NSSize(width: 对比.原图.width, height: 对比.原图.height)),
                          耗时毫秒: 登记.耗时毫秒,
                          输入宽: 对比.原图.width,
                          输入高: 对比.原图.height,
                          低清宽: 登记.低清宽,
                          低清高: 登记.低清高,
                          输出宽: 对比.成片.width,
                          输出高: 对比.成片.height,
                          分段毫秒: [:],
                          调色: 调色参数)
            单帧文件 = nil
            // 恢复的成片来自动图：文案口径同步回「动图」，否则会误标成整段视频。
            动图运行 = true
        case (.图片, false):
            guard let 成片 = Self.原图(url: 产物地址, 备用: nil) else { return }
            guard let 原图 = Self.原图(url: 条目.url, 备用: 条目.缩略图) else { return }
            guard 任务代次 == 代次 else { return }
            产物 = 超分产物(输出图: NSImage(cgImage: 成片,
                                       size: NSSize(width: 成片.width, height: 成片.height)),
                          输出CG: 成片,
                          输入图: NSImage(cgImage: 原图,
                                       size: NSSize(width: 原图.width, height: 原图.height)),
                          耗时毫秒: 登记.耗时毫秒,
                          输入宽: 原图.width,
                          输入高: 原图.height,
                          低清宽: 登记.低清宽,
                          低清高: 登记.低清高,
                          输出宽: 成片.width,
                          输出高: 成片.height,
                          分段毫秒: [:],
                          调色: 调色参数)
            单帧文件 = 产物地址
            视频产物 = nil
            动图运行 = false
        case (.视频, _):
            guard let 对比 = await Self.对比帧(源: 条目.url, 输出: 产物地址) else { return }
            guard 任务代次 == 代次 else { return }
            视频产物 = 视频超分成片(输出URL: 产物地址,
                                   帧数: 登记.帧数,
                                   输入宽: 登记.输入宽,
                                   输入高: 登记.输入高,
                                   低清宽: 登记.低清宽,
                                   低清高: 登记.低清高,
                                   输出宽: 登记.输出宽,
                                   输出高: 登记.输出高,
                                   耗时秒: 登记.耗时秒,
                                   含音轨: 登记.含音轨,
                                   体积字节: 登记.体积字节)
            产物 = 超分产物(输出图: NSImage(cgImage: 对比.成片,
                                       size: NSSize(width: 对比.成片.width, height: 对比.成片.height)),
                          输出CG: 对比.成片,
                          输入图: NSImage(cgImage: 对比.原图,
                                       size: NSSize(width: 对比.原图.width, height: 对比.原图.height)),
                          耗时毫秒: 登记.耗时毫秒,
                          输入宽: 对比.原图.width,
                          输入高: 对比.原图.height,
                          低清宽: 登记.低清宽,
                          低清高: 登记.低清高,
                          输出宽: 对比.成片.width,
                          输出高: 对比.成片.height,
                          分段毫秒: [:],
                          调色: 调色参数)
            单帧文件 = nil
            动图运行 = false
        }

        显示超分 = true
        视频运行 = false
        视频已处理 = 0
        视频总数 = 0
        提示 = nil
        缓存命中 = true
        阶段 = .完成
    }

    /// 索引命中并确保磁盘产物已就绪（PNG 由后台编码队列写入，短暂轮询等待）。
    private func 等待缓存登记(源 url: URL, 等待: Double) async -> 超分缓存条目? {
        let 起 = Date()
        while true {
            if let 命中 = 超分缓存索引.共享.条目(源: url) { return 命中 }
            if Date().timeIntervalSince(起) >= max(0, 等待) { return nil }
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
    }

    // MARK: 验收用

    /// 从参考张量里取一张真实低清帧（624x360），用于自动化验收整条超分链路。
    nonisolated static func 自检帧() throws -> CGImage {
        let 路径 = 文档目录.模型路径("ref/flash_selftest.safetensors")
        guard FileManager.default.fileExists(atPath: 路径) else {
            throw 超分错误(说明: "找不到参考张量：" + 路径)
        }
        let 库 = try SafeTensors权重库(路径: 路径)
        let lr = try 库.Float32张量("in.lr")
        guard lr.形状.count == 4, lr.形状[1] == 3 else {
            throw 超分错误(说明: "in.lr 形状异常")
        }
        let 高 = lr.形状[2]
        let 宽 = lr.形状[3]
        guard let 图 = 转图(lr.数值, 宽: 宽, 高: 高) else {
            throw 超分错误(说明: "参考帧转图失败")
        }
        return 图
    }

    // MARK: 像素搬运

    /// 任意尺寸输入的自适应低清目标尺寸：短边超过上限才等比缩小，源本身更小则原样不放大；
    /// 宽高同乘一个比例后就近对齐到 8 的倍数（Wan latent = 低清/4、token = 低清/8 都要整除，
    /// 官方 stride = 16 × max(patch)/4 = 8），故竖屏 / 横屏 / 方屏 / 奇数尺寸都能直接跑，
    /// 不存在「宽高必须手动凑成 8 的倍数」的输入限制。
    nonisolated static func 目标尺寸(_ 图: CGImage, 短边上限: Int) -> (宽: Int, 高: Int)? {
        let 原宽 = 图.width
        let 原高 = 图.height
        guard 原宽 > 0, 原高 > 0 else { return nil }
        let 比例 = min(1.0, Double(短边上限) / Double(min(原宽, 原高)))
        let 宽 = 对齐8(Int((Double(原宽) * 比例).rounded()))
        let 高 = 对齐8(Int((Double(原高) * 比例).rounded()))
        return (宽, 高)
    }

    /// 就近对齐到 8 的倍数（最小 8，向下取整，绝不放大）：对齐误差 ≤ 7 px，肉眼与比例上不可辨。
    nonisolated static func 对齐8(_ 值: Int) -> Int {
        max(8, (max(值, 1) / 8) * 8)
    }

    /// CGImage → RGBA8（缩放到长边，宽高对齐 8）。行 0 为图像顶部。
    nonisolated static func 缩放(_ 图: CGImage, 短边上限: Int) -> (字节: [UInt8], 宽: Int, 高: Int)? {
        let 原宽 = 图.width
        let 原高 = 图.height
        guard 原宽 > 0, 原高 > 0 else { return nil }

        let 比例 = min(1.0, Double(短边上限) / Double(min(原宽, 原高)))
        let 宽 = 对齐8(Int((Double(原宽) * 比例).rounded()))
        let 高 = 对齐8(Int((Double(原高) * 比例).rounded()))

        guard let 色彩空间 = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var 字节 = [UInt8](repeating: 0, count: 宽 * 高 * 4)
        let 成功: Bool = 字节.withUnsafeMutableBytes { raw in
            guard let 上下文 = CGContext(data: raw.baseAddress,
                                     width: 宽,
                                     height: 高,
                                     bitsPerComponent: 8,
                                     bytesPerRow: 宽 * 4,
                                     space: 色彩空间,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                return false
            }
            上下文.interpolationQuality = .high
            上下文.draw(图, in: CGRect(x: 0, y: 0, width: 宽, height: 高))
            return true
        }
        guard 成功 else { return nil }
        return (字节, 宽, 高)
    }

    /// NCHW float [-1, 1] → CGImage。
    nonisolated static func 转图(_ 值: [Float], 宽: Int, 高: Int) -> CGImage? {
        let 平面 = 宽 * 高
        guard 宽 > 0, 高 > 0, 值.count >= 3 * 平面 else { return nil }
        guard let 色彩空间 = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        var 字节 = [UInt8](repeating: 255, count: 平面 * 4)
        for i in 0..<平面 {
            let 基 = i * 4
            for 通道 in 0..<3 {
                let 缩放值 = (值[通道 * 平面 + i] + 1.0) * 127.5
                let 截断 = min(max(缩放值, 0), 255)
                字节[基 + 通道] = UInt8(截断.rounded())
            }
            字节[基 + 3] = 255
        }

        // 关键：让 CGContext 自持像素内存并显式拷贝，避免 CGImage 悬空引用局部数组，
        // 保证异步 PNG 编码期间像素仍然有效（原实现直接借用栈外数组，异步化后会读到已释放内存）
        guard let 上下文 = CGContext(data: nil,
                                 width: 宽,
                                 height: 高,
                                 bitsPerComponent: 8,
                                 bytesPerRow: 宽 * 4,
                                 space: 色彩空间,
                                 bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let 目标 = 上下文.data else {
            return nil
        }
        字节.withUnsafeBytes { 目标.copyMemory(from: $0.baseAddress!, byteCount: 平面 * 4) }
        return 上下文.makeImage()
    }
}
