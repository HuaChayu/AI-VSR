import AVFoundation
import CoreImage
import Darwin
import CoreGraphics
import Metal
import QuartzCore

/// 整段视频超分：AVAssetReader 逐帧取帧 → 与单帧预览完全相同的超分核心 → AVAssetWriter 编码落盘。
///
/// 与单帧预览的唯一差别是「输出来源」：单帧只把结果留给预览区，这里每帧都写进编码器。
/// 超分本身走 超分通道.超分单帧，两条通路像素级一致，不存在预览与成片两套实现。
nonisolated enum 视频超分 {

    struct 结果 {
        var 输出URL: URL
        var 帧数: Int
        /// 源视频显示尺寸（转正后）：对比层右侧源帧的真实像素尺寸。
        var 输入宽: Int
        var 输入高: Int
        /// 送模型推理的低清尺寸（短边封顶后对齐 8），与 输入宽高 不同口径。
        var 低清宽: Int
        var 低清高: Int
        var 输出宽: Int
        var 输出高: Int
        var 耗时秒: Double
        var 含音轨: Bool
        var 输出字节: Int64
    }

    /// 逐帧进度回调（已处理帧数, 预计总帧数）。
    typealias 进度回调 = @Sendable (Int, Int) -> Void

    /// 跨线程取消开关（渲染线程每帧读一次）。
    final class 取消开关: @unchecked Sendable {
        private let 锁 = NSLock()
        private var 已取消 = false

        var 值: Bool {
            锁.lock()
            defer { 锁.unlock() }
            return 已取消
        }

        func 取消() {
            锁.lock()
            已取消 = true
            锁.unlock()
        }

        func 重置() {
            锁.lock()
            已取消 = false
            锁.unlock()
        }
    }

    /// 成片落地路径：默认落 文档目录.超分输出目录，源名 + 超分4x 后缀，重名自动加序号（不覆盖既有文件）。
    static func 输出地址(源: URL) -> URL {
        // 自动化验收可用 UPSAMPLE_OUT_DIR 覆盖落盘目录；常规 GUI 运行统一落 ~/Documents/AI超分/超分输出。
        let 覆盖目录 = ProcessInfo.processInfo.environment["UPSAMPLE_OUT_DIR"]
        let 目录 = 覆盖目录.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? 文档目录.超分输出目录
        try? FileManager.default.createDirectory(at: 目录, withIntermediateDirectories: true)
        let 基名 = 源.deletingPathExtension().lastPathComponent + "_超分4x"
        var 路径 = 目录.appendingPathComponent(基名 + ".mp4")
        var 序号 = 2
        while FileManager.default.fileExists(atPath: 路径.path) {
            路径 = 目录.appendingPathComponent("\(基名)_\(序号).mp4")
            序号 += 1
        }
        return 路径
    }

    /// 跑完整段。全部工作在本函数所在线程同步完成（调用方负责放到后台 Task）。
    static func 运行(源: URL,
                    输出: URL,
                    短边上限: Int,
                    用Wan: Bool,
                    调色: 色彩通路.调色参数,
                    抖动: Bool,
                    取消: @escaping @Sendable () -> Bool,
                    进度: @escaping 进度回调) async throws -> 结果 {

        let 开始时刻 = Date()
        let 资源 = AVURLAsset(url: 源)
        guard let 视频轨 = try await 资源.loadTracks(withMediaType: .video).first else {
            throw 超分错误(说明: "找不到视频轨：\(源.lastPathComponent)")
        }

        let 时长 = try await 资源.load(.duration)
        let 时长秒 = max(0, CMTimeGetSeconds(时长))
        let 帧率 = max(1.0, Double(try await 视频轨.load(.nominalFrameRate)))
        let 变换 = try await 视频轨.load(.preferredTransform)
        let 自然尺寸 = try await 视频轨.load(.naturalSize)

        // 转正后的显示尺寸（手机竖拍素材的 preferredTransform 会旋转 90°，宽高需互换）
        let 转正矩形 = CGRect(origin: .zero, size: 自然尺寸).applying(变换)
        let 显示宽 = Int(abs(转正矩形.width).rounded())
        let 显示高 = Int(abs(转正矩形.height).rounded())
        guard 显示宽 > 0, 显示高 > 0 else {
            throw 超分错误(说明: "视频尺寸异常：\(自然尺寸)")
        }

        // 低清目标尺寸与单帧预览同口径：短边封顶、只缩不放、对齐 8；输出恒为 4 倍
        let 比例 = min(1.0, Double(短边上限) / Double(min(显示宽, 显示高)))
        let 低宽 = 超分通道.对齐8(Int((Double(显示宽) * 比例).rounded()))
        let 低高 = 超分通道.对齐8(Int((Double(显示高) * 比例).rounded()))
        let 输出宽 = 低宽 * 4
        let 输出高 = 低高 * 4

        // 音轨不再与视频同写入器交错受理：两条轨道共用一个写入器时，任一方跑在前/后都会互相
        // 关闭对方入口（实测在部分素材的音频尾部时间戳异常时必然互锁）。改为视频单独写码，
        // 收尾后一次透传混流并入原音轨——与素材内容/尺寸/时长无关，也不再依赖音频时间戳比较。
        let 有音轨 = !(try await 资源.loadTracks(withMediaType: .audio)).isEmpty

        // 视频读取通道：BGRA 全尺寸出帧，旋转与缩放交给后面的 CIImage / 线性光通路
        let 读设置: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        // 单点建立并启动解码通道：主循环与端点抽样共用同一段建立逻辑，
        // 抽取端点不需要另写一套读取设置，也不会与主循环的实现漂移。
        func 建读器() throws -> (读: AVAssetReader, 输出: AVAssetReaderTrackOutput) {
            let 读 = try AVAssetReader(asset: 资源)
            let 输出 = AVAssetReaderTrackOutput(track: 视频轨, outputSettings: 读设置)
            输出.alwaysCopiesSampleData = false
            guard 读.canAdd(输出) else {
                throw 超分错误(说明: "无法建立视频读取通道")
            }
            读.add(输出)
            guard 读.startReading() else {
                throw 超分错误(说明: "无法启动解码器：\(读.error?.localizedDescription ?? "未知原因")")
            }
            return (读, 输出)
        }
        let (读, 视频输出) = try 建读器()
        // 取样本统一走这一处：主循环与端点抽样共用，解码口径不会分叉。
        func 取样本(_ 输出: AVAssetReaderTrackOutput) -> CMSampleBuffer? {
            输出.copyNextSampleBuffer()
        }

        // 写成片：HEVC 高码率，保留 4 倍像素的信息量，8K 级输出也不至于糊成一片
        // 中间视频（无音轨）放 cache 目录，绝不落在超分输出目录，避免半成品混进成片清单
        let 中间目录 = 文档目录.缓存目录
        try? FileManager.default.createDirectory(at: 中间目录, withIntermediateDirectories: true)
        let 中间视频 = 中间目录
            .appendingPathComponent("超分中间-无音轨-" + UUID().uuidString + ".mp4")
        let 写 = try AVAssetWriter(outputURL: 中间视频, fileType: .mp4)
        let 码率 = max(12_000_000, Int(Double(输出宽 * 输出高) * 帧率 * 0.12))
        let 压缩属性: [String: Any] = [
            AVVideoAverageBitRateKey: 码率,
            AVVideoMaxKeyFrameIntervalKey: max(1, Int(帧率 * 2))
        ]
        // 色彩口径跟源走：源轨声明了什么传递函数，8bit 编码曲线与写码标签就用同一族。
        // 标签与字节必须成对，否则播放器按标签反解，成片就会整体偏亮或偏暗。
        let 源口径 = 源色彩.读取(try await 视频轨.load(.formatDescriptions).first)
        let 写设置: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 输出宽,
            AVVideoHeightKey: 输出高,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: 源口径.原语,
                AVVideoTransferFunctionKey: 源口径.传递函数,
                AVVideoYCbCrMatrixKey: 源口径.矩阵
            ],
            AVVideoCompressionPropertiesKey: 压缩属性
        ]
        let 视频输入 = AVAssetWriterInput(mediaType: .video, outputSettings: 写设置)
        视频输入.expectsMediaDataInRealTime = false
        let 适配器 = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: 视频输入,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: 输出宽,
                kCVPixelBufferHeightKey as String: 输出高,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                kCVPixelBufferCGImageCompatibilityKey as String: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
            ])
        guard 写.canAdd(视频输入) else {
            throw 超分错误(说明: "编码器不接受视频输入")
        }
        写.add(视频输入)

        guard 写.startWriting() else {
            throw 超分错误(说明: "无法启动编码器：\(写.error?.localizedDescription ?? "未知原因")")
        }
        写.startSession(atSourceTime: .zero)

        let ci上下文: CIContext
        if let 图形设备 = MTLCreateSystemDefaultDevice() {
            ci上下文 = CIContext(mtlDevice: 图形设备, options: [.cacheIntermediates: false])
        } else {
            ci上下文 = CIContext(options: [.cacheIntermediates: false])
        }

        var 已处理 = 0
        let 预计 = max(1, Int((时长秒 * 帧率).rounded()))
        let 日志地址 = 输出.deletingPathExtension().appendingPathExtension("log")
        var 日志 = "整段超分 日志\n输入 \(显示宽)×\(显示高) → 低清 \(低宽)×\(低高) → 输出 \(输出宽)×\(输出高) 帧率 \(帧率)\n"
        try? 日志.write(to: 日志地址, atomically: true, encoding: .utf8)

        // 基线采样：只累计各阶段真实耗时与峰值占用，不改变任何计算路径。
        性能探针.共享.重置()
        var 解码累计毫秒 = 0.0
        var 编码累计毫秒 = 0.0
        var 窗口累计毫秒 = 0.0
        var 封装累计毫秒 = 0.0
        var 混流累计毫秒 = 0.0
        var 内存峰值MB = 0.0

        func 等编码就绪(_ 输入: AVAssetWriterInput, 超时秒: Double = 30) throws {
            let 起始 = DispatchTime.now().uptimeNanoseconds
            while !输入.isReadyForMoreMediaData {
                if 写.status == .failed {
                    throw 超分错误(说明: "编码失败：\(写.error?.localizedDescription ?? "未知原因")")
                }
                if 取消() {
                    throw 超分错误(说明: "已取消")
                }
                // 编码器长时间不给入口视为卡死：抛错中止并保留已写内容，避免无限自旋挂住进程。
                if DispatchTime.now().uptimeNanoseconds - 起始 > UInt64(超时秒 * 1e9) {
                    throw 超分错误(说明: "编码器 \(Int(超时秒)) 秒未就绪（status=\(写.status.rawValue)，错误=\(写.error?.localizedDescription ?? "无")），已中止")
                }
                usleep(2_000)
            }
        }

        // 连续流式调度：滚动视频 KV 从整段首帧一路延伸到末帧，全程只建立一个流式会话、中途不重置，
        // 于是第 N 帧的历史深度恒为「窗口帧数 − 1」，不存在周期性重置带来的边界跳变。
        // 端点条件取整段的真实首帧与真实末帧各跑一次 Wan（官方 streaming.py 的端点语义），
        // 全程固定不变，因此任意源帧的生成条件完全一致——既不缺后向上下文，也没有每窗口重复一次的端点脉冲。
        // 窗口帧数 只决定条件 RoPE 跨度与滚动历史深度，与素材帧数/内容/分辨率无关。
        // 不启用 Wan 时退化为逐帧一次性超分。
        let 窗口帧数: Int = 用Wan ? 16 : 1
        var 待处理: [(帧: CGImage, 时间: CMTime)] = []
        var 流式会话: Flash宿主.流式会话?
        var 本帧分段: [String: Double] = [:]
        var 起始潜变量: [Float]?
        var 结束潜变量: [Float]?

        // 全局端点条件：整段只在真实首帧与真实末帧各跑一次 Wan，之后全程复用同一对端点。
        // 端点帧走与主解码完全相同的读取设置、转正与色彩回退口径，因此与主循环所见像素一致。
        var 端点Wan毫秒: Double = 0
        if 用Wan {
            let 端起点 = CACurrentMediaTime()
            // 端点抽样：整段解码一遍，只留真实首帧与真实末帧。转正与色彩回退口径与主循环完全一致，
            // 因此端点帧与主循环看到的像素相同；抽样开销只在整段开头发生一次。
            let (端点读, 端点输出) = try 建读器()
            var 端点首: CGImage?
            var 端点末: CGImage?
            while let 端点样本 = 取样本(端点输出) {
                // 大视频抽首末帧要扫完整个文件，这段没有检查点，会让「取消」看起来完全没反应。
                if 取消() {
                    端点读.cancelReading()
                    throw 超分错误(说明: "已取消")
                }
                guard let 端点像素 = CMSampleBufferGetImageBuffer(端点样本),
                      let 端点图 = 转正图(端点像素, 变换: 变换, 上下文: ci上下文, 回退空间: 源口径.空间) else { continue }
                if 端点首 == nil { 端点首 = 端点图 }
                端点末 = 端点图
            }
            端点读.cancelReading()
            guard let 首帧图 = 端点首, let 末帧图 = 端点末 else {
                throw 超分错误(说明: "端点帧抽取失败")
            }
            // 两次端点 Wan 前向是单帧级等待，同样给一个取消出口。
            if 取消() { throw 超分错误(说明: "已取消") }
            let 首端 = try 超分通道.端点潜变量(首帧图, 短边上限: 短边上限, 输入解码: 源口径.编码)
            let 末端 = try 超分通道.端点潜变量(末帧图, 短边上限: 短边上限, 输入解码: 源口径.编码)
            起始潜变量 = 首端.潜变量
            结束潜变量 = 末端.潜变量
            端点Wan毫秒 = 首端.耗时毫秒 + 末端.耗时毫秒
            窗口累计毫秒 += (CACurrentMediaTime() - 端起点) * 1000
        }


        func 冲刷窗口() throws {
            guard !待处理.isEmpty else { return }

            let 窗口 = 待处理
            let 推理起 = CACurrentMediaTime()
            let 产出: [超分单帧结果]
            if !用Wan {
                产出 = [try 超分通道.超分单帧(窗口[0].帧,
                                            短边上限: 短边上限,
                                            用Wan: 用Wan,
                                            调色: 调色,
                                            抖动: 抖动,
                                            输出编码: 源口径.编码,
                                            输入解码: 源口径.编码)]
            } else {
                guard let 起 = 起始潜变量, let 止 = 结束潜变量 else {
                    throw 超分错误(说明: "端点条件未就绪")
                }
                // 滚动 KV 由 流式会话 跨帧持有：首帧建立窗口，之后每帧接续，历史深度恒为 窗口帧数 − 1。
                产出 = [try 超分通道.超分流式帧(窗口[0].帧,
                                            短边上限: 短边上限,
                                            调色: 调色,
                                            抖动: 抖动,
                                            输出编码: 源口径.编码,
                                            输入解码: 源口径.编码,
                                            起始条件: 起,
                                            结束条件: 止,
                                            窗口总帧数: 窗口帧数,
                                            会话: &流式会话)]
            }
            窗口累计毫秒 += (CACurrentMediaTime() - 推理起) * 1000

            // 缓冲只是「一帧进、一帧出」的搬运单元：历史连续性由流式会话负责，与出图数无关。
            let 出图数 = 窗口.count
            for 索引 in 0..<min(出图数, 产出.count) {
                let 单帧 = 产出[索引]
                if 用Wan { 本帧分段 = 单帧.分段 }
                let 时间戳 = 窗口[索引].时间
                // 每帧自动释放池：转正图得到的 CGImage、色彩通路纹理与编码池取出的 CVPixelBuffer
                // 逐帧排空，避免 append 路径在无池线程上永久累积。
                try autoreleasepool {
            let 编码起 = CACurrentMediaTime()
                    try 等编码就绪(视频输入)
                    guard let 池 = 适配器.pixelBufferPool else {
                        throw 超分错误(说明: "编码缓冲池未就绪")
                    }
                    var 缓冲: CVPixelBuffer?
                    let 建结果 = CVPixelBufferPoolCreatePixelBuffer(nil, 池, &缓冲)
                    guard 建结果 == kCVReturnSuccess, let 缓冲 else {
                        throw 超分错误(说明: "无法分配编码缓冲（\(建结果)）")
                    }
                    写入(缓冲, 图: 单帧.图, 空间: 源口径.编码.色彩空间)

                    if !适配器.append(缓冲, withPresentationTime: 时间戳) {
                        throw 超分错误(说明: "第 \(已处理 + 1) 帧写码失败：\(写.error?.localizedDescription ?? "未知原因")")
                    }

                    已处理 += 1
            编码累计毫秒 += (CACurrentMediaTime() - 编码起) * 1000
            性能探针.共享.累加(["编码": (CACurrentMediaTime() - 编码起) * 1000])
            let 杂项起 = CACurrentMediaTime()
                    进度(已处理, 预计)

            let 当前占用 = 内存占用MB()
            内存峰值MB = max(内存峰值MB, 当前占用)
                    let 分段 = 单帧.分段
                    日志 += String(format: "帧 %d | Wan %.0f ms | Flash %.0f ms | 前置 %.0f ms | 回归 %.0f ms | 占用 %.0f MB\n",
                                  已处理,
                                  单帧.wan毫秒 ?? 0,
                                  单帧.flash毫秒,
                                  分段["前置"] ?? 0,
                                  分段["回归"] ?? 0,
                                  当前占用)
                    try? 日志.write(to: 日志地址, atomically: true, encoding: .utf8)
                    性能探针.共享.累加(["杂项": (CACurrentMediaTime() - 杂项起) * 1000])
                }
            }

            // 逐帧出图，缓冲随即清空：滚动 KV 已留在流式会话里，无需再用交接帧传递上下文。
            待处理 = []
        }

        while true {
            // 解码口径：取样本（含解封装）+ 像素转正图（含真实解码与色彩转换），两段都算解码开销。
            let 取样本起 = CACurrentMediaTime()
            let 样本 = 取样本(视频输出)
            解码累计毫秒 += (CACurrentMediaTime() - 取样本起) * 1000
            guard let 样本 else { break }
            if 取消() {
                读.cancelReading()
                写.cancelWriting()
                throw 超分错误(说明: "已取消")
            }
            guard let 像素 = CMSampleBufferGetImageBuffer(样本) else { continue }
            let 转图起 = CACurrentMediaTime()
            guard let 帧 = 转正图(像素, 变换: 变换, 上下文: ci上下文, 回退空间: 源口径.空间) else {
                throw 超分错误(说明: "第 \(已处理 + 1) 帧解码失败")
            }
            解码累计毫秒 += (CACurrentMediaTime() - 转图起) * 1000

            let 时间戳 = CMSampleBufferGetPresentationTimeStamp(样本)
            // 窗口级自动释放池：本窗内的 CGImage / CVPixelBuffer / Metal 编码产物在冲刷后立即排空。
            // 推理线程没有 RunLoop 兜底池，不显式排空会一路攒到进程结束（历史 40GB 占用的主因之一）。
            let 池起 = CACurrentMediaTime()
            let 窗口快照 = 窗口累计毫秒
            try autoreleasepool {
                // 逐帧即出图：上一帧的滚动 KV 已在流式会话里，不需要攒窗，也不存在攒窗相位差。
                待处理 = [(帧, 时间戳)]
                try 冲刷窗口()
            }
            let 池块毫秒 = (CACurrentMediaTime() - 池起) * 1000
            var 池增量: [String: Double] = ["池残余": 池块毫秒 - (窗口累计毫秒 - 窗口快照)]
            if 用Wan {
                池增量["前置"] = 本帧分段["前置"] ?? 0
                池增量["Flash"] = 本帧分段["Flash"] ?? 0
                池增量["回归"] = 本帧分段["回归"] ?? 0
                池增量["Wan"] = 端点Wan毫秒
            }
            性能探针.共享.累加(池增量)
            端点Wan毫秒 = 0
        }
        try 冲刷窗口()
        读.cancelReading()
        视频输入.markAsFinished()
        // 封装收尾带超时与取消：写入器排空编码器时若卡住，await finishWriting 会永久挂起、
        // 且不占用任何线程（表现为进程空转），因此用回调 + 轮询加 120 秒上限兜底。
        let 封装计时起 = CACurrentMediaTime()
        let 封装 = DispatchSemaphore(value: 0)
        写.finishWriting { 封装.signal() }
        let 封装起 = DispatchTime.now().uptimeNanoseconds
        while 封装.wait(timeout: .now() + 0.2) == .timedOut {
            if 取消() { throw 超分错误(说明: "已取消") }
            if DispatchTime.now().uptimeNanoseconds - 封装起 > 120_000_000_000 {
                throw 超分错误(说明: "封装 120 秒未完成（status=\(写.status.rawValue)，错误=\(写.error?.localizedDescription ?? "无")），已中止")
            }
        }
        封装累计毫秒 = (CACurrentMediaTime() - 封装计时起) * 1000

        guard 写.status == .completed else {
            throw 超分错误(说明: "成片封装失败：\(写.error?.localizedDescription ?? "未知原因")")
        }

        // 音轨并入：视频写码完成后一次透传混流，音轨时间轴由合成器接管，不参与编码交错调度。
        if 有音轨 {
            let 混流起 = CACurrentMediaTime()
            try 混流(视频: 中间视频, 源: 源, 输出: 输出, 取消: 取消)
            混流累计毫秒 = (CACurrentMediaTime() - 混流起) * 1000
            try? FileManager.default.removeItem(at: 中间视频)
        } else {
            try? FileManager.default.removeItem(at: 输出)
            try FileManager.default.moveItem(at: 中间视频, to: 输出)
        }

        // 基线摘要：把总耗时按「解码 / 推理 / 编码 / 封装 / 混流」拆开写回同一份日志，
        // 供后续优化做同口径对照（推理内部再按 前置/Wan/Flash/回归 细分）。
        let 总耗时秒 = Date().timeIntervalSince(开始时刻)
        let 探针 = 性能探针.共享.快照()
        let 推理合计 = 探针["前置", default: 0] + 探针["Wan", default: 0]
            + 探针["Flash", default: 0] + 探针["回归", default: 0]
        let 总毫秒 = max(1.0, 总耗时秒 * 1000)
        func 占比(_ 毫秒: Double) -> String { String(format: "%.1f%%", 毫秒 / 总毫秒 * 100) }
        // 单任务完成耗时报告：一行结论写进日志、输出目录 report.txt（追加）与控制台，手动测试直接取用。
        let 完成行 = String(format: "完成 %@ | 总耗时 %.1f s | 帧数 %d | 单帧均耗 %.0f ms\n",
                         输出.lastPathComponent, 总耗时秒, 已处理, 总毫秒 / Double(max(1, 已处理)))
        日志 += 完成行
        print("=== 完成报告 " + 完成行.trimmingCharacters(in: .newlines))
        let 报告地址 = 输出.deletingLastPathComponent().appendingPathComponent("report.txt")
        let 旧报告 = (try? String(contentsOf: 报告地址, encoding: .utf8)) ?? ""
        try? (旧报告 + 完成行).write(to: 报告地址, atomically: true, encoding: .utf8)

        var 摘要 = "\n=== 基线摘要 ===\n"
        摘要 += String(format: "吞吐 总耗时 %.1f s | 帧数 %d | 单帧均耗 %.0f ms | 内存峰值 %.0f MB\n",
                      总耗时秒, 已处理, 总毫秒 / Double(max(1, 已处理)), 内存峰值MB)
        摘要 += String(format: "解码 %.0f ms (%@) | 编码 %.0f ms (%@) | 封装 %.0f ms (%@) | 混流 %.0f ms (%@)\n",
                      解码累计毫秒, 占比(解码累计毫秒),
                      编码累计毫秒, 占比(编码累计毫秒),
                      封装累计毫秒, 占比(封装累计毫秒),
                      混流累计毫秒, 占比(混流累计毫秒))
        摘要 += String(format: "推理 %.0f ms (%@) = 前置 %.0f (%@) + Wan %.0f (%@) + Flash %.0f (%@) + 回归 %.0f (%@)\n",
                      推理合计, 占比(推理合计),
                      探针["前置", default: 0], 占比(探针["前置", default: 0]),
                      探针["Wan", default: 0], 占比(探针["Wan", default: 0]),
                      探针["Flash", default: 0], 占比(探针["Flash", default: 0]),
                      探针["回归", default: 0], 占比(探针["回归", default: 0]))
        let 未归因 = 总毫秒 - (解码累计毫秒 + 编码累计毫秒 + 封装累计毫秒 + 混流累计毫秒 + 推理合计)
        摘要 += String(format: "未归因 %.0f ms (%@) | 窗口调用累计 %.0f ms | 单帧均耗(推理) %.0f ms\n",
                      未归因, 占比(未归因), 窗口累计毫秒, 推理合计 / Double(max(1, 已处理)))
        let 杂项 = 探针["杂项", default: 0]
        let 窗口其它 = 窗口累计毫秒 - (推理合计 + 编码累计毫秒 + 杂项)
        摘要 += String(format: "细分 回归上传 %.0f ms (%@) | 回归调色 %.0f ms (%@) | 回归输出 %.0f ms (%@)\n",
                      探针["回归上传", default: 0], 占比(探针["回归上传", default: 0]),
                      探针["回归调色", default: 0], 占比(探针["回归调色", default: 0]),
                      探针["回归输出", default: 0], 占比(探针["回归输出", default: 0]))
        摘要 += String(format: "细分 帧杂项(进度/内存/日志) %.0f ms (%@) | 窗口其它残余 %.0f ms (%@)\n",
                      杂项, 占比(杂项), 窗口其它, 占比(窗口其它))
        摘要 += String(format: "细分 池残余(含排空) %.0f ms (%@)\n",
                      探针["池残余", default: 0], 占比(探针["池残余", default: 0]))
        日志 += 摘要
        try? 日志.write(to: 日志地址, atomically: true, encoding: .utf8)
        print(摘要)

        let 体积 = (try? FileManager.default.attributesOfItem(atPath: 输出.path)[.size] as? NSNumber)?.int64Value ?? 0
        return 结果(输出URL: 输出,
                  帧数: 已处理,
                  输入宽: 显示宽,
                  输入高: 显示高,
                  低清宽: 低宽,
                  低清高: 低高,
                  输出宽: 输出宽,
                  输出高: 输出高,
                  耗时秒: Date().timeIntervalSince(开始时刻),
                  含音轨: 有音轨,
                  输出字节: 体积)
    }

    /// 把成片视频与源音轨合成一份：透传不重编码，画质与音质均按原样搬运。
    private static func 混流(视频: URL, 源: URL, 输出: URL,
                          取消: @escaping @Sendable () -> Bool) throws {
        let 合成 = AVMutableComposition()
        let 视源 = AVURLAsset(url: 视频)
        let 视轨源列表 = 视源.tracks(withMediaType: .video)
        guard let 视轨源 = 视轨源列表.first,
              let 视轨 = 合成.addMutableTrack(withMediaType: .video,
                                          preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw 超分错误(说明: "音轨混流失败：中间视频轨不可用")
        }
        let 视时长 = 视源.duration
        guard CMTimeGetSeconds(视时长) > 0 else {
            throw 超分错误(说明: "音轨混流失败：中间视频时长为 0")
        }
        try 视轨.insertTimeRange(CMTimeRange(start: .zero, duration: 视时长), of: 视轨源, at: .zero)
        视轨.preferredTransform = 视轨源.preferredTransform

        let 音源 = AVURLAsset(url: 源)
        if let 音频源轨 = 音源.tracks(withMediaType: .audio).first,
           let 音轨 = 合成.addMutableTrack(withMediaType: .audio,
                                      preferredTrackID: kCMPersistentTrackID_Invalid) {
            let 音范围 = 音频源轨.timeRange
            let 可用时长 = CMTimeMinimum(音范围.duration, 视时长)
            if CMTimeGetSeconds(可用时长) > 0 {
                try 音轨.insertTimeRange(CMTimeRange(start: 音范围.start, duration: 可用时长),
                                       of: 音频源轨, at: .zero)
            }
        }

        guard let 导出 = AVAssetExportSession(asset: 合成, presetName: AVAssetExportPresetPassthrough) else {
            throw 超分错误(说明: "音轨混流失败：无法建立导出会话")
        }
        try? FileManager.default.removeItem(at: 输出)
        导出.outputURL = 输出
        导出.outputFileType = .mp4
        导出.shouldOptimizeForNetworkUse = false
        let 号 = DispatchSemaphore(value: 0)
        导出.exportAsynchronously { 号.signal() }
        let 起 = DispatchTime.now().uptimeNanoseconds
        while 号.wait(timeout: .now() + 0.2) == .timedOut {
            if 取消() {
                导出.cancelExport()
                throw 超分错误(说明: "已取消")
            }
            if DispatchTime.now().uptimeNanoseconds - 起 > 180_000_000_000 {
                导出.cancelExport()
                throw 超分错误(说明: "音轨混流 180 秒未完成，已中止")
            }
        }
        guard 导出.status == .completed else {
            throw 超分错误(说明: "音轨混流失败：\(导出.error?.localizedDescription ?? "未知原因")")
        }
    }

    /// 当前进程真实物理占用（MB，口径同活动监视器"内存"列）。
    /// 用 task_vm_info.phys_footprint 而非 resident_size：后者不含已被 Metal/驱动记账
    /// 但尚未成为常驻页的脏页，会把数十 GB 的占用显示成几百 MB，掩盖真实问题。
    private static func 内存占用MB() -> Double {
        var 信息 = task_vm_info_data_t()
        var 计数 = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let 结果 = withUnsafeMutablePointer(to: &信息) { 指针 in
            指针.withMemoryRebound(to: integer_t.self, capacity: Int(计数)) { 整数指针 in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), 整数指针, &计数)
            }
        }
        guard 结果 == KERN_SUCCESS else { return 0 }
        return Double(信息.phys_footprint) / 1024.0 / 1024.0
    }

    // MARK: 色彩口径

    /// 源色彩口径：8bit 编码曲线与写码标签都由源轨实际声明的传递函数决定，一端定义、两端共用。
    /// 通路内部的中转域固定为 sRGB（模型训练域），只做传递函数层面的互逆编解码，不做色域映射；
    /// BT.709 与 sRGB 主色域相同（同 D65、同三基色），故原语/矩阵恒按 709 声明，
    /// 传递函数按源声明在 709 与 sRGB 之间选择——依据只有元数据，没有任何按素材/尺寸/时长的分支。
    private struct 源色彩 {
        let 编码: 色彩通路.传递编码

        var 原语: String { AVVideoColorPrimaries_ITU_R_709_2 }
        var 矩阵: String { AVVideoYCbCrMatrix_ITU_R_709_2 }
        var 传递函数: String {
            编码 == .bt709 ? AVVideoTransferFunction_ITU_R_709_2 : AVVideoTransferFunction_IEC_sRGB
        }
        /// 解释源像素时的兜底空间（像素缓冲自身没带色彩附件时使用）
        var 空间: CGColorSpace { 编码.色彩空间 }

        init(传递函数 源传递: String?) {
            // 8bit SDR 通路只有两种可复现的传递函数：sRGB 系与 BT.709 系。
            // 其它声明（PQ / HLG / 未标注）超出本通路的表达能力，按 BT.709 处理（与既有行为一致）。
            编码 = (源传递 == AVVideoTransferFunction_IEC_sRGB) ? .sRGB : .bt709
        }

        static func 读取(_ 格式: CMFormatDescription?) -> 源色彩 {
            let 扩展 = 格式.flatMap { CMFormatDescriptionGetExtensions($0) as? [String: Any] } ?? [:]
            return 源色彩(传递函数: 扩展[kCMFormatDescriptionExtension_TransferFunction as String] as? String)
        }
    }

    /// 像素缓冲自身声明的色彩空间（CoreVideo 已按 primaries/transfer/matrix 附件解析好）；
    /// 没有附件时返回 nil，由调用方以轨道声明的空间兜底。
    private static func 像素空间(_ 像素: CVPixelBuffer) -> CGColorSpace? {
        if let 空间 = CVImageBufferGetColorSpace(像素) { return 空间.takeUnretainedValue() }
        let 原语 = CVBufferCopyAttachment(像素, kCVImageBufferColorPrimariesKey, nil) as? String
        let 传递 = CVBufferCopyAttachment(像素, kCVImageBufferTransferFunctionKey, nil) as? String
        guard 原语 != nil || 传递 != nil else { return nil }
        return 源色彩(传递函数: 传递).空间
    }

    // MARK: 像素搬运

    /// 视频帧 → 转正（应用 preferredTransform）后的 CGImage，色彩空间取源像素附件的声明。
    private static func 转正图(_ 像素: CVPixelBuffer,
                             变换: CGAffineTransform,
                             上下文: CIContext,
                             回退空间: CGColorSpace) -> CGImage? {
        var 图 = CIImage(cvPixelBuffer: 像素)
        if !变换.isIdentity {
            图 = 图.transformed(by: 变换)
        }
        let 范围 = 图.extent
        guard 范围.width >= 1, 范围.height >= 1 else { return nil }
        if 范围.origin != .zero {
            图 = 图.transformed(by: CGAffineTransform(translationX: -范围.origin.x, y: -范围.origin.y))
        }
        let 目标 = CGRect(x: 0, y: 0, width: 范围.width.rounded(), height: 范围.height.rounded())
        // 按源像素自己声明的色彩空间解释像素（缺失时用轨道声明的空间兜底），
        // 让 CoreGraphics 走恒等映射，不再隐式做一次 709→sRGB 传递函数转换。
        // 真正的 sRGB 化只在线性光域（色彩通路.线性图）做一次，与输出端的编码曲线严格互逆。
        let 空间 = 像素空间(像素) ?? 回退空间
        return 上下文.createCGImage(图, from: 目标, format: .RGBA8, colorSpace: 空间)
    }

    /// CGImage → BGRA CVPixelBuffer（就地绘制，像素级不缩放即 1:1 拷入）。
    /// 绘制上下文固定取图自身的色彩空间，CoreGraphics 只会做逐字节恒等映射，
    /// 不会把已经按目标传递函数编码好的码值再重新解释一遍；`空间` 仅在图未声明空间时兜底。
    private static func 写入(_ 缓冲: CVPixelBuffer, 图: CGImage, 空间: CGColorSpace) {
        CVPixelBufferLockBaseAddress(缓冲, [])
        defer { CVPixelBufferUnlockBaseAddress(缓冲, []) }
        let 宽 = CVPixelBufferGetWidth(缓冲)
        let 高 = CVPixelBufferGetHeight(缓冲)
        let 行字节 = CVPixelBufferGetBytesPerRow(缓冲)
        guard let 基 = CVPixelBufferGetBaseAddress(缓冲),
              let 上下文 = CGContext(data: 基,
                                 width: 宽,
                                 height: 高,
                                 bitsPerComponent: 8,
                                 bytesPerRow: 行字节,
                                 space: 图.colorSpace ?? 空间,
                                 bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                     | CGBitmapInfo.byteOrder32Little.rawValue) else {
            return
        }
        上下文.interpolationQuality = .none
        上下文.draw(图, in: CGRect(x: 0, y: 0, width: 宽, height: 高))
    }
}
