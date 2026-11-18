import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// 动图（GIF 等多帧图片）超分：CGImageSource 逐帧解出全分辨率帧 → 与单帧预览、整段视频
/// 完全相同的 超分通道.超分单帧 前向 → ImageIO 逐帧写回 GIF。
///
/// 与 视频超分 的两点差别：①帧从 CGImageSourceCreateImageAtIndex 来，不是 AVAssetReader 解码；
/// ②成片保持源格式写成 GIF（逐帧原始延时 + 源循环次数），不转成 mp4。
/// 超分核心与落盘收尾口径全部复用同一条实现，因此动图成片与图片单帧、视频成片像素级一致。
nonisolated enum 动图超分 {

    typealias 结果 = 视频超分.结果
    typealias 进度回调 = 视频超分.进度回调

    /// 缺省帧率：GIF 未声明 DelayTime 时按 10fps 编码（动图惯例档位）。
    nonisolated static let 默认帧率 = 10.0

    /// 逐帧显示时长（秒）：优先用 GIF 的 UnclampedDelayTime，其次 DelayTime，两者缺失按默认帧率折算。
    /// 浏览器与 QuickTime 均以 UnclampedDelayTime 为准，故取同一优先级，避免与播放器判定不一致。
    nonisolated static func 帧时延(源: CGImageSource, 帧数: Int) -> [Double] {
        let 缺省 = 1.0 / 默认帧率
        var 时延: [Double] = []
        时延.reserveCapacity(帧数)
        for 序号 in 0..<帧数 {
            let 属性 = CGImageSourceCopyPropertiesAtIndex(源, 序号, nil) as? [String: Any]
            let 动图 = 属性?[kCGImagePropertyGIFDictionary as String] as? [String: Any]
            let 秒 = (动图?[kCGImagePropertyGIFUnclampedDelayTime as String] as? Double)
                ?? (动图?[kCGImagePropertyGIFDelayTime as String] as? Double)
                ?? 缺省
            时延.append(秒.isFinite && 秒 > 0 ? 秒 : 缺省)
        }
        return 时延
    }

    /// 容器循环次数：从源动图的 GIF 容器属性里读回（0 表示无限循环）。
    /// 读不到时按 0 处理——GIF 的行业惯例就是缺省无限循环，写 1 会让成片只播一遍。
    nonisolated static func 循环次数(源: CGImageSource) -> Int {
        let 属性 = CGImageSourceCopyProperties(源, nil) as? [String: Any]
        let 动图 = 属性?[kCGImagePropertyGIFDictionary as String] as? [String: Any]
        if let 数 = 动图?[kCGImagePropertyGIFLoopCount as String] as? NSNumber {
            return max(0, 数.intValue)
        }
        return 0
    }

    /// 成片落地路径：与整段视频同目录，但保持动图格式（源名_超分4x.gif，重名自动加序号，不覆盖既有文件）。
    nonisolated static func 输出地址(源: URL) -> URL {
        let 覆盖目录 = ProcessInfo.processInfo.environment["UPSAMPLE_OUT_DIR"]
        let 目录 = 覆盖目录.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? 文档目录.超分输出目录
        try? FileManager.default.createDirectory(at: 目录, withIntermediateDirectories: true)
        let 基名 = 源.deletingPathExtension().lastPathComponent + "_超分4x"
        var 路径 = 目录.appendingPathComponent(基名 + ".gif")
        var 序号 = 2
        while FileManager.default.fileExists(atPath: 路径.path) {
            路径 = 目录.appendingPathComponent("\(基名)_\(序号).gif")
            序号 += 1
        }
        return 路径
    }

    /// 跑完整段动图。全部工作在本函数所在线程同步完成（调用方负责放到后台 Task）。
    /// 取消或失败时删除中间文件，不残留半成品；成片落盘成功后才把文件移动到 输出。
    static func 运行(源: URL,
                    输出: URL,
                    短边上限: Int,
                    用Wan: Bool,
                    调色: 色彩通路.调色参数,
                    抖动: Bool,
                    取消: @escaping @Sendable () -> Bool,
                    进度: @escaping 进度回调) async throws -> 结果 {

        let 开始时刻 = Date()
        guard let 图源 = CGImageSourceCreateWithURL(源 as CFURL, nil) else {
            throw 超分错误(说明: "读不到动图：\(源.lastPathComponent)")
        }
        let 总帧数 = CGImageSourceGetCount(图源)
        guard 总帧数 > 1 else {
            throw 超分错误(说明: "动图帧数不足（\(总帧数) 帧）：\(源.lastPathComponent)")
        }
        // 首帧只用来量显示尺寸：动图各帧同尺寸，逐帧解码时再取全分辨率位图。
        guard let 首帧 = CGImageSourceCreateImageAtIndex(图源, 0, nil) else {
            throw 超分错误(说明: "动图首帧解码失败：\(源.lastPathComponent)")
        }
        let 显示宽 = 首帧.width
        let 显示高 = 首帧.height
        guard 显示宽 > 0, 显示高 > 0 else {
            throw 超分错误(说明: "动图尺寸异常：\(显示宽)×\(显示高)")
        }

        let 时延 = 帧时延(源: 图源, 帧数: 总帧数)

        // 低清目标尺寸与单帧预览、整段视频同口径：短边封顶、只缩不放、对齐 8；输出恒为 4 倍
        let 比例 = min(1.0, Double(短边上限) / Double(min(显示宽, 显示高)))
        let 低宽 = 超分通道.对齐8(Int((Double(显示宽) * 比例).rounded()))
        let 低高 = 超分通道.对齐8(Int((Double(显示高) * 比例).rounded()))
        let 输出宽 = 低宽 * 4
        let 输出高 = 低高 * 4

        // 中间文件放 cache 目录，绝不直接写超分输出目录，避免取消后留下半成品动图。
        let 中间目录 = 文档目录.缓存目录
        try? FileManager.default.createDirectory(at: 中间目录, withIntermediateDirectories: true)
        let 中间动图 = 中间目录
            .appendingPathComponent("动图中间-" + UUID().uuidString + ".gif")
        // 唯一的半成品清理点：取消、解码失败、编码失败、落盘失败都从这里走。
        func 清理() { try? FileManager.default.removeItem(at: 中间动图) }

        // 输出保持源格式：ImageIO 直接写 GIF，帧序列与源一一对应，不经过视频转码。
        guard let 写 = CGImageDestinationCreateWithURL(中间动图 as CFURL,
                                                    UTType.gif.identifier as CFString,
                                                    总帧数,
                                                    nil) else {
            清理()
            throw 超分错误(说明: "无法创建动图编码器：\(源.lastPathComponent)")
        }
        // 容器级属性：循环次数与源动图一致（读不到按 0 无限循环），否则成片只会播一遍。
        let 容器属性: [String: Any] = [
            kCGImagePropertyGIFDictionary as String: [
                kCGImagePropertyGIFLoopCount as String: 循环次数(源: 图源)
            ]
        ]
        CGImageDestinationSetProperties(写, 容器属性 as CFDictionary)

        var 已处理 = 0

        do {
            for 序号 in 0..<总帧数 {
                if 取消() { throw 超分错误(说明: "已取消") }
                // 逐帧从原文件解全分辨率位图：不在内存里缓存整段，长动图也不会撑爆内存。
                guard let 帧 = CGImageSourceCreateImageAtIndex(图源, 序号, nil) else {
                    throw 超分错误(说明: "第 \(序号 + 1) 帧解码失败")
                }
                // 复用单帧超分同一条前向（与视频逐帧、图片单帧完全同源）
                let 单帧 = try 超分通道.超分单帧(帧,
                                              短边上限: 短边上限,
                                              用Wan: 用Wan,
                                              调色: 调色,
                                              抖动: 抖动)
                // 每帧写回它自己的原始延时：GIF 允许逐帧不同时长，压成固定帧率会改变动图节奏。
                let 秒 = 时延[序号]
                let 帧属性: [String: Any] = [
                    kCGImagePropertyGIFDictionary as String: [
                        kCGImagePropertyGIFDelayTime as String: 秒,
                        kCGImagePropertyGIFUnclampedDelayTime as String: 秒
                    ]
                ]
                CGImageDestinationAddImage(写, 单帧.图, 帧属性 as CFDictionary)
                已处理 = 序号 + 1
                进度(已处理, 总帧数)
            }

            guard CGImageDestinationFinalize(写) else {
                throw 超分错误(说明: "动图编码收尾失败：已写入 \(已处理)/\(总帧数) 帧")
            }

            // 收尾成功后才落到超分输出目录：取消/失败路径不会在成片目录留下半成品。
            if FileManager.default.fileExists(atPath: 输出.path) {
                try? FileManager.default.removeItem(at: 输出)
            }
            do {
                try FileManager.default.moveItem(at: 中间动图, to: 输出)
            } catch {
                throw 超分错误(说明: "成片落盘失败：\(error.localizedDescription)")
            }
        } catch {
            清理()
            throw error
        }

        let 体积 = (try? FileManager.default.attributesOfItem(atPath: 输出.path)[.size] as? NSNumber)?
            .int64Value ?? 0
        return 结果(输出URL: 输出,
                  帧数: 总帧数,
                  输入宽: 显示宽,
                  输入高: 显示高,
                  低清宽: 低宽,
                  低清高: 低高,
                  输出宽: 输出宽,
                  输出高: 输出高,
                  耗时秒: Date().timeIntervalSince(开始时刻),
                  含音轨: false,
                  输出字节: 体积)
    }

}
