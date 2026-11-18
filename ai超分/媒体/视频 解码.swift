import AVFoundation
import AppKit

/// 视频帧抽取：只取「某时刻的一帧」，用 AVAssetImageGenerator，
/// 不拉播放器、不建 AVPlayerItemVideoOutput——超分页面只读帧。
enum 视频解码 {
    static func 抽帧(url: URL, 时间: Double, 最大边长: CGFloat = 1600) async throws -> CGImage {
        let 资源 = AVURLAsset(url: url)
        let 生成器 = AVAssetImageGenerator(asset: 资源)
        生成器.appliesPreferredTrackTransform = true
        生成器.requestedTimeToleranceBefore = .zero
        生成器.requestedTimeToleranceAfter = .zero
        if 最大边长 > 0 {
            生成器.maximumSize = CGSize(width: 最大边长, height: 最大边长)
        }

        var 目标秒 = max(0, 时间)
        if let 时长 = try? await 资源.load(.duration) {
            let 总秒 = CMTimeGetSeconds(时长)
            if 总秒.isFinite, 总秒 > 0 {
                目标秒 = min(目标秒, max(0, 总秒 - 0.02))
            }
        }

        let 结果 = try await 生成器.image(at: CMTime(seconds: 目标秒, preferredTimescale: 600))
        return 结果.image
    }

    /// 视频首帧缩略图（素材列表用，尺寸压到 480 以内）。
    static func 缩略图(url: URL) async -> NSImage? {
        guard let 时长 = try? await AVURLAsset(url: url).load(.duration) else { return nil }
        let 总秒 = max(0, CMTimeGetSeconds(时长))
        let 取样 = 总秒 > 0 ? min(1.0, 总秒 * 0.1) : 0
        guard let 图 = try? await 抽帧(url: url, 时间: 取样, 最大边长: 480) else { return nil }
        return NSImage.以(cg图: 图)
    }
}
