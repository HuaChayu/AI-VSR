import AppKit
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

enum 素材类型: String, CaseIterable, Identifiable {
    case 视频
    case 图片

    var id: String { rawValue }
    var 标题: String { rawValue }
    var 图标: String { self == .视频 ? "film" : "photo" }
}

/// 素材条目：素材箱里的一条。只读元数据 + 缩略图，不含任何编辑指令。
struct 素材条目: Identifiable {
    let id = UUID()
    var url: URL
    var 类型: 素材类型
    var 名称: String
    var 时长: Double?
    var 像素尺寸: CGSize?
    var 帧率: Double?
    /// 图片素材的帧数：GIF 等多帧容器用 CGImageSourceGetCount 读出，静态图为 1，读不到为 nil。
    /// 帧数 > 1 的图片即「动图」，超分走逐帧通路并保持源格式输出 GIF；筛选仍归 .图片 不变。
    var 帧数: Int?
    var 体积: Int64?
    var 缩略图: NSImage?
    var 导入时间: Date
    /// 是否已处理完成（超分跑完即标记；批量处理时逐条置位），素材栏右上角显示对勾。
    var 完成 = false

    init(url: URL, 类型: 素材类型, 名称: String? = nil) {
        self.url = url
        self.类型 = 类型
        self.名称 = 名称 ?? url.deletingPathExtension().lastPathComponent
        self.导入时间 = Date()
    }

    var 时长文本: String {
        guard let 时长, 时长.isFinite, 时长 > 0 else { return "—" }
        let 总秒 = Int(时长.rounded())
        return String(format: "%d:%02d", 总秒 / 60, 总秒 % 60)
    }

    var 分辨率文本: String {
        guard let 像素尺寸 else { return "—" }
        return "\(Int(像素尺寸.width)) × \(Int(像素尺寸.height))"
    }

    var 帧率文本: String {
        guard let 帧率, 帧率 > 0 else { return "—" }
        return String(format: "%.2f fps", 帧率)
    }

    var 体积文本: String {
        guard let 体积 else { return "—" }
        return ByteCountFormatter.string(fromByteCount: 体积, countStyle: .file)
    }

    /// 是否多帧动图：类型仍是 .图片（筛选口径不变），只是帧数多于 1。
    var 是动图: Bool { 类型 == .图片 && (帧数 ?? 1) > 1 }

    /// 动图帧数（非动图返回 0）：多帧通路据此取总帧数。
    var 动图帧数: Int { 是动图 ? (帧数 ?? 0) : 0 }

    /// 「动图 · N 帧」标注：素材栏副标题与详情元数据共用同一口径。
    var 动图文本: String { 是动图 ? "动图 · \(帧数 ?? 0) 帧" : "" }

    /// 素材栏 / 详情副标题文本：动图用「动图 · N 帧」替代类型名，静态图片与视频保持原样。
    var 副标题文本: String { 是动图 ? 动图文本 : 类型.标题 }
}

/// 素材读取：只负责「把文件变成 素材条目」，不碰 UI，不碰 Store。
enum 素材读取 {
    static func 判定类型(_ url: URL) -> 素材类型? {
        guard let 类型 = UTType(filenameExtension: url.pathExtension.lowercased()) else { return nil }
        if 类型.conforms(to: .movie) || 类型.conforms(to: .video) { return .视频 }
        if 类型.conforms(to: .image) { return .图片 }
        return nil
    }

    /// 图片帧数：CGImageSourceGetCount 对 GIF / APNG 等多帧容器返回真实帧数，
    /// 静态图片返回 1；容器读不出来时返回 nil（此时按静态图处理）。
    static func 图片帧数(url: URL) -> Int? {
        guard let 源 = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let 数量 = CGImageSourceGetCount(源)
        return 数量 > 0 ? 数量 : nil
    }

    static func 读取(url: URL) async -> 素材条目? {
        guard let 类型 = 判定类型(url) else { return nil }
        var 条目 = 素材条目(url: url, 类型: 类型)

        if let 属性 = try? FileManager.default.attributesOfItem(atPath: url.path),
           let 大小 = 属性[.size] as? NSNumber {
            条目.体积 = 大小.int64Value
        }

        switch 类型 {
        case .视频:
            let 资源 = AVURLAsset(url: url)
            if let 时长 = try? await 资源.load(.duration) {
                条目.时长 = CMTimeGetSeconds(时长)
            }
            if let 轨道 = try? await 资源.loadTracks(withMediaType: .video).first {
                if let 尺寸 = try? await 轨道.load(.naturalSize) {
                    条目.像素尺寸 = 尺寸
                }
                if let 帧率 = try? await 轨道.load(.nominalFrameRate) {
                    条目.帧率 = Double(帧率)
                }
            }
            条目.缩略图 = await 视频解码.缩略图(url: url)

        case .图片:
            // 先记帧数：GIF 等多帧容器读到 > 1 时超分走逐帧通路，不再只解首帧出一张 PNG。
            条目.帧数 = 图片帧数(url: url)
            if let 图 = NSImage(contentsOf: url) {
                条目.缩略图 = 图
                条目.像素尺寸 = 图.像素尺寸
            }
        }

        return 条目
    }
}
