import SwiftUI
import UniformTypeIdentifiers

/// 主窗口：超分 / 占位 两个工作区并列（第二个 tab 功能待定，暂用占位名）。
///
/// 整个窗口都是素材投放区：从访达把文件拖到任意位置，等同于点「添加素材」/「添加视频」——
/// 统一走 素材Store.导入(urls:)，所以类型判定、去重、不支持即跳过全部与按钮同一条路。
/// 只认文件 URL：文本、颜色、内部列表等其它拖放不匹配这里的类型，仍由原控件自己处理。
struct ContentView: View {
    @State private var 输出 = 输出Store()
    @State private var 素材 = 素材Store()
    /// 有文件悬在窗口上：只用于画一圈提示描边，不参与任何判定。
    @State private var 拖放悬停 = false

    var body: some View {
        TabView {
            超分工作区()
                .tabItem { Label("超分", systemImage: "wand.and.stars") }

            输出工作区()
                .tabItem { Label("占位", systemImage: "square.dashed") }
        }
        .environment(输出)
        .environment(素材)
        .frame(minWidth: 1080, minHeight: 680)
        .onDrop(of: [UTType.fileURL], isTargeted: $拖放悬停) { 提供者 in
            收下文件(提供者)
        }
        .overlay {
            if 拖放悬停 {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.accent.opacity(0.7), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .allowsHitTesting(false)
                    .padding(2)
            }
        }
    }

    // MARK: 拖放收口

    /// 只接「文件 URL」这一类拖放；没有可用的文件就返回 false，把这次拖放原样还给系统/子控件。
    /// 多文件一起拖入时逐个取 URL，能取到几个就交给 导入 几个。
    private func 收下文件(_ 提供者: [NSItemProvider]) -> Bool {
        let 文件型 = 提供者.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }
        guard !文件型.isEmpty else { return false }

        Task { @MainActor in
            var 路径们: [URL] = []
            for 一项 in 文件型 {
                if let 路径 = await Self.文件路径(一项), !路径们.contains(路径) {
                    路径们.append(路径)
                }
            }
            // 与两个导入按钮同一个入口：内部按扩展名判定类型、自动去重、不支持的直接跳过继续下一个。
            guard !路径们.isEmpty else { return }
            await 素材.导入(urls: 路径们)
        }
        return true
    }

    /// 从拖放项里取出文件 URL：系统一般给 Data 形式的 file:// 字节串，个别来源直接给 URL 或字符串。
    private static func 文件路径(_ 提供者: NSItemProvider) async -> URL? {
        await withCheckedContinuation { 续 in
            提供者.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { 载入, _ in
                var 结果: URL?
                if let 数据 = 载入 as? Data {
                    结果 = URL(dataRepresentation: 数据, relativeTo: nil)
                } else if let 地址 = 载入 as? URL {
                    结果 = 地址
                } else if let 字符串 = 载入 as? String {
                    结果 = URL(string: 字符串)
                }
                续.resume(returning: 结果?.standardizedFileURL)
            }
        }
    }
}
