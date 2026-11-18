import SwiftUI

/// 主窗口：超分 / 占位 两个工作区并列（第二个 tab 功能待定，暂用占位名）。
struct ContentView: View {
    @State private var 输出 = 输出Store()
    @State private var 素材 = 素材Store()

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
    }
}
