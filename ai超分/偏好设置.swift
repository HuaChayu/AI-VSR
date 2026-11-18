// ============================================================
//  文件作用：偏好设置面板（照抄「无限画布」的 PreferencesView 结构）。
//  左侧导航栏 + 右侧内容区，620×440，FloatingWindow 背景，
//  由 MyApp.swift 的 Settings 场景承载，菜单「偏好设置…」（⌘,）打开。
//  本轮暂无真实参数：各分节内容均为占位文本，后续按需补齐控件。
//  互动文件：MyApp.swift（Settings 场景引用 PreferencesView）。
// ============================================================

import SwiftUI

// ============================================================

// MARK: - 偏好设置面板（左右分栏）

struct PreferencesView: View {
    @State private var selectedSection = 0

    /// 暂只需两页：模型管理 / API / 授权设置 / 快捷键 暂无真实参数，先去掉，用到再补。
    private let sections = ["通用设置", "关于"]

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            content
        }
        .frame(width: 620, height: 440)
        .background(Color(red: 0.95, green: 0.95, blue: 0.97))
        .background(FloatingWindow())
    }

    // 左侧导航栏
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(sections.indices, id: \.self) { index in
                Button {
                    selectedSection = index
                } label: {
                    HStack {
                        Text(sections[index])
                            .font(.system(size: 13))
                            .foregroundColor(selectedSection == index ? .white : .primary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(selectedSection == index ? Color.accentColor : Color.clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Spacer()
        }
        .padding(10)
        .frame(width: 180)
        .background(Color(red: 0.90, green: 0.90, blue: 0.93))
    }

    // 右侧内容区
    private var content: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(sections[selectedSection])
                .font(.title2)
                .bold()
            Divider()
            Text("这里是「\(sections[selectedSection])」的内容区域，占位文本，后续补充。")
                .font(.system(size: 13))
                .foregroundColor(.secondary)
            Spacer()
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color.white)
    }
}

// ============================================================

// MARK: - 偏好设置窗口置顶

/// 让偏好设置窗口保持 floating 层级，点击主体窗口时不会沉到后面。
struct FloatingWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            view.window?.level = .floating
        }
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
