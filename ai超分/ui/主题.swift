import SwiftUI
import AppKit

// MARK: - 动态配色工具
//
// 深浅色都用同一套语义名，靠 NSColor 动态 provider 跟随系统外观切换，
// 避免每个视图里再判断 colorScheme。

private func 动态色(_ 浅: (CGFloat, CGFloat, CGFloat), _ 深: (CGFloat, CGFloat, CGFloat)) -> Color {
    Color(nsColor: NSColor(name: nil) { 外观 in
        let 是深 = 外观.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        let 值 = 是深 ? 深 : 浅
        return NSColor(srgbRed: 值.0, green: 值.1, blue: 值.2, alpha: 1)
    })
}

private func 动态透明色(_ 浅: (CGFloat, CGFloat, CGFloat, CGFloat),
                     _ 深: (CGFloat, CGFloat, CGFloat, CGFloat)) -> Color {
    Color(nsColor: NSColor(name: nil) { 外观 in
        let 是深 = 外观.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        let 值 = 是深 ? 深 : 浅
        return NSColor(srgbRed: 值.0, green: 值.1, blue: 值.2, alpha: 值.3)
    })
}

// MARK: - 主题

enum Theme {
    // 尺寸
    static let panelRadius: CGFloat = 16
    static let padding: CGFloat = 14
    static let tightPadding: CGFloat = 10
    static let hairline: CGFloat = 1

    // 四级背景：画布 / 面板 / 检查器 / 视口
    static var canvasBackground: Color {
        动态色((0.949, 0.953, 0.961), (0.086, 0.090, 0.100))
    }
    static var panelBackground: Color {
        动态色((0.984, 0.986, 0.992), (0.125, 0.132, 0.146))
    }
    static var inspectorBackground: Color {
        动态色((1.000, 1.000, 1.000), (0.108, 0.114, 0.126))
    }
    static var viewportBackground: Color {
        动态色((0.898, 0.906, 0.918), (0.070, 0.074, 0.082))
    }

    // 控件与线条
    static var fieldBackground: Color {
        动态透明色((0, 0, 0, 0.06), (1, 1, 1, 0.10))
    }
    static var stroke: Color {
        动态透明色((0, 0, 0, 0.14), (1, 1, 1, 0.13))
    }
    static var divider: Color {
        动态透明色((0, 0, 0, 0.08), (1, 1, 1, 0.075))
    }

    // 文字
    static var textPrimary: Color {
        动态色((0.086, 0.094, 0.106), (1.000, 1.000, 1.000))
    }
    static var textSecondary: Color {
        动态色((0.290, 0.310, 0.340), (1.000, 1.000, 1.000))
            .opacity(0.70)
    }
    static var textTertiary: Color {
        动态色((0.480, 0.510, 0.550), (1.000, 1.000, 1.000))
            .opacity(0.46)
    }

    // 强调色与危险色
    static var accent: Color { .accentColor }
    static var danger: Color {
        动态色((0.780, 0.220, 0.200), (0.960, 0.360, 0.320))
    }
}

// MARK: - 细线

/// 1 像素分隔线：默认竖直（宽 1），竖直 false 时横向（高 1）。
struct 细线: View {
    var 竖直: Bool = true

    var body: some View {
        Rectangle()
            .fill(Theme.divider)
            .frame(width: 竖直 ? Theme.hairline : nil,
                   height: 竖直 ? nil : Theme.hairline)
    }
}

// MARK: - 空态

/// 列表 / 详情区的空状态占位：大图标 + 标题 + 一句说明。
struct 空态提示: View {
    var symbol: String
    var 标题: String
    var 说明: String

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(Theme.textTertiary)
            Text(标题)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.textSecondary)
            Text(说明)
                .font(.system(size: 11))
                .foregroundStyle(Theme.textTertiary)
                .multilineTextAlignment(.center)
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - 小徽章

/// 标签式小徽章：弱染色胶囊，用于阶段、格式、只读等短标记。
struct 小徽章: View {
    var 文本: String
    var 颜色: Color = Theme.accent

    var body: some View {
        Text(文本)
            .font(.system(size: 10.5, weight: .semibold))
            .foregroundStyle(颜色)
            .padding(.horizontal, 6)
            .padding(.vertical, 2.5)
            .background(颜色.opacity(0.14), in: Capsule())
            .overlay(Capsule().strokeBorder(颜色.opacity(0.28), lineWidth: 1))
    }
}

// MARK: - 面板标题

/// 面板统一标题条：图标 + 标题 + 右侧副标题。
struct PanelHeader: View {
    var title: String
    var symbol: String
    var 副标题: String?

    init(title: String, symbol: String, 副标题: String? = nil) {
        self.title = title
        self.symbol = symbol
        self.副标题 = 副标题
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.textSecondary)
            Text(title)
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
            Spacer(minLength: 4)
            if let 副标题, !副标题.isEmpty {
                Text(副标题)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panelBackground)
    }
}
