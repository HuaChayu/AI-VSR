// ============================================================
//  文件作用：预览区与底部信息区之间的播放条（Liquid Glass 玻璃面板）。
//  ① 播放 / 暂停（与全局空格快捷键同一入口，按钮状态跟随播放内核）
//  ② 可拖动进度条：拖动即暂停并 seek 定格，松手按拖动前的播放状态决定是否续播
//  ③ 当前时间 / 总时长（动图额外显示帧号）
//  ④ 右侧「一次播放 / 无限循环」切换，状态持久化到 UserDefaults（预览播放模式）
//  互动文件：视频 播放.swift（预览播放器）、素材 详情.swift（插入位置与素材上下文）。
// ============================================================

import SwiftUI

struct 播放条: View {
    let 播放器: 预览播放器
    /// 当前选中的素材：决定禁用态与「静态图片不支持播放」的说明文案
    let 条目: 素材条目?

    @State private var 拖动中 = false
    @State private var 拖动值: Double = 0
    @State private var 拖动前播放中 = false

    var body: some View {
        HStack(spacing: 12) {
            播放按钮

            VStack(alignment: .leading, spacing: 2) {
                进度轨道
                HStack(spacing: 8) {
                    Text("\(时间文本(显示时间)) / \(时间文本(播放器.总时长))")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.textSecondary)
                    if 播放器.种类 == .动图, 播放器.动图总帧 > 0 {
                        Text("帧 \(播放器.动图帧号)/\(播放器.动图总帧)")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(Theme.textTertiary)
                    }
                    if let 说明 {
                        Text(说明)
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.textTertiary)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
            }

            音量控件

            模式按钮
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .modifier(玻璃面板())
        .padding(.horizontal, Theme.padding)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.panelBackground)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Theme.divider)
                .frame(height: Theme.hairline)
        }
    }

    // MARK: 控件

    private var 播放按钮: some View {
        Button {
            播放器.切换()
        } label: {
            Image(systemName: 播放器.播放中 ? "pause.fill" : "play.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(播放器.可播放 ? Theme.textPrimary : Theme.textTertiary)
                .frame(width: 30, height: 30)
                .background(
                    Circle()
                        .fill(播放器.可播放 ? Theme.fieldBackground : Color.clear)
                        .overlay(Circle().strokeBorder(Theme.stroke, lineWidth: Theme.hairline))
                )
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!播放器.可播放)
        .help(播放器.可播放 ? "播放 / 暂停（空格）" : "当前素材没有可播放内容")
    }

    private var 进度轨道: some View {
        Slider(value: 进度绑定,
               in: 0...max(播放器.总时长, 0.001),
               onEditingChanged: 拖动状态改变)
            .controlSize(.mini)
            .tint(Theme.accent)
            .disabled(!播放器.可播放 || 播放器.总时长 <= 0)
            .help("拖动进度：暂停并定格到该位置，松手后按原状态续播")
    }

    /// 音量区：静音按钮 + 0…1 音量滑杆，两者与播放内核的 静音 / 音量 双向绑定。
    private var 音量控件: some View {
        HStack(spacing: 6) {
            静音按钮
            Slider(value: 音量绑定, in: 0...1)
                .controlSize(.mini)
                .tint(Theme.accent)
                .frame(width: 72)
                .disabled(!有声源)
                .help("预览音量：拉到最右会在原始音轨上再增益约 +6dB，拉到最左等同静音")
        }
    }

    private var 静音按钮: some View {
        Button {
            let 原来静音 = 播放器.静音
            播放器.静音 = !播放器.静音
            // 从静音（或因滑杆拉到 0 而静音）恢复时给一个能听见的起点，否则点了像没反应。
            if 原来静音, 播放器.音量 <= 0.001 { 播放器.音量 = 0.8 }
        } label: {
            Image(systemName: 音量图标)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(有声源 ? Theme.textPrimary : Theme.textTertiary)
                .frame(width: 26, height: 26)
                .background(
                    Circle()
                        .fill(有声源 ? Theme.fieldBackground : Color.clear)
                        .overlay(Circle().strokeBorder(Theme.stroke, lineWidth: Theme.hairline))
                )
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(!有声源)
        .help(播放器.静音 ? "取消静音" : "静音")
    }

    /// 只有视频才有音轨；动图、静态图、空素材一律禁用音量控件。
    private var 有声源: Bool { 播放器.种类 == .视频 }

    private var 音量图标: String {
        if 播放器.静音 || 播放器.音量 <= 0.001 { return "speaker.slash.fill" }
        if 播放器.音量 < 0.34 { return "speaker.fill" }
        if 播放器.音量 < 0.67 { return "speaker.wave.1.fill" }
        return "speaker.wave.2.fill"
    }

    /// 滑杆 ↔ 内核：静音时滑杆显示 0；把滑杆推离 0 会自动解除静音，拉到 0 视为静音。
    private var 音量绑定: Binding<Double> {
        Binding(get: {
            播放器.静音 ? 0 : Double(播放器.音量)
        }, set: { 值 in
            let 目标 = Float(min(max(值, 0), 1))
            if 目标 <= 0.001 {
                播放器.音量 = 0
                播放器.静音 = true
            } else {
                播放器.音量 = 目标
                播放器.静音 = false
            }
        })
    }

    private var 模式按钮: some View {
        Button {
            播放器.模式 = 播放器.模式.另一
        } label: {
            HStack(spacing: 4) {
                Image(systemName: 播放器.模式.图标)
                    .font(.system(size: 10, weight: .semibold))
                Text(播放器.模式.标题)
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(Theme.textSecondary)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(Theme.fieldBackground)
                    .overlay(Capsule().strokeBorder(Theme.stroke, lineWidth: Theme.hairline))
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help("\(播放器.模式.提示)（点一下切到\(播放器.模式.另一.标题)）")
    }

    // MARK: 绑定与文案

    private var 显示时间: Double {
        if 拖动中 { return 拖动值 }
        return min(max(播放器.当前时间, 0), max(播放器.总时长, 0))
    }

    private var 进度绑定: Binding<Double> {
        Binding(get: { 显示时间 },
                set: { 值 in
                    拖动值 = 值
                    播放器.定位(到: 值)
                })
    }

    private func 拖动状态改变(_ 编辑中: Bool) {
        if 编辑中 {
            拖动前播放中 = 播放器.播放中
            拖动值 = 播放器.当前时间
            播放器.暂停()
            拖动中 = true
        } else {
            播放器.定位(到: 拖动值)
            拖动中 = false
            if 拖动前播放中 { 播放器.播放() }
        }
    }

    private var 说明: String? {
        switch 播放器.种类 {
        case .视频, .动图:
            return nil
        case .静态图:
            return "静态图片不支持播放"
        case .空:
            return 条目 == nil ? "未选中素材" : nil
        }
    }

    private func 时间文本(_ 秒: Double) -> String {
        guard 秒.isFinite, 秒 > 0 else { return "0:00.0" }
        let 整秒 = Int(秒)
        return String(format: "%d:%04.1f", 整秒 / 60, 秒 - Double(整秒 / 60 * 60))
    }
}

// ============================================================

/// Liquid Glass 面板：macOS 26 起用系统玻璃材质；更早的 SDK 回退到 ultraThinMaterial + 细描边。
private struct 玻璃面板: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content
                .glassEffect(.regular, in: .rect(cornerRadius: 14))
        } else {
            content
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(.ultraThinMaterial)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(Theme.stroke, lineWidth: Theme.hairline)
                )
        }
    }
}
