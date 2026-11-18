import SwiftUI

/// 输出页：占位。参数模型已就位但全部禁用，等编码器接入后再打开。
struct 输出工作区: View {
    @Environment(输出Store.self) private var 输出

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "输出", symbol: "square.and.arrow.up", 副标题: "占位")

            ScrollView {
                VStack(spacing: 16) {
                    占位说明
                    参数卡
                    计划卡
                }
                .padding(Theme.padding)
                .frame(maxWidth: 720, alignment: .center)
                .frame(maxWidth: .infinity)
            }
        }
        .background(Theme.canvasBackground)
    }

    private var 占位说明: some View {
        VStack(spacing: 10) {
            Image(systemName: "square.and.arrow.up.trianglebadge.exclamationmark")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(Theme.textTertiary)
            Text("输出通道占位")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Theme.textPrimary)
            Text("超分与分离的产物如何合成导出，等两条通道的通路定下来再落地。当前页面只固化参数口径，不做编码。")
                .font(.system(size: 11))
                .foregroundStyle(Theme.textTertiary)
                .multilineTextAlignment(.center)
                .lineSpacing(3)
                .frame(maxWidth: 460)
        }
        .padding(.vertical, 26)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .fill(Theme.panelBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .strokeBorder(Theme.stroke, lineWidth: 1)
        )
    }

    private var 参数卡: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text("输出参数")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                小徽章(文本: "只读")
                Spacer(minLength: 0)
            }

            参数行(标题: "容器", 内容: 输出.容器格式.rawValue)
            参数行(标题: "编码", 内容: 输出.编码器.rawValue)
            参数行(标题: "分辨率", 内容: 输出.分辨率.rawValue)
            参数行(标题: "码率", 内容: String(format: "%.0f Mbps", 输出.码率))
            参数行(标题: "文件名", 内容: 输出.文件名)

            Divider().overlay(Theme.divider)

            HStack(spacing: 8) {
                Button("选择输出位置") { }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(true)
                Button("开始输出") { 输出.试导出() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .tint(Theme.accent)
                    .disabled(true)
                Spacer(minLength: 0)
            }
        }
        .padding(Theme.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .fill(Theme.panelBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .strokeBorder(Theme.stroke, lineWidth: 1)
        )
    }

    private var 计划卡: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("后续要接的东西")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Theme.textPrimary)

            计划行("一", "超分与分离的中间产物落盘格式（帧序列 / 带 alpha 的序列）")
            计划行("二", "AVAssetWriter 编码通路 + 色彩空间与 HDR 元数据透传")
            计划行("三", "批量导出队列与失败重试")
        }
        .padding(Theme.padding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .fill(Theme.panelBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.panelRadius, style: .continuous)
                .strokeBorder(Theme.stroke, lineWidth: 1)
        )
    }

    private func 参数行(标题: String, 内容: String) -> some View {
        HStack(spacing: 10) {
            Text(标题)
                .font(.system(size: 11))
                .foregroundStyle(Theme.textSecondary)
                .frame(width: 68, alignment: .leading)
            Text(内容)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
            Spacer(minLength: 0)
        }
    }

    private func 计划行(_ 序号: String, _ 文本: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(序号)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Theme.accent)
                .frame(width: 12)
            Text(文本)
                .font(.system(size: 11))
                .foregroundStyle(Theme.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }
}
