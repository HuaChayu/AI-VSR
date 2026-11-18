import SwiftUI

/// 超分页底部的一条自检条：展示 App 内 Flash 主干自检（FLASH_SELFTEST=1 启动触发）的状态与对拍指标。
struct Flash自检条: View {
    @Bindable var 模型: Flash自检模型

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "bolt.horizontal.circle")
                .font(.system(size: 12))
                .foregroundStyle(Theme.textSecondary)

            Text("Flash 主干自检")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.textSecondary)

            Text("（仅对拍解码器，非超分入口）")
                .font(.system(size: 10))
                .foregroundStyle(Theme.textTertiary)

            Text(模型.状态)
                .font(.system(size: 10))
                .foregroundStyle(Theme.textTertiary)

            if 模型.运行中 {
                ProgressView().controlSize(.mini)
            }

            if 模型.耗时文本 != "—" {
                指标(模型.耗时文本)
            }
            if 模型.保真文本 != "—" {
                指标(模型.保真文本)
            }

            Spacer(minLength: 8)

        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Theme.inspectorBackground)
        .overlay(alignment: .top) { 细线() }
    }

    private func 指标(_ 文本: String) -> some View {
        Text(文本)
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(Theme.textSecondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Theme.panelBackground)
            )
    }
}
