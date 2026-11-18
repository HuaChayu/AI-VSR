import SwiftUI
import UniformTypeIdentifiers
import Observation

/// 左侧素材栏：导入（视频 / 素材两路）、搜索、类型筛选、条目列表，
/// 底部统计行右侧是「批量超分」与「清空素材箱」两个动作。
struct 素材栏: View {
    @Environment(素材Store.self) private var 素材
    @Environment(超分通道.self) private var 超分
    @State private var 导入视频中 = false
    @State private var 导入素材中 = false
    @State private var 垃圾桶悬停 = false

    /// 批量队列挂在超分通道上，视图重建不会丢进度
    private var 批量: 批量超分队列 { 超分.批量队列 }

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "素材箱", symbol: "tray.full")
            导入区
            细线(竖直: false)
            列表
        }
        .background(Theme.panelBackground)
        .fileImporter(isPresented: $导入视频中,
                      allowedContentTypes: [.movie, .video],
                      allowsMultipleSelection: true) { 结果 in
            switch 结果 {
            case .success(let urls):
                Task { await 素材.导入(urls: urls) }
            case .failure(let 错误):
                素材.错误信息 = 错误.localizedDescription
            }
        }
        .fileImporter(isPresented: $导入素材中,
                      allowedContentTypes: [.image],
                      allowsMultipleSelection: true) { 结果 in
            switch 结果 {
            case .success(let urls):
                Task { await 素材.导入(urls: urls) }
            case .failure(let 错误):
                素材.错误信息 = 错误.localizedDescription
            }
        }
    }

    // MARK: 导入区

    private var 导入区: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                导入按钮(标题: "导入视频", 图标: "film") { 导入视频中 = true }
                导入按钮(标题: "导入素材", 图标: "photo") { 导入素材中 = true }
            }

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.textTertiary)
                TextField("搜索素材名", text: 搜索绑定)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.textPrimary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Theme.fieldBackground)
            )

            Picker("", selection: 筛选绑定) {
                ForEach(素材Store.素材筛选.allCases) { 项 in
                    Text(项.标题).tag(项)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)

            if 素材.导入中 {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("正在解析素材…")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.textSecondary)
                    Spacer(minLength: 0)
                }
            }

            if let 错误 = 素材.错误信息 {
                Text(错误)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.danger)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineLimit(2)
            }
        }
        .padding(Theme.tightPadding)
    }

    private func 导入按钮(标题: String, 图标: String, 动作: @escaping () -> Void) -> some View {
        Button(action: 动作) {
            HStack(spacing: 5) {
                Image(systemName: 图标)
                    .font(.system(size: 11, weight: .medium))
                Text(标题)
                    .font(.system(size: 11, weight: .medium))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Theme.fieldBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(Theme.stroke, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(Theme.textPrimary)
    }

    // MARK: 列表

    private var 列表: some View {
        ScrollView {
            LazyVStack(spacing: 3) {
                ForEach(素材.显示条目) { 条目 in
                    素材行(条目: 条目)
                }
            }
            .padding(8)
            // 给底部统计行（含两个动作按钮）让出高度，最后一行不会被压住
            .padding(.bottom, 22)
        }
        .frame(maxHeight: .infinity)
        .overlay(alignment: .bottom) {
            HStack(spacing: 6) {
                Text(素材.统计文本)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 4)
                批量超分按钮
                清空按钮
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(Theme.panelBackground.opacity(0.95))
        }
        .overlay {
            if 素材.显示条目.isEmpty {
                空态提示(symbol: "tray",
                       标题: 素材.条目.isEmpty ? "素材箱是空的" : "没有匹配的素材",
                       说明: 素材.条目.isEmpty ? "导入视频或素材后在这里管理" : "换个关键词或筛选条件")
            }
        }
    }

    // MARK: 底部动作（批量超分 / 清空素材箱）

    /// 批量超分：空闲时把未超分素材按列表顺序排队，运行中显示「批量 n/总数」并支持再次点击中止。
    private var 批量超分按钮: some View {
        Button {
            批量.切换(素材: 素材, 超分: 超分)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: 批量.运行 ? "hourglass" : "square.stack.3d.up.fill")
                    .font(.system(size: 9, weight: .semibold))
                Text(批量.按钮标题)
                    .font(.system(size: 10, weight: .medium))
                    .lineLimit(1)
                    .monospacedDigit()
            }
            .foregroundStyle(批量.运行 ? Theme.accent : Theme.textSecondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Theme.fieldBackground)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(批量.运行 ? Theme.accent.opacity(0.55) : Theme.stroke, lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!批量按钮可用)
        .opacity(素材.条目.isEmpty ? 0.45 : 1)
        .help(批量.悬停提示)
    }

    /// 队列自己在跑时必须可点（它是中止入口），只有「别处正在超分」或空列表才禁用
    private var 批量按钮可用: Bool {
        guard !素材.条目.isEmpty else { return false }
        if 批量.运行 { return true }
        return !超分.运行中 && !超分.视频运行
    }

    /// 清空素材箱：仅移出素材列表，不删除磁盘上的源文件与已生成的超分产物。
    private var 清空按钮: some View {
        Button {
            清空素材箱()
        } label: {
            Image(systemName: "trash")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(垃圾桶悬停 ? Theme.danger : Theme.textSecondary)
                .frame(width: 22, height: 18)
                .background {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Theme.fieldBackground)
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(垃圾桶悬停 ? Theme.danger.opacity(0.5) : Theme.stroke, lineWidth: 1)
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(素材.条目.isEmpty)
        .opacity(素材.条目.isEmpty ? 0.45 : 1)
        .onHover { 垃圾桶悬停 = $0 }
        .help("清空素材箱：移出全部素材并中止正在进行的超分与批量队列（只清列表，不删磁盘上的源文件与已生成产物）")
    }

    private func 清空素材箱() {
        // 顺序：先中止批量队列 → 再中止单帧/整段超分并作废在途结果 → 最后清空列表
        guard !素材.条目.isEmpty else { return }
        批量.取消全部()
        超分.中止全部()
        素材.清空()
    }

    /// 素材栏副行：视频给时长+分辨率，动图在分辨率前标「动图 · N 帧」，静态图片只给分辨率。
    private func 副行文本(_ 条目: 素材条目) -> String {
        switch 条目.类型 {
        case .视频:
            return "\(条目.时长文本) · \(条目.分辨率文本)"
        case .图片:
            return 条目.是动图 ? "\(条目.动图文本) · \(条目.分辨率文本)" : 条目.分辨率文本
        }
    }

    private func 素材行(条目: 素材条目) -> some View {
        let 选中 = 素材.选中 == 条目.id
        return HStack(spacing: 8) {
            缩略图(条目: 条目)
            VStack(alignment: .leading, spacing: 2) {
                Text(条目.名称)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(副行文本(条目))
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.textTertiary)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(选中 ? Theme.fieldBackground : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(选中 ? Theme.accent.opacity(0.55) : Color.clear, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture { 素材.选择(条目.id) }
        .contextMenu {
            Button("在访达中显示") {
                NSWorkspace.shared.activateFileViewerSelecting([条目.url])
            }
            Button("从素材箱移除", role: .destructive) {
                素材.移除(条目.id)
            }
        }
    }

    @ViewBuilder
    private func 缩略图(条目: 素材条目) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .fill(Theme.viewportBackground)
            if let 图 = 条目.缩略图 {
                Image(nsImage: 图)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: 条目.类型.图标)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.textTertiary)
            }
        }
        .frame(width: 46, height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .strokeBorder(Theme.stroke, lineWidth: 0.5)
        )
        .overlay(alignment: .topTrailing) {
            if 条目.完成 {
                完成标记
            }
        }
    }

    /// 处理完成角标：缩略图右上角的对勾（批量处理时逐条点亮）。
    private var 完成标记: some View {
        Image(systemName: "checkmark")
            .font(.system(size: 7, weight: .heavy))
            .foregroundStyle(Theme.canvasBackground)
            .frame(width: 12, height: 12)
            .background(Circle().fill(Theme.accent))
            .overlay(Circle().strokeBorder(Theme.panelBackground, lineWidth: 1))
            .offset(x: -2, y: 2)
    }

    // MARK: 绑定

    private var 搜索绑定: Binding<String> {
        Binding(get: { 素材.搜索词 }, set: { 素材.搜索词 = $0 })
    }

    private var 筛选绑定: Binding<素材Store.素材筛选> {
        Binding(get: { 素材.筛选 }, set: { 素材.筛选 = $0 })
    }
}

// MARK: - 批量超分队列

/// 把素材箱里尚未超分的条目按列表顺序串行送进 超分通道：一条跑完自动跑下一条，
/// 每条成功后回写 素材.完成（素材栏缩略图右上角对勾）。运行中再次点击可中止剩余队列。
@MainActor
@Observable
final class 批量超分队列 {
    /// 剩余待跑条目（按列表顺序，队首是下一条）
    private(set) var 队列: [UUID] = []
    private(set) var 总数 = 0
    private(set) var 完成数 = 0
    private(set) var 失败数 = 0
    private(set) var 运行 = false

    @ObservationIgnored private var 任务: Task<Void, Never>?
    @ObservationIgnored private var 已请求中止 = false

    /// 当前正在跑第几条（1 起），未运行时为 0
    var 当前序号: Int {
        guard 运行 else { return 0 }
        return min(总数, 完成数 + 失败数 + 1)
    }

    var 按钮标题: String {
        运行 ? "批量 \(当前序号)/\(总数)" : "批量超分"
    }

    var 悬停提示: String {
        if 运行 {
            return "正在批量超分：第 \(当前序号)/\(总数) 条，再次点击可中止剩余队列（当前这条跑完即停）"
        }
        return "把未超分的素材按列表顺序逐条超分（已完成的自动跳过）"
    }

    /// 按钮入口：空闲时开始排队，运行中则中止剩余队列。
    func 切换(素材: 素材Store, 超分: 超分通道) {
        if 运行 { 中止(); return }
        开始(素材: 素材, 超分: 超分)
    }

    /// 中止剩余队列（当前这条不被打断，跑完即停）
    func 中止() {
        guard 运行 else { return }
        已请求中止 = true
        队列.removeAll()
    }

    /// 清空素材箱时调用：立即丢弃队列并解除运行态（单帧 / 整段的中止由 超分.清空() 负责）
    func 取消全部() {
        已请求中止 = true
        队列.removeAll()
        运行 = false
        任务?.cancel()
        任务 = nil
        总数 = 0
        完成数 = 0
        失败数 = 0
    }

    private func 开始(素材: 素材Store, 超分: 超分通道) {
        guard !超分.运行中, !超分.视频运行 else {
            素材.onNote?("正在超分中，等这一条跑完再批量")
            return
        }
        let 待跑 = 素材.条目.filter { !$0.完成 }
        guard !待跑.isEmpty else {
            素材.onNote?("素材都已超分完成，没有需要批量处理的条目")
            return
        }
        guard 超分.可运行 else {
            素材.onNote?("超分权重未就绪，先点「开始超分」完成下载再批量")
            return
        }

        队列 = 待跑.map(\.id)
        总数 = 待跑.count
        完成数 = 0
        失败数 = 0
        已请求中止 = false
        运行 = true
        素材.onNote?("开始批量超分：共 \(总数) 条待跑")

        任务 = Task { @MainActor in
            await 逐条执行(素材: 素材, 超分: 超分)
        }
    }

    private func 逐条执行(素材: 素材Store, 超分: 超分通道) async {
        while !队列.isEmpty, !已请求中止, !Task.isCancelled {
            let id = 队列.removeFirst()
            guard let 条目 = 素材.条目.first(where: { $0.id == id }), !条目.完成 else { continue }

            // 每条开始前清空一次，避免上一条的产物与提示串到这一条上
            超分.清空()
            超分.开始(素材: 条目, 预览帧: 条目.缩略图)

            let 结果 = await 等待落定(超分: 超分)
            switch 结果 {
            case .完成:
                素材.标记完成(id)
                完成数 += 1
            case .失败:
                失败数 += 1
            default:
                // 被换素材或清空素材箱打断：剩余队列直接作废
                已请求中止 = true
            }
        }

        let 实际完成 = 完成数
        let 实际失败 = 失败数
        运行 = false
        队列.removeAll()
        任务 = nil
        guard !Task.isCancelled else { return }
        var 文本 = "批量超分结束：完成 \(实际完成) 条"
        if 实际失败 > 0 { 文本 += "，失败 \(实际失败) 条" }
        if 已请求中止, 实际完成 + 实际失败 < 总数 { 文本 += "（已中止剩余队列）" }
        素材.onNote?(文本)
    }

    /// 等这次超分落到终态（完成 / 失败 / 空闲）并返回该状态
    private func 等待落定(超分: 超分通道) async -> 超分通道.进度 {
        while 超分.阶段 == .运行, !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        return 超分.阶段
    }
}
