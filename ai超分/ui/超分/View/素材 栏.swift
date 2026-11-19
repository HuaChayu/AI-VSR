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

    /// 本批还有活：只影响图标与标题写法，不影响点击语义（点击永远是「把没进队列的排进去」）。
    private var 批量在跑: Bool { 批量.在途(超分) > 0 }

    /// 批量超分：把素材库里还没进队列的素材按列表顺序逐条排进同一条串行队列。
    /// 再次点击不会撤出任何条目，只是把新出现的待办补进队列——手动取消掉的那条也会在这一步被重新排进去。
    private var 批量超分按钮: some View {
        Button {
            批量.入队(素材: 素材, 超分: 超分)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: 批量在跑 ? "hourglass" : "square.stack.3d.up.fill")
                    .font(.system(size: 9, weight: .semibold))
                Text(批量.按钮标题(超分))
                    .font(.system(size: 10, weight: .medium))
                    .lineLimit(1)
                    .monospacedDigit()
            }
            .foregroundStyle(批量在跑 ? Theme.accent : Theme.textSecondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Theme.fieldBackground)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(批量在跑 ? Theme.accent.opacity(0.55) : Theme.stroke, lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!批量按钮可用)
        .opacity(批量按钮可用 ? 1 : 0.45)
        .help(批量.悬停提示)
    }

    /// 有「未完成 且 不在跑 且 不在排队」的素材才点亮：点了就有东西入队，没有就置灰。
    /// 判定只看素材库现状，不掺批次、已落定之类的记账，所以取消掉一条后会立刻重新点亮。
    private var 批量按钮可用: Bool {
        素材.条目.contains { 条目 in
            !条目.完成 && 条目.id != 超分.运行素材id && !超分.在排队(条目.id)
        }
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

/// 批量按钮的唯一语义：把素材库里「未完成 且 不在跑 且 不在排队」的素材按列表顺序逐条排进
/// 超分通道的同一条按 id 队列，与单条点击共用一套排队/取消语义：被排上的素材在详情页就是「已排队」，
/// 可单独撤出；跑完由 刚完成 → 素材.完成 点亮对勾。
/// 再次点击不会撤出任何东西——它只是把又一次出现的待办补进队列，
/// 所以手动取消掉的某条只要还没完成，下一次点击就会被重新排进去。
@MainActor
@Observable
final class 批量超分队列 {
    /// 本批纳入的素材 id（按列表顺序）。
    /// 批次只回答「这一批是谁」，真正的排队、串行执行、按 id 取消全部交给 超分.排队 那一套队列，
    /// 不再有第二条独立执行路线——所以被批量带上的素材在详情页就是「已排队」，可单独取消。
    private(set) var 批次: [UUID] = []

    var 总数: Int { 批次.count }

    /// 这批还有几条在队里等着或正在跑（0 = 这批没有活的了）。只用于显示进度，不参与按钮可用性判断。
    func 在途(_ 超分: 超分通道) -> Int {
        批次.filter { $0 == 超分.运行素材id || 超分.在排队($0) }.count
    }

    /// 这批已经落定的条数（跑完、失败、被单独撤出队列都算落定）
    func 已落定(_ 超分: 超分通道) -> Int {
        max(0, 批次.count - 在途(超分))
    }

    /// 标题只有两种形态：「批量超分」或「批量 n/总数」，没有第二种语义。
    func 按钮标题(_ 超分: 超分通道) -> String {
        guard 总数 > 0, 在途(超分) > 0 else { return "批量超分" }
        return "批量 \(已落定(超分))/\(总数)"
    }

    var 悬停提示: String {
        "把素材库里还没超分、也不在队列里的素材按列表顺序逐条排进同一条串行队列（已完成的、已在队列里的自动跳过）；再次点击只是把新的待办补进队列，不会撤出任何条目"
    }

    /// 点击入口：永远只做「把还没有进队列的素材排进队列」，没有第二种行为。
    func 入队(素材: 素材Store, 超分: 超分通道) {
        guard 超分.可运行 else {
            素材.onNote?("超分权重未就绪，先点「开始超分」完成下载再批量")
            return
        }
        // 只挑真正的待办：没完成、不是正在跑的那条、也不在队列里等着。
        // 手动取消排队或被取消掉的条目在这里会被重新捡起来，因为取消只动队列、不动 完成 标记。
        let 待跑 = 素材.条目.filter { 条目 in
            !条目.完成 && 条目.id != 超分.运行素材id && !超分.在排队(条目.id)
        }
        guard !待跑.isEmpty else {
            素材.onNote?("没有需要批量处理的条目（都已完成或已在队列里）")
            return
        }
        // 批次只记账，用于显示「批量 n/总数」：上一批还在队里的继续算，再把这一批新排的接上。
        批次 = 批次.filter { $0 == 超分.运行素材id || 超分.在排队($0) }
        for 条目 in 待跑 where !批次.contains(条目.id) {
            批次.append(条目.id)
        }
        // 逐条走统一入口，自带「在跑/在队则跳过，忙则入队，闲则直跑」，仍然严格串行独占引擎。
        for 条目 in 待跑 {
            超分.开始(素材: 条目, 预览帧: 条目.缩略图)
        }
        素材.onNote?("已把 \(待跑.count) 条排进队列，按顺序逐条超分")
    }

    /// 清空素材箱时调用：丢掉批次账（真正的中止由 超分.中止全部() 负责）
    func 取消全部() {
        批次.removeAll()
    }
}
