import AppKit
import Observation

/// 超分页 Store：本质是个素材箱——导入、元数据、筛选、选中与预览帧。
/// 超分推理由 超分通道 调 Flash 引擎完成，此处只负责「输入侧」。
@Observable
final class 素材Store {
    enum 素材筛选: String, CaseIterable, Identifiable {
        case 全部
        case 视频
        case 图片

        var id: String { rawValue }
        var 标题: String { rawValue }
    }

    var 条目: [素材条目] = []
    var 选中: UUID?
    var 导入中 = false
    var 预览中 = false
    var 预览帧: NSImage?
    var 筛选: 素材筛选 = .全部
    var 搜索词 = ""
    var 错误信息: String?
    var onNote: ((String) -> Void)?

    // MARK: 派生

    var 视频们: [素材条目] { 条目.filter { $0.类型 == .视频 } }
    var 图片们: [素材条目] { 条目.filter { $0.类型 == .图片 } }

    var 选中条目: 素材条目? { 条目.first { $0.id == 选中 } }

    var 显示条目: [素材条目] {
        var 结果 = 条目
        switch 筛选 {
        case .全部: break
        case .视频: 结果 = 结果.filter { $0.类型 == .视频 }
        case .图片: 结果 = 结果.filter { $0.类型 == .图片 }
        }
        let 关键词 = 搜索词.trimmingCharacters(in: .whitespacesAndNewlines)
        if !关键词.isEmpty {
            结果 = 结果.filter { $0.名称.localizedStandardContains(关键词) }
        }
        return 结果
    }

    var 统计文本: String {
        "\(条目.count) 个素材 · 视频 \(视频们.count) · 图片 \(图片们.count)"
    }

    // MARK: 动作

    func 导入(urls: [URL]) async {
        guard !urls.isEmpty else { return }
        导入中 = true
        错误信息 = nil
        defer { 导入中 = false }

        var 新增 = 0
        var 重复 = 0
        var 不支持 = 0

        for url in urls {
            let 标准路径 = url.standardizedFileURL
            if 条目.contains(where: { $0.url.standardizedFileURL == 标准路径 }) {
                重复 += 1
                continue
            }
            guard let 新条目 = await 素材读取.读取(url: 标准路径) else {
                不支持 += 1
                continue
            }
            条目.append(新条目)
            if 选中 == nil { 选中 = 新条目.id }
            新增 += 1
        }

        if 不支持 > 0 {
            错误信息 = "跳过 \(不支持) 个不支持的文件（仅支持视频 / 图片）"
        }
        if 新增 > 0 {
            onNote?("已导入 \(新增) 个素材")
            await 载入预览(时间: 0.5)
        } else if 重复 > 0 {
            onNote?("\(重复) 个素材已在素材箱中，未重复导入")
        }
    }

    func 选择(_ id: UUID?) {
        选中 = id
        Task { await 载入预览(时间: 0.5) }
    }

    /// 标记某条素材的处理完成状态（素材栏右上角对勾）；批量处理时可按 id 逐条置位。
    func 标记完成(_ id: UUID, _ 完成: Bool = true) {
        guard let 下标 = 条目.firstIndex(where: { $0.id == id }) else { return }
        guard 条目[下标].完成 != 完成 else { return }
        条目[下标].完成 = 完成
    }

    /// 抽一帧给预览区：图片直接复用缩略图，视频按时间抽帧（1920 上限）。
    func 载入预览(时间: Double = 0.5) async {
        guard let 条目 = 选中条目 else {
            预览帧 = nil
            return
        }
        if 条目.类型 == .图片 {
            预览帧 = 条目.缩略图
            return
        }
        预览中 = true
        defer { 预览中 = false }
        if let 图 = try? await 视频解码.抽帧(url: 条目.url, 时间: 时间, 最大边长: 1920) {
            预览帧 = NSImage.以(cg图: 图)
        } else {
            预览帧 = 条目.缩略图
        }
    }

    func 移除(_ id: UUID) {
        guard let 下标 = 条目.firstIndex(where: { $0.id == id }) else { return }
        let 名称 = 条目[下标].名称
        // 同步清掉该素材的产物落盘登记（磁盘上的产物文件按既有语义保留，不删除）
        超分缓存索引.共享.移除(源: 条目[下标].url)
        条目.remove(at: 下标)
        if 选中 == id {
            选中 = 条目.first?.id
            预览帧 = nil
            Task { await 载入预览(时间: 0.5) }
        }
        onNote?("已从素材箱移除《\(名称)》（原文件仍在磁盘上）")
    }

    func 清空() {
        // 清空素材箱时整表清掉产物落盘登记（磁盘上的产物文件按既有语义保留，不删除）
        超分缓存索引.共享.清空()
        条目.removeAll()
        选中 = nil
        预览帧 = nil
        onNote?("素材箱已清空（原文件仍在磁盘上）")
    }
}
