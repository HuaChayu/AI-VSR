// ============================================================
// 文件作用：超分产物落盘索引（缓存）。
// 详情页切换素材时会调用 超分.清空() 把内存里的产物置空，导致切回已超分素材时对比层消失。
// 这里把「素材标识（源文件标准路径）→ 产物落盘路径 + 尺寸口径」持久化到 文档目录.缓存目录，
// 切回素材时按索引从磁盘 PNG / MP4 重建对比层；产物文件本身仍落在 文档目录.超分输出目录，不复制不删除。
// ============================================================
import Foundation

/// 一次超分产物的落盘登记：按素材源文件路径索引，记录产物路径与全部尺寸/耗时口径。
/// 视频条目额外记录整段成片信息（帧数、耗时、音轨、体积），用于切回时重建 视频产物。
struct 超分缓存条目: Codable {
    /// 素材标识：源文件标准路径（跨 App 重启稳定，不依赖每次导入都变的 UUID）。
    var 源路径: String
    /// 素材类型标题：图片 / 视频。
    var 类型: String
    /// 产物落盘绝对路径（图片为 PNG，视频为 MP4）。
    var 产物路径: String
    var 输入宽: Int
    var 输入高: Int
    var 低清宽: Int
    var 低清高: Int
    var 输出宽: Int
    var 输出高: Int
    /// 单帧：Flash 段毫秒；整段：平均每帧毫秒。
    var 耗时毫秒: Double
    /// 视频专属：成片帧数。
    var 帧数: Int = 0
    /// 视频专属：整段耗时秒。
    var 耗时秒: Double = 0
    /// 视频专属：是否保留音轨。
    var 含音轨: Bool = false
    /// 视频专属：成片体积字节。
    var 体积字节: Int64 = 0
    /// 登记时间，仅供排查用。
    var 记录时间: Date = Date()
}

/// 超分产物落盘索引：JSON 持久化在 文档目录.缓存目录 下，App 重启后仍可恢复对比层。
/// 线程安全（内部 NSLock），既可从 Worker/后台回写，也可从 Store 与视图读取。
final class 超分缓存索引: @unchecked Sendable {
    static let 共享 = 超分缓存索引()
    nonisolated static let 文件名 = "超分缓存索引.json"

    private let 锁 = NSLock()
    private var 表: [String: 超分缓存条目] = [:]
    private var 已载入 = false

    nonisolated static var 索引文件: URL {
        文档目录.缓存目录.appendingPathComponent(文件名)
    }

    /// 素材标识：源文件标准路径（同一文件不同写法归一到同一个键）。
    nonisolated static func 键(_ url: URL) -> String {
        url.standardizedFileURL.path
    }

    /// 命中判定：登记存在且磁盘产物还在（用户手动删掉产物就当作没缓存，后续超分会重新登记）。
    func 条目(源 url: URL) -> 超分缓存条目? {
        let 键 = Self.键(url)
        锁.lock()
        载入锁内()
        let 命中 = 表[键]
        锁.unlock()
        guard let 命中 else { return nil }
        guard FileManager.default.fileExists(atPath: 命中.产物路径) else { return nil }
        return 命中
    }

    /// 登记（覆盖同源旧记录）并立即落盘。
    func 记录(_ 条目: 超分缓存条目) {
        锁.lock()
        载入锁内()
        表[条目.源路径] = 条目
        落盘锁内()
        锁.unlock()
    }

    /// 素材被移除时清掉对应登记；磁盘上的产物文件按既有语义保留，不删除。
    func 移除(源 url: URL) {
        let 键 = Self.键(url)
        锁.lock()
        载入锁内()
        guard 表.removeValue(forKey: 键) != nil else {
            锁.unlock()
            return
        }
        落盘锁内()
        锁.unlock()
    }

    /// 清空素材箱时清掉全部登记（产物文件不删）。
    func 清空() {
        锁.lock()
        载入锁内()
        表.removeAll()
        落盘锁内()
        锁.unlock()
    }

    /// 当前登记快照，供排查与验收使用。
    func 全部() -> [超分缓存条目] {
        锁.lock()
        载入锁内()
        let 快照 = 表.values.sorted { $0.记录时间 > $1.记录时间 }
        锁.unlock()
        return 快照
    }

    // MARK: 内部（调用方已持锁）

    private func 载入锁内() {
        guard !已载入 else { return }
        已载入 = true
        let 地址 = Self.索引文件
        guard let 数据 = try? Data(contentsOf: 地址) else { return }
        // 写入侧用 ISO8601，读取侧必须同口径，否则任何一条含日期字段解不出来整表都会丢
        let 解码器 = JSONDecoder()
        解码器.dateDecodingStrategy = .iso8601
        if let 解码 = try? 解码器.decode([超分缓存条目].self, from: 数据) {
            表 = Dictionary(uniqueKeysWithValues: 解码.map { ($0.源路径, $0) })
            return
        }
        // 兼容早期按默认策略写出的文件
        if let 解码 = try? JSONDecoder().decode([超分缓存条目].self, from: 数据) {
            表 = Dictionary(uniqueKeysWithValues: 解码.map { ($0.源路径, $0) })
            return
        }
        print("[超分缓存] 索引解析失败，按空索引继续：" + 地址.path)
    }

    private func 落盘锁内() {
        let 地址 = Self.索引文件
        do {
            try FileManager.default.createDirectory(at: 文档目录.缓存目录, withIntermediateDirectories: true)
            let 编码 = JSONEncoder()
            编码.outputFormatting = [.prettyPrinted, .sortedKeys]
            编码.dateEncodingStrategy = .iso8601
            let 清单 = 表.values.sorted { $0.记录时间 > $1.记录时间 }
            let 数据 = try 编码.encode(清单)
            try 数据.write(to: 地址, options: .atomic)
        } catch {
            print("[超分缓存] 索引写入失败：" + 地址.path + " → " + error.localizedDescription)
        }
    }
}
