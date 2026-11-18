//
//  模型下载.swift
//  ai超分
//
//  Swift 原生 ModelScope 权重下载器（零第三方依赖，仅 Foundation + CryptoKit）
//  移植来源：无限画布/模型下载.swift，按本项目权重规格裁剪。
//
//  仓库：https://modelscope.cn/models/HuaCHayu/RelayVSR/files
//  端点（已实测可用）：
//    1) 文件清单: GET https://modelscope.cn/api/v1/models/{owner}/{repo}/repo/files?Revision=master&Recursive=true
//    2) 下载:     GET https://modelscope.cn/models/{owner}/{repo}/resolve/{revision}/{filepath}
//
//  落地位置：文档目录.模型目录（~/Documents/AI超分/model），与加载侧 文档目录.模型路径() 同一处。
//  必需文件仅两个：wan.safetensors(10066399024 B)、flash.safetensors(127759784 B)；
//  ref/flash_selftest.safetensors 与超分主流程无关，不纳入下载清单。
//
//  可靠性：
//    1) 已存在且 SHA256 与远端一致 -> 直接跳过，绝不重复下载；
//    2) 断点续传：下载中断时把 resumeData 落到 cache/续传/，下次触发自动从断点继续；
//    3) 失败重试：单文件最多 4 次尝试，续传数据失效则退回全新下载；
//    4) 校验失败：删除半成品并重下，保证落盘文件可用。
//

import Foundation
import CryptoKit
import Observation

// MARK: - 远端文件条目

struct ModelScope文件: Decodable {
    let path: String
    let size: Int
    let sha256: String
    let type: String      // "tree" | "blob"

    // ModelScope 清单返回的键是大写（Path/Size/Sha256/Type），必须显式映射，
    // 否则解码失败会抛 NSCocoaErrorDomain "The data couldn't be read because it is missing."
    enum CodingKeys: String, CodingKey {
        case path = "Path"
        case size = "Size"
        case sha256 = "Sha256"
        case type = "Type"
    }

    var 名: String { (path as NSString).lastPathComponent }
    var 是目录: Bool { type == "tree" }
}

/// 清单外壳：{ "Data": { "Files": [...] } }
private struct ModelScope清单: Decodable {
    struct 载荷: Decodable { let Files: [ModelScope文件] }
    let Data: 载荷
}

// MARK: - 下载错误

struct 权重下载错误: LocalizedError {
    let 说明: String
    var errorDescription: String? { 说明 }
}

// MARK: - RelayVSR 权重规格

/// 本项目超分需要的权重：仓库坐标 + 必需文件名 + 就绪判断。
/// 所有路径都指向 文档目录.模型目录，禁止出现硬编码绝对路径。
enum RelayVSR权重 {
    nonisolated static let owner = "HuaCHayu"
    nonisolated static let repo = "RelayVSR"
    nonisolated static let 版本 = "master"

    nonisolated static let wan文件 = "wan.safetensors"
    nonisolated static let flash文件 = "flash.safetensors"

    /// 全部必需文件（Wan 条件 latent + Flash 4× 解码）
    nonisolated static var 全部: [String] { [wan文件, flash文件] }

    /// 按是否启用 Wan 分支算出真正需要的文件
    nonisolated static func 需要(启用Wan: Bool) -> [String] {
        启用Wan ? [wan文件, flash文件] : [flash文件]
    }

    nonisolated static func 文件(_ 名: String) -> URL {
        文档目录.模型文件(名)
    }

    nonisolated static func 路径(_ 名: String) -> String {
        文档目录.模型路径(名)
    }

    /// 就绪判定：文件存在且非空（半截文件交给 SHA256 校验环节处理）
    nonisolated static func 就绪(_ 名: String) -> Bool {
        let 文件管理 = FileManager.default
        let url = 文件(名)
        guard 文件管理.fileExists(atPath: url.path) else { return false }
        let 属性 = try? 文件管理.attributesOfItem(atPath: url.path)
        let 大小 = (属性?[.size] as? NSNumber)?.int64Value ?? 0
        return 大小 > 0
    }

    /// 返回给定文件里缺失的那些，顺序保持入参顺序
    nonisolated static func 缺失(_ 名列表: [String]) -> [String] {
        名列表.filter { !就绪($0) }
    }
}

// MARK: - 活动下载文件（UI 列表项：文件名 + 该文件独立实时速度）

struct 活动下载文件 {
    let 名: String
    let 已写: Int64
    let 总: Int64
    let 速度: Double
}

// MARK: - 下载器

/// 单个 ModelScope 仓库的下载器：只下载调用方点名的文件，落到目标目录。
nonisolated final class ModelScope下载器: NSObject, URLSessionDownloadDelegate {
    let owner: String
    let repo: String
    var revision: String

    /// 整包进度回调 (累计已下载字节, 总字节, 当前文件名)
    var onProgress: ((Int64, Int64, String) -> Void)?
    /// 单文件进度回调 (已下载, 总字节, 文件名, 实时速度 B/s)
    var onFileProgress: ((Int64, Int64, String, Double) -> Void)?
    /// 活动下载文件列表回调
    var onActiveFilesChanged: (([活动下载文件]) -> Void)?

    /// 最大并发下载文件数（本项目仅 2 个权重，2 路并行足够吃满带宽）
    private let 最大并发文件数 = 2
    /// 单文件最多尝试次数（失败后续传重试）
    private let 最多尝试次数 = 4

    private var session: URLSession!

    // MARK: 任务状态（锁保护）

    private final class 文件任务 {
        let file: ModelScope文件
        let dest: URL
        var 已写: Int64 = 0
        var 速度样本: [速度采样] = []
        var 迁移错误: Error?
        var 续延: CheckedContinuation<URL, Error>?
        init(file: ModelScope文件, dest: URL) {
            self.file = file
            self.dest = dest
        }
    }

    private struct 速度采样 {
        let 时间: Date
        let 字节: Int64
    }

    private let 锁 = NSLock()
    private var 活动任务: [URLSessionDownloadTask: 文件任务] = [:]
    private var 已完成字节: Int64 = 0
    private var 总字节: Int64 = 0
    private var 已取消 = false

    /// 进度节流：同一文件内至少前进 512 KB 或换文件才回调，避免刷爆主线程
    private var 上次回报字节: Int64 = -1

    // MARK: 全局实时速度（滑动窗口）

    private var 速度样本: [速度采样] = []
    private let 速度窗口: TimeInterval = 5.0

    var currentSpeedBytesPerSec: Double {
        guard 速度样本.count >= 2, let 首 = 速度样本.first, let 末 = 速度样本.last else { return 0 }
        let dt = 末.时间.timeIntervalSince(首.时间)
        guard dt > 0.1 else { return 0 }
        return Double(末.字节 - 首.字节) / dt
    }

    private func 记录速度(_ 全局字节: Int64) {
        let now = Date()
        速度样本.append(速度采样(时间: now, 字节: 全局字节))
        while let 首 = 速度样本.first, now.timeIntervalSince(首.时间) > 速度窗口 {
            速度样本.removeFirst()
        }
    }

    private static func 求速度(_ 样本: [速度采样]) -> Double {
        guard 样本.count >= 2, let 首 = 样本.first, let 末 = 样本.last else { return 0 }
        let dt = 末.时间.timeIntervalSince(首.时间)
        guard dt > 0.1 else { return 0 }
        return Double(末.字节 - 首.字节) / dt
    }

    /// 当前活动文件快照（调用方需已持有 锁）
    private func 活动快照() -> [活动下载文件] {
        活动任务.values.map { ctx in
            活动下载文件(名: ctx.file.名,
                        已写: ctx.已写,
                        总: Int64(ctx.file.size),
                        速度: Self.求速度(ctx.速度样本))
        }
    }

    init(owner: String, repo: String, revision: String = "master") {
        self.owner = owner
        self.repo = repo
        self.revision = revision
        super.init()
        // 前台会话：权重下载是用户点「开始超分」即时触发的任务。
        // 不用后台会话，避免同一 identifier 复用导致过期签名 URL 被系统路由回来（NSCocoaErrorDomain 260）。
        let 配置 = URLSessionConfiguration.default
        配置.timeoutIntervalForRequest = 60
        配置.timeoutIntervalForResource = 7 * 24 * 3600   // 10 GB 级文件给足时间
        session = URLSession(configuration: 配置, delegate: self, delegateQueue: nil)
    }

    deinit {
        session?.invalidateAndCancel()
    }

    // MARK: 1. 远端文件清单

    func 清单() async throws -> [ModelScope文件] {
        var comps = URLComponents(string: "https://modelscope.cn/api/v1/models/\(owner)/\(repo)/repo/files")!
        comps.queryItems = [URLQueryItem(name: "Revision", value: revision),
                            URLQueryItem(name: "Recursive", value: "true")]
        var 请求 = URLRequest(url: comps.url!)
        请求.timeoutInterval = 40
        let (数据, 响应) = try await URLSession.shared.data(for: 请求)
        let 码 = (响应 as? HTTPURLResponse)?.statusCode ?? -1
        guard 码 == 200 else {
            throw 权重下载错误(说明: "拉取模型清单失败（HTTP \(码)）")
        }
        let 清单 = try JSONDecoder().decode(ModelScope清单.self, from: 数据)
        return 清单.Data.Files.filter { !$0.是目录 && !$0.path.hasSuffix(".gitattributes") }
    }

    // MARK: 2. 下载指定文件

    /// 只下载 名称列表 里的文件到 root；已存在且校验通过的自动跳过。
    func 下载(文件 名称列表: [String], 到 root: URL) async throws {
        已取消 = false
        上次回报字节 = -1
        let 远端 = try await 清单()
        var 目标: [ModelScope文件] = []
        for 名 in 名称列表 {
            if let 命中 = 远端.first(where: { $0.path == 名 }) ?? 远端.first(where: { $0.名 == 名 }) {
                目标.append(命中)
            } else {
                throw 权重下载错误(说明: "仓库 \(owner)/\(repo) 里找不到文件：\(名)")
            }
        }
        let 合计 = 目标.reduce(Int64(0)) { $0 + Int64($1.size) }
        锁.lock(); 总字节 = 合计; 已完成字节 = 0; 锁.unlock()

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        var 累计: Int64 = 0
        try await withThrowingTaskGroup(of: (Int64, String).self) { 组 in
            var 迭代 = 目标.makeIterator()
            for _ in 0..<max(1, 最大并发文件数) {
                if let f = 迭代.next() {
                    组.addTask { [weak self] in
                        guard let self else { return (0, f.名) }
                        return (try await self.单文件(f, root: root), f.名)
                    }
                }
            }
            while let (尺寸, 名) = try await 组.next() {
                累计 += 尺寸
                锁.lock(); 已完成字节 = 累计; 锁.unlock()
                onProgress?(累计, 合计, 名)
                if let f = 迭代.next() {
                    组.addTask { [weak self] in
                        guard let self else { return (0, f.名) }
                        return (try await self.单文件(f, root: root), f.名)
                    }
                }
            }
        }
    }

    /// 主动取消：终止所有在途任务
    func 取消() {
        已取消 = true
        锁.lock(); let 任务们 = Array(活动任务.keys); 锁.unlock()
        for t in 任务们 { t.cancel() }
    }

    private func 单文件(_ f: ModelScope文件, root: URL) async throws -> Int64 {
        let dest = root.appendingPathComponent(f.path)
        if FileManager.default.fileExists(atPath: dest.path),
           let 和 = try? await Self.sha256(of: dest),
           和.lowercased() == f.sha256.lowercased() {
            // 已有且完好：直接计入进度，不重复下载
            return Int64(f.size)
        }
        try await 下载一个(f, 到: dest)
        return Int64(f.size)
    }

    private func 下载一个(_ f: ModelScope文件, 到 dest: URL) async throws {
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let url = URL(string: "https://modelscope.cn/models/\(owner)/\(repo)/resolve/\(revision)/\(f.path)") else {
            throw 权重下载错误(说明: "下载地址构造失败：\(f.path)")
        }

        var 续传数据 = Self.读续传数据(名: f.名)
        var 最后错误: Error = 权重下载错误(说明: "下载失败：\(f.名)")

        for 尝试 in 1...最多尝试次数 {
            if 已取消 { throw CancellationError() }
            do {
                try await 执行下载(f, url: url, 到: dest, 续传数据: 续传数据)
                if let 和 = try? await Self.sha256(of: dest), 和.lowercased() == f.sha256.lowercased() {
                    Self.清续传数据(名: f.名)
                    return
                }
                // 校验不一致：删掉半成品，下次从头来
                try? FileManager.default.removeItem(at: dest)
                Self.清续传数据(名: f.名)
                续传数据 = nil
                最后错误 = 权重下载错误(说明: "SHA256 校验不一致：\(f.名)")
            } catch {
                最后错误 = error
                if (error as? CancellationError) != nil || 已取消 { break }
                // 失败时 delegate 已把 resumeData 落盘，下一轮优先续传
                续传数据 = Self.读续传数据(名: f.名)
                print("[模型下载] \(f.名) 第 \(尝试) 次尝试失败：\(error.localizedDescription)")
            }
        }
        throw 最后错误
    }

    /// 发起单个 URLSession 下载任务并等待结束（续传数据可为 nil）
    private func 执行下载(_ f: ModelScope文件, url: URL, 到 dest: URL, 续传数据: Data?) async throws {
        let 任务: URLSessionDownloadTask
        if let 数据 = 续传数据 {
            任务 = session.downloadTask(withResumeData: 数据)
        } else {
            任务 = session.downloadTask(with: url)
        }
        let ctx = 文件任务(file: f, dest: dest)
        // withResumeData 生成的请求没有原始 URL，进度回调仍可用；若失效会立刻报错并退回全新下载
        锁.lock(); 活动任务[任务] = ctx; let 列表 = 活动快照(); 锁.unlock()
        onActiveFilesChanged?(列表)

        do {
            _ = try await withCheckedThrowingContinuation { (c: CheckedContinuation<URL, Error>) in
                ctx.续延 = c
                任务.resume()
            }
        } catch {
            // 续传数据可能已随签名 URL 过期，清掉以便下一轮全新下载
            if 续传数据 != nil { Self.清续传数据(名: f.名) }
            throw error
        }
    }

    // MARK: 3. 进度与完成回调

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let now = Date()
        锁.lock()
        guard let ctx = 活动任务[downloadTask] else { 锁.unlock(); return }
        ctx.已写 = totalBytesWritten
        ctx.速度样本.append(速度采样(时间: now, 字节: totalBytesWritten))
        while let 首 = ctx.速度样本.first, now.timeIntervalSince(首.时间) > 速度窗口 {
            ctx.速度样本.removeFirst()
        }
        let 全局 = 已完成字节 + 活动任务.values.reduce(Int64(0)) { $0 + $1.已写 }
        let 总 = 总字节
        let 速度 = Self.求速度(ctx.速度样本)
        let 名 = ctx.file.名
        let 列表 = 活动快照()
        锁.unlock()

        记录速度(全局)

        // 节流：换文件或字节前进 ≥512KB 才回调
        let 应该回报 = 名 != 上次回报文件名 || 全局 - 上次回报字节 >= 512 * 1024
        if 应该回报 {
            上次回报字节 = 全局
            上次回报文件名 = 名
            onProgress?(全局, 总, 名)
            onFileProgress?(totalBytesWritten, totalBytesExpectedToWrite, 名, 速度)
            onActiveFilesChanged?(列表)
        }
    }

    private var 上次回报文件名: String = ""

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        锁.lock()
        guard let ctx = 活动任务[downloadTask] else { 锁.unlock(); return }
        let dest = ctx.dest
        锁.unlock()

        // 临时文件必须在回调返回前搬走，否则系统会删掉它
        let 文件管理 = FileManager.default
        try? 文件管理.removeItem(at: dest)
        do {
            try 文件管理.moveItem(at: location, to: dest)
        } catch {
            ctx.迁移错误 = error
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let 下载任务 = task as? URLSessionDownloadTask else { return }
        锁.lock()
        guard let ctx = 活动任务[下载任务] else { 锁.unlock(); return }
        活动任务[下载任务] = nil
        let 列表 = 活动快照()
        锁.unlock()
        onActiveFilesChanged?(列表)

        if let error {
            // 保存续传数据，下次从断点继续
            if let 数据 = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                Self.存续传数据(数据, 名: ctx.file.名)
            }
            ctx.续延?.resume(throwing: error)
        } else if let 迁移错误 = ctx.迁移错误 {
            ctx.续延?.resume(throwing: 迁移错误)
        } else {
            ctx.续延?.resume(returning: ctx.dest)
        }
    }

    // MARK: 4. 工具

    /// 流式 SHA256（大文件不整读进内存）
    static func sha256(of url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            let 句柄 = try FileHandle(forReadingFrom: url)
            defer { try? 句柄.close() }
            var 哈希 = SHA256()
            while let 块 = try 句柄.read(upToCount: 1 << 20), !块.isEmpty {
                哈希.update(data: 块)
            }
            return 哈希.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    // 续传数据落在 cache/续传/ 下，随「文档目录」走，不写系统临时目录
    nonisolated static func 续传文件(_ 名: String) -> URL {
        文档目录.缓存目录
            .appendingPathComponent("续传", isDirectory: true)
            .appendingPathComponent(名 + ".resume")
    }

    static func 存续传数据(_ 数据: Data, 名: String) {
        let url = 续传文件(名)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? 数据.write(to: url, options: .atomic)
    }

    static func 读续传数据(名: String) -> Data? {
        try? Data(contentsOf: 续传文件(名))
    }

    static func 清续传数据(名: String) {
        try? FileManager.default.removeItem(at: 续传文件(名))
    }
}

// MARK: - 下载管理（UI 状态的唯一来源）

/// 超分入口点「开始超分」时调用：只下缺失的权重，下载完自动续跑。
@MainActor
@Observable
final class 权重下载管理 {
    static let 共享 = 权重下载管理()

    /// 是否正在下载（驱动进度浮层与按钮禁用）
    private(set) var 下载中 = false
    /// 0~1 整包进度
    private(set) var 进度: Double = 0
    /// 当前文件名
    private(set) var 当前文件 = ""
    /// 已下载 / 总量 文本
    private(set) var 体积文本 = ""
    /// 实时速度文本
    private(set) var 速度文本 = ""
    /// 失败原因（非 nil 时 UI 展示并可重试）
    private(set) var 错误提示: String?

    @ObservationIgnored private var 下载器: ModelScope下载器?
    @ObservationIgnored private var 待下载: [String] = []
    @ObservationIgnored private var 完成后动作: (() -> Void)?

    private init() {}

    /// 当前需要下载但还缺失的文件
    var 缺失文件: [String] { RelayVSR权重.缺失(RelayVSR权重.全部) }

    /// 全部权重就绪
    var 就绪: Bool { 缺失文件.isEmpty }

    /// UI 一句话状态
    var 状态文本: String {
        if let 错误提示 { return "权重下载失败：" + 错误提示 }
        if 下载中 {
            var 文本 = 进度 > 0 ? "正在下载 \(当前文件) \(Int((进度 * 100).rounded()))%" : "正在准备下载 \(当前文件)"
            if !体积文本.isEmpty { 文本 += " · " + 体积文本 }
            if !速度文本.isEmpty { 文本 += " · " + 速度文本 }
            return 文本
        }
        if !就绪 { return "缺少权重：" + 缺失文件.joined(separator: "、") + "（点「开始超分」自动下载）" }
        return ""
    }

    /// 就绪检查 + 只下缺失项；全部就绪直接执行完成动作。
    func 确保权重(_ 名列表: [String], 完成后 动作: @escaping () -> Void) {
        let 缺失 = RelayVSR权重.缺失(名列表)
        guard !缺失.isEmpty else { 动作(); return }
        guard !下载中 else { return }   // 已在下载，忽略重复触发（按钮此时已禁用）

        错误提示 = nil
        待下载 = 缺失
        完成后动作 = 动作
        进度 = 0
        当前文件 = 缺失.first ?? ""
        体积文本 = ""
        速度文本 = ""
        下载中 = true

        let 下载器 = ModelScope下载器(owner: RelayVSR权重.owner,
                                    repo: RelayVSR权重.repo,
                                    revision: RelayVSR权重.版本)
        self.下载器 = 下载器

        下载器.onProgress = { [weak self, weak 下载器] 已, 总, 名 in
            let 速度 = 下载器?.currentSpeedBytesPerSec ?? 0
            Task { @MainActor in
                guard let self else { return }
                self.进度 = 总 > 0 ? min(1, Double(已) / Double(总)) : 0
                self.当前文件 = 名
                self.体积文本 = Self.体积(已) + " / " + Self.体积(总)
                self.速度文本 = Self.速度(速度)
            }
        }

        let 根 = 文档目录.模型目录
        Task { @MainActor [weak self] in
            do {
                try await 下载器.下载(文件: 缺失, 到: 根)
                guard let self else { return }
                self.下载中 = false
                self.进度 = 1
                self.当前文件 = ""
                self.体积文本 = ""
                self.速度文本 = ""
                self.下载器 = nil
                let 动作 = self.完成后动作
                self.完成后动作 = nil
                self.待下载 = []
                动作?()
            } catch {
                guard let self else { return }
                self.下载中 = false
                self.当前文件 = ""
                self.体积文本 = ""
                self.速度文本 = ""
                self.下载器 = nil
                if (error as? CancellationError) != nil {
                    self.错误提示 = nil
                    self.完成后动作 = nil
                } else {
                    self.错误提示 = error.localizedDescription
                }
            }
        }
    }

    /// 失败后重试同一批缺失文件（下载完成后仍会续跑原来的超分）
    func 重试() {
        guard !下载中 else { return }
        let 名 = 待下载.isEmpty ? RelayVSR权重.缺失(RelayVSR权重.全部) : 待下载
        guard !名.isEmpty else { return }
        let 动作 = 完成后动作
        错误提示 = nil
        确保权重(名) { 动作?() }
    }

    /// 取消当前下载（不进入超分流程）
    func 取消() {
        下载器?.取消()
        下载器 = nil
        下载中 = false
        当前文件 = ""
        体积文本 = ""
        速度文本 = ""
        进度 = 0
        错误提示 = nil
        完成后动作 = nil
        待下载 = []
    }

    /// 清理失败提示
    func 清空错误() {
        错误提示 = nil
    }

    // MARK: 文案

    static func 体积(_ 字节: Int64) -> String {
        let 兆 = Double(字节) / 1_048_576
        if 兆 >= 1024 { return String(format: "%.2f GB", 兆 / 1024) }
        return String(format: "%.0f MB", 兆)
    }

    static func 速度(_ 字节每秒: Double) -> String {
        guard 字节每秒 > 1 else { return "" }
        if 字节每秒 >= 1_048_576 { return String(format: "%.1f MB/s", 字节每秒 / 1_048_576) }
        return String(format: "%.0f KB/s", 字节每秒 / 1024)
    }
}
