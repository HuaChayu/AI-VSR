// ============================================================
//  文件作用：预览区的播放内核与画面宿主。
//  预览播放器 统一持有「视频（AVQueuePlayer）/ 动图（逐帧延时驱动）」两套播放，
//  对外只暴露 播放 / 暂停 / 定位(seek) / 当前时间 / 总时长 / 循环模式 / 音量 / 静音 / 动图当前帧，
//  播放条与预览区都只跟它交互，因此拖动定格、跨素材重置、一次/循环切换只有一处实现。
//  视频播放视图 只负责把 AVPlayerLayer 贴在预览区，自己不再持有播放器与循环器。
//  互动文件：播放 条.swift（控件驱动）、素材 详情.swift（装入目标 + 渲染画面）、全局快捷键.swift。
// ============================================================

import AVFoundation
import AppKit
import ImageIO
import SwiftUI

/// 播放循环口径：一次播放（播完停在末帧） / 无限循环（播到结尾接回开头继续播）。
enum 播放模式: String, CaseIterable, Identifiable {
    case 一次
    case 循环

    var id: String { rawValue }

    var 标题: String {
        switch self {
        case .一次: return "一次播放"
        case .循环: return "无限循环"
        }
    }

    var 图标: String {
        switch self {
        case .一次: return "arrow.right.to.line"
        case .循环: return "repeat"
        }
    }

    var 提示: String {
        switch self {
        case .一次: return "播完停在末帧，按钮回到暂停态，便于对比最后一帧"
        case .循环: return "播到结尾自动从头续播，反复观察整段差异"
        }
    }

    var 另一: 播放模式 { self == .一次 ? .循环 : .一次 }
}

/// 预览播放器：视频与动图共用一套播放 / 进度 / 循环模式接口。
@MainActor
@Observable
final class 预览播放器 {
    enum 播放源类型: String {
        case 空
        case 视频
        case 动图
        case 静态图
    }

    // MARK: 状态

    /// 是否已经可以接收「同刻起跑」请求。
    /// AVPlayer 的 status 不是 readyToPlay 时，setRate(_:time:atHostTime:) 会直接抛
    /// NSInvalidArgumentException 崩掉整个 App（实测：即便关掉 automaticallyWaitsToMinimizeStalling 也不行）。
    private(set) var 可同步起播 = false

    private(set) var 种类: 播放源类型 = .空
    private(set) var 地址: URL?
    var 播放中 = false
    /// 播放头（秒）：视频取播放器当前时刻，动图取累计帧时长。
    var 当前时间: Double = 0
    /// 总时长（秒）：动图按逐帧延时求和。
    var 总时长: Double = 0
    /// 一次播放 / 无限循环：写入 UserDefaults，跨启动保留。
    var 模式: 播放模式 = 预览播放器.读取模式() {
        didSet {
            guard 模式 != oldValue else { return }
            UserDefaults.standard.set(模式.rawValue, forKey: 预览播放器.模式键)
        }
    }
    /// 播放音量（0…1）：由播放条的滑杆双向绑定，写入 UserDefaults 跨启动保留。
    /// AVPlayer.volume 的物理上限就是 1.0，光靠它「拉满」只有原始响度；
    /// 因此这里固定 volume = 1.0，实际响度改由 AVAudioMix 的输入参数做增益（1.0 → 2 倍 ≈ +6dB）。
    var 音量: Float = 预览播放器.读取音量() {
        didSet {
            guard 音量 != oldValue else { return }
            UserDefaults.standard.set(音量, forKey: 预览播放器.音量键)
            应用音量()
        }
    }
    /// 静音开关：由播放条的静音按钮驱动，与 音量 各自独立（取消静音自动回到原音量）。
    var 静音 = false {
        didSet {
            guard 静音 != oldValue else { return }
            应用音量()
        }
    }
    /// 动图当前帧（预览区直接画这张）
    private(set) var 动图帧图: NSImage?
    /// 动图当前帧号（1 起）与总帧数，供播放条显示「帧 3/12」。
    private(set) var 动图帧号 = 0
    private(set) var 动图总帧 = 0

    // MARK: 内部

    let 播放器 = AVQueuePlayer()
    private var 载入任务: Task<Void, Never>?
    private var 播放任务: Task<Void, Never>?
    private var 动图帧: [NSImage] = []
    private var 动图时延: [Double] = []
    private var 动图起点: [Double] = []
    /// 循环模式下已发出「回到开头」的 seek，等播放头真的回到前半段再解除，避免连续 seek。
    private var 归零中 = false
    /// 音频增益安装序号：异步载入音轨期间音量又被拖动时，只让最后一次生效。
    private var 音频序号 = 0

    nonisolated static let 模式键 = "预览播放模式"
    nonisolated static let 音量键 = "预览播放音量"

    nonisolated static func 读取模式() -> 播放模式 {
        guard let 值 = UserDefaults.standard.string(forKey: 模式键),
              let 模式 = 播放模式(rawValue: 值) else { return .循环 }
        return 模式
    }

    nonisolated static func 读取音量() -> Float {
        guard UserDefaults.standard.object(forKey: 音量键) != nil else { return 0.8 }
        return min(max(UserDefaults.standard.float(forKey: 音量键), 0), 1)
    }

    /// 滑杆值(0…1) → AVAudioMix 输入增益：1.0 给到 2 倍（+6dB），
    /// 保证「拉满」确实比原始音轨更响，而不是卡在 AVPlayer.volume 的 1.0 原始响度上。
    nonisolated static func 增益(音量: Float) -> Float {
        min(max(音量, 0), 1) * 2
    }

    /// 造一个「只播放、恒定静音」的内核，给对比层的源片那一路用。
    /// 不写 UserDefaults，所以不会污染主预览的音量设置。
    static func 静音内核() -> 预览播放器 {
        let 内核 = 预览播放器()
        内核.静音 = true
        return 内核
    }

    init() {
        播放器.actionAtItemEnd = .pause
        // 不再写死静音：静音状态与音量全部下放给播放条控件，由 应用音量() 落到播放器上。
        播放器.volume = 1
        播放器.isMuted = false
        // 关键：默认 YES 时 AVPlayer 会直接拒绝 setRate(_:time:atHostTime:) 并抛
        // NSInvalidArgumentException（"cannot service a synchronized playback request..."），
        // 也就是一播放就闪退的真凶。本地文件不需要等缓冲，关掉即可正常同步起跑。
        播放器.automaticallyWaitsToMinimizeStalling = false
    }

    /// 调试日志（临时排查用）
    nonisolated static func 调试日志(_ 文字: String) {
        let 行 = "\(Date().timeIntervalSince1970) \(文字)\n"
        let 路径 = "/tmp/playcheck/run.log"
        if !FileManager.default.fileExists(atPath: 路径) { FileManager.default.createFile(atPath: 路径, contents: nil) }
        if let 句柄 = FileHandle(forWritingAtPath: 路径) {
            句柄.seekToEndOfFile()
            句柄.write(Data(行.utf8))
            try? 句柄.close()
        }
    }

    // MARK: 派生

    var 可播放: Bool { 种类 == .视频 || 种类 == .动图 }

    /// 画面是否已经就绪（预览区据此决定画播放层还是画静态帧）。
    var 已就绪: Bool {
        switch 种类 {
        case .视频: return 播放器.currentItem != nil
        case .动图: return !动图帧.isEmpty
        default: return false
        }
    }

    // MARK: 音量 / 静音

    /// 把 静音、音量 落到真实播放器上。
    /// 静音走 isMuted（立即生效、且不影响混音设置）；音量固定 volume = 1.0，响度由 AVAudioMix 增益给出。
    func 应用音量() {
        播放器.isMuted = 静音
        播放器.volume = 1
        安装音频增益()
    }

    /// 给当前 item 装一份「按 音量 换算增益」的音频混音。
    /// AVPlayer.volume 只能到 1.0，超过原始响度必须靠 AVMutableAudioMixInputParameters 的 volume 放大。
    private func 安装音频增益() {
        let 增益 = 预览播放器.增益(音量: 音量)
        // 滑杆拖动会连发多次，只认最后一次：否则先发的异步任务可能后回来，把新增益覆盖成旧值。
        音频序号 += 1
        let 序号 = 音频序号
        Task { @MainActor [weak self] in
            guard let self, let 条目 = self.播放器.currentItem else { return }
            let 资源 = 条目.asset
            let 轨道组 = (try? await 资源.loadTracks(withMediaType: .audio)) ?? []
            // 没有音轨（无声视频/动图）就什么都不用做
            guard let 轨道 = 轨道组.first,
                  序号 == self.音频序号,
                  条目 === self.播放器.currentItem else { return }
            let 参数 = AVMutableAudioMixInputParameters(track: 轨道)
            参数.setVolume(增益, at: .zero)
            let 混音 = AVMutableAudioMix()
            混音.inputParameters = [参数]
            条目.audioMix = 混音
        }
    }

    // MARK: 装入

    /// 切换播放目标。地址与种类都没变时直接返回，避免 SwiftUI 反复刷新打断正在播的内容。
    func 装入(地址 新地址: URL?, 种类 新种类: 播放源类型) {
        预览播放器.调试日志("装入 新=\(新地址?.lastPathComponent ?? "空") 种类=\(新种类.rawValue) 旧=\(地址?.lastPathComponent ?? "空")")
        let 目标 = (新种类 == .视频 || 新种类 == .动图) ? 新地址 : nil
        // 地址没变但片源已被卸掉（比如对比层退出时清过 item）也必须重装，否则画面永远空着。
        // 注意：只有 AVPlayer 那一路才谈得上「item 被卸掉」。动图走逐帧缓存、currentItem 恒为 nil，
        // 拿它当依据会让每次装入都清理重载，把正在播的动图打回第一帧（表现为「按播放没反应」）。
        let 缺片源 = 新种类 == .视频 && 播放器.currentItem == nil
        guard 目标 != 地址 || 新种类 != 种类 || 缺片源 else { return }
        清理()
        种类 = 新种类
        地址 = 目标
        guard let 目标 else { return }
        switch 新种类 {
        case .视频:
            载入任务 = Task { [weak self] in await self?.装入视频(目标) }
        case .动图:
            载入任务 = Task { [weak self] in await self?.装入动图(目标) }
        default:
            break
        }
    }

    /// 停掉并清空一切播放状态（换素材 / 素材被移除时调用）。
    func 清理() {
        播放中 = false
        可同步起播 = false
        载入任务?.cancel()
        载入任务 = nil
        播放任务?.cancel()
        播放任务 = nil
        播放器.pause()
        播放器.replaceCurrentItem(with: nil)
        归零中 = false
        当前时间 = 0
        总时长 = 0
        动图帧 = []
        动图时延 = []
        动图起点 = []
        动图帧图 = nil
        动图帧号 = 0
        动图总帧 = 0
    }

    private func 装入视频(_ url: URL) async {
        let 资源 = AVURLAsset(url: url)
        let 秒 = (try? await 资源.load(.duration)).map { CMTimeGetSeconds($0) } ?? 0
        guard !Task.isCancelled, 地址 == url, 种类 == .视频 else { return }
        总时长 = 秒.isFinite && 秒 > 0 ? 秒 : 0
        当前时间 = 0
        播放器.replaceCurrentItem(with: AVPlayerItem(asset: 资源))
        // 换片源后混音/静音都挂在新 item 上，必须重新套一次，否则新视频会按默认音量出声或没声。
        应用音量()
        等待起播就绪()
    }

    /// 等 AVPlayer 真正 readyToPlay 后才放开「同刻起跑」权限，避免 setRate(atHostTime:) 抛异常崩溃。
    private func 等待起播就绪() {
        可同步起播 = false
        Task { @MainActor in
            for _ in 0..<300 {                       // 最多等 6 秒，就绪即退出
                if Task.isCancelled { return }
                switch 播放器.status {
                case .readyToPlay:
                    可同步起播 = true
                    return
                case .failed:
                    可同步起播 = false
                    return
                default:
                    try? await Task.sleep(nanoseconds: 20_000_000)
                }
            }
        }
    }

    private func 装入动图(_ url: URL) async {
        let 载入 = await Task.detached(priority: .userInitiated) { 预览播放器.解动图(url) }.value
        guard !Task.isCancelled, 地址 == url, 种类 == .动图 else { return }
        let 帧 = 载入.帧.map { NSImage.以(cg图: $0) }
        guard !帧.isEmpty else { return }
        动图帧 = 帧
        动图时延 = 载入.时延
        var 起点: [Double] = []
        var 累计 = 0.0
        for 序号 in 帧.indices {
            起点.append(累计)
            累计 += 载入.时延.indices.contains(序号) ? max(0.02, 载入.时延[序号]) : 0.1
        }
        动图起点 = 起点
        动图总帧 = 帧.count
        总时长 = 累计
        当前时间 = 0
        显示帧(0)
    }

    /// 逐帧解出 GIF 全分辨率帧与逐帧延时（与动图超分同一套延时口径，播放节奏与成片一致）。
    nonisolated static func 解动图(_ url: URL) -> (帧: [CGImage], 时延: [Double]) {
        guard let 源 = CGImageSourceCreateWithURL(url as CFURL, nil) else { return ([], []) }
        let 数量 = CGImageSourceGetCount(源)
        guard 数量 > 0 else { return ([], []) }
        let 时延 = 动图超分.帧时延(源: 源, 帧数: 数量)
        var 帧: [CGImage] = []
        帧.reserveCapacity(数量)
        for 序号 in 0..<数量 {
            if let 图 = CGImageSourceCreateImageAtIndex(源, 序号, nil) { 帧.append(图) }
        }
        return (帧, 时延)
    }

    // MARK: 全局空格

    /// 把「空格 → 播放 / 暂停」挂到全局快捷键上。
    /// 返回 false 表示这次空格不消费：静态图、无素材、片源还没载入好时让空格照常下发。
    func 挂到全局快捷键() {
        全局快捷键.共享.挂载(标识: ObjectIdentifier(self)) { [weak self] in
            guard let self, self.可播放, self.已就绪 else { return false }
            self.切换()
            return true
        }
    }

    func 摘下全局快捷键() {
        全局快捷键.共享.摘下(标识: ObjectIdentifier(self))
    }

    // MARK: 播放控制

    func 播放() {
        预览播放器.调试日志("播放() 种类=\(种类.rawValue) 当前=\(当前时间) 总=\(总时长) 模式=\(模式.rawValue)")
        guard 可播放 else { return }
        switch 种类 {
        case .视频:
            guard 播放器.currentItem != nil else { return }
            // 一次播放停在末帧后再点播放：先回到开头，否则 AVPlayer 会原地不动
            if 总时长 > 0, 当前时间 >= 总时长 - 0.05 { 定位(到: 0) }
            归零中 = false
            播放器.play()
        case .动图:
            guard !动图帧.isEmpty else { return }
            if 总时长 > 0, 当前时间 >= 总时长 - 0.001 { 定位(到: 0) }
        default:
            return
        }
        播放中 = true
        启动任务()
    }

    func 暂停() {
        预览播放器.调试日志("暂停() 当前=\(当前时间) 总=\(总时长)")
        播放中 = false
        播放任务?.cancel()
        播放任务 = nil
        播放器.pause()
    }

    func 切换() {
        if 播放中 { 暂停() } else { 播放() }
    }

    /// 把播放头硬定位到某一刻（容差归零）：拖动进度条时用来定格看某一帧的差异。
    func 定位(到 秒: Double) {
        let 目标 = min(max(0, 秒), max(总时长, 0))
        当前时间 = 目标
        switch 种类 {
        case .视频:
            归零中 = false
            播放器.seek(to: CMTime(seconds: 目标, preferredTimescale: 600),
                     toleranceBefore: .zero,
                     toleranceAfter: .zero)
        case .动图:
            显示帧(帧序号(时间: 目标))
        default:
            break
        }
    }

    // MARK: 计时任务（视频按时间轮询、动图按帧延时推进）

    private func 启动任务() {
        播放任务?.cancel()
        播放任务 = Task { [weak self] in
            guard let self else { return }
            while self.播放中, !Task.isCancelled {
                switch self.种类 {
                case .视频:
                    self.刷新视频()
                    try? await Task.sleep(nanoseconds: 30_000_000)
                case .动图:
                    let 等待 = self.当前帧时长()
                    try? await Task.sleep(nanoseconds: UInt64(等待 * 1_000_000_000))
                    if Task.isCancelled { return }
                    if self.播放中 { self.推进一帧() }
                default:
                    return
                }
            }
        }
    }

    private func 刷新视频() {
        guard 种类 == .视频 else { return }
        let 秒 = CMTimeGetSeconds(播放器.currentTime())
        if 秒.isFinite { 当前时间 = max(0, 秒) }
        guard 总时长 > 0 else { return }
        if 归零中 {
            if 当前时间 < 总时长 * 0.5 { 归零中 = false }
            return
        }
        guard 当前时间 >= 总时长 - 0.03 else { return }
        switch 模式 {
        case .一次:
            预览播放器.调试日志("到末尾·一次播放停末帧 当前=\(当前时间) 总=\(总时长)")
            当前时间 = 总时长
            播放中 = false
            播放器.pause()
            播放任务 = nil
        case .循环:
            预览播放器.调试日志("到末尾·循环回零 当前=\(当前时间) 总=\(总时长)")
            归零中 = true
            当前时间 = 0
            播放器.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
            播放器.play()
        }
    }

    // MARK: 动图帧推进

    private func 当前帧时长() -> Double {
        let 序号 = 动图帧号 - 1
        guard 动图时延.indices.contains(序号) else { return 0.1 }
        return max(0.02, 动图时延[序号])
    }

    private func 帧序号(时间 秒: Double) -> Int {
        guard !动图起点.isEmpty else { return 0 }
        var 序号 = 0
        for (下标, 起点) in 动图起点.enumerated() where 起点 <= 秒 { 序号 = 下标 }
        return 序号
    }

    private func 显示帧(_ 序号: Int) {
        let 目标 = min(max(0, 序号), max(动图帧.count - 1, 0))
        guard 动图帧.indices.contains(目标) else { return }
        动图帧号 = 目标 + 1
        动图帧图 = 动图帧[目标]
        if 动图起点.indices.contains(目标) { 当前时间 = 动图起点[目标] }
    }

    private func 推进一帧() {
        guard !动图帧.isEmpty else { return }
        let 下一 = 动图帧号 + 1
        if 下一 > 动图帧.count {
            switch 模式 {
            case .一次:
                播放中 = false
                播放任务 = nil
                动图帧号 = 动图帧.count
                动图帧图 = 动图帧[动图帧.count - 1]
                当前时间 = 总时长
            case .循环:
                显示帧(0)
            }
            return
        }
        显示帧(下一 - 1)
    }
}

// ============================================================

/// 预览区的视频画面宿主：只把预览播放器的 AVPlayerLayer 贴上来，播放状态全在 预览播放器 里。
struct 视频播放视图: NSViewRepresentable {
    let url: URL
    let 播放器: 预览播放器

    func makeNSView(context: Context) -> 播放宿主视图 {
        let 视图 = 播放宿主视图()
        视图.装入(播放器: 播放器)
        视图.装入(url: url)
        return 视图
    }

    func updateNSView(_ nsView: 播放宿主视图, context: Context) {
        nsView.装入(播放器: 播放器)
        nsView.装入(url: url)
    }

    static func dismantleNSView(_ nsView: 播放宿主视图, coordinator: ()) {
        nsView.停止()
    }
}

final class 播放宿主视图: NSView {
    private let 画面层 = AVPlayerLayer()
    private var 预览播放器: 预览播放器?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        let 根层 = CALayer()
        根层.backgroundColor = NSColor.clear.cgColor
        layer = 根层
        画面层.videoGravity = .resizeAspect
        根层.addSublayer(画面层)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) 未实现")
    }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        画面层.frame = bounds
        CATransaction.commit()
    }

    /// 绑定播放内核：画面层永远显示内核里的那个播放器。
    func 装入(播放器: 预览播放器) {
        guard 预览播放器 !== 播放器 else { return }
        预览播放器 = 播放器
        画面层.player = 播放器.播放器
    }

    /// 换素材时让内核重装片源（同地址重复调用在内核里已被拦掉）。
    func 装入(url: URL) {
        guard let 预览播放器 else { return }
        guard 预览播放器.地址 != url || 预览播放器.种类 != .视频 else { return }
        预览播放器.装入(地址: url, 种类: .视频)
    }

    func 停止() {
        预览播放器?.暂停()
    }
}
