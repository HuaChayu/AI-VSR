import AVFoundation
import AppKit
import SwiftUI

/// 右侧主区：当前素材的大图预览 + 元数据 + Flash 超分通道。
struct 素材详情: View {
    @Environment(素材Store.self) private var 素材
    @Environment(超分通道.self) private var 超分

    @State private var 提示条: String?
    /// 预览播放内核：视频 / 动图共用一套播放、进度、循环模式，播放条与全局空格都作用于它。
    @State private var 播放器 = 预览播放器()
    /// 对比用的左路源片：复用一个只播放的静音播放内核，退出对比即清空 item。
    /// 注意：预览播放器 已不再在 init 里写死静音（音量交给播放条控制），
    /// 这一路必须显式静音，否则左右对比时会和成片叠成双声。
    @State private var 源片播放器 = 预览播放器.静音内核()
    @State private var 鼠标在内 = false
    @State private var 悬停点: CGPoint?
    /// 对齐请求序号：源片 seek 是异步的，多次对齐只认最后一次的回调。
    @State private var 对齐序号 = 0
    /// 本次超分对应的素材 id（超分跑完据此点亮完成角标）。
    @State private var 超分素材: UUID?
    /// 对比层显示倍率：1 = 整图等比适配；往上放大可做 1:1 像素级局部查看。
    @State private var 对比缩放: CGFloat = 1

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: 素材.选中条目?.名称 ?? "预览",
                        symbol: "play.rectangle",
                        副标题: 素材.选中条目?.副标题文本)

            预览区

            if let 条目 = 素材.选中条目 {
                播放条(播放器: 播放器, 条目: 条目)
                元数据(条目)
            }

            超分控制
        }
        .task(id: 播放目标键) {
            // 选中啥播啥：有产物播成片，无产物播源文件；切素材时内核自会清空进度。
            guard let 条目 = 素材.选中条目 else {
                播放器.装入(地址: nil, 种类: .空)
                return
            }
            let 目标 = 播放目标(条目)
            播放器.装入(地址: 目标.url, 种类: 目标.种类)
        }
        .onAppear {
            // 空格监听已在 App 启动时装好（MyApp.swift），这里只把「空格做什么」挂到播放内核上
            播放器.挂到全局快捷键()
            超分.展示素材id = 素材.选中
            // 视图重新挂载时把已经跑完的素材补一次角标：onChange 只听「变化」，
            // 重建期间跑完的那条不会重放，这里按 刚完成 里的 id 补点亮。
            for id in 超分.刚完成 { 素材.标记完成(id) }
        }
        .onDisappear {
            播放器.暂停()
            播放器.摘下全局快捷键()
        }
        .onChange(of: 素材.选中条目?.url) { _, _ in
            提示条 = nil
            对比缩放 = 1
            // 展示的成品跟着选中走：切到谁就只看谁自己的成品，别的素材的成果不会串过来。
            超分.展示素材id = 素材.选中
            超分.清空()
            // 切回已超分过的素材：按落盘索引从磁盘产物重建对比层，不必重新推理。
            guard let 条目 = 素材.选中条目 else { return }
            Task { @MainActor in
                guard 素材.选中 == 条目.id else { return }
                await 超分.恢复缓存(素材: 条目)
                guard 超分.产物 != nil, 素材.选中 == 条目.id else { return }
                素材.标记完成(条目.id)
            }
        }
        .onChange(of: 超分.产物?.输出宽) { _, _ in
            // 换了一批超分结果就回到整图适配，避免沿上一张的放大倍率跳变
            对比缩放 = 1
        }
        .onChange(of: 超分.刚完成) { _, 完成表 in
            // 跑完哪条就点亮哪条：队列连跑时会连着跑完好几条，按 刚完成 里的 id 逐个点亮，
            // 不按「当前选中」猜，免得把上一条的结果记到下一条头上。
            for id in 完成表 { 素材.标记完成(id) }
        }
    }

    /// 播放目标键：素材 id + 播放地址 + 类型，任一变化才重装播放源，避免 SwiftUI 刷新打断正在播的内容。
    private var 播放目标键: String {
        guard let 条目 = 素材.选中条目 else { return "空" }
        let 目标 = 播放目标(条目)
        return "\(条目.id.uuidString)|\(目标.种类.rawValue)|\(目标.url?.path ?? "空")"
    }

    /// 选中啥播啥：有超分成片（视频 mp4 / 动图 gif）播成片，没有产物播源文件；
    /// 静态图片给播放内核一个「静态图」状态，播放条据此禁用按钮与进度。
    private func 播放目标(_ 条目: 素材条目) -> (url: URL?, 种类: 预览播放器.播放源类型) {
        switch 条目.类型 {
        case .视频:
            return (播放地址(条目), .视频)
        case .图片:
            return (播放地址(条目), 条目.是动图 ? .动图 : .静态图)
        }
    }

    // MARK: 预览

    private var 预览区: some View {
        GeometryReader { 几何 in
            ZStack {
                Theme.viewportBackground

                画面(容器: 几何.size)
                    .clipped()

                // 只要引擎在干活就显示：切到别的素材也照样看到「轮到谁、第几帧」。
                if 超分.忙 {
                    VStack(spacing: 8) {
                        if let 比例 = 超分.视频进度比例 {
                            ProgressView(value: 比例)
                                .progressViewStyle(.linear)
                                .frame(width: 180)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                        Text(超分.进度摘要)
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.textSecondary)
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Theme.panelBackground.opacity(0.9))
                    )
                    // 贴右上角：进度是全局信息，别压在画面中间挡着看素材。
                    .padding(12)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }

                // 权重缺失时：先下载再超分，下载期间不进入超分流程
                if 权重下载管理.共享.下载中 {
                    VStack(spacing: 8) {
                        ProgressView(value: 权重下载管理.共享.进度)
                            .progressViewStyle(.linear)
                            .frame(width: 200)
                        Text(权重下载管理.共享.状态文本)
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.textSecondary)
                        Button("取消下载") { 权重下载管理.共享.取消() }
                            .buttonStyle(.plain)
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.textSecondary)
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Theme.panelBackground.opacity(0.9))
                    )
                }

                // 放大倍率角标：视频态下也能立刻看到滚轮是否吃到
                if 视频对比中, 对比缩放 > 1.02 {
                    Text(String(format: "×%.2f · 滚轮还原", 对比缩放))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(Theme.panelBackground.opacity(0.85))
                        )
                        .allowsHitTesting(false)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                        .padding(10)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onHover { 在内 in
            // 只控对比层显隐，不碰播放状态：播放中滑进来也能对比
            鼠标在内 = 在内
            if 在内, 可对比 { NSCursor.resizeLeftRight.set() } else { NSCursor.arrow.set() }
        }
        .onContinuousHover(coordinateSpace: .local) { 阶段 in
            switch 阶段 {
            case .active(let 点):
                // 分割线跟手：关掉隐式动画
                withTransaction(Transaction(animation: nil)) { 悬停点 = 点 }
                if 可对比 { NSCursor.resizeLeftRight.set() }
            case .ended:
                withTransaction(Transaction(animation: nil)) { 悬停点 = nil }
                NSCursor.arrow.set()
            }
        }
        .background {
            滚轮缩放(启用: { 可对比 }, 处理: { 增量 in 缩放对比(增量) })
        }
        .onChange(of: 视频对比中) { _, 开 in
            if 开, let 条目 = 素材.选中条目 {
                源片播放器.装入(地址: 条目.url, 种类: .视频)
                // 源片已在（同一素材反复进出）就直接对齐；否则等载入完成再对齐
                if 源片播放器.已就绪 { 对齐源片(并起跑: true) }
            } else {
                // 只暂停不卸载片源：卸掉 item 而地址还在，再进对比会被「同地址不重装」拦掉，左侧就成空的了
                源片播放器.暂停()
            }
        }
        .onChange(of: 源片播放器.总时长) { _, 时长 in
            guard 时长 > 0 else { return }
            if 视频对比中 { 对齐源片(并起跑: true) }
            else if 动图对比中, 源片播放器.已就绪 { 源片播放器.定位(到: 播放器.当前时间) }
        }
        .onChange(of: 动图对比中) { _, 开 in
            guard 开, let 条目 = 素材.选中条目 else { return }
            if 源片播放器.地址 != 条目.url || 源片播放器.种类 != .动图 {
                源片播放器.装入(地址: 条目.url, 种类: .动图)
            }
            if 源片播放器.动图帧图 != nil { 源片播放器.定位(到: 播放器.当前时间) }
        }
        .onChange(of: 播放器.动图帧号) { _, _ in
            // 动图没有 AVPlayer 时钟，源路帧号直接跟着成片帧号走，播放中每帧都能对上
            guard 动图对比中, 源片播放器.动图帧图 != nil else { return }
            源片播放器.定位(到: 播放器.当前时间)
        }
        .onChange(of: 播放器.播放中) { _, 在播 in
            guard 视频对比中 else { return }
            if 在播 { 对齐源片(并起跑: true) } else { 源片播放器.播放器.pause() }
        }
        .onChange(of: 播放器.当前时间) { 旧, 新 in
            // 循环回零 / 拖回进度条时补一次对齐，正常播放不介入
            guard 视频对比中, 播放器.播放中, 新 + 0.5 < 旧 else { return }
            对齐源片(并起跑: true)
        }
        .onChange(of: 播放器.可同步起播) { _, 可以 in
            // 点播时片源还没 ready，先普通起播；一旦 ready 立刻补一次同刻对齐
            guard 可以, 视频对比中, 播放器.播放中 else { return }
            对齐源片(并起跑: true)
        }
        .onChange(of: 源片播放器.可同步起播) { _, 可以 in
            guard 可以, 视频对比中, 播放器.播放中 else { return }
            对齐源片(并起跑: true)
        }
    }

    /// 把源片锚到宿主时钟并精确 seek 到成片当前时间（只在进入对比、载入完成、回零时各做一次）。
    /// 关键：seek 是异步的，必须等它落地（completion）再起跑；否则 setRate 用的是 seek 前的旧位置，
    /// 源片会停在旧帧不动 —— 这正是「循环时右边开播、左边卡在第一帧」的直接原因。
    private func 对齐源片(并起跑: Bool) {
        源片播放器.播放器.masterClock = CMClockGetHostTimeClock()
        let 目标 = 播放器.播放器.currentTime()
        对齐序号 &+= 1
        let 本次 = 对齐序号
        源片播放器.播放器.seek(to: 目标, toleranceBefore: .zero, toleranceAfter: .zero) { _ in
            Task { @MainActor in
                guard 本次 == 对齐序号 else { return }   // 期间又发起了新对齐，以最新一次为准
                guard 并起跑, 播放器.播放中 else { return }
                双路同刻起跑()
            }
        }
    }

    /// 唯一的一处起跑实现：两路各取自身播放头，绑同一个 hostTime 起跑。
    private func 双路同刻起跑() {
        let 起点 = CMTimeAdd(CMClockGetTime(CMClockGetHostTimeClock()),
                          CMTime(value: 60, timescale: 1000))
        let 源时 = 源片播放器.播放器.currentTime()
        if 源片播放器.可同步起播, 源时.isValid, 源时.isNumeric {
            源片播放器.播放器.setRate(1, time: 源时, atHostTime: 起点)
        } else {
            源片播放器.播放器.play()
        }
        let 成时 = 播放器.播放器.currentTime()
        if 播放器.可同步起播, 成时.isValid, 成时.isNumeric {
            播放器.播放器.setRate(1, time: 成时, atHostTime: 起点)
        } else {
            播放器.播放器.play()
        }
    }

    /// 滚轮调对比层倍率：向下缩小回到整图适配，向上放大到像素级（上限 16×）。
    private func 缩放对比(_ 增量: CGFloat) {
        guard abs(增量) > 0.01 else { return }
        let 倍 = pow(1.0022, 增量)
        对比缩放 = min(max(对比缩放 * 倍, 1), 16)
    }

    /// 有超分产物 + 鼠标在预览区 + 未播放时，进入「竖线前后对比」。
    private var 可对比: Bool {
        素材.选中条目 != nil && 超分.产物 != nil && 超分.显示超分 && 鼠标在内
    }

    /// 视频双路对比：播放中也成立，左右两侧都随播放实时出帧。
    private var 视频对比中: Bool {
        可对比 && 超分.视频产物 != nil && 播放器.已就绪 && 素材.选中条目?.类型 == .视频
    }

    /// 动图双路对比：和视频同一套口径，播放中两侧各自逐帧出图，鼠标进出不再定格。
    private var 动图对比中: Bool {
        可对比 && 素材.选中条目?.是动图 == true && 播放器.动图帧图 != nil && 超分.产物 != nil
    }

    /// 分割线横坐标：取鼠标悬停点，没悬停就在正中。
    private func 分割位置(_ 容器: CGSize) -> CGFloat {
        let 宽 = max(容器.width, 1)
        return max(0, min(悬停点?.x ?? 宽 / 2, 宽))
    }

    /// 缩放锚点：以指针所在位置为原点放大，没悬停就居中放大。
    private func 缩放锚点(_ 容器: CGSize) -> UnitPoint {
        let 宽 = max(容器.width, 1)
        let 高 = max(容器.height, 1)
        let 点 = 悬停点 ?? CGPoint(x: 宽 / 2, y: 高 / 2)
        return UnitPoint(x: max(0, min(点.x / 宽, 1)),
                         y: max(0, min(点.y / 高, 1)))
    }

    @ViewBuilder
    private func 画面(容器: CGSize) -> some View {
        if let 条目 = 素材.选中条目 {
            if 条目.类型 == .视频 {
                // 视频画面全交给播放内核：暂停时定格在当前帧，播放时连续出帧
                if 播放器.已就绪 {
                    ZStack(alignment: .topLeading) {
                        ZStack(alignment: .topLeading) {
                            视频播放视图(url: 播放地址(条目), 播放器: 播放器)
                                .id("预览成片-\(条目.id.uuidString)")
                                .allowsHitTesting(false)
                            // 播放中对比：再叠一路源片，用 SwiftUI 裁到分割线左边，右侧仍是成片
                            if 视频对比中 {
                                视频播放视图(url: 条目.url, 播放器: 源片播放器)
                                    .id("预览源片-\(条目.id.uuidString)")
                                    .allowsHitTesting(false)
                                    .frame(width: max(容器.width, 1), height: max(容器.height, 1))
                                    .frame(width: max(分割位置(容器), 0.5),
                                           height: max(容器.height, 1),
                                           alignment: .leading)
                                    .clipped()
                            }
                        }
                        .frame(width: max(容器.width, 1), height: max(容器.height, 1), alignment: .topLeading)
                        // 滚轮放大/还原同样作用于视频画面：只变换画面两层，分割线保持屏幕坐标
                        .scaleEffect(对比缩放, anchor: 缩放锚点(容器))
                        .clipped()
                        if 视频对比中 {
                            分割线(容器: 容器)
                        }
                    }
                } else if let 图 = (超分.显示超分 ? 超分.产物?.输出图 : nil) ?? 素材.预览帧 ?? 条目.缩略图 {
                    单图(图)
                } else {
                    ProgressView().controlSize(.small)
                }
            } else if 条目.是动图 {
                // 动图按逐帧延时在播放内核里推进；鼠标进对比时再叠一路源 gif 同帧对齐，播放中也能左右比
                if 播放器.已就绪, let 图 = 播放器.动图帧图 {
                    if 动图对比中, let 源图 = 源片播放器.动图帧图 {
                        ZStack(alignment: .topLeading) {
                            ZStack(alignment: .topLeading) {
                                单图(图)
                                    .frame(width: max(容器.width, 1), height: max(容器.height, 1))
                                单图(源图)
                                    .frame(width: max(容器.width, 1), height: max(容器.height, 1))
                                    .frame(width: max(分割位置(容器), 0.5),
                                           height: max(容器.height, 1),
                                           alignment: .leading)
                                    .clipped()
                            }
                            .frame(width: max(容器.width, 1), height: max(容器.height, 1), alignment: .topLeading)
                            .scaleEffect(对比缩放, anchor: 缩放锚点(容器))
                            .clipped()
                            分割线(容器: 容器)
                        }
                    } else {
                        单图(图)
                    }
                } else if let 图 = 素材.预览帧 ?? 条目.缩略图 {
                    单图(图)
                } else {
                    ProgressView().controlSize(.small)
                }
            } else if 可对比, let 产物 = 超分.产物 {
                对比层(产物: 产物, 原图: 产物.输入图, 容器: 容器)
            } else if let 产物 = 超分.产物, 超分.显示超分 {
                单图(产物.输出图)
            } else if let 图 = 素材.预览帧 ?? 条目.缩略图 {
                单图(图)
            } else {
                ProgressView().controlSize(.small)
            }
        } else {
            空态提示(symbol: "film.stack",
                       标题: "还没有选中的素材",
                       说明: "从左侧导入视频或素材，点选任意一条查看画面")
        }
    }

    /// 视频对比的分割线：竖线 + 左右两侧尺寸标签，全部纯 SwiftUI 叠加。
    private func 分割线(容器: CGSize) -> some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.white.opacity(0.85))
                .frame(width: 1.5, height: max(容器.height, 1))
                .offset(x: 分割位置(容器))
                .shadow(color: .black.opacity(0.4), radius: 4)
            VStack {
                HStack(spacing: 6) {
                    对比标签("源帧 " + (超分.产物?.输入尺寸文本 ?? ""))
                    Spacer(minLength: 0)
                    对比标签("超分 " + (超分.产物?.输出尺寸文本 ?? ""))
                }
                Spacer(minLength: 0)
            }
            .padding(10)
        }
        .frame(width: max(容器.width, 1), height: max(容器.height, 1))
        .allowsHitTesting(false)
    }

    private func 单图(_ 图: NSImage) -> some View {
        Image(nsImage: 图)
            .resizable()
            .scaledToFit()
            .padding(20)
            .shadow(color: .black.opacity(0.55), radius: 16, y: 8)
    }

    /// 源素材 / 超分成片对比：竖线右侧恒为超分结果，左侧为源素材（与视频对比同一口径）。
    /// 默认整图等比适配、不裁切；滚轮放大时以鼠标位置为锚点，指针下的内容保持不动，便于像素级 1:1 查看。
    private func 对比层(产物: 超分通道.超分产物, 原图: NSImage, 容器: CGSize) -> some View {
        let 比例 = CGFloat(产物.输出宽) / CGFloat(max(1, 产物.输出高))
        let 适配 = Self.适配矩形(比例: 比例, 容器: 容器)

        // 每 1 个超分像素占多少屏幕点：整图适配 = 适配宽 / 输出宽
        let 基础缩放 = 适配.width / CGFloat(max(1, 产物.输出宽))
        let 缩放 = 基础缩放 * 对比缩放
        let 图宽 = CGFloat(产物.输出宽) * 缩放
        let 图高 = CGFloat(产物.输出高) * 缩放

        // 两侧共用同一显示矩形，才保证像素级 1:1 对齐。
        // 显示宽超出该层原生像素宽属于「放大查看」，用最近邻保住像素块不糊；
        // 否则属于缩小适配，用高质量重采样避免摩尔纹。
        let 超分插值: Image.Interpolation = 图宽 > CGFloat(产物.输出宽) ? .none : .high
        let 原图插值: Image.Interpolation = 图宽 > CGFloat(产物.输入宽) ? .none : .high

        let 指针 = 悬停点 ?? CGPoint(x: 容器.width / 2, y: 容器.height / 2)
        let 分割 = max(0, min(指针.x, 容器.width))

        // 以指针为锚点放大：指针指向的那一点在屏幕上的位置保持不变
        let 原点X = 指针.x - (指针.x - 适配.minX) * 对比缩放
        let 原点Y = 指针.y - (指针.y - 适配.minY) * 对比缩放

        // 两层都铺满整个容器（position 让图片自身尺寸不参与容器布局）。
        // 左右口径与视频统一：右侧恒为超分成片，左侧为源素材 —— 因此裁左的是原帧层，且它压在最上面。
        let 超分层 = ZStack(alignment: .topLeading) {
            Image(nsImage: 产物.输出图)
                .resizable()
                .interpolation(超分插值)
                .scaledToFill()
                .frame(width: 图宽, height: 图高)
                .position(x: 原点X + 图宽 / 2, y: 原点Y + 图高 / 2)
        }
        .frame(width: 容器.width, height: 容器.height)

        let 原帧层 = ZStack(alignment: .topLeading) {
            Image(nsImage: 原图)
                .resizable()
                .interpolation(原图插值)
                .scaledToFill()
                .frame(width: 图宽, height: 图高)
                .position(x: 原点X + 图宽 / 2, y: 原点Y + 图高 / 2)
        }
        .frame(width: 容器.width, height: 容器.height)
        .mask(alignment: .topLeading) {
            Rectangle().frame(width: 分割, height: 容器.height)
        }

        return ZStack(alignment: .topLeading) {
            超分层
            原帧层

            Rectangle()
                .fill(Color.white.opacity(0.85))
                .frame(width: 1.5, height: 容器.height)
                .offset(x: 分割)
                .shadow(color: .black.opacity(0.4), radius: 4)

            VStack {
                HStack(spacing: 6) {
                    对比标签((超分.视频产物 == nil ? "原图 " : "源帧 ") + 产物.输入尺寸文本)
                    Spacer(minLength: 0)
                    // 右侧恒为超分成片，报它的原生输出尺寸；左侧报源素材自身尺寸。
                    对比标签("超分 " + 产物.输出尺寸文本)
                }
                Spacer(minLength: 0)
                HStack(spacing: 6) {
                    对比标签(缩放文本(基础缩放: 基础缩放))
                    Spacer(minLength: 0)
                    if 对比缩放 > 1.02 {
                        对比标签("滚轮缩放 · 指针处放大")
                    }
                }
            }
            .padding(10)
            .allowsHitTesting(false)
        }
        .frame(width: 容器.width, height: 容器.height)
        .clipped()
    }

    /// 左下角倍率文案：整图适配 / ×N / 1:1（屏幕 1 点正好 1 个超分像素）。
    private func 缩放文本(基础缩放: CGFloat) -> String {
        if 对比缩放 <= 1.02 { return "整图适配" }
        let 每像素点 = 基础缩放 * 对比缩放
        let 倍率 = String(format: "×%.2f", 对比缩放)
        return abs(每像素点 - 1) < 0.05 ? "1:1 · " + 倍率 : 倍率
    }

    private func 对比标签(_ 文本: String) -> some View {
        Text(文本)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(Theme.textSecondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(Theme.panelBackground.opacity(0.85))
            )
    }

    /// 等比适配的可视矩形（scaledToFit 之后图像真实占位）。
    private static func 适配矩形(比例: CGFloat, 容器: CGSize) -> CGRect {
        guard 容器.width > 0, 容器.height > 0, 比例 > 0 else { return .zero }
        let 容器比例 = 容器.width / 容器.height
        if 比例 > 容器比例 {
            let 宽 = 容器.width
            let 高 = 宽 / 比例
            return CGRect(x: 0, y: (容器.height - 高) / 2, width: 宽, height: 高)
        }
        let 高 = 容器.height
        let 宽 = 高 * 比例
        return CGRect(x: (容器.width - 宽) / 2, y: 0, width: 宽, height: 高)
    }

    // MARK: 元数据

    private func 元数据(_ 条目: 素材条目) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 18) {
                信息项(标题: "类型", 值: 条目.副标题文本)
                信息项(标题: "时长", 值: 条目.时长文本)
                信息项(标题: "分辨率", 值: 条目.分辨率文本)
                信息项(标题: "帧率", 值: 条目.帧率文本)
                信息项(标题: "体积", 值: 条目.体积文本)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, Theme.padding)
            .padding(.vertical, 10)

            HStack(spacing: 6) {
                Image(systemName: "folder")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.textTertiary)
                Text(条目.url.path)
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
                Button("在访达中显示") {
                    NSWorkspace.shared.activateFileViewerSelecting([条目.url])
                }
                .buttonStyle(.plain)
                .font(.system(size: 10))
                .foregroundStyle(Theme.textSecondary)
            }
            .padding(.horizontal, Theme.padding)
            .padding(.bottom, 10)
        }
        .background(Theme.panelBackground)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.divider).frame(height: Theme.hairline)
        }
    }

    private func 信息项(标题: String, 值: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(标题)
                .font(.system(size: 10))
                .foregroundStyle(Theme.textTertiary)
            Text(值)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.textPrimary)
        }
    }

    // MARK: 超分控制条

    private var 超分控制: some View {
        HStack(spacing: 10) {
            小徽章(文本: 超分.启用Wan ? "Wan + Flash · 4×" : "Flash · 4×",
                   颜色: 超分.阶段 == .失败 ? Theme.danger : Theme.accent)

            VStack(alignment: .leading, spacing: 2) {
                Text(权重下载管理.共享.下载中 ? "正在下载权重" : 超分.状态文本)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(超分.阶段 == .失败 ? Theme.danger : Theme.textPrimary)

                Text(副信息)
                    .font(.system(size: 10))
                    .foregroundStyle(超分.阶段 == .失败 ? Theme.danger : Theme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer(minLength: 8)

            if 超分.运行中 {
                ProgressView().controlSize(.small)
            }

            if let 成片 = 超分.视频产物 {
                Button("在访达中显示") {
                    NSWorkspace.shared.activateFileViewerSelecting([成片.输出URL])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("成片信息") {
                    提示条 = 成片.尺寸文本 + " · " + 成片.体积文本
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else if 超分.产物 != nil {
                if let 文件 = 超分.单帧文件 {
                    Button("在访达中显示") {
                        NSWorkspace.shared.activateFileViewerSelecting([文件])
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Button("保存 PNG") {
                    if let 路径 = 超分.保存() {
                        提示条 = "已保存 " + 路径.lastPathComponent
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // 取消只属于「正在被引擎跑的那一条」：按钮按本条素材 id 判，跟全局 视频运行 无关，
            // 别人的素材在跑不会在本页冒出取消按钮。
            if 超分.本条在跑 {
                Button(超分.取消中 ? "取消中…" : "取消") {
                    if let 目标 = 素材.选中 { 超分.取消(素材id: 目标) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(超分.取消中)
            }

            // 本条已在队里等着：单独的撤回入口，不连累正在跑的那条。
            if 超分.本条排队中, let 待撤 = 素材.选中 {
                Button("取消排队") { 超分.取消排队(待撤) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }

            Button(超分.按钮标题) { 开始超分() }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                // 忙的时候也要能点：点下去是排队，不是没反应。只有本条自己在跑、或本条已经在队里才禁用。
                .disabled(超分.本条在跑 || 超分.本条排队中 || 权重下载管理.共享.下载中 || !可开始当前素材)
        }
        .padding(.horizontal, Theme.padding)
        .padding(.vertical, 10)
        .background(Theme.inspectorBackground)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.divider).frame(height: Theme.hairline)
        }
    }

    private var 副信息: String {
        let 下载 = 权重下载管理.共享
        if 下载.下载中 { return 下载.状态文本 }
        if let 错误 = 下载.错误提示, 超分.阶段 != .失败 {
            return "权重下载失败：" + 错误 + " · 再点一次「开始超分」重试"
        }
        if 超分.阶段 == .失败, let 提示 = 超分.提示 { return 提示 }
        if let 产物 = 超分.产物 {
            // 两段耗时分别标出：Wan 出条件 latent、Flash 4× 解码，避免被误读成只有 Flash 在跑。
            var 段: [String] = []
            if let wan = 超分.wan耗时毫秒 { 段.append(String(format: "Wan %.0f ms", wan)) }
            段.append("Flash " + 产物.耗时文本)
            // 三段尺寸口径分开写清：输入素材原生尺寸 · 送模型的低清尺寸 → 4× 输出尺寸。
            var 文本 = 段.joined(separator: " + ") + " · "
                + 产物.输入尺寸文本 + " · 低清 " + 产物.低清尺寸文本
                + " → " + 产物.输出尺寸文本
            if let 提示条 { 文本 += " · " + 提示条 }
            return 文本
        }
        if 超分.运行中 { return "首次运行会先加载 Wan 5B 与 Flash 权重" }
        if !超分.可运行 {
            let 缺 = 超分.缺权重提示 ?? "缺少权重"
            return 缺 + " · 点「开始超分」自动从 ModelScope 下载到 " + 文档目录.模型目录.path
        }
        return "Wan 5B 单步出条件 → Flash 4× 解码 · 低清短边上限 " + String(超分.输入短边)
    }

    private var 可开始当前素材: Bool {
        素材.选中条目 != nil
    }

    /// 成片已生成时播放成片，否则播源片。
    private func 播放地址(_ 条目: 素材条目) -> URL {
        超分.视频产物?.输出URL ?? 条目.url
    }

    /// 视频走整段逐帧超分 + 编码落盘；图片按原文件取全分辨率位图走单帧通道并自动落成 PNG。
    /// 进流程前先做权重就绪检查：只下载缺失的那个（wan / flash），
    /// 下载期间只展示进度、不进入超分，下载完成后自动续跑本次超分。
    private func 开始超分() {
        提示条 = nil
        guard let 条目 = 素材.选中条目 else { return }
        超分素材 = 条目.id

        let 需要 = RelayVSR权重.需要(启用Wan: 超分.启用Wan)
        let 预览帧 = 素材.预览帧
        权重下载管理.共享.确保权重(需要) { [超分] in
            超分.开始(素材: 条目, 预览帧: 预览帧)
        }
    }
}

/// 预览区滚轮探针：只在「可对比」时消费滚轮事件做放大/还原，其余情况原样放行，不影响其它区域滚动。
private struct 滚轮缩放: NSViewRepresentable {
    let 启用: () -> Bool
    let 处理: (CGFloat) -> Void

    func makeNSView(context: Context) -> 探针 {
        let 视图 = 探针()
        视图.启用 = 启用
        视图.处理 = 处理
        return 视图
    }

    func updateNSView(_ nsView: 探针, context: Context) {
        nsView.启用 = 启用
        nsView.处理 = 处理
    }

    final class 探针: NSView {
        var 启用: (() -> Bool)?
        var 处理: ((CGFloat) -> Void)?
        private var 监听: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            移除监听()
            guard window != nil else { return }
            监听 = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] 事件 in
                guard let self, let 启用 = self.启用, 启用() else { return 事件 }
                // 触控板给的是精确增量（量级小），普通滚轮一格约 1～3，统一换算成同一尺度
                let 增量 = 事件.hasPreciseScrollingDeltas ? 事件.scrollingDeltaY : 事件.scrollingDeltaY * 6
                self.处理?(增量)
                return nil
            }
        }

        deinit {
            if let 监听 { NSEvent.removeMonitor(监听) }
        }

        private func 移除监听() {
            if let 监听 { NSEvent.removeMonitor(监听) }
            监听 = nil
        }
    }
}
