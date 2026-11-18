// ============================================================
//  文件作用：App 内全局快捷键——空格播放 / 暂停当前素材预览。
//  用 NSEvent.addLocalMonitorForEvents(matching: .keyDown) 在事件进入响应链之前拦截，
//  因此焦点在任意界面（素材栏、空白处、其它 Tab 之外的本窗口区域）都能生效；
//  唯一例外是当前第一响应者是文本输入（NSTextView 字段编辑器 / NSTextField / 搜索框），
//  此时必须放行，否则搜索框里打不出空格。
//  ⚠️ 但 SwiftUI 的 TextField 一旦编辑过，字段编辑器会一直霸占 firstResponder：
//  点列表、预览区这些不接收键盘焦点的区域并不会让它 resign，于是空格永远被当成文字输入。
//  所以另装一个鼠标按下监听：按点不在输入框内就主动收回窗口焦点（ESC 同样退出输入）。
//  生命周期：只在 App 启动时装一次监听，退出时移除（见 MyApp.swift 的应用委托）。
//  互动文件：MyApp.swift（安装 / 移除）、素材 详情.swift（挂载播放处理）。
// ============================================================

import AppKit

@MainActor
final class 全局快捷键 {
    static let 共享 = 全局快捷键()

    /// 当前挂载的播放处理：返回 true 表示已消费这次空格，false 表示放行（例如静态图片没有可播放内容）。
    private var 切换播放: (() -> Bool)?
    /// 挂载者标识，避免 A 视图卸载时把 B 视图刚挂上的处理摘掉。
    private var 挂载标识: ObjectIdentifier?
    private var 键盘监听: Any?
    private var 鼠标监听: Any?
    private var 已安装 = false

    // MARK: 生命周期

    /// 安装空格监听（幂等：重复调用不会重复装）。
    func 安装() {
        guard !已安装 else { return }
        已安装 = true
        键盘监听 = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] 事件 in
            guard let self else { return 事件 }
            return self.处理(事件)
        }
        鼠标监听 = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] 事件 in
            guard let self else { return 事件 }
            self.处理点击(事件)
            return 事件
        }
    }

    /// 移除监听（App 退出时调用）。
    func 移除() {
        if let 键盘监听 { NSEvent.removeMonitor(键盘监听) }
        if let 鼠标监听 { NSEvent.removeMonitor(鼠标监听) }
        键盘监听 = nil
        鼠标监听 = nil
        已安装 = false
        切换播放 = nil
        挂载标识 = nil
    }

    // MARK: 挂载

    /// 由当前显示预览的视图挂上「空格做什么」。
    func 挂载(标识: ObjectIdentifier, 处理: @escaping () -> Bool) {
        挂载标识 = 标识
        切换播放 = 处理
    }

    /// 视图消失时摘掉；只摘自己挂的那一份。
    func 摘下(标识: ObjectIdentifier) {
        guard 挂载标识 == 标识 else { return }
        挂载标识 = nil
        切换播放 = nil
    }

    // MARK: 事件处理

    /// 空格 → 播放 / 暂停。返回 nil 表示已消费（不再下发），返回事件表示放行。
    private func 处理(_ 事件: NSEvent) -> NSEvent? {
        // ESC：正在输入时退出输入，把键盘交还给快捷键
        if 事件.keyCode == 53,
           let 窗口 = 事件.window,
           Self.文本输入拥有焦点(窗口) {
            窗口.makeFirstResponder(nil)
            return nil
        }
        guard 事件.keyCode == 49 else { return 事件 }        // 49 = 空格
        // 带修饰键的空格另有含义（⌘空格切换输入法、⇧空格等），一律不拦
        let 修饰 = 事件.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard 修饰.isEmpty else { return 事件 }
        // 文本输入有焦点：放行，让空格进到输入框
        guard !Self.文本输入拥有焦点(事件.window) else { return 事件 }
        guard let 切换播放, 切换播放() else { return 事件 }
        return nil
    }

    /// 鼠标按下：按点不在输入框上时，主动收回窗口焦点（即「输入框以外的地方都算失焦」）。
    /// 必须这么做——macOS 的 SwiftUI TextField 编辑后，字段编辑器（NSTextView）会一直占着
    /// firstResponder，点列表、预览区这些不接收键盘焦点的区域不会让它 resign。
    private func 处理点击(_ 事件: NSEvent) {
        guard let 窗口 = 事件.window,
              let 响应者 = 窗口.firstResponder,
              Self.是文本控件(响应者) else { return }
        let 点 = 事件.locationInWindow
        // 字段编辑器的 delegate 一般是真正那个 NSTextField，以它的 frame 为准（编辑器 frame 可能被放大）
        if let 编辑器 = 响应者 as? NSTextView,
           let 文本框 = 编辑器.delegate as? NSView {
            if Self.命中(文本框, 窗口点: 点) { return }
            窗口.makeFirstResponder(nil)
            return
        }
        if Self.命中(响应者 as? NSView, 窗口点: 点) { return }
        窗口.makeFirstResponder(nil)
    }

    /// 第一响应者是文本输入控件时返回 true。
    /// SwiftUI 的 TextField / 搜索框在 macOS 上都由字段编辑器（NSTextView）持有焦点，故先判 NSTextView。
    nonisolated static func 文本输入拥有焦点(_ 窗口: NSWindow?) -> Bool {
        guard let 窗口, let 响应者 = 窗口.firstResponder else { return false }
        return 是文本控件(响应者)
    }

    /// 响应者本身是文本输入（或其字段编辑器）。
    private nonisolated static func 是文本控件(_ 响应者: NSResponder) -> Bool {
        if 响应者 is NSTextView { return true }
        if 响应者 is NSTextField { return true }
        if 响应者 is NSText { return true }
        return false
    }

    /// 窗口坐标点是否落在该视图的可见范围内。
    private nonisolated static func 命中(_ 视图: NSView?, 窗口点: NSPoint) -> Bool {
        guard let 视图 else { return false }
        return 视图.convert(视图.bounds, to: nil).contains(窗口点)
    }
}
