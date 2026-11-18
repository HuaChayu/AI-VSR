import SwiftUI
import AppKit

@main
struct MyApp: App {
    @NSApplicationDelegateAdaptor(应用委托.self) private var 委托

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 1280, height: 820)

        // 偏好设置：菜单栏应用菜单自动出现「偏好设置…」（⌘,），打开 偏好设置.swift 的 PreferencesView
        Settings {
            PreferencesView()
        }
    }
}

final class 应用委托: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // 首次启动：幂等创建 ~/Documents/AI超分 下的「输入输出」「超分输出」「model」「cache」
        文档目录.确保存在()
        // 空格全局播放：整个 App 生命周期只装这一次监听，退出时在 applicationWillTerminate 移除
        全局快捷键.共享.安装()
    }

    func applicationWillTerminate(_ notification: Notification) {
        全局快捷键.共享.移除()
    }
}
