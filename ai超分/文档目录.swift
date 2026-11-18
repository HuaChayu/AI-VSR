// ============================================================
//  文件作用：项目文档目录定义与首次启动初始化。
//  根目录固定在 ~/Documents/AI超分，下含四个子目录：
//  「输入输出」「超分输出」「model」「cache」。
//  初始化幂等：目录已存在则跳过，绝不覆盖或清空已有内容。
//  互动文件：MyApp.swift（启动时调用 确保存在()）；Wan 宿主.swift、
//  超分 通道.swift、Flash 自检.swift（权重与参考张量路径取自 模型目录）。
// ============================================================

import Foundation

enum 文档目录 {
    /// 根目录名（位于用户的「文档」下）
    nonisolated static let 根目录名 = "AI超分"

    /// 四个子目录名，顺序即创建顺序
    nonisolated static let 子目录名 = ["输入输出", "超分输出", "model", "cache"]

    /// 权重目录名：所有模型权重与参考张量都放在这里
    nonisolated static let 模型子目录名 = "model"

    nonisolated static var 根目录: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/Documents/" + 根目录名, isDirectory: true)
    }

    nonisolated static var 输入输出目录: URL { 根目录.appendingPathComponent("输入输出", isDirectory: true) }
    nonisolated static var 超分输出目录: URL { 根目录.appendingPathComponent("超分输出", isDirectory: true) }
    nonisolated static var 模型目录: URL { 根目录.appendingPathComponent(模型子目录名, isDirectory: true) }
    nonisolated static var 缓存目录: URL { 根目录.appendingPathComponent("cache", isDirectory: true) }

    /// 全部目录（根 + 四个子目录），供初始化遍历
    nonisolated static var 全部目录: [URL] {
        [根目录] + 子目录名.map { 根目录.appendingPathComponent($0, isDirectory: true) }
    }

    /// 模型目录下某个文件的路径（如 "wan.safetensors"、"ref/flash_selftest.safetensors"）
    nonisolated static func 模型文件(_ 相对路径: String) -> URL {
        模型目录.appendingPathComponent(相对路径)
    }

    /// 模型目录下某个文件的绝对路径字符串（现有代码按字符串拼接使用）
    nonisolated static func 模型路径(_ 相对路径: String) -> String {
        模型文件(相对路径).path
    }

    /// 首次启动初始化：幂等创建根目录与四个子目录。
    /// 已存在的目录保持原样，不删除、不搬迁、不覆盖任何已有文件。
    /// - Returns: 是否全部目录均可用（存在且为目录）。
    @discardableResult
    nonisolated static func 确保存在() -> Bool {
        let 文件管理 = FileManager.default
        var 全部可用 = true
        for 目录 in 全部目录 {
            var 是目录: ObjCBool = false
            if 文件管理.fileExists(atPath: 目录.path, isDirectory: &是目录) {
                if 是目录.boolValue { continue }
                // 同名文件占位：不覆盖，仅记录并继续
                全部可用 = false
                print("[文档目录] 已存在同名文件，跳过创建：" + 目录.path)
                continue
            }
            do {
                try 文件管理.createDirectory(at: 目录, withIntermediateDirectories: true)
                print("[文档目录] 创建：" + 目录.path)
            } catch {
                全部可用 = false
                print("[文档目录] 创建失败：" + 目录.path + " → " + error.localizedDescription)
            }
        }
        return 全部可用
    }
}
