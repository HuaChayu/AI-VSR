import SwiftUI

struct 超分工作区: View {
    @Environment(素材Store.self) private var 素材
    @State private var 自检 = Flash自检模型()
    @State private var 超分 = 超分通道()

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                素材栏()
                    .frame(width: 272)
                细线()
                素材详情()
            }

            Flash自检条(模型: 自检)
        }
        .environment(超分)
        .background(Theme.canvasBackground)
        .task {
            // 便于自动化验收：FLASH_UPSCALE=1 启动时自动跑一帧真实超分并落盘
            if ProcessInfo.processInfo.environment["FLASH_UPSCALE"] == "1" {
                await 跑超分验收()
            }
            // 便于自动化验收：UPSAMPLE_VIDEO=1 启动时自动跑一整段视频超分并落盘
            if ProcessInfo.processInfo.environment["UPSAMPLE_VIDEO"] == "1" {
                await 跑视频验收()
            }
            // 便于自动化验收：UPSAMPLE_IMAGE=<图片路径> 启动时按真实素材走一遍图片超分
            if let 图路径 = ProcessInfo.processInfo.environment["UPSAMPLE_IMAGE"], !图路径.isEmpty {
                await 跑图片验收(路径: 图路径)
            }
            // 便于自动化验收：UPSAMPLE_SEQ="图片,视频,图片" 启动时按序连跑，验证切换素材不复用旧尺寸
            if let 序列 = ProcessInfo.processInfo.environment["UPSAMPLE_SEQ"], !序列.isEmpty {
                await 跑连续验收(序列: 序列)
            }
            // 便于自动化验收：UPSAMPLE_GIF=<动图路径> 走一遍「导入 → 动图超分 → 切走切回恢复」
            if let 动图路径 = ProcessInfo.processInfo.environment["UPSAMPLE_GIF"], !动图路径.isEmpty {
                await 跑动图验收(路径: 动图路径)
            }
            // 便于自动化验收：UPSAMPLE_CACHE=1 验证「超分 → 切走 → 切回」的落盘缓存恢复
            if ProcessInfo.processInfo.environment["UPSAMPLE_CACHE"] == "1" {
                await 跑缓存恢复验收()
            }
            // 便于自动化验收：FLASH_SELFTEST=1 启动时自动跑一帧对拍
            guard ProcessInfo.processInfo.environment["FLASH_SELFTEST"] == "1" else { return }
            自检.运行()
        }
    }

    /// 自动化验收：整段视频逐帧超分 + 编码落盘，输出成片与结果文本。
    private func 跑视频验收(序号: String = "") async {
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        var 文本 = ""
        do {
            try FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
            let 源地址 = ProcessInfo.processInfo.environment["UPSAMPLE_SRC"] ?? (目录 + "/src.mp4")
            let 源 = URL(fileURLWithPath: 源地址)
            // 连续验收时先把上一素材的结果清掉，等价于用户在界面上切换素材
            if !序号.isEmpty { 超分.清空() }
            超分.开始视频(源: 源)
            while 超分.运行中 {
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            if let 成片 = 超分.视频产物 {
                文本 = "视频超分 完成 | " + 成片.尺寸文本 + " | " + 成片.耗时文本
                    + " | " + 成片.体积文本 + " | 音轨" + (成片.含音轨 ? "已保留" : "缺失")
                    + " | " + 成片.输出URL.path
            } else {
                文本 = "视频超分 失败：" + (超分.提示 ?? "未知")
            }
        } catch {
            文本 = "视频超分 异常：\(error)"
        }
        print("[整段超分] " + 文本)
        try? (文本 + "\n").write(toFile: 目录 + "/upscale_video_result" + 序号 + ".txt",
                                atomically: true,
                                encoding: .utf8)
    }

    /// 自动化验收：把一张真实图片当成素材导入，走与「开始超分」按钮完全相同的入口，等完成并记录落地文件。
    private func 跑图片验收(路径: String, 序号: String = "") async {
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        try? FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
        // .task 可能随视图重建重跑；用锁文件确保一次启动只跑一遍
        let 锁文件 = 目录 + "/upscale_image_running" + 序号 + ".lock"
        guard !FileManager.default.fileExists(atPath: 锁文件) else { return }
        FileManager.default.createFile(atPath: 锁文件, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: 锁文件) }
        await 素材.导入(urls: [URL(fileURLWithPath: 路径)])
        guard let 条目 = 素材.选中条目 else {
            print("[图片超分] 导入失败：\(路径)")
            return
        }
        // 选中素材会触发详情视图的「换素材即清空」，等这波清空落地后再发起，避免把刚跑出的结果冲掉
        超分.清空()
        try? await Task.sleep(nanoseconds: 1_200_000_000)
        超分.开始(素材: 条目, 预览帧: 素材.预览帧)
        let 起 = Date()
        while 超分.产物 == nil, 超分.提示 == nil, Date().timeIntervalSince(起) < 900 {
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        var 文本: String
        if let 产物 = 超分.产物 {
            文本 = "图片超分 完成 | 类型" + 条目.类型.标题 + " | 原图 "
                + 产物.输入尺寸文本 + " | 低清 " + 产物.低清尺寸文本 + " -> 输出 " + 产物.输出尺寸文本
                + " | " + 产物.耗时文本
                + " | " + (超分.单帧文件?.path ?? "未落盘")
        } else {
            文本 = "图片超分 失败：" + (超分.提示 ?? "未知")
        }
        print("[图片超分] " + 文本)
        try? (文本 + "\n").write(toFile: 目录 + "/upscale_image_result" + 序号 + ".txt",
                                atomically: true,
                                encoding: .utf8)
    }

    /// 自动化验收：动图（GIF）导入 → 逐帧超分 → GIF 落盘 → 切走切回靠落盘索引恢复对比层。
    /// 走与界面「开始超分」完全相同的 开始(素材:预览帧:) 入口，因此同时验证了分流与批量通路共用的一段代码。
    private func 跑动图验收(路径: String) async {
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        try? FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
        let 锁文件 = 目录 + "/upscale_gif_running.lock"
        guard !FileManager.default.fileExists(atPath: 锁文件) else { return }
        FileManager.default.createFile(atPath: 锁文件, contents: nil)
        defer { try? FileManager.default.removeItem(atPath: 锁文件) }

        await 素材.导入(urls: [URL(fileURLWithPath: 路径)])
        guard let 条目 = 素材.选中条目 else {
            print("[动图超分] 导入失败：\(路径)")
            return
        }
        var 文本 = "导入 | 类型" + 条目.类型.标题 + " | 帧数" + String(条目.帧数 ?? -1)
            + " | 是动图" + (条目.是动图 ? "是" : "否") + " | 副标题「" + 条目.副标题文本 + "」"

        // 选中素材会触发详情视图的「换素材即清空」，等这波清空落地后再发起
        超分.清空()
        try? await Task.sleep(nanoseconds: 1_200_000_000)
        超分.开始(素材: 条目, 预览帧: 素材.预览帧)
        // 自动化验收：UPSAMPLE_GIF_CANCEL=<秒> 在跑动图途中触发取消，验证取消后不残留半成品动图。
        if let 取消秒 = ProcessInfo.processInfo.environment["UPSAMPLE_GIF_CANCEL"],
           let 秒 = Double(取消秒), 秒 > 0 {
            try? await Task.sleep(nanoseconds: UInt64(秒 * 1_000_000_000))
            超分.取消视频()
        }
        let 起 = Date()
        while 超分.运行中, Date().timeIntervalSince(起) < 3_600 {
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        if let 成片 = 超分.视频产物 {
            文本 += "\n超分 | " + 成片.尺寸文本 + " | 帧数" + String(成片.帧数)
                + " | " + 成片.耗时文本 + " | " + 成片.体积文本
                + " | 对比层" + (超分.产物 == nil ? "缺失" : "可滑")
                + " | " + 成片.输出URL.path
        } else {
            文本 += "\n超分 | 失败：" + (超分.提示 ?? "未知")
        }

        // 切走切回：清空后按落盘索引恢复，模拟用户在素材栏切到别的素材再切回来
        let 成片地址 = 超分.视频产物?.输出URL
        let 产物帧数 = 超分.视频产物?.帧数 ?? 0
        超分.清空()
        try? await Task.sleep(nanoseconds: 300_000_000)
        await 超分.恢复缓存(素材: 条目)
        let 恢复 = 超分.视频产物
        文本 += "\n恢复 | 缓存命中" + (超分.缓存命中 ? "是" : "否")
            + " | 产物" + (恢复 == nil ? "缺失" : "在")
            + " | 帧数" + String(恢复?.帧数 ?? 0) + "/" + String(产物帧数)
            + " | 对比层" + (超分.产物 == nil ? "缺失" : "可滑")
            + " | 地址一致" + ((恢复?.输出URL == 成片地址) ? "是" : "否")
            + " | 状态「" + 超分.状态文本 + "」"

        print("[动图超分] " + 文本)
        try? (文本 + "\n").write(toFile: 目录 + "/upsample_gif_result.txt",
                                 atomically: true,
                                 encoding: .utf8)
    }

    /// 自动化验收：按 UPSAMPLE_SEQ 指定的素材顺序连跑（如「图片,视频,图片」），
    /// 每次跑完把结果追加到 upsample_seq.log，用于验证任意顺序切换素材都不复用上一任务的尺寸。
    private func 跑连续验收(序列: String) async {
        let 环境 = ProcessInfo.processInfo.environment
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        try? FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
        let 图一 = 环境["UPSAMPLE_IMAGE"] ?? ""
        let 图二 = 环境["UPSAMPLE_IMAGE2"] ?? 图一
        let 日志 = 目录 + "/upsample_seq.log"
        try? "".write(toFile: 日志, atomically: true, encoding: .utf8)
        var 图序 = 0
        let 步骤 = 序列.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        for (序, 步) in 步骤.enumerated() {
            let 标号 = String(format: "%02d", 序 + 1)
            if 步 == "图片" {
                let 路径 = 图序 == 0 ? 图一 : 图二
                图序 += 1
                if 路径.isEmpty {
                    print("[连续验收] 缺少 UPSAMPLE_IMAGE，跳过第" + 标号 + "步")
                    continue
                }
                await 跑图片验收(路径: 路径, 序号: "_" + 标号)
            } else if 步 == "视频" {
                await 跑视频验收(序号: "_" + 标号)
            } else {
                continue
            }
            let 结果文件 = 目录 + "/upscale_" + (步 == "图片" ? "image" : "video") + "_result_" + 标号 + ".txt"
            let 行 = (try? String(contentsOfFile: 结果文件, encoding: .utf8)) ?? "无结果文件"
            let 汇总 = "第" + 标号 + "步 " + 步 + "：" + 行.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
            print("[连续验收] " + 汇总)
            if let 句柄 = FileHandle(forWritingAtPath: 日志) {
                句柄.seekToEndOfFile()
                句柄.write(汇总.data(using: .utf8) ?? Data())
                try? 句柄.close()
            } else {
                try? 汇总.write(toFile: 日志, atomically: true, encoding: .utf8)
            }
        }
    }

    /// 自动化验收：用参考张量里的真实低清帧走完整超分链路，落盘 PNG 与结果文本。
    private func 跑超分验收() async {
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        var 文本 = ""
        do {
            try FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
            let 帧 = try 超分通道.自检帧()
            // 自检帧短边低于上限时按「只缩不放」原样送入，无需覆盖用户的短边设置
            超分.开始(帧: 帧)
            while 超分.运行中 {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if let 产物 = 超分.产物 {
                let 路径 = 超分.保存(目录: URL(fileURLWithPath: 目录))
                文本 = "超分验收 完成 | " + 产物.输入尺寸文本 + " -> " + 产物.输出尺寸文本
                    + " | " + 产物.耗时文本 + " | " + (路径?.path ?? "保存失败")
            } else {
                文本 = "超分验收 失败：" + (超分.提示 ?? "未知")
            }
        } catch {
            文本 = "超分验收 异常：\(error)"
        }
        print("[FLASH超分] " + 文本)
        try? (文本 + "\n").write(toFile: 目录 + "/upscale_result.txt",
                                atomically: true,
                                encoding: .utf8)
    }

    /// 自动化验收：UPSAMPLE_CACHE=1 时跑「超分 → 切到别的素材 → 切回」链路，
    /// 验证切走后产物被清空、切回时按落盘索引从磁盘恢复对比层。
    /// 素材来源：UPSAMPLE_CACHE_IMAGE（图片）、UPSAMPLE_CACHE_SRC（视频，可缺省）；结果写 upsample_cache.log。
    private func 跑缓存恢复验收() async {
        let 环境 = ProcessInfo.processInfo.environment
        let 目录 = 文档目录.超分输出目录.appendingPathComponent("验收").path
        try? FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
        let 日志 = 目录 + "/upsample_cache.log"
        var 行: [String] = []
        行.append("索引文件 " + 超分缓存索引.索引文件.path)
        // 进程刚启动读到的条目数：> 0 说明索引文件跨重启持久化生效
        行.append("启动时索引条目数=" + String(超分缓存索引.共享.全部().count))

        var 名单: [URL] = []
        if let 图 = 环境["UPSAMPLE_CACHE_IMAGE"], !图.isEmpty { 名单.append(URL(fileURLWithPath: 图)) }
        if let 影 = 环境["UPSAMPLE_CACHE_SRC"], !影.isEmpty { 名单.append(URL(fileURLWithPath: 影)) }
        guard !名单.isEmpty else {
            行.append("缺少 UPSAMPLE_CACHE_IMAGE / UPSAMPLE_CACHE_SRC，验收跳过")
            try? 行.joined(separator: "\n").write(toFile: 日志, atomically: true, encoding: .utf8)
            print("[缓存验收] 缺少素材，跳过")
            return
        }
        await 素材.导入(urls: 名单)
        guard 素材.条目.count >= 1 else {
            行.append("导入失败，条目为空")
            try? 行.joined(separator: "\n").write(toFile: 日志, atomically: true, encoding: .utf8)
            return
        }

        /// 当前对比层对应的产物文件（图片取单帧 PNG，视频取成片 MP4）
        func 当前产物路径() -> String { 超分.视频产物?.输出URL.path ?? 超分.单帧文件?.path ?? "无" }

        for (序, 条目) in 素材.条目.enumerated() {
            let 标号 = String(format: "%02d", 序 + 1)
            // 1) 选中当前素材并跑完超分（与用户点击「开始超分」同一入口）
            素材.选择(条目.id)
            try? await Task.sleep(nanoseconds: 800_000_000)
            超分.清空()
            try? await Task.sleep(nanoseconds: 200_000_000)
            超分.开始(素材: 条目, 预览帧: 素材.预览帧)
            let 起 = Date()
            while 超分.阶段 == .运行 || 超分.视频运行, Date().timeIntervalSince(起) < 1800 {
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            let 自己产物 = 当前产物路径()
            行.append("第" + 标号 + "步 " + 条目.类型.标题 + "《" + 条目.名称 + "》 超分后 产物="
                      + (超分.产物 != nil ? "有" : "无")
                      + " 缓存命中=" + (超分.缓存命中 ? "是" : "否")
                      + " 落盘=" + 自己产物)

            guard 素材.条目.count > 1 else {
                行.append("第" + 标号 + "步 只有一条素材，跳过切换验证")
                continue
            }

            // 2) 切到别的素材：对比层必须换掉（复现原 bug 的前半段）
            let 别的 = 素材.条目[(序 + 1) % 素材.条目.count]
            素材.选择(别的.id)
            // 轮询到对比层确实换掉（视图 onChange 与落盘恢复都在主线程排队，耗时不定）
            var 切走产物 = 当前产物路径()
            let 切走起 = Date()
            while 切走产物 == 自己产物, Date().timeIntervalSince(切走起) < 20 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                切走产物 = 当前产物路径()
            }
            行.append("第" + 标号 + "步 切到 " + 别的.类型.标题 + "《" + 别的.名称 + "》后 产物="
                      + (超分.产物 != nil ? "有" : "无")
                      + " 缓存命中=" + (超分.缓存命中 ? "是" : "否")
                      + " 当前产物=" + 切走产物
                      + " | " + (切走产物 == 自己产物 ? "仍显示上一素材（异常）" : "已切换"))

            // 3) 切回：由 素材详情 的 onChange 触发落盘缓存恢复，应重新具备对比层
            素材.选择(条目.id)
            // 轮询到该素材的产物恢复为止（最多等 20s）
            var 恢复产物 = 当前产物路径()
            let 恢复起 = Date()
            while 恢复产物 != 自己产物, Date().timeIntervalSince(恢复起) < 20 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                恢复产物 = 当前产物路径()
            }
            let 可对比 = 超分.产物 != nil && 恢复产物 == 自己产物
            行.append("第" + 标号 + "步 切回《" + 条目.名称 + "》后 产物="
                      + (可对比 ? "已恢复" : "未恢复")
                      + " 视频产物=" + (超分.视频产物 != nil ? "已恢复" : "无")
                      + " 缓存命中=" + (超分.缓存命中 ? "是" : "否")
                      + " 状态=" + 超分.状态文本
                      + " | " + (可对比 ? "可滑动对比" : "不可对比（失败）"))
        }

        // 4) 移除素材时同步清理索引项（UPSAMPLE_CACHE_PURGE=1 才跑，避免动到真实索引）
        if 环境["UPSAMPLE_CACHE_PURGE"] == "1", let 首条 = 素材.条目.first {
            let 清理前 = 超分缓存索引.共享.全部().count
            素材.移除(首条.id)
            let 清理后 = 超分缓存索引.共享.全部().count
            let 剩余 = 超分缓存索引.共享.条目(源: 首条.url)
            行.append("移除《" + 首条.名称 + "》后 索引条目 " + String(清理前) + " -> " + String(清理后)
                      + " | 该素材索引=" + (剩余 == nil ? "已清理" : "仍存在（异常）"))
        }

        let 文本 = 行.joined(separator: "\n")
        print("[缓存验收]\n" + 文本)
        try? (文本 + "\n").write(toFile: 日志, atomically: true, encoding: .utf8)
    }
}
