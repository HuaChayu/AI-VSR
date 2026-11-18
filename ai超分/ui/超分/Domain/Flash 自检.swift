import AppKit
import Metal
import Observation

/// App 内 Flash 主干自检：用与 Python 参考完全相同的输入跑一帧，
/// 把像素结果与参考对拍，确认 Swift + Metal 移植与参考实现一致。
@MainActor
@Observable
final class Flash自检模型 {
    struct 结果 {
        var 耗时毫秒: Double
        var psnr: Double
        var 余弦: Double
        var 参考图: NSImage?
        var 实测图: NSImage?
        var 说明: String
    }

    var 运行中 = false
    var 状态 = "尚未运行"
    var 耗时文本 = "—"
    var 保真文本 = "—"
    var 提示: String?
    var 参考图: NSImage?
    var 实测图: NSImage?

    /// 权重与参考张量落点：~/Documents/AI超分/model/（参考张量在 ref 子目录）
    static let 权重路径 = 文档目录.模型路径("flash.safetensors")
    static let 张量路径 = 文档目录.模型路径("ref/flash_selftest.safetensors")

    var 可运行: Bool {
        FileManager.default.fileExists(atPath: Self.权重路径)
            && FileManager.default.fileExists(atPath: Self.张量路径)
    }

    func 运行() {
        guard !运行中, 可运行 else { return }
        运行中 = true
        提示 = nil
        状态 = "正在加载权重与算子…"

        Task.detached(priority: .userInitiated) {
            do {
                let 结果 = try Self.推理()
                await MainActor.run {
                    self.耗时文本 = String(format: "%.0f ms", 结果.耗时毫秒)
                    self.保真文本 = String(format: "PSNR %.2f dB · 余弦 %.6f", 结果.psnr, 结果.余弦)
                    self.参考图 = 结果.参考图
                    self.实测图 = 结果.实测图
                    self.状态 = 结果.psnr >= 38 ? "与参考一致" : "与参考有偏差"
                    self.提示 = 结果.说明
                    self.运行中 = false
                    self.落盘("自检 \(self.状态) | \(self.耗时文本) | \(self.保真文本) | \(结果.说明)")
                }
            } catch {
                await MainActor.run {
                    self.状态 = "失败"
                    self.提示 = "\(error)"
                    self.运行中 = false
                    self.落盘("自检失败：\(error)")
                }
            }
        }
    }

    /// 自动化验收用：把结果同时写进固定路径的日志文件。
    private func 落盘(_ 文本: String) {
        print("[FLASH自检] " + 文本)
        let 目录 = 文档目录.模型路径("ref")
        try? FileManager.default.createDirectory(atPath: 目录, withIntermediateDirectories: true)
        try? (文本 + "\n").write(toFile: 目录 + "/selftest_result.txt",
                                atomically: true,
                                encoding: .utf8)
    }

    // MARK: - 推理（后台线程）

    nonisolated private static func 推理() throws -> 结果 {
        guard let 设备 = MTLCreateSystemDefaultDevice() else {
            throw 自检错误.环境("找不到 Metal 设备")
        }
        let 引擎 = try Flash引擎(设备: 设备, 权重路径: 权重路径)
        let 张量 = try SafeTensors权重库(路径: 张量路径)

        let lr = try 张量.Float32张量("in.lr")
        let 起始 = try 张量.Float32张量("in.cond")
        let 结束 = try 张量.Float32张量("in.end_cond")
        let 参考像素 = try 张量.Float32张量("out.pixels")

        func 上传(_ 值: [Float]) throws -> MTLBuffer {
            guard let 缓冲 = 设备.makeBuffer(length: 值.count * MemoryLayout<Float>.stride,
                                          options: .storageModeShared) else {
                throw 自检错误.环境("无法分配显存缓冲")
            }
            memcpy(缓冲.contents(), 值, 值.count * MemoryLayout<Float>.stride)
            return 缓冲
        }

        let 高 = lr.形状[2]
        let 宽 = lr.形状[3]
        let 输入 = Flash引擎.帧输入(低清: try 上传(lr.数值),
                                  高: 高,
                                  宽: 宽,
                                  起始条件: try 上传(起始.数值),
                                  结束条件: try 上传(结束.数值))

        let 开始 = Date()
        let 输出 = try 引擎.前向(输入)
        let 耗时 = Date().timeIntervalSince(开始) * 1000

        let 像素指针 = 输出.像素.contents().bindMemory(to: Float.self,
                                                     capacity: 3 * 输出.高 * 输出.宽)
        let 实测 = Array(UnsafeBufferPointer(start: 像素指针, count: 3 * 输出.高 * 输出.宽))
        let 参考 = 参考像素.数值
        guard 参考.count == 实测.count else {
            throw 自检错误.环境("像素数量不一致：参考 \(参考.count) / 实测 \(实测.count)")
        }

        var 平方差 = 0.0
        var 点积 = 0.0
        var 参考平方 = 0.0
        var 实测平方 = 0.0
        for i in 0..<参考.count {
            let a = Double(参考[i])
            let b = Double(实测[i])
            let d = a - b
            平方差 += d * d
            点积 += a * b
            参考平方 += a * a
            实测平方 += b * b
        }
        let 均方误差 = 平方差 / Double(参考.count) / 4.0   // 值域 [-1,1] → 归一化后差缩小 4 倍
        let psnr = 10 * log10(1.0 / max(均方误差, 1e-12))
        let 余弦 = 点积 / max(sqrt(参考平方 * 实测平方), 1e-20)

        let 参考图 = 图像(参考, 宽: 输出.宽, 高: 输出.高)
        let 实测图 = 图像(实测, 宽: 输出.宽, 高: 输出.高)

        return 结果(耗时毫秒: 耗时,
                  psnr: psnr,
                  余弦: 余弦,
                  参考图: 参考图,
                  实测图: 实测图,
                  说明: "输入 \(宽)×\(高) → 输出 \(输出.宽)×\(输出.高)，单帧 \(String(format: "%.0f", 耗时)) ms")
    }

    /// NCHW float [-1,1] → NSImage
    nonisolated private static func 图像(_ 值: [Float], 宽: Int, 高: Int) -> NSImage? {
        guard 值.count >= 3 * 宽 * 高, 宽 > 0, 高 > 0 else { return nil }
        var 字节 = [UInt8](repeating: 255, count: 宽 * 高 * 4)
        let 平面 = 宽 * 高
        for i in 0..<平面 {
            for c in 0..<3 {
                let v = 值[c * 平面 + i] * 0.5 + 0.5
                字节[i * 4 + c] = UInt8(max(0, min(255, v * 255)))
            }
        }
        let 空间 = CGColorSpaceCreateDeviceRGB()
        return 字节.withUnsafeMutableBytes { 缓冲 -> NSImage? in
            guard let 上下文 = CGContext(data: 缓冲.baseAddress,
                                     width: 宽,
                                     height: 高,
                                     bitsPerComponent: 8,
                                     bytesPerRow: 宽 * 4,
                                     space: 空间,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
                  let 图 = 上下文.makeImage() else { return nil }
            return NSImage(cgImage: 图, size: NSSize(width: 宽, height: 高))
        }
    }
}

enum 自检错误: Error, CustomStringConvertible {
    case 环境(String)

    var description: String {
        switch self {
        case let .环境(说明): 说明
        }
    }
}
