import Foundation
import Metal

/// Wan 主干常驻宿主：权重只加载一次（fp16 约 5.03 GB），产出 Flash 所需的端点条件 latent。
nonisolated final class Wan宿主: @unchecked Sendable {
    static let 共享 = Wan宿主()
    /// 权重落点：~/Documents/AI超分/model/wan.safetensors（fp16 缓存同名同目录，见 Wan 引擎.缓存路径）
    static let 权重路径 = 文档目录.模型路径("wan.safetensors")
    /// 单步流式采样（RelayVSR 官方设定）：timesteps = [1000]、无 CFG。
    static let 噪声种子: UInt64 = 20251031

    private let 锁 = NSLock()
    private var 引擎: Wan引擎?
    private var 设备: MTLDevice?

    static var 权重就绪: Bool {
        FileManager.default.fileExists(atPath: 权重路径)
    }

    /// 初次加载权重（约 5–8 秒，10 GB 读盘 + GPU 转换），供界面预热调用。
    func 预热() throws {
        锁.lock()
        defer { 锁.unlock() }
        _ = try 取引擎()
    }

    private func 取引擎() throws -> (Wan引擎, MTLDevice) {
        if let 引擎, let 设备 { return (引擎, 设备) }
        guard Self.权重就绪 else {
            throw 超分错误(说明: "找不到 Wan 权重：\(Self.权重路径)")
        }
        guard let 设备 = MTLCreateSystemDefaultDevice() else {
            throw 超分错误(说明: "找不到 Metal 设备")
        }
        let 引擎 = try Wan引擎(设备: 设备, 权重路径: Self.权重路径)
        self.设备 = 设备
        self.引擎 = 引擎
        return (引擎, 设备)
    }

    /// 低清 RGBA8（sRGB，行 0 = 顶部，4 字节/像素）→ 端点条件 latent [48, 高/4, 宽/4]（float，planar）。
    ///
    /// 流程与官方 streaming.py 一致：低清先 4× 双线性上采样到输出分辨率，再 16× PixelUnshuffle
    /// 进 LQ4xProj，故 latent = 4×低清/16 = 低清/4、token = latent/2 = 低清/8（引擎内已按此口径推导）。
    /// 内核吃的是逐通道 planar RGB，故此处先把 RGBA8 去交织成 3 平面（不能用 RGBA8 直接当 planar 用）。
    /// 确定性噪声 latent 经单步 timestep=1000 前向得速度预测，超分 latent = 噪声 − 预测。
    /// 该结果正是 Flash 在 endpoints 模式下需要的条件，非零条件。
    func 条件(低清: [UInt8], 宽: Int, 高: Int) throws -> (潜变量: [Float], 高: Int, 宽: Int, 耗时毫秒: Double) {
        锁.lock()
        defer { 锁.unlock() }

        // latent = 低清/4、token = 低清/8 都要整除（官方 stride = 16 × max(patch)/4 = 8）。
        // 尺寸收敛统一由 超分通道.目标尺寸 完成（任意输入视频都自动等比缩放到 8 的倍数），
        // 这里的校验只是内部不变量断言，正常调用不会触发。
        guard 宽 > 0, 高 > 0, 宽 % 8 == 0, 高 % 8 == 0 else {
            throw 超分错误(说明: "内部尺寸未对齐 8（\(宽)×\(高)），请检查 超分通道.目标尺寸")
        }
        guard 低清.count >= 4 * 宽 * 高 else {
            throw 超分错误(说明: "低清缓冲长度不足（需 RGBA8，\(宽)×\(高)×4 字节）")
        }

        let 开始 = Date()
        let (引擎, 设备) = try 取引擎()

        let 潜高 = 高 / 4
        let 潜宽 = 宽 / 4
        let 潜元素 = 48 * 潜高 * 潜宽

        guard let 潜变量缓冲 = 设备.makeBuffer(length: 潜元素 * 2, options: .storageModeShared) else {
            throw 超分错误(说明: "无法分配 latent 缓冲")
        }
        let 潜指针 = 潜变量缓冲.contents().assumingMemoryBound(to: Float16.self)
        var 状态 = Self.噪声种子
        var 缓存: Float = 0
        var 有缓存 = false
        for i in 0..<潜元素 {
            var 值: Float
            if 有缓存 {
                值 = 缓存
                有缓存 = false
            } else {
                var u1: Float = 0
                repeat {
                    状态 = 状态 &* 6364136223846793005 &+ 1442695040888963407
                    u1 = Float(UInt32(truncatingIfNeeded: 状态 >> 32) & 0xFFFFFF) / Float(1 << 24)
                } while u1 <= 1e-7
                状态 = 状态 &* 6364136223846793005 &+ 1442695040888963407
                let u2 = Float(UInt32(truncatingIfNeeded: 状态 >> 32) & 0xFFFFFF) / Float(1 << 24)
                let 半径 = sqrt(-2.0 * log(u1))
                值 = 半径 * cos(2.0 * Float.pi * u2)
                缓存 = 半径 * sin(2.0 * Float.pi * u2)
                有缓存 = true
            }
            潜指针[i] = Float16(值)
        }

        guard let 低清缓冲 = 设备.makeBuffer(length: 3 * 宽 * 高, options: .storageModeShared) else {
            throw 超分错误(说明: "无法分配低清缓冲")
        }
        let 平面 = 宽 * 高
        低清.withUnsafeBytes { raw in
            guard let 源 = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let 目标 = 低清缓冲.contents().assumingMemoryBound(to: UInt8.self)
            for c in 0..<3 {
                let 基 = 目标 + c * 平面
                for i in 0..<平面 {
                    基[i] = 源[i * 4 + c]
                }
            }
        }

        let 预测 = try 引擎.前向(潜变量: 潜变量缓冲, 低清: 低清缓冲, 源高: 高, 源宽: 宽)

        let 预测指针 = 预测.contents().assumingMemoryBound(to: Float16.self)
        var 结果 = [Float](repeating: 0, count: 潜元素)
        for i in 0..<潜元素 {
            结果[i] = Float(潜指针[i]) - Float(预测指针[i])
        }

        return (结果, 潜高, 潜宽, Date().timeIntervalSince(开始) * 1000)
    }
}
