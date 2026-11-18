import Foundation
import Metal
import CoreGraphics
import QuartzCore
import ImageIO
import UniformTypeIdentifiers

nonisolated struct 色彩通路错误: LocalizedError {
    let 描述: String
    var errorDescription: String? { 描述 }
}

/// 分段线性化色彩通路。
///
/// 事实依据（2026-11-14 查证）：
/// - Wan 主 DiT 训练域为 sRGB gamma 域（DDA/Wan 官方 encode_rgb_gamma 走 srgb_to_linear 仅用于把
///   gamma 值映射到 0-1，权重端未做真正的传递函数还原）。
/// - Flash 4× VSR（RelayVSR-main/inference.py）训练时输入 uint8/255、输出 mul(255)，
///   配合 streaming.py 的 x.div(127.5).sub(1.0)，全程无 EOTF 线性化，属 sRGB gamma 域。
/// 因此模型段必须保持 sRGB 域：进模型前编码回 sRGB、出模型后解码到线性光。
///
/// 全部中转/调色/缩放都在线性光 fp16（rgba16Float）上完成，8bit 只在外部接口
/// （解码入参 / Wan 兼容接口 / 最终输出）出现，浮点链路内部不再有 8bit 往返。
nonisolated enum 色彩通路 {
    /// 线性光中转格式：fp16 足够宽（iOS/macOS GPU 原生半精度存储），且带宽远低于 fp32。
    static let 光栅格式: MTLPixelFormat = .rgba16Float

    // MARK: - 生命周期与公共工具

    /// 色彩通路自用的 Metal 上下文（与各引擎共用同一物理 GPU，队列独立）。
    static func 上下文() throws -> (MTLDevice, MTLCommandQueue) {
        guard let 设备 = MTLCreateSystemDefaultDevice() else {
            throw 色彩通路错误(描述: "找不到 Metal 设备")
        }
        guard let 队 = 设备.makeCommandQueue() else {
            throw 色彩通路错误(描述: "无法创建命令队列")
        }
        return (设备, 队)
    }

    /// fp32 NCHW 平面数组 → 共享缓冲（模型输出回归线性光用）。
    static func 平面缓冲(_ 值: [Float], 设备: MTLDevice) throws -> MTLBuffer {
        let 字节 = 值.count * MemoryLayout<Float>.stride
        let 缓冲 = try 共享缓冲(字节: 字节, 设备: 设备)
        值.withUnsafeBytes { 源 in
            if let 首 = 源.baseAddress { memcpy(缓冲.contents(), 首, 字节) }
        }
        return 缓冲
    }

    /// 一次性收尾：提交并同步等待，保证 GPU 时间可直接用于同基准测速。
    @discardableResult
    static func 收尾(_ 命令: MTLCommandBuffer) -> Double {
        let 起 = CACurrentMediaTime()
        命令.commit()
        命令.waitUntilCompleted()
        if let 错 = 命令.error {
            FileHandle.standardError.write("色彩通路 GPU 错误：\(错.localizedDescription)\n".data(using: .utf8)!)
        }
        return (CACurrentMediaTime() - 起) * 1000.0
    }

    static func 新命令(_ 队: MTLCommandQueue) throws -> MTLCommandBuffer {
        guard let 命令 = 队.makeCommandBuffer() else {
            throw 色彩通路错误(描述: "无法创建命令缓冲")
        }
        return 命令
    }

    static func 光栅纹理(宽: Int, 高: Int, 设备: MTLDevice,
                          用途: MTLTextureUsage = [.shaderRead, .shaderWrite, .renderTarget]) throws -> MTLTexture {
        let 描述 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: 光栅格式,
                                                         width: max(1, 宽), height: max(1, 高),
                                                         mipmapped: false)
        描述.usage = 用途
        描述.storageMode = .private
        guard let 纹理 = 设备.makeTexture(descriptor: 描述) else {
            throw 色彩通路错误(描述: "无法创建 \(宽)x\(高) 线性光纹理")
        }
        return 纹理
    }

    static func 共享缓冲(字节: Int, 设备: MTLDevice) throws -> MTLBuffer {
        guard let 缓冲 = 设备.makeBuffer(length: max(16, 字节), options: .storageModeShared) else {
            throw 色彩通路错误(描述: "无法创建 \(字节) 字节共享缓冲")
        }
        return 缓冲
    }

    static func 读回(_ 缓冲: MTLBuffer, 字节: Int) -> [UInt8] {
        var 结果 = [UInt8](repeating: 0, count: 字节)
        结果.withUnsafeMutableBytes { 目标 in
            if let 源 = 目标.baseAddress { memcpy(源, 缓冲.contents(), 字节) }
        }
        return 结果
    }

    static func 线程组(_ 管线: MTLComputePipelineState, 宽: Int, 高: Int) -> MTLSize {
        let w = 管线.threadExecutionWidth
        let 高1 = max(1, 管线.maxTotalThreadsPerThreadgroup / w)
        return MTLSize(width: w, height: min(高1, max(1, 高)), depth: 1)
    }

    // MARK: - 1) 解码（8bit 传递函数编码）→ 线性光 fp16

    /// CGImage → 线性光纹理。**只搬运字节，不让 CoreGraphics 做任何色彩空间转换**：
    /// 提取上下文固定取图自身声明的空间，绘制退化为逐字节恒等映射；
    /// 8bit 码值究竟按哪条传递函数解，由 `解码`（源轨声明的传递函数）交给着色器显式处理后决定。
    /// 这样两端曲线严格互逆，避免 CG 对「视频色域」与标准 BT.709 OETF 的解释差异带来的整体偏亮/偏暗。
    static func 线性图(from 图: CGImage, 设备: MTLDevice, 队: MTLCommandQueue,
                    解码: 传递编码 = .sRGB) throws -> MTLTexture {
        let 宽 = 图.width
        let 高 = 图.height
        let 行字节 = 宽 * 4
        var 原始 = [UInt8](repeating: 0, count: max(16, 行字节 * 高))
        let 色彩 = 图.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let 信息 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        原始.withUnsafeMutableBytes { 缓冲 in
            guard let 上下文 = CGContext(data: 缓冲.baseAddress, width: 宽, height: 高,
                                        bitsPerComponent: 8, bytesPerRow: 行字节,
                                        space: 色彩, bitmapInfo: 信息) else { return }
            上下文.interpolationQuality = .high
            上下文.draw(图, in: CGRect(x: 0, y: 0, width: 宽, height: 高))
        }
        let 源 = try 共享缓冲(字节: 原始.count, 设备: 设备)
        memcpy(源.contents(), 原始, 原始.count)
        return try 线性纹理(fromRGBA8: 源, 宽: 宽, 高: 高, 设备: 设备, 队: 队, 解码: 解码)
    }

    /// 交织 RGBA8（按 `解码` 声明的传递函数编码）缓冲 → 线性光纹理。
    static func 线性纹理(fromRGBA8 源: MTLBuffer, 宽: Int, 高: Int,
                    设备: MTLDevice, 队: MTLCommandQueue,
                    解码: 传递编码 = .sRGB) throws -> MTLTexture {
        let 出 = try 光栅纹理(宽: 宽, 高: 高, 设备: 设备)
        let 命 = try 新命令(队)
        guard let 编码 = 命.makeComputeCommandEncoder() else {
            throw 色彩通路错误(描述: "无法创建色阶解码编码器")
        }
        let 管 = try 图形管线.取(设备, 解码 == .bt709 ? "k_bt7098_to_linear" : "k_srgb8_to_linear")
        编码.setComputePipelineState(管)
        编码.setBuffer(源, offset: 0, index: 0)
        编码.setTexture(出, index: 0)
        var 尺寸 = SIMD2<UInt32>(UInt32(宽), UInt32(高))
        编码.setBytes(&尺寸, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        编码.dispatchThreads(MTLSize(width: 宽, height: 高, depth: 1), threadsPerThreadgroup: 线程组(管, 宽: 宽, 高: 高))
        编码.endEncoding()
        收尾(命)
        return 出
    }

    // MARK: - 2) 线性光重采样（Catmull-Rom）

    static func 线性缩放(_ 源: MTLTexture, 到 宽: Int, 高: Int, 队: MTLCommandQueue) throws -> MTLTexture {
        if 源.width == 宽 && 源.height == 高 {
            return try 线性拷贝(源, 队: 队)
        }
        let 出 = try 光栅纹理(宽: 宽, 高: 高, 设备: 源.device)
        let 命 = try 新命令(队)
        guard let 编码 = 命.makeComputeCommandEncoder() else { throw 色彩通路错误(描述: "无法创建重采样编码器") }
        let 管 = try 图形管线.取(源.device, "k_resample_linear")
        编码.setComputePipelineState(管)
        编码.setTexture(源, index: 0)
        编码.setTexture(出, index: 1)
        var 比例 = SIMD2<Float>(Float(源.width) / Float(宽), Float(源.height) / Float(高))
        编码.setBytes(&比例, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
        var 尺寸 = SIMD2<UInt32>(UInt32(宽), UInt32(高))
        编码.setBytes(&尺寸, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        编码.dispatchThreads(MTLSize(width: 宽, height: 高, depth: 1), threadsPerThreadgroup: 线程组(管, 宽: 宽, 高: 高))
        编码.endEncoding()
        收尾(命)
        return 出
    }

    static func 线性拷贝(_ 源: MTLTexture, 队: MTLCommandQueue) throws -> MTLTexture {
        出口(源, 队: 队)
    }

    private static func 出口(_ 源: MTLTexture, 队: MTLCommandQueue) -> MTLTexture {
        let 出 = 源.device.makeTexture(descriptor: {
            let 描述 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: 源.pixelFormat,
                                                             width: 源.width, height: 源.height,
                                                             mipmapped: false)
            描述.usage = [.shaderRead, .shaderWrite, .renderTarget]
            描述.storageMode = .private
            return 描述
        }()) ?? 源
        // blit 拷贝（同尺寸、同格式，走最省带宽的路径）
        if let 命 = 队.makeCommandBuffer(), let 拷 = 命.makeBlitCommandEncoder() {
            拷.copy(from: 源, sourceSlice: 0, sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: 源.width, height: 源.height, depth: 1),
                    to: 出, destinationSlice: 0, destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            拷.endEncoding()
            收尾(命)
        }
        return 出
    }

    // MARK: - 3) 模型域（sRGB）桥接

    /// NCHW 平面 fp32 [-1,1]（sRGB 域模型输出）→ 线性光纹理。
    static func 线性纹理(from平面 平面缓冲: MTLBuffer, 宽: Int, 高: Int,
                    设备: MTLDevice, 队: MTLCommandQueue) throws -> MTLTexture {
        let 出 = try 光栅纹理(宽: 宽, 高: 高, 设备: 设备)
        let 命 = try 新命令(队)
        guard let 编码 = 命.makeComputeCommandEncoder() else { throw 色彩通路错误(描述: "无法创建模型域解码器") }
        let 管 = try 图形管线.取(设备, "k_planes_to_linear")
        编码.setComputePipelineState(管)
        编码.setBuffer(平面缓冲, offset: 0, index: 0)
        编码.setTexture(出, index: 0)
        var 尺寸 = SIMD2<UInt32>(UInt32(宽), UInt32(高))
        编码.setBytes(&尺寸, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        编码.dispatchThreads(MTLSize(width: 宽, height: 高, depth: 1), threadsPerThreadgroup: 线程组(管, 宽: 宽, 高: 高))
        编码.endEncoding()
        收尾(命)
        return 出
    }

    // MARK: - 4) 线性调色

    struct 调色参数 {
        var 曝光: Float = 0
        var 对比: Float = 0
        var 饱和: Float = 1
    }

    /// 线性光调色：曝光（线性增益）/ 对比（以 18% 灰为支点）/ 饱和度（线性亮度轴）。
    static func 调色(_ 源: MTLTexture, 参数: 调色参数, 队: MTLCommandQueue) throws -> MTLTexture {
        if 参数.曝光 == 0 && 参数.对比 == 0 && 参数.饱和 == 1 {
            return try 线性拷贝(源, 队: 队)
        }
        let 出 = try 光栅纹理(宽: 源.width, 高: 源.height, 设备: 源.device)
        let 命 = try 新命令(队)
        guard let 编码 = 命.makeComputeCommandEncoder() else { throw 色彩通路错误(描述: "无法创建调色编码器") }
        let 管 = try 图形管线.取(源.device, "k_grade_linear")
        编码.setComputePipelineState(管)
        编码.setTexture(源, index: 0)
        编码.setTexture(出, index: 1)
        var 值 = SIMD3<Float>(参数.曝光, 参数.对比, 参数.饱和)
        编码.setBytes(&值, length: MemoryLayout<SIMD3<Float>>.size, index: 0)
        编码.dispatchThreads(MTLSize(width: 源.width, height: 源.height, depth: 1),
                          threadsPerThreadgroup: 线程组(管, 宽: 源.width, 高: 源.height))
        编码.endEncoding()
        收尾(命)
        return 出
    }

    // MARK: - 5) 输出编码：sRGB 8bit（TPDF 抖动）

    /// 8bit 输出的编码曲线——必须与写码时声明的传递函数同一族。
    /// 否则「字节按 A 曲线编码、容器按 B 曲线声明」，播放器按标签反解就会整体偏亮/偏暗。
    enum 传递编码: UInt32 {
        /// IEC 61966-2-1（sRGB 曲线），对应 AVVideoTransferFunction_IEC_sRGB
        case sRGB = 0
        /// ITU-R BT.709-6 推荐 8bit 传递函数，对应 AVVideoTransferFunction_ITU_R_709_2
        case bt709 = 1

        var 名称: String {
            switch self {
            case .sRGB: return "IEC 61966-2-1 (sRGB)"
            case .bt709: return "ITU-R BT.709"
            }
        }

        /// 与该编码曲线配套的色彩空间：8bit 字节挂同族空间后，
        /// 下游 CGImage ↔ CGContext 之间绘制才是恒等映射，不会被再解一次。
        var 色彩空间: CGColorSpace {
            switch self {
            case .sRGB:
                return CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            case .bt709:
                return CGColorSpace(name: CGColorSpace.itur_709) ?? CGColorSpaceCreateDeviceRGB()
            }
        }
    }

    /// 线性光 → 交织 RGBA8 字节，编码曲线由 `传递` 指定，带 TPDF 抖动（幅度 ±1 LSB）。
    static func 输出RGBA8(from 线性: MTLTexture,
                      队: MTLCommandQueue,
                      抖动: Bool = true,
                      传递: 传递编码 = .sRGB) throws -> [UInt8] {
        let 宽 = 线性.width
        let 高 = 线性.height
        let 字节 = 宽 * 4 * 高
        let 缓冲 = try 共享缓冲(字节: 字节, 设备: 线性.device)
        let 命 = try 新命令(队)
        guard let 编码 = 命.makeComputeCommandEncoder() else { throw 色彩通路错误(描述: "无法创建输出编码器") }
        let 管 = try 图形管线.取(线性.device, "k_linear_to_out8")
        编码.setComputePipelineState(管)
        编码.setTexture(线性, index: 0)
        编码.setBuffer(缓冲, offset: 0, index: 0)
        var 尺寸 = SIMD2<UInt32>(UInt32(宽), UInt32(高))
        编码.setBytes(&尺寸, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        var 开关 = 抖动 ? UInt32(1) : UInt32(0)
        编码.setBytes(&开关, length: MemoryLayout<UInt32>.size, index: 2)
        var 曲线 = 传递.rawValue
        编码.setBytes(&曲线, length: MemoryLayout<UInt32>.size, index: 3)
        编码.dispatchThreads(MTLSize(width: 宽, height: 高, depth: 1), threadsPerThreadgroup: 线程组(管, 宽: 宽, 高: 高))
        编码.endEncoding()
        收尾(命)
        return 读回(缓冲, 字节: 字节)
    }

    // MARK: - 6) CGImage

    /// 8bit 字节 → CGImage。`色彩` 传入与字节编码同族的空间（见 传递编码.色彩空间），
    /// 下游 CGContext 又用同一空间绘制时整条搬运路径是恒等映射，不会再被重解一次。
    static func CG图(fromRGBA8 字节: [UInt8], 宽: Int, 高: Int, 色彩 指定: CGColorSpace? = nil) -> CGImage? {
        guard 字节.count >= 宽 * 4 * 高 else { return nil }
        let 数据 = Data(字节)
        guard let 提供者 = CGDataProvider(data: 数据 as CFData) else { return nil }
        let 色彩 = 指定 ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        return CGImage(width: 宽, height: 高, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: 宽 * 4, space: 色彩,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: 提供者, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }
}
