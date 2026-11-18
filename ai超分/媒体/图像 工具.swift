import AppKit
import CoreGraphics

extension NSImage {
    /// 真实像素尺寸（rep 的 pixelsWide/High，比 size 更可靠，尤其对 @2x 资源）。
    var 像素尺寸: CGSize {
        if let 表示 = representations.first, 表示.pixelsWide > 0, 表示.pixelsHigh > 0 {
            return CGSize(width: 表示.pixelsWide, height: 表示.pixelsHigh)
        }
        return size
    }

    /// 取底层 CGImage（Vision / CoreImage 链路需要）。
    /// 用 CGImage 构造 NSImage，尺寸以像素为单位，避免被自动当 @2x 缩小一半。
    static func 以(cg图: CGImage) -> NSImage {
        NSImage(cgImage: cg图, size: NSSize(width: cg图.width, height: cg图.height))
    }
}
