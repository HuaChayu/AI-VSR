import Foundation
import Metal

nonisolated struct 图形管线错误: LocalizedError {
    let 描述: String
    var errorDescription: String? { 描述 }
}

/// 运行时编译并缓存色彩通路的内核（色彩着色器.源码）。
/// 与产线预编译的 .metal 内核互相独立：只服务于线性光转换这类轻量 kernel。
nonisolated enum 图形管线 {
    private static let 锁 = NSLock()
    private static var 库表: [ObjectIdentifier: MTLLibrary] = [:]
    private static var 管线表: [String: MTLComputePipelineState] = [:]

    static func 取(_ 设备: MTLDevice, _ 函数名: String) throws -> MTLComputePipelineState {
        锁.lock()
        defer { 锁.unlock() }
        if let 已有 = 管线表[函数名] { return 已有 }
        let 库: MTLLibrary
        if let 缓存库 = 库表[ObjectIdentifier(设备)] {
            库 = 缓存库
        } else {
            let 选项 = MTLCompileOptions()
            选项.fastMathEnabled = true
            库 = try 设备.makeLibrary(source: 色彩着色器.源码, options: 选项)
            库表[ObjectIdentifier(设备)] = 库
        }
        guard let 函数 = 库.makeFunction(name: 函数名) else {
            throw 图形管线错误(描述: "色彩着色器缺少内核：\(函数名)")
        }
        let 管线 = try 设备.makeComputePipelineState(function: 函数)
        管线表[函数名] = 管线
        return 管线
    }
}
