import Foundation

// 与 Flash 算子.metal 中的同名结构逐字段对齐（全为 4 字节标量，顺序必须一致）。

nonisolated struct Im2ColParams {
    var inH: Int32
    var inW: Int32
    var inC: Int32
    var kernelSize: Int32
    var stride: Int32
    var pad: Int32
    var outH: Int32
    var outW: Int32
    var outCols: Int32
    var realCols: Int32
    var channelsLast: Int32
}

nonisolated struct BiasParams {
    var rows: Int32
    var cols: Int32
    var hasBias: Int32
    var act: Int32
}

nonisolated struct NormParams {
    var rows: Int32
    var cols: Int32
    var eps: Float
}

nonisolated struct RoPEParams {
    var tokens: Int32
    var heads: Int32
    var width: Int32
    var temporalPosition: Int32
    var theta: Float
    var temporalDim: Int32
    var heightDim: Int32
    var widthDim: Int32
    var headDim: Int32
    var planeTokens: Int32
    /// 每平面时间位置步长：流式历史帧首平面 t=temporalPosition，逐平面 +planeStep
    var planeStep: Int32
}

nonisolated struct AttnParams {
    var batchTokens: Int32
    var batches: Int32
    var height: Int32
    var width: Int32
    var heads: Int32
    var kvGroups: Int32
    var headDim: Int32
    var planes: Int32
    var condPlanes: Int32
    var temporalWindow: Int32
    var hasPlaneMask: Int32
}

nonisolated struct PixelShuffleParams {
    var tokens: Int32
    var tokenWidth: Int32
    var scale: Int32
    var channels: Int32
    var outWidth: Int32
    var outHeight: Int32
}

nonisolated struct BicubicParams {
    var inH: Int32
    var inW: Int32
    var outH: Int32
    var outW: Int32
    var scale: Int32
    var channels: Int32
}

nonisolated struct CopyParams {
    var count: Int32
}
