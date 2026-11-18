import Foundation

// 与 Wan 算子.metal 中的同名结构逐字段对齐（全部 4 字节标量，顺序一致，禁止插队）。

nonisolated struct WanBiasParams {
    var cols: Int32
    var hasBias: Int32
    var act: Int32
    var rows: Int32
}

nonisolated struct WanNormParams {
    var rows: Int32
    var cols: Int32
    var eps: Float
    var hasWeight: Int32
    var hasBias: Int32
}

nonisolated struct WanColParams {
    var cols: Int32      // 输入通道数（像素行主序时的行宽）
    var kOut: Int32      // 展开列数 = C * 9
    var H: Int32
    var W: Int32
    var OH: Int32
    var OW: Int32
    var stride: Int32
    var pad: Int32
}

nonisolated struct WanPatchUnfoldParams {
    var patchW: Int32
    var patchH: Int32
    var C: Int32
    var H: Int32
    var W: Int32
}

nonisolated struct WanModParams {
    var count: Int32
    var cols: Int32
    var shiftIdx: Int32
    var scaleIdx: Int32
}

nonisolated struct WanGateParams {
    var rows: Int32
    var cols: Int32
    var idx: Int32
}

nonisolated struct WanRopeParams {
    var tokens: Int32
    var heads: Int32
    var dim: Int32
    var planeW: Int32
    var planeH: Int32
    var frameIndex: Int32
    var theta: Float
    var pad: Int32
}

nonisolated struct WanMaskParams {
    var span: Int32      // 每帧令牌数
    var H: Int32         // 令牌网格高
    var W: Int32         // 令牌网格宽
    var wh: Int32        // 空间窗高
    var ww: Int32        // 空间窗宽
    var window: Int32    // 时间窗帧数
    var qOffset: Int32   // query 帧在窗口内的槽位
    var kv: Int32        // 逻辑键数
    var kvStride: Int32  // 行跨距（8 元素对齐，满足 MPS rowBytes）
    var scale: Float
    var pad0: Float
    var pad1: Float
}

nonisolated struct WanUnpatchParams {
    var ph: Int32        // 令牌网格高
    var pw: Int32        // 令牌网格宽
    var C: Int32         // latent 通道
    var planeW: Int32    // latent 宽（= pw * 2）
    var planeH: Int32    // latent 高（= ph * 2）
    var pad0: Int32
    var pad1: Int32
    var pad2: Int32
}

nonisolated struct WanLQParams {
    var srcH: Int32
    var srcW: Int32
    var dstH: Int32
    var dstW: Int32
    var outW: Int32
    var outH: Int32
    var pad0: Int32
    var pad1: Int32
}

/// 半精度 GEMM：C[M,N] = A[M,K] · B[N,K]ᵀ，行跨距显式给出。
struct WanGemmParams {
    var M: Int32
    var N: Int32
    var K: Int32
    var aRowStride: Int32
    var bRowStride: Int32
    var cRowStride: Int32
}

/// bf16 → fp16 整表搬运：总元素量超 4G，偏移必须 64 位。
struct WanCopyParams {
    var n: UInt32
    var pad0: UInt32
    var srcOff: UInt64
    var dstOff: UInt64
}
