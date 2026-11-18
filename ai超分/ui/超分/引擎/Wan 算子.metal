// Wan 主干（5B DiT）Metal 算子集合 —— RelayVSR 移植
// 张量一律 half（fp16）行主序；归一化 / softmax 内部走 fp32。
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct WanBiasParams { int cols; int hasBias; int act; int rows; };
struct WanNormParams { int rows; int cols; float eps; int hasWeight; int hasBias; };
struct WanColParams { int cols; int kOut; int H; int W; int OH; int OW; int stride; int pad; };
struct WanPatchUnfoldParams { int patchW; int patchH; int C; int H; int W; };
struct WanModParams { int count; int cols; int shiftIdx; int scaleIdx; };
struct WanGateParams { int rows; int cols; int idx; };
struct WanRopeParams { int tokens; int heads; int dim; int planeW; int planeH; int frameIndex; float theta; int pad; };
struct WanMaskParams { int span; int H; int W; int wh; int ww; int window; int qOffset; int kv; int kvStride; float scale; float pad0; float pad1; };
struct WanUnpatchParams { int ph; int pw; int C; int planeW; int planeH; int pad0; int pad1; int pad2; };
struct WanLQParams { int srcH; int srcW; int dstH; int dstW; int outW; int outH; int pad0; int pad1; };
struct WanCopyParams { uint n; uint pad0; ulong srcOff; ulong dstOff; };

// MARK: - 权重解码 bf16 → fp16

kernel void wan_bf16_to_half(device const ushort* src [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant WanCopyParams& r [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= r.n) { return; }
    dst[r.dstOff + ulong(gid)] = half(as_type<float>(uint(src[r.srcOff + ulong(gid)]) << 16));
}

// MARK: - 低清帧：4x 双线性上采样 + 16x unshuffle，直出 [像素, 768]
//
// 口径与官方 streaming.py 一致：native(低清) --F.interpolate(scale=4, bilinear, align_corners=False)-->
// LQ4x(4×低清 = 输出分辨率) --PixelUnshuffle(16)--> latent(4×低清/16 = 低清/4) --conv2 stride2--> token(低清/8)。
// 故本内核每输出一个 latent 像素 (x,y) 的 16x16 块元素 (i,j)，实际取的是 4× 上采样图上的像素
// (16x+i, 16y+j)，即对低清做双线性采样（不做任何最近邻近似），与官方 F.interpolate 结果对齐。

kernel void wan_lq_prepare(device const uchar* rgb [[buffer(0)]],
                           device half* out [[buffer(1)]],
                           constant WanLQParams& p [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]]) {
    int x = int(gid.x), y = int(gid.y), z = int(gid.z);
    if (x >= p.outW || y >= p.outH || z >= 768) { return; }
    int c = z >> 8;
    int rem = z & 255;
    int i = rem >> 4;
    int j = rem & 15;
    // 4× 上采样网格坐标（dstH/dstW = 4*srcH/4*srcW）
    int oy = y * 16 + i;
    int ox = x * 16 + j;
    // align_corners=False 的坐标映射：src = (dst + 0.5) * (srcSize/dstSize) - 0.5
    float sy = (float(oy) + 0.5f) * float(p.srcH) / float(p.dstH) - 0.5f;
    float sx = (float(ox) + 0.5f) * float(p.srcW) / float(p.dstW) - 0.5f;
    int y0 = int(floor(sy));
    int x0 = int(floor(sx));
    float fy = sy - float(y0);
    float fx = sx - float(x0);
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 > p.srcH - 1) { y0 = p.srcH - 1; fy = 0.0f; }
    if (x0 > p.srcW - 1) { x0 = p.srcW - 1; fx = 0.0f; }
    int y1 = min(y0 + 1, p.srcH - 1);
    int x1 = min(x0 + 1, p.srcW - 1);
    device const uchar* plane = rgb + c * p.srcH * p.srcW;
    float v00 = float(plane[y0 * p.srcW + x0]);
    float v01 = float(plane[y0 * p.srcW + x1]);
    float v10 = float(plane[y1 * p.srcW + x0]);
    float v11 = float(plane[y1 * p.srcW + x1]);
    float 上 = v00 + (v01 - v00) * fx;
    float 下 = v10 + (v11 - v10) * fx;
    float v = 上 + (下 - 上) * fy;
    out[(size_t(y) * p.outW + x) * 768 + z] = half(v / 127.5f - 1.0f);
}

// MARK: - 逐元素偏置 / SiLU

kernel void wan_bias_act(device half* x [[buffer(0)]],
                         device const half* b [[buffer(1)]],
                         constant WanBiasParams& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int col = int(gid.x), row = int(gid.y);
    if (col >= p.cols || row >= p.rows) { return; }
    size_t idx = size_t(row) * p.cols + col;
    float v = float(x[idx]);
    if (p.hasBias != 0) { v += float(b[col]); }
    if (p.act != 0) { v = v / (1.0f + exp(-v)); }
    x[idx] = half(v);
}

// MARK: - 3x3 stride2 pad1 im2col（输入为像素行主序 [pix, C]）

kernel void wan_im2col3(device const half* x [[buffer(0)]],
                        device half* out [[buffer(1)]],
                        constant WanColParams& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    int k = int(gid.x), tok = int(gid.y);
    if (k >= p.kOut) { return; }
    int c = k / 9, r = (k % 9) / 3, s = k % 3;
    int oh = tok / p.OW, ow = tok % p.OW;
    int ih = oh * p.stride - p.pad + r;
    int iw = ow * p.stride - p.pad + s;
    half v = half(0.0f);
    if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W) {
        v = x[(size_t(ih) * p.W + iw) * p.cols + c];
    }
    out[size_t(tok) * p.kOut + k] = v;
}

// MARK: - DiT patch unfold：[C,90,156] → [3510, C*4]

kernel void wan_patch_unfold(device const half* latent [[buffer(0)]],
                             device half* out [[buffer(1)]],
                             constant WanPatchUnfoldParams& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    int k = int(gid.x), tok = int(gid.y);
    int total = p.C * 4;
    if (k >= total) { return; }
    int c = k / 4, r = (k % 4) / 2, s = k % 2;
    int ph = tok / p.patchW, pw = tok % p.patchW;
    int ih = ph * 2 + r, iw = pw * 2 + s;
    out[size_t(tok) * total + k] = latent[size_t(c) * p.H * p.W + size_t(ih) * p.W + iw];
}

// MARK: - LayerNorm（fp32 内部，可选仿射）

kernel void wan_layernorm(device const half* x [[buffer(0)]],
                          device const half* w [[buffer(1)]],
                          device const half* b [[buffer(2)]],
                          device half* out [[buffer(3)]],
                          constant WanNormParams& p [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgsize [[threads_per_threadgroup]]) {
    float sum = 0.0f, sumsq = 0.0f;
    for (int c = int(tid); c < p.cols; c += int(tgsize)) {
        float v = float(x[size_t(row) * p.cols + c]);
        sum += v; sumsq += v * v;
    }
    threadgroup float 共享[64];
    float sv = simd_sum(sum);
    float sq = simd_sum(sumsq);
    uint lane = tid % 32, group = tid / 32;
    uint 组数 = (tgsize + 31) / 32;
    if (lane == 0) { 共享[group] = sv; 共享[32 + group] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = 0.0f, 总平方 = 0.0f;
        for (uint i = 0; i < 组数; ++i) { 总 += 共享[i]; 总平方 += 共享[32 + i]; }
        共享[0] = 总; 共享[1] = 总平方;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 共享[0] / float(p.cols);
    float var = 共享[1] / float(p.cols) - mean * mean;
    float inv = 1.0f / sqrt(max(var, 0.0f) + p.eps);
    for (int c = int(tid); c < p.cols; c += int(tgsize)) {
        size_t idx = size_t(row) * p.cols + c;
        float v = (float(x[idx]) - mean) * inv;
        if (p.hasWeight != 0) { v *= float(w[c]); }
        if (p.hasBias != 0) { v += float(b[c]); }
        out[idx] = half(v);
    }
}

// MARK: - 整行 RMSNorm（qk norm：跨全部 heads；也用于 lq_proj 的跨通道归一化）

kernel void wan_rmsnorm_row(device const half* x [[buffer(0)]],
                            device const half* w [[buffer(1)]],
                            device half* out [[buffer(2)]],
                            constant WanNormParams& p [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgsize [[threads_per_threadgroup]]) {
    float sumsq = 0.0f;
    for (int c = int(tid); c < p.cols; c += int(tgsize)) {
        float v = float(x[size_t(row) * p.cols + c]);
        sumsq += v * v;
    }
    threadgroup float 共享[64];
    float sq = simd_sum(sumsq);
    uint lane = tid % 32, group = tid / 32;
    uint 组数 = (tgsize + 31) / 32;
    if (lane == 0) { 共享[group] = sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = 0.0f;
        for (uint i = 0; i < 组数; ++i) { 总 += 共享[i]; }
        共享[0] = 总;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sqrt(共享[0] / float(p.cols) + p.eps);
    for (int c = int(tid); c < p.cols; c += int(tgsize)) {
        size_t idx = size_t(row) * p.cols + c;
        out[idx] = half(float(x[idx]) * inv * float(w[c]));
    }
}

// MARK: - AdaLN 调制：out = normed * (1 + scale) + shift

kernel void wan_modulate(device const half* normed [[buffer(0)]],
                         device const float* mods [[buffer(1)]],
                         device half* out [[buffer(2)]],
                         constant WanModParams& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int col = int(gid.x), row = int(gid.y);
    if (col >= p.cols) { return; }
    size_t idx = size_t(row) * p.cols + col;
    float scale = mods[size_t(p.scaleIdx) * p.cols + col];
    float shift = mods[size_t(p.shiftIdx) * p.cols + col];
    out[idx] = half(float(normed[idx]) * (1.0f + scale) + shift);
}

// MARK: - 残差叠加（可带 gate）

kernel void wan_gate_add(device half* hidden [[buffer(0)]],
                         device const half* x [[buffer(1)]],
                         device const float* mods [[buffer(2)]],
                         constant WanGateParams& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int col = int(gid.x), row = int(gid.y);
    if (col >= p.cols) { return; }
    size_t idx = size_t(row) * p.cols + col;
    float gate = 1.0f;
    if (p.idx >= 0) { gate = mods[size_t(p.idx) * p.cols + col]; }
    hidden[idx] = half(float(hidden[idx]) + float(x[idx]) * gate);
}

// MARK: - 3D RoPE（t/h/w 分段，streaming 绝对帧号）

kernel void wan_rope(device half* x [[buffer(0)]],
                     constant WanRopeParams& p [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int j = int(gid.x), tok = int(gid.y);
    int halfDim = p.dim / 2;
    if (j >= halfDim) { return; }
    int tDim = p.dim - 4 * (p.dim / 6);
    int hDim = (p.dim - tDim) / 2;
    int segLen = 0, segIdx = 0;
    float pos = 0.0f;
    int plane = p.planeW * p.planeH;
    if (j * 2 < tDim) {
        segLen = tDim; segIdx = j; pos = float(p.frameIndex);
    } else if (j * 2 < tDim + hDim) {
        segLen = hDim; segIdx = j - tDim / 2; pos = float((tok % plane) / p.planeW);
    } else {
        segLen = hDim; segIdx = j - (tDim + hDim) / 2; pos = float((tok % plane) % p.planeW);
    }
    float freq = pow(p.theta, -2.0f * float(segIdx) / float(segLen));
    float angle = pos * freq;
    float c = cos(angle), s = sin(angle);
    for (int h = 0; h < p.heads; ++h) {
        size_t base = (size_t(tok) * p.heads + h) * p.dim + 2 * j;
        float x0 = float(x[base]), x1 = float(x[base + 1]);
        x[base] = half(x0 * c - x1 * s);
        x[base + 1] = half(x0 * s + x1 * c);
    }
}

// MARK: - 窗口掩码 softmax（自注意力）

kernel void wan_mask_softmax(device const half* s [[buffer(0)]],
                             device half* pOut [[buffer(1)]],
                             constant WanMaskParams& p [[buffer(2)]],
                             uint q [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgsize [[threads_per_threadgroup]]) {
    int span = p.span;
    int qi = int(q);
    int qt = qi / span + p.qOffset;
    int qy = (qi % span) / p.W;
    int qx = (qi % span) % p.W;
    int top = clamp(qy - p.wh / 2, 0, max(0, p.H - p.wh));
    int left = clamp(qx - p.ww / 2, 0, max(0, p.W - p.ww));
    int lo = max(0, (qt - p.window + 1)) * span;
    int hi = min((qt + 1) * span, p.kv);
    if (lo > hi) { lo = hi; }

    float 局部最大 = -INFINITY;
    for (int k = lo + int(tid); k < hi; k += int(tgsize)) {
        int rel = k % span;
        int ky = rel / p.W, kx = rel % p.W;
        if (ky >= top && ky < top + p.wh && kx >= left && kx < left + p.ww) {
            局部最大 = max(局部最大, float(s[size_t(qi) * p.kvStride + k]) * p.scale);
        }
    }
    threadgroup float 共享[64];
    float m = simd_max(局部最大);
    uint lane = tid % 32, group = tid / 32;
    uint 组数 = (tgsize + 31) / 32;
    if (lane == 0) { 共享[group] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = -INFINITY;
        for (uint i = 0; i < 组数; ++i) { 总 = max(总, 共享[i]); }
        共享[0] = 总;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float 最大值 = isfinite(共享[0]) ? 共享[0] : 0.0f;

    float 局部和 = 0.0f;
    for (int k = lo + int(tid); k < hi; k += int(tgsize)) {
        int rel = k % span;
        int ky = rel / p.W, kx = rel % p.W;
        if (ky >= top && ky < top + p.wh && kx >= left && kx < left + p.ww) {
            局部和 += exp(float(s[size_t(qi) * p.kvStride + k]) * p.scale - 最大值);
        }
    }
    float 和 = simd_sum(局部和);
    if (lane == 0) { 共享[group] = 和; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = 0.0f;
        for (uint i = 0; i < 组数; ++i) { 总 += 共享[i]; }
        共享[0] = (总 > 0.0f && isfinite(总)) ? 总 : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float 归一 = 1.0f / 共享[0];
    for (int k = int(tid); k < p.kv; k += int(tgsize)) {
        size_t idx = size_t(qi) * p.kvStride + k;
        int rel = k % span;
        int ky = rel / p.W, kx = rel % p.W;
        bool 可见 = (k >= lo && k < hi && ky >= top && ky < top + p.wh && kx >= left && kx < left + p.ww);
        pOut[idx] = half(可见 ? exp(float(s[idx]) * p.scale - 最大值) * 归一 : 0.0f);
    }
}

// MARK: - 稠密 softmax（跨注意力，无掩码）

kernel void wan_softmax_dense(device const half* s [[buffer(0)]],
                              device half* pOut [[buffer(1)]],
                              constant WanMaskParams& p [[buffer(2)]],
                              uint q [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tgsize [[threads_per_threadgroup]]) {
    int qi = int(q);
    float 局部最大 = -INFINITY;
    for (int k = int(tid); k < p.kv; k += int(tgsize)) {
        局部最大 = max(局部最大, float(s[size_t(qi) * p.kvStride + k]) * p.scale);
    }
    threadgroup float 共享[64];
    float m = simd_max(局部最大);
    uint lane = tid % 32, group = tid / 32;
    uint 组数 = (tgsize + 31) / 32;
    if (lane == 0) { 共享[group] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = -INFINITY;
        for (uint i = 0; i < 组数; ++i) { 总 = max(总, 共享[i]); }
        共享[0] = 总;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float 最大值 = isfinite(共享[0]) ? 共享[0] : 0.0f;

    float 局部和 = 0.0f;
    for (int k = int(tid); k < p.kv; k += int(tgsize)) {
        局部和 += exp(float(s[size_t(qi) * p.kvStride + k]) * p.scale - 最大值);
    }
    float 和 = simd_sum(局部和);
    if (lane == 0) { 共享[group] = 和; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float 总 = 0.0f;
        for (uint i = 0; i < 组数; ++i) { 总 += 共享[i]; }
        共享[0] = (总 > 0.0f && isfinite(总)) ? 总 : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float 归一 = 1.0f / 共享[0];
    for (int k = int(tid); k < p.kv; k += int(tgsize)) {
        size_t idx = size_t(qi) * p.kvStride + k;
        pOut[idx] = half(exp(float(s[idx]) * p.scale - 最大值) * 归一);
    }
}

// MARK: - GELU(tanh) 就地

kernel void wan_gelu(device half* x [[buffer(0)]],
                     constant uint& n [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    size_t idx = size_t(gid);
    float v = float(x[idx]);
    float a = 0.7978845608f * (v + 0.044715f * v * v * v);
    a = clamp(a, -15.0f, 15.0f);
    float t = precise::tanh(a);
    float r = 0.5f * v * (1.0f + t);
    x[idx] = half(r);
}

// MARK: - unpatch：[3510,192] → latent [48,90,156]

kernel void wan_unpatch(device const half* x [[buffer(0)]],
                        device half* latent [[buffer(1)]],
                        constant WanUnpatchParams& p [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
    int w = int(gid.x), h = int(gid.y), c = int(gid.z);
    if (w >= p.planeW || h >= p.planeH || c >= p.C) { return; }
    int ph = h >> 1, ph2 = h & 1;
    int pw = w >> 1, pw2 = w & 1;
    size_t tok = size_t(ph) * p.pw + pw;
    int total = p.C * 4;
    size_t src = tok * total + size_t(ph2 * 2 + pw2) * p.C + c;
    latent[size_t(c) * p.planeH * p.planeW + size_t(h) * p.planeW + w] = x[src];
}

// MARK: - 单步流式采样：latent -= pred

kernel void wan_latent_step(device half* latent [[buffer(0)]],
                            device const half* pred [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    latent[gid] = half(float(latent[gid]) - float(pred[gid]));
}

kernel void wan_copy_half(device const half* src [[buffer(0)]],
                          device half* dst [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    dst[gid] = src[gid];
}

// MARK: - 单头转置：[tokens, srcStride] 中从 srcOffset 起的 dim 维 → [dim, tokens] 的 dstOffset 处
// 供 MPS 逐头 GEMM 使用（MPS 需右侧矩阵行连续，无法直接跨 head 取列）

struct WanHeadParams { int tokens; int dim; int srcStride; int dstStride; int srcOffset; int dstOffset; };

kernel void wan_head_transpose(device const half* src [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant WanHeadParams& p [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    int d = int(gid.x), tok = int(gid.y);
    if (d >= p.dim || tok >= p.tokens) { return; }
    dst[size_t(p.dstOffset) + size_t(d) * size_t(p.dstStride) + size_t(tok)] =
        src[size_t(tok) * size_t(p.srcStride) + size_t(p.srcOffset) + size_t(d)];
}

// MARK: - 半精度 GEMM：C[M,N] = A[M,K] · B[N,K]ᵀ
// MPS 的矩阵乘只能挂到 command buffer，无法嵌进单条编码器里逐层串联，故自备 tiled 实现。
// 行跨距全部显式给出，便于直接对多头切片做偏移（免掉 K/V 转置）。

struct WanGemmParams { int M; int N; int K; int aRowStride; int bRowStride; int cRowStride; };

#define WAN_LOADK(Aa, Bb, KK)                                                  \
    _Pragma("unroll") for (int i = 0; i < 2; ++i) {                            \
        simdgroup_load(Aa[i], As + (行起 + i * 8) * 行跨 + (KK), 行跨);        \
    }                                                                          \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) {                            \
        simdgroup_load(Bb[j], Bs + (KK) * B跨 + 列起 + j * 8, B跨);            \
    }

#define WAN_MMAK(Aa, Bb)                                                       \
    _Pragma("unroll") for (int i = 0; i < 2; ++i) {                            \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                        \
            simdgroup_multiply_accumulate(acc[i][j], Aa[i], Bb[j], acc[i][j]); \
        }                                                                      \
    }

kernel void wan_gemm(device const half* A [[buffer(0)]],
                     device const half* B [[buffer(1)]],
                     device half* C [[buffer(2)]],
                     constant WanGemmParams& p [[buffer(3)]],
                     uint2 组 [[threadgroup_position_in_grid]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
    const int BM = 64, BN = 64, BK = 32;
    // 共享缓存行跨取 40（8 的倍数，且 40*sizeof(half)=80B 是 16 字节整数倍）：
    // 每个 8x8 块的起始地址天然对齐，同时避开整 bank 对齐造成的冲突。
    const int 行跨 = 40;
    // As / Bs 合用一块共享内存；GEMM 主循环结束后这块内存再复用为 fp32 结果瓦片，
    // 使 threadgroup 静态占用压到 10KB，换取更高的 threadgroup 并发度。
    threadgroup half 共享[64 * 行跨 * 2];
    threadgroup float* 写回 = (threadgroup float*)共享;   // 可容纳 32x68 的 fp32 瓦片
    threadgroup half* As = 共享;
    threadgroup half* Bs = 共享 + 64 * 行跨;  // Bs 转置存为 [K][N]，行跨另取 72（>=64 且 16 字节对齐）
    const int B跨 = 72;
    const int 写回跨 = 68;

    int tx = int(tid.x), ty = int(tid.y);
    int 行基 = int(组.y) * BM;
    int 列基 = int(组.x) * BN;

    // 16x16 = 256 线程 = 8 个 simdgroup（线程按 x 最快线性排列，每 32 个一组）。
    // 每个 simdgroup 承担 16 行 x 32 列 = 2x4 个 8x8 累加块。
    int 线 = ty * 16 + tx;
    int 群 = 线 >> 5;
    int 行起 = (群 >> 1) * 16;
    int 列起 = (群 & 1) * 32;

    simdgroup_matrix<float, 8, 8> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < p.K; k0 += BK) {
        for (int i = ty; i < BM; i += 16) {
            int gi = 行基 + i;
            for (int k = tx; k < BK; k += 16) {
                int gk = k0 + k;
                As[i * 行跨 + k] = (gi < p.M && gk < p.K) ? A[size_t(gi) * p.aRowStride + gk] : half(0);
            }
        }
        for (int j = ty; j < BN; j += 16) {
            int gj = 列基 + j;
            for (int k = tx; k < BK; k += 16) {
                int gk = k0 + k;
                // 写入即转置成 [K][N] 布局，后续 MMA 直接顺序加载
                Bs[k * B跨 + j] = (gj < p.N && gk < p.K) ? B[size_t(gj) * p.bRowStride + gk] : half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 硬件 MMA：fp16 输入、fp32 累加，直接走矩阵单元。
        // BK=32 固定为 4 个 K 步，双缓冲乒乓：加载下一段与当前段 MMA 交错发射以隐藏共享内存延迟。
        simdgroup_matrix<half, 8, 8> aA[2], bA[4], aB[2], bB[4];
        WAN_LOADK(aA, bA, 0);
        WAN_LOADK(aB, bB, 8);
        WAN_MMAK(aA, bA);
        WAN_LOADK(aA, bA, 16);
        WAN_MMAK(aB, bB);
        WAN_LOADK(aB, bB, 24);
        WAN_MMAK(aA, bA);
        WAN_MMAK(aB, bB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 回写：fp32 累加结果分两批落到复用的共享瓦片（此处完成唯一的 float→half 转换），
    // 再按 M/N 越界裁剪写回 C，语义与逐元素版本完全一致。
    for (int 阶段 = 0; 阶段 < 2; ++阶段) {
        if ((群 >> 2) == 阶段) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    simdgroup_store(acc[i][j], 写回 + (行起 + i * 8 - 阶段 * 32) * 写回跨 + 列起 + j * 8, 写回跨);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int n = 0; n < 8; ++n) {
            int e = 线 + n * 256;
            int i = e / BN;
            int j = e % BN;
            int gi = 行基 + 阶段 * 32 + i;
            int gj = 列基 + j;
            if (gi >= p.M || gj >= p.N) { continue; }
            C[size_t(gi) * p.cRowStride + gj] = half(写回[i * 写回跨 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void wan_gemm_v4(device const half* A [[buffer(0)]],
                     device const half* B [[buffer(1)]],
                     device half* C [[buffer(2)]],
                     constant WanGemmParams& p [[buffer(3)]],
                     uint2 组 [[threadgroup_position_in_grid]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
    const int BM = 64, BN = 64, BK = 32;
    // 共享缓存行跨取 40（8 的倍数，且 40*sizeof(half)=80B 是 16 字节整数倍）：
    // 每个 8x8 块的起始地址天然对齐，同时避开整 bank 对齐造成的冲突。
    const int 行跨 = 40;
    // As / Bs 合用一块共享内存；GEMM 主循环结束后这块内存再复用为 fp32 结果瓦片，
    // 使 threadgroup 静态占用压到 10KB，换取更高的 threadgroup 并发度。
    threadgroup half 共享[64 * 行跨 * 2];
    threadgroup float* 写回 = (threadgroup float*)共享;   // 可容纳 32x68 的 fp32 瓦片
    threadgroup half* As = 共享;
    threadgroup half* Bs = 共享 + 64 * 行跨;  // Bs 转置存为 [K][N]，行跨另取 72（>=64 且 16 字节对齐）
    const int B跨 = 72;
    const int 写回跨 = 68;

    int tx = int(tid.x), ty = int(tid.y);
    int 行基 = int(组.y) * BM;
    int 列基 = int(组.x) * BN;

    // 16x16 = 256 线程 = 8 个 simdgroup（线程按 x 最快线性排列，每 32 个一组）。
    // 每个 simdgroup 承担 16 行 x 32 列 = 2x4 个 8x8 累加块。
    int 线 = ty * 16 + tx;
    int 群 = 线 >> 5;
    int 行起 = (群 >> 1) * 16;
    int 列起 = (群 & 1) * 32;

    simdgroup_matrix<float, 8, 8> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < p.K; k0 += BK) {
        // v4：全局加载改 half4 向量化（16 线程一次取满 128 B），共享内存写入保持原布局。
        // 前提：行跨与切片偏移均为 4 的倍数、K 为 4 的倍数，由 Swift 侧派发前判定。
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            int f = 线 + n * 256;
            int i = f >> 3;
            int k4 = (f & 7) * 4;
            int gi = 行基 + i;
            int gk = k0 + k4;
            half4 v = half4(0);
            if (gi < p.M && gk + 3 < p.K) { v = *(device const half4*)(A + size_t(gi) * p.aRowStride + gk); }
            *(threadgroup half4*)(As + i * 行跨 + k4) = v;
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            int f = 线 + n * 256;
            int j = f >> 3;
            int k4 = (f & 7) * 4;
            int gj = 列基 + j;
            int gk = k0 + k4;
            half4 v = half4(0);
            if (gj < p.N && gk + 3 < p.K) { v = *(device const half4*)(B + size_t(gj) * p.bRowStride + gk); }
            Bs[(k4 + 0) * B跨 + j] = v.x;
            Bs[(k4 + 1) * B跨 + j] = v.y;
            Bs[(k4 + 2) * B跨 + j] = v.z;
            Bs[(k4 + 3) * B跨 + j] = v.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 硬件 MMA：fp16 输入、fp32 累加，直接走矩阵单元。
        // BK=32 固定为 4 个 K 步，双缓冲乒乓：加载下一段与当前段 MMA 交错发射以隐藏共享内存延迟。
        simdgroup_matrix<half, 8, 8> aA[2], bA[4], aB[2], bB[4];
        WAN_LOADK(aA, bA, 0);
        WAN_LOADK(aB, bB, 8);
        WAN_MMAK(aA, bA);
        WAN_LOADK(aA, bA, 16);
        WAN_MMAK(aB, bB);
        WAN_LOADK(aB, bB, 24);
        WAN_MMAK(aA, bA);
        WAN_MMAK(aB, bB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 回写：fp32 累加结果分两批落到复用的共享瓦片（此处完成唯一的 float→half 转换），
    // 再按 M/N 越界裁剪写回 C，语义与逐元素版本完全一致。
    for (int 阶段 = 0; 阶段 < 2; ++阶段) {
        if ((群 >> 2) == 阶段) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    simdgroup_store(acc[i][j], 写回 + (行起 + i * 8 - 阶段 * 32) * 写回跨 + 列起 + j * 8, 写回跨);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int n = 0; n < 8; ++n) {
            int e = 线 + n * 256;
            int i = e / BN;
            int j = e % BN;
            int gi = 行基 + 阶段 * 32 + i;
            int gj = 列基 + j;
            if (gi >= p.M || gj >= p.N) { continue; }
            C[size_t(gi) * p.cRowStride + gj] = half(写回[i * 写回跨 + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


// ================= wan_gemm_v5：大 M 场景重构 =================
// 128x64 threadgroup tile，8 simdgroup（每 simdgroup 16x64 = 2x8 累加器阵），BK=32，
// K 维寄存器双缓冲流水，共享行跨 +4 消除 bank 冲突，fp32 累加器分批复用共享区写回。
// 线程组 256x1；调度 tile=(128, 64)。
template <int BM, int BN, int BK, int SNR, int SNC, int APAD = 8, int BPAD = 8, int WB = 0>
inline void gemm_body(device const half* A, device const half* B, device half* C,
                      constant WanGemmParams& p, uint2 组, uint tid, threadgroup half* 共享) {
    constexpr int NT   = SNR * SNC * 32;          // 每 threadgroup 线程数
    constexpr int AS   = BK + APAD;               // A 共享行跨（half）
    constexpr int BS   = BN + BPAD;               // B 共享行跨（half，转置为 [K][N]）
    constexpr int SMEMB = BM * AS * 2 + BK * BS * 2;
    constexpr int A4   = BM * BK / 4 / NT;        // 每线程加载的 half4 数
    constexpr int B4   = BK * BN / 4 / NT;
    constexpr int RB   = BM / SNR;                // 每 simdgroup 行数
    constexpr int CB   = BN / SNC;                // 每 simdgroup 列数
    constexpr int RA   = RB / 8, CA = CB / 8;     // 累加器阵列 RA x CA
    constexpr int K4   = BK / 4;
    constexpr int KSTEP = BK / 8;
    constexpr int 批   = (SMEMB >= RB * BN * 4) ? RB : (RB / 2);

    threadgroup half* As = 共享;
    threadgroup half* Bs = 共享 + BM * AS;
    threadgroup float* 写回 = (threadgroup float*)共享;   // K 循环结束后复用为 fp32 写回瓦片

    int 线 = int(tid);
    int 群 = 线 >> 5;
    int 行起 = (群 / SNC) * RB;
    int 列起 = (群 % SNC) * CB;
    int 行基 = int(组.y) * BM;
    int 列基 = int(组.x) * BN;

    simdgroup_matrix<float, 8, 8> acc[RA][CA];
#pragma unroll
    for (int i = 0; i < RA; ++i)
#pragma unroll
        for (int j = 0; j < CA; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // 全局 -> 寄存器（K 维流水的一级缓冲）
    half4 rA[A4], rB[B4];

#define 载入(K0)                                                               \
    _Pragma("unroll") for (int n = 0; n < A4; ++n) {                           \
        int f = 线 + n * NT;                                                   \
        int i = f / K4, k4 = (f % K4) * 4;                                      \
        int gi = 行基 + i, gk = (K0) + k4;                                      \
        rA[n] = (gi < p.M && gk + 3 < p.K)                                      \
              ? *(device const half4*)(A + size_t(gi) * p.aRowStride + gk) : half4(0); \
    }                                                                          \
    _Pragma("unroll") for (int n = 0; n < B4; ++n) {                           \
        int f = 线 + n * NT;                                                   \
        int j = f / K4, k4 = (f % K4) * 4;                                      \
        int gj = 列基 + j, gk = (K0) + k4;                                      \
        rB[n] = (gj < p.N && gk + 3 < p.K)                                      \
              ? *(device const half4*)(B + size_t(gj) * p.bRowStride + gk) : half4(0); \
    }

#define 落盘()                                                                 \
    _Pragma("unroll") for (int n = 0; n < A4; ++n) {                           \
        int f = 线 + n * NT;                                                   \
        int i = f / K4, k4 = (f % K4) * 4;                                      \
        *(threadgroup half4*)(As + i * AS + k4) = rA[n];                        \
    }                                                                          \
    _Pragma("unroll") for (int n = 0; n < B4; ++n) {                           \
        int f = 线 + n * NT;                                                   \
        int j = f / K4, k4 = (f % K4) * 4;                                      \
        Bs[(k4 + 0) * BS + j] = rB[n].x;                                        \
        Bs[(k4 + 1) * BS + j] = rB[n].y;                                        \
        Bs[(k4 + 2) * BS + j] = rB[n].z;                                        \
        Bs[(k4 + 3) * BS + j] = rB[n].w;                                        \
    }

    载入(0);
    落盘();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int 段数 = (p.K + BK - 1) / BK;
    for (int s = 0; s < 段数; ++s) {
        int 下一 = (s + 1) * BK;
        // 先发射下一 K 块的全局加载到寄存器（与随后的 MMA 在硬件上重叠），
        // 共享内存侧不再放第二份，避免 18KB+ 占用压垮 occupancy。
        if (下一 < p.K) 载入(下一);

        // 双组累加器输入：显式两套 frag（避免 simdgroup_matrix 动态索引）
        simdgroup_matrix<half, 8, 8> g0A[RA], g0B[CA], g1A[RA], g1B[CA];
#define LOADG(G, OFF)                                                          \
        _Pragma("unroll") for (int i = 0; i < RA; ++i)                         \
            simdgroup_load(G##A[i], As + (行起 + i * 8) * AS + (OFF), AS);     \
        _Pragma("unroll") for (int j = 0; j < CA; ++j)                         \
            simdgroup_load(G##B[j], Bs + (OFF) * BS + 列起 + j * 8, BS);
#define MMAG(G)                                                                \
        _Pragma("unroll") for (int i = 0; i < RA; ++i)                         \
            _Pragma("unroll") for (int j = 0; j < CA; ++j)                     \
                simdgroup_multiply_accumulate(acc[i][j], G##A[i], G##B[j], acc[i][j]);

        LOADG(g0, 0)
        LOADG(g1, 8)
#pragma unroll
        for (int q = 0; q < KSTEP / 4; ++q) {
            MMAG(g0)
            if ((4 * q + 2) < KSTEP) LOADG(g0, (4 * q + 2) * 8)
            MMAG(g1)
            if ((4 * q + 3) < KSTEP) LOADG(g1, (4 * q + 3) * 8)
            MMAG(g0)
            if ((4 * q + 4) < KSTEP) LOADG(g0, (4 * q + 4) * 8)
            MMAG(g1)
            if ((4 * q + 5) < KSTEP) LOADG(g1, (4 * q + 5) * 8)
        }
#undef LOADG
#undef MMAG
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (下一 < p.K) {
            落盘();
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 写回模式 WB=1：simdgroup 直接落 device，省掉共享中转与批次同步
    if (WB == 1) {
#pragma unroll
        for (int i = 0; i < RA; ++i)
#pragma unroll
            for (int j = 0; j < CA; ++j) {
                int gi = 行基 + 行起 + i * 8, gj = 列基 + 列起 + j * 8;
                if (gi + 7 < p.M && gj + 7 < p.N) {
                    simdgroup_matrix<half, 8, 8> h;
                    for (int t = 0; t < 2; ++t)
                        h.thread_elements()[t] = half(acc[i][j].thread_elements()[t]);
                    simdgroup_store(h, C + size_t(gi) * p.cRowStride + gj, p.cRowStride);
                }
            }
        return;
    }

    // 写回：fp32 累加器分批落到复用共享区，再 half4 向量化合并写全局
    for (int b0 = 0; b0 < BM; b0 += 批) {
        int 本批 = BM - b0 < 批 ? BM - b0 : 批;
        int sg0 = 行起 - b0;                     // 本 simdgroup 在该批内的起始行（可能越界）
        if (sg0 >= 0 && sg0 + RB <= 本批) {
#pragma unroll
            for (int i = 0; i < RA; ++i)
#pragma unroll
                for (int j = 0; j < CA; ++j)
                    simdgroup_store(acc[i][j], 写回 + (sg0 + i * 8) * BN + 列起 + j * 8, BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr int C4 = 批 * BN / 4 / NT;
        _Pragma("unroll") for (int n = 0; n < C4; ++n) {
            int e = (线 + n * NT) * 4;
            int i = e / BN, j = e % BN;
            if (i < 本批) {
                int gi = 行基 + b0 + i, gj = 列基 + j;
                if (gi < p.M && gj + 3 < p.N) {
                    half4 o = half4(half(写回[i * BN + j]), half(写回[i * BN + j + 1]),
                                    half(写回[i * BN + j + 2]), half(写回[i * BN + j + 3]));
                    *(device half4*)(C + size_t(gi) * p.cRowStride + gj) = o;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define KDEFP(NAME, BM, BN, BK, SNR, SNC, APAD, BPAD)                           \
kernel void NAME(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], \
                 device half* C [[buffer(2)]], constant WanGemmParams& p [[buffer(3)]],   \
                 uint2 组 [[threadgroup_position_in_grid]],                   \
                 uint2 tid [[thread_position_in_threadgroup]]) {              \
    threadgroup half 共享区[(BM * (BK + APAD) * 2 + BK * (BN + BPAD) * 2) / 2]; \
    gemm_body<BM, BN, BK, SNR, SNC, APAD, BPAD>(A, B, C, p, 组, tid.x, 共享区); \
}
KDEFP(wan_gemm_v5, 128, 64, 32, 8, 1, 4, 4)
