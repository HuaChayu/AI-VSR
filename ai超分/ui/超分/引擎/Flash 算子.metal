//
//  Flash 算子.metal
//  ai超分
//
//  RelayVSR Flash 条件主干在 Metal 上的算子集合。
//  所有激活张量统一为按行主序的 float32：[tokens, 通道]。
//  与 Python 参考实现（refvsr_inference/models/flash）逐位对齐，
//  数值约定：fp32、RMSNorm 用 rsqrt(var + eps)、RoPE 为相邻奇偶通道成对旋转。
//

#include <metal_stdlib>
using namespace metal;

#include <metal_simdgroup_matrix>

// MARK: - 通用参数

struct Im2ColParams {
    int inH;
    int inW;
    int inC;
    int kernelSize;
    int stride;
    int pad;
    int outH;
    int outW;
    int outCols;   // 补齐到 16 字节对齐后的列数
    int realCols;  // inC * kernel * kernel
    int channelsLast;  // 1 表示输入为 [H,W,C] 布局
};

struct BiasParams {
    int rows;
    int cols;
    int hasBias;
    int act;       // 0 = 无激活，1 = SiLU
};

struct NormParams {
    int rows;
    int cols;
    float eps;
};

struct RoPEParams {
    int tokens;
    int heads;
    int width;
    int temporalPosition;
    float theta;
    int temporalDim;
    int heightDim;
    int widthDim;
    int headDim;
    int planeTokens;
    int planeStep;      // 每平面时间位置步长：流式历史帧首平面 t=temporalPosition，逐平面 +planeStep
};

struct AttnParams {
    int batchTokens;    // 每个 batch 的 token 数（height * width）
    int batches;
    int height;
    int width;
    int heads;
    int kvGroups;
    int headDim;
    int planes;         // 每个 batch 的 KV 平面数
    int condPlanes;     // 其中的条件平面数
    int temporalWindow;
    int hasPlaneMask;
};

struct PixelShuffleParams {
    int tokens;
    int tokenWidth;
    int scale;          // 16
    int channels;       // 3
    int outWidth;
    int outHeight;
};

struct BicubicParams {
    int inH;
    int inW;
    int outH;
    int outW;
    int scale;          // 4
    int channels;       // 3
};

struct CopyParams {
    int count;
};

// MARK: - im2col

kernel void flash_im2col(device const float* src [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         constant Im2ColParams& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]])
{
    int row = gid.x;
    int col = gid.y;
    if (row >= p.outH * p.outW || col >= p.outCols) return;

    float value = 0.0f;
    if (col < p.realCols) {
        int oy = row / p.outW;
        int ox = row - oy * p.outW;
        int kernelArea = p.kernelSize * p.kernelSize;
        int channel = col / kernelArea;
        int rest = col - channel * kernelArea;
        int ky = rest / p.kernelSize;
        int kx = rest - ky * p.kernelSize;
        int iy = oy * p.stride - p.pad + ky;
        int ix = ox * p.stride - p.pad + kx;
        if (iy >= 0 && iy < p.inH && ix >= 0 && ix < p.inW) {
            int index = (p.channelsLast != 0)
                ? ((iy * p.inW + ix) * p.inC + channel)
                : ((channel * p.inH + iy) * p.inW + ix);
            value = src[index];
        }
    }
    dst[row * p.outCols + col] = value;
}

// MARK: - 偏置与激活

kernel void flash_bias_act(device float* x [[buffer(0)]],
                           device const float* bias [[buffer(1)]],
                           constant BiasParams& p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    int row = gid.x;
    int col = gid.y;
    if (row >= p.rows || col >= p.cols) return;
    int index = row * p.cols + col;
    float value = x[index];
    if (p.hasBias != 0) value += bias[col];
    if (p.act == 1) value = value / (1.0f + exp(-value));
    x[index] = value;
}

// MARK: - RMSNorm

kernel void flash_rmsnorm(device const float* x [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant NormParams& p [[buffer(3)]],
                          uint groupId [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]],
                          uint groupSize [[threads_per_threadgroup]])
{
    int row = int(groupId);
    int base = row * p.cols;

    float accSum = 0.0f;
    for (int i = int(lane); i < p.cols; i += int(groupSize)) {
        float v = x[base + i];
        accSum += v * v;
    }
    accSum = simd_sum(accSum);

    threadgroup float partial[32];
    int warp = int(lane) / 32;
    int warpLane = int(lane) % 32;
    if (warpLane == 0) partial[warp] = accSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        int warps = int(groupSize) / 32;
        float v = (warpLane < warps) ? partial[warpLane] : 0.0f;
        v = simd_sum(v);
        if (warpLane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / float(p.cols) + p.eps);
    for (int i = int(lane); i < p.cols; i += int(groupSize)) {
        out[base + i] = x[base + i] * scale * weight[i];
    }
}

// MARK: - 3D RoPE

static inline float flash_inv_freq(float theta, int pairIndex, int dim)
{
    float exponent = float(2 * pairIndex) / float(dim);
    return 1.0f / pow(theta, exponent);
}

kernel void flash_rope(device float* x [[buffer(0)]],
                       constant RoPEParams& p [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    int token = int(gid.x);
    int head = int(gid.y);
    if (token >= p.tokens || head >= p.heads) return;

    int local_token = p.planeTokens > 0 ? token % p.planeTokens : token;
    int plane = p.planeTokens > 0 ? token / p.planeTokens : 0;
    // 流式历史帧：每平面用各自的时间位置（首平面 temporalPosition，逐平面 +planeStep）
    int temporalPosition = p.temporalPosition + plane * p.planeStep;
    int row = local_token / p.width;
    int col = local_token - row * p.width;
    int halfDim = p.headDim / 2;
    device float* base = x + (token * p.heads + head) * p.headDim;

    for (int pair = 0; pair < halfDim; pair++) {
        int d0 = pair * 2;
        float angle = 0.0f;
        if (d0 < p.temporalDim) {
            angle = float(temporalPosition) * flash_inv_freq(p.theta, pair, p.temporalDim);
        } else if (d0 < p.temporalDim + p.heightDim) {
            int local = d0 - p.temporalDim;
            angle = float(row) * flash_inv_freq(p.theta, local / 2, p.heightDim);
        } else {
            int local = d0 - p.temporalDim - p.heightDim;
            angle = float(col) * flash_inv_freq(p.theta, local / 2, p.widthDim);
        }
        float c = cos(angle);
        float s = sin(angle);
        float even = base[d0];
        float odd = base[d0 + 1];
        base[d0] = even * c - odd * s;
        base[d0 + 1] = even * s + odd * c;
    }
}

// MARK: - SwiGLU

kernel void flash_swiglu(device const float* gateUp [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant BiasParams& p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]])
{
    int row = int(gid.x);
    int col = int(gid.y);
    if (row >= p.rows || col >= p.cols) return;
    int hidden = p.cols;
    float gate = gateUp[row * (2 * hidden) + col];
    float up = gateUp[row * (2 * hidden) + hidden + col];
    float silu = gate / (1.0f + exp(-gate));
    out[row * hidden + col] = silu * up;
}

// MARK: - 逐元素运算

kernel void flash_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant CopyParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= p.count) return;
    out[i] = a[i] + b[i];
}

kernel void flash_add_inplace(device float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              constant CopyParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= p.count) return;
    a[i] = a[i] + b[i];
}

kernel void flash_add_row_vector(device float* x [[buffer(0)]],
                                 device const float* vec [[buffer(1)]],
                                 constant BiasParams& p [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    int row = int(gid.x);
    int col = int(gid.y);
    if (row >= p.rows || col >= p.cols) return;
    x[row * p.cols + col] += vec[col];
}

kernel void flash_copy(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant CopyParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= p.count) return;
    dst[i] = src[i];
}

// MARK: - pixel shuffle

kernel void flash_pixel_shuffle(device const float* src [[buffer(0)]],
                                device float* dst [[buffer(1)]],
                                constant PixelShuffleParams& p [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    int ox = int(gid.x);
    int oy = int(gid.y);
    if (ox >= p.outWidth || oy >= p.outHeight) return;

    int tokenX = ox / p.scale;
    int subX = ox - tokenX * p.scale;
    int tokenY = oy / p.scale;
    int subY = oy - tokenY * p.scale;
    int token = tokenY * p.tokenWidth + tokenX;
    int sub = subY * p.scale + subX;
    int channelStride = p.scale * p.scale;

    for (int c = 0; c < p.channels; c++) {
        dst[(c * p.outHeight + oy) * p.outWidth + ox] =
            src[token * (p.channels * channelStride) + c * channelStride + sub];
    }
}

// MARK: - 双三次上采样 + 残差

static inline float flash_cubic_weight(float t)
{
    // PyTorch F.interpolate(mode="bicubic") 使用 a = -0.75
    const float a = -0.75f;
    float x = fabs(t);
    if (x <= 1.0f) {
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return ((x - 5.0f) * x + 8.0f) * x * a - 4.0f * a;
    }
    return 0.0f;
}

kernel void flash_bicubic_add(device const float* lr [[buffer(0)]],
                              device const float* residual [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant BicubicParams& p [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]])
{
    int ox = int(gid.x);
    int oy = int(gid.y);
    if (ox >= p.outW || oy >= p.outH) return;

    float sx = (float(ox) + 0.5f) / float(p.scale) - 0.5f;
    float sy = (float(oy) + 0.5f) / float(p.scale) - 0.5f;
    int ix0 = int(floor(sx));
    int iy0 = int(floor(sy));
    float fx = sx - float(ix0);
    float fy = sy - float(iy0);

    float wx[4];
    float wy[4];
    wx[0] = flash_cubic_weight(1.0f + fx);
    wx[1] = flash_cubic_weight(fx);
    wx[2] = flash_cubic_weight(1.0f - fx);
    wx[3] = flash_cubic_weight(2.0f - fx);
    wy[0] = flash_cubic_weight(1.0f + fy);
    wy[1] = flash_cubic_weight(fy);
    wy[2] = flash_cubic_weight(1.0f - fy);
    wy[3] = flash_cubic_weight(2.0f - fy);

    int xs[4];
    int ys[4];
    for (int i = 0; i < 4; i++) {
        xs[i] = min(max(ix0 - 1 + i, 0), p.inW - 1);
        ys[i] = min(max(iy0 - 1 + i, 0), p.inH - 1);
    }

    for (int c = 0; c < p.channels; c++) {
        float acc = 0.0f;
        device const float* plane = lr + c * p.inH * p.inW;
        for (int j = 0; j < 4; j++) {
            float rowAcc = 0.0f;
            for (int i = 0; i < 4; i++) {
                rowAcc += wx[i] * plane[ys[j] * p.inW + xs[i]];
            }
            acc += wy[j] * rowAcc;
        }
        int index = (c * p.outH + oy) * p.outW + ox;
        out[index] = acc + residual[index];
    }
}

// ===== simdgroup 硬件 MMA 版窗口注意力（单遍在线 softmax：运行最大只增不减，边算边归一化）=====

// MARK: - 窗口化 GQA：simdgroup 硬件 MMA 版（fp16 输入 / fp32 累加，保留 online softmax 与窗口掩码）
//
//  设计要点
//  1) 查询瓦片按 QP 编译期固定（32 / 64 / 96），线程数 = QP×4，simdgroup 数 = QP/8，
//     每个 simdgroup 用 simdgroup_matrix<half,8,8> 承担 8 个查询。QP=96 时一个线程组覆盖整行
//     88 个令牌，K/V 从全局只读一遍（QP=32 时要按列组读三遍），全局读取量降为 1/3。
//  2) QKᵀ（8 查询 × 64 维 · 64 维 × 32 键）与 PV（8 查询 × 32 键 · 32 键 × 64 维）都用
//     simdgroup_multiply_accumulate，fp16 输入、fp32 累加。
//  3) 分数/P 的掩码与 softmax 一律在共享内存上做，不依赖 fragment 的线程内元素布局；
//     O 的在线 rescale 是整组统一标量，直接乘到 thread_elements 上（与布局无关）。
//  4) Q 只在装载期借用分数缓冲，不单独占共享内存：QP=96 时静态占用 24.1KB（15.4+8.7）。
//     共享行跨取 68/36/40（136B/72B/160B）均为 16 字节整数倍且非 128B 整倍数，避开 bank 冲突；
//     8×8 瓦片的起始行恒为 8 的倍数，故每次 simdgroup_load/store 地址天然 16B 对齐。



#define ATTN_QS  68    // Q 暂存行跨（half）
#define ATTN_KS(KB) ((KB) + 4)   // Kᵀ 键跨（half）
#define ATTN_VS  68    // V 行跨（half）
// K/V 全局取数：float 与 half 两种输入各一版，供 f32q / f32q_h 共用同一实现
static inline float4 attn_取四(device const float* 基) { return *(device const float4*)基; }
static inline float4 attn_取四(device const half* 基) { half4 h = *(device const half4*)基; return float4(h); }

// 分数/P 行跨：既要放得下 KB 个分数，也要放得下 Q 暂存（ATTN_QS/2 = 34 个 float），
// 取两者较大值，避免 QP 较大时 Q 暂存越界写坏后续缓冲。行字节 144/160 非 128 整倍数，避开 bank 冲突。
#define ATTN_SS(KB) (((KB) + 8) > 36 ? ((KB) + 8) : 36)

template <int QP, int KB, typename KVT>
static inline void attn_mma_impl(device const float* q,
                                 device const KVT* k,
                                 device const KVT* v,
                                 device float* out,
                                 device const int2* windows,
                                 device const uchar* planeMask,
                                 constant AttnParams& p,
                                 uint gidRaw,
                                 uint 线程,
                                 uint 组大小,
                                 threadgroup float* Ss,
                                 threadgroup half* Ps,
                                 threadgroup half* Kt,
                                 threadgroup half* Vs)
{
    if (p.headDim != 64) { return; }
    if (int(组大小) != QP * 4) { return; }

    const int headsPerGroup = p.heads / p.kvGroups;
    const int 列组数 = (p.width + QP - 1) / QP;
    const int 总组数 = p.batches * p.height * p.heads * 列组数;
    const int gid = int(gidRaw);
    if (gid >= 总组数) { return; }

    const int 列组 = gid % 列组数;
    int 剩余 = gid / 列组数;
    const int head = 剩余 % p.heads;
    剩余 /= p.heads;
    const int row = 剩余 % p.height;
    const int batch = 剩余 / p.height;
    const int group = head / headsPerGroup;
    const int colFirst = 列组 * QP;

    const int 群 = int(线程 >> 5);
    const int 行起 = 群 * 8;
    const int qStride = p.heads * p.headDim;
    const int colStride = p.kvGroups * p.headDim;
    const int groupOff = group * p.headDim;

    // ---- Q 暂存（借用分数缓冲），随后装入 fragment ----
    threadgroup half* Qs = (threadgroup half*)Ss;
    for (int idx = int(线程); idx < QP * 64; idx += QP * 4) {
        int r = idx >> 6;
        int d = idx & 63;
        int c = colFirst + r;
        half 值 = half(0.0f);
        if (c < p.width) {
            int 令牌 = row * p.width + c;
            值 = half(q[(batch * p.batchTokens + 令牌) * qStride + head * 64 + d]);
        }
        Qs[r * ATTN_QS + d] = 值;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half, 8, 8> Qf[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        simdgroup_load(Qf[kk], Qs + 行起 * ATTN_QS + kk * 8, ATTN_QS);
    }

    simdgroup_matrix<float, 8, 8> Of[8];
#pragma unroll
    for (int jd = 0; jd < 8; ++jd) {
        Of[jd] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const int 我查询 = int(线程) >> 2;
    const int 我键片 = int(线程) & 3;

    float 行最大 = -INFINITY;
    float 行和 = 0.0f;
    const float invDimLog2 = rsqrt(float(p.headDim)) * 1.4426950408889634f;
    const int 平面基 = batch * p.planes;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 单遍在线 softmax（FA2 式）：运行最大只增不减 ⇒ 权重恒 ≤ 1、重标定因子恒 ≤ 1，
    // 累加器为 float，数值安全；省掉原两遍法重复的一遍 K 载入与 QK^T（微基准实测 −13%）。
    // 累加器元素行归属：同线程两元素同属一行，行 = 4*(lane/16) + (lane%8)/2
    const int 我元素行 = ((int(线程) & 16) != 0 ? 4 : 0) + ((int(线程) & 7) >> 1);
    {
    for (int plane = 0; plane < p.planes; plane++) {
        if (p.hasPlaneMask != 0 && planeMask[平面基 + plane] == 0) { continue; }

        if (plane >= p.condPlanes) {
            int relative = plane - (p.planes - 1);
            if (relative > 0) { continue; }
            if (p.temporalWindow > 0 && relative <= -p.temporalWindow) { continue; }
        }

        int2 窗口 = windows[plane];
        int winH = min(窗口.x, p.height);
        int winW = min(窗口.y, p.width);
        if (winH <= 0 || winW <= 0) { continue; }
        int rowLow = min(max(row - winH / 2, 0), p.height - winH);

        int uLow = p.width;
        int uHigh = 0;
        for (int r = 0; r < QP; r++) {
            int c = min(colFirst + r, p.width - 1);
            int lo = min(max(c - winW / 2, 0), p.width - winW);
            uLow = min(uLow, lo);
            uHigh = max(uHigh, lo + winW);
        }
        const bool 满窗 = (winW >= p.width);

        int 我列 = min(colFirst + 我查询, p.width - 1);
        int 我低 = min(max(我列 - winW / 2, 0), p.width - winW);
        int 我高 = 我低 + winW;

        int 平面令牌 = (平面基 + plane) * p.batchTokens;

        for (int rr = rowLow; rr < rowLow + winH; rr++) {
            int 键行基 = (平面令牌 + rr * p.width) * colStride + groupOff;
            for (int c0 = uLow; c0 < uHigh; c0 += KB) {
                for (int idx = int(线程); idx < KB * 64 / 4; idx += QP * 4) {
                    int kk = idx / 16;
                    int d = (idx % 16) * 4;
                    int c = c0 + kk;
                    const bool 有效 = (c < uHigh && c < p.width);
                    float4 kv = 有效 ? attn_取四(k + 键行基 + c * colStride + d)
                                     : float4(0.0f, 0.0f, 0.0f, 0.0f);
                    Kt[(d + 0) * ATTN_KS(KB) + kk] = half(kv.x);
                    Kt[(d + 1) * ATTN_KS(KB) + kk] = half(kv.y);
                    Kt[(d + 2) * ATTN_KS(KB) + kk] = half(kv.z);
                    Kt[(d + 3) * ATTN_KS(KB) + kk] = half(kv.w);
                }
                {
                    for (int idx = int(线程); idx < KB * 64 / 4; idx += QP * 4) {
                        int kk = idx / 16;
                        int d = (idx % 16) * 4;
                        int c = c0 + kk;
                        const bool 有效 = (c < uHigh && c < p.width);
                        float4 vv = 有效 ? attn_取四(v + 键行基 + c * colStride + d)
                                         : float4(0.0f, 0.0f, 0.0f, 0.0f);
                        *(threadgroup half4*)(&Vs[kk * ATTN_VS + d]) = half4(half(vv.x), half(vv.y), half(vv.z), half(vv.w));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_matrix<float, 8, 8> Sf[KB / 8];
#pragma unroll
                for (int ct = 0; ct < KB / 8; ++ct) {
                    Sf[ct] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
                simdgroup_matrix<half, 8, 8> Kf[KB / 8];
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                    for (int ct = 0; ct < KB / 8; ++ct) {
                        simdgroup_load(Kf[ct], Kt + (kk * 8) * ATTN_KS(KB) + ct * 8, ATTN_KS(KB));
                    }
#pragma unroll
                    for (int ct = 0; ct < KB / 8; ++ct) {
                        simdgroup_multiply_accumulate(Sf[ct], Qf[kk], Kf[ct], Sf[ct]);
                    }
                }
#pragma unroll
                for (int ct = 0; ct < KB / 8; ++ct) {
                    auto& 元素 = Sf[ct].thread_elements();
#pragma unroll
                    for (int e = 0; e < 2; ++e) { 元素[e] *= invDimLog2; }
                    simdgroup_store(Sf[ct], Ss + 行起 * ATTN_SS(KB) + ct * 8, ATTN_SS(KB));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float 分数[KB / 4];
#pragma unroll
                for (int e = 0; e < KB / 4; ++e) {
                    int c = c0 + 我键片 * (KB / 4) + e;
                    float s = Ss[我查询 * ATTN_SS(KB) + 我键片 * (KB / 4) + e];
                    if (c >= 我高 || (!满窗 && c < 我低)) { s = -1.0e4f; }
                    分数[e] = s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                {
                    float 局最大 = -INFINITY;
#pragma unroll
                    for (int e = 0; e < KB / 4; ++e) { 局最大 = max(局最大, 分数[e]); }
                    局最大 = max(局最大, simd_shuffle_xor(局最大, 1));
                    局最大 = max(局最大, simd_shuffle_xor(局最大, 2));
                    const float 新最大 = max(行最大, 局最大);
                    const bool 首瓦 = (行最大 == -INFINITY);
                    const float 因子 = 首瓦 ? 0.0f : exp2(行最大 - 新最大);
                    if (首瓦) { 行和 = 0.0f; }
                    // 条件整 warp 一致，不产生发散；仅在运行最大被刷新时重标定
                    if (!首瓦 && simd_any(因子 < 1.0f)) {
                        // 查询行 r 的因子由 lane 4r 广播给持有该行累加器元素的线程
                        const float 我因子 = simd_shuffle(因子, ushort(4 * 我元素行));
                        行和 *= 因子;
#pragma unroll
                        for (int jd = 0; jd < 8; ++jd) {
                            auto& o元素 = Of[jd].thread_elements();
#pragma unroll
                            for (int oe = 0; oe < 2; ++oe) { o元素[oe] *= 我因子; }
                        }
                    }
                    行最大 = 新最大;

                    float 局部和 = 0.0f;
#pragma unroll
                    for (int e = 0; e < KB / 4; ++e) {
                        float w = exp2(分数[e] - 行最大);
                        局部和 += w;
                        Ps[我查询 * ATTN_SS(KB) + 我键片 * (KB / 4) + e] = half(w);
                    }
                    行和 += 局部和;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_matrix<half, 8, 8> Pf[KB / 8];
#pragma unroll
                    for (int kc = 0; kc < KB / 8; ++kc) {
                        simdgroup_load(Pf[kc], Ps + 行起 * ATTN_SS(KB) + kc * 8, ATTN_SS(KB));
                    }
                    simdgroup_matrix<half, 8, 8> Vf[8];
#pragma unroll
                    for (int kc = 0; kc < KB / 8; ++kc) {
#pragma unroll
                        for (int jd = 0; jd < 8; ++jd) {
                            simdgroup_load(Vf[jd], Vs + (kc * 8) * ATTN_VS + jd * 8, ATTN_VS);
                        }
#pragma unroll
                        for (int jd = 0; jd < 8; ++jd) {
                            simdgroup_multiply_accumulate(Of[jd], Pf[kc], Vf[jd], Of[jd]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }
    }

    // 分担同一查询的 4 个线程合并权重和
    float 总和 = 行和 + simd_shuffle_xor(行和, 1);
    总和 += simd_shuffle_xor(总和, 2);
    float 倒数 = (总和 > 0.0f) ? (1.0f / 总和) : 0.0f;

    // 输出暂存借 Ss（64 维分两批，避免申请 QP*64 float 的大块共享内存）
    threadgroup float* Ob = (threadgroup float*)Ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int 半 = 0; 半 < 2; ++半) {
#pragma unroll
        for (int jd = 0; jd < 4; ++jd) {
            simdgroup_store(Of[半 * 4 + jd], Ob + 行起 * 32 + jd * 8, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int c = colFirst + 我查询;
        if (c < p.width) {
            int 令牌 = row * p.width + c;
            int 基 = (batch * p.batchTokens + 令牌) * qStride + head * 64;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                int d = 我键片 * 16 + e;
                if (d >= 半 * 32 && d < 半 * 32 + 32) {
                    out[基 + d] = Ob[我查询 * 32 + (d - 半 * 32)] * 倒数;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define ATTN_KERNEL(名称, QP, KB, 类型)                                                       \
kernel void 名称(device const float* q [[buffer(0)]],                                    \
                 device const 类型* k [[buffer(1)]],                                      \
                 device const 类型* v [[buffer(2)]],                                      \
                 device float* out [[buffer(3)]],                                         \
                 device const int2* windows [[buffer(4)]],                                \
                 device const uchar* planeMask [[buffer(5)]],                             \
                 constant AttnParams& p [[buffer(6)]],                                    \
                 uint groupId [[threadgroup_position_in_grid]],                            \
                 uint 线程 [[thread_position_in_threadgroup]],                             \
                 uint 组大小 [[threads_per_threadgroup]])                                  \
{                                                                                         \
    threadgroup float Ss[QP * ((ATTN_SS(KB) > 34) ? ATTN_SS(KB) : 34)];                       \
    threadgroup half Ps[QP * ATTN_SS(KB)];                                                \
    threadgroup half Kt[64 * ATTN_KS(KB)];                                                \
    threadgroup half Vs[KB * ATTN_VS];                                               \
    attn_mma_impl<QP, KB, 类型>(q, k, v, out, windows, planeMask, p, groupId, 线程, 组大小,   \
                            Ss, Ps, Kt, Vs);                                               \
}

// ---- 实例化：每线程组 32 查询（8 查询/simdgroup，4 个 simdgroup），KB 16 / 32 两档 ----
ATTN_KERNEL(flash_attention_mma_32_16,   32, 16, float)
ATTN_KERNEL(flash_attention_mma_h_32_16, 32, 16, half)
ATTN_KERNEL(flash_attention_mma_h_16_16, 16, 16, half)
ATTN_KERNEL(flash_attention_mma_16_16,   16, 16, float)
ATTN_KERNEL(flash_attention_mma_32_32,   32, 32, float)
ATTN_KERNEL(flash_attention_mma_h_32_32, 32, 32, half)


// MARK: - KV 半精度（与官方 KV cache 的 bf16 等宽表示，省一半显存）

kernel void flash_cast_f2h(device const float* src [[buffer(0)]],
                           device half* dst [[buffer(1)]],
                           constant CopyParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.count) return;
    dst[gid] = half(src[gid]);
}

kernel void flash_rope_half(device half* x [[buffer(0)]],
                            constant RoPEParams& p [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]])
{
    int token = int(gid.x);
    int head = int(gid.y);
    if (token >= p.tokens || head >= p.heads) return;

    int local_token = p.planeTokens > 0 ? token % p.planeTokens : token;
    int plane = p.planeTokens > 0 ? token / p.planeTokens : 0;
    int temporalPosition = p.temporalPosition + plane * p.planeStep;
    int row = local_token / p.width;
    int col = local_token - row * p.width;
    int halfDim = p.headDim / 2;
    device half* base = x + (token * p.heads + head) * p.headDim;

    for (int pair = 0; pair < halfDim; pair++) {
        int d0 = pair * 2;
        float angle = 0.0f;
        if (d0 < p.temporalDim) {
            angle = float(temporalPosition) * flash_inv_freq(p.theta, pair, p.temporalDim);
        } else if (d0 < p.temporalDim + p.heightDim) {
            int local = d0 - p.temporalDim;
            angle = float(row) * flash_inv_freq(p.theta, local / 2, p.heightDim);
        } else {
            int local = d0 - p.temporalDim - p.heightDim;
            angle = float(col) * flash_inv_freq(p.theta, local / 2, p.widthDim);
        }
        float c = cos(angle);
        float s = sin(angle);
        float even = float(base[d0]);
        float odd = float(base[d0 + 1]);
        base[d0] = half(even * c - odd * s);
        base[d0 + 1] = half(even * s + odd * c);
    }
}

// ---------------------------------------------------------------------------
// 向量化版激活/门控：每线程处理 4 个连续列（float4），线程组内沿列方向相邻线程
// 访问连续地址，退化为 128B 整块事务。旧标量版每线程组仅 16 线程、且沿行方向
// 取址，实测 128 次 bias_act 合计 136 ms、单次最高 61 ms，改为向量化后显著下降。
// 数值路径与标量版完全一致（同样的 fp32 加偏置、同样的 exp 形式 SiLU）。
// ---------------------------------------------------------------------------
kernel void flash_bias_act_v4(device float* xPtr [[buffer(0)]],
                              device const float* bPtr [[buffer(1)]],
                              constant BiasParams& p [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]])
{
    int c4 = int(gid.x), row = int(gid.y);
    int 列组 = p.cols >> 2;
    if (c4 >= 列组 || row >= p.rows) return;
    device float4* x = (device float4*)xPtr;
    device const float4* b = (device const float4*)bPtr;
    size_t idx = size_t(row) * 列组 + c4;
    float4 v = x[idx];
    if (p.hasBias != 0) {
        float4 bb = b[c4];
        v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
    }
    if (p.act == 1) {
        v.x = v.x / (1.0f + exp(-v.x));
        v.y = v.y / (1.0f + exp(-v.y));
        v.z = v.z / (1.0f + exp(-v.z));
        v.w = v.w / (1.0f + exp(-v.w));
    }
    x[idx] = v;
}

kernel void flash_swiglu_v4(device const float* inPtr [[buffer(0)]],
                            device float* outPtr [[buffer(1)]],
                            constant BiasParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    int c4 = int(gid.x), row = int(gid.y);
    int 列组 = p.cols >> 2;
    if (c4 >= 列组 || row >= p.rows) return;
    device const float4* in = (device const float4*)inPtr;
    device float4* out = (device float4*)outPtr;
    size_t 行基 = size_t(row) * 列组 * 2;
    float4 g = in[行基 + c4];
    float4 u = in[行基 + 列组 + c4];
    float4 r;
    r.x = (g.x / (1.0f + exp(-g.x))) * u.x;
    r.y = (g.y / (1.0f + exp(-g.y))) * u.y;
    r.z = (g.z / (1.0f + exp(-g.z))) * u.z;
    r.w = (g.w / (1.0f + exp(-g.w))) * u.w;
    out[size_t(row) * 列组 + c4] = r;
}

// 融合：gate 侧就地加偏置并 SiLU，再与 up 相乘，省掉一次 2H 的读改写
kernel void flash_swiglu_bias_v4(device const float* inPtr [[buffer(0)]],
                                 device const float* bPtr [[buffer(1)]],
                                 device float* outPtr [[buffer(2)]],
                                 constant BiasParams& p [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    int c4 = int(gid.x), row = int(gid.y);
    int 列组 = p.cols >> 2;
    if (c4 >= 列组 || row >= p.rows) return;
    device const float4* in = (device const float4*)inPtr;
    device const float4* b = (device const float4*)bPtr;
    device float4* out = (device float4*)outPtr;
    size_t 行基 = size_t(row) * 列组 * 2;
    float4 g = in[行基 + c4];
    float4 u = in[行基 + 列组 + c4];
    if (p.hasBias != 0) {
        float4 bb = b[c4];
        g.x += bb.x; g.y += bb.y; g.z += bb.z; g.w += bb.w;
    }
    float4 r;
    r.x = (g.x / (1.0f + exp(-g.x))) * u.x;
    r.y = (g.y / (1.0f + exp(-g.y))) * u.y;
    r.z = (g.z / (1.0f + exp(-g.z))) * u.z;
    r.w = (g.w / (1.0f + exp(-g.w))) * u.w;
    out[size_t(row) * 列组 + c4] = r;
}
