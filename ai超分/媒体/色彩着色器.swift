import Foundation

/// 色彩通路用到的 Metal 内核源码（运行时编译，不参与 Xcode 的 .metal 构建阶段）。
///
/// 只做三件事：sRGB↔线性光互转、线性空间的 Catmull-Rom 重采样、8/16bit 输出量化（带 TPDF 抖动）。
/// 8bit 端点一律用 `device uchar*` / `device ushort*` 缓冲直写，
/// 不经过 UNORM 纹理的硬件取整，保证字节精确、可比对。
nonisolated enum 色彩着色器 {
    static let 源码 = """
    #include <metal_stdlib>
    using namespace metal;

    // MARK: - 色彩传递函数（IEC 61966-2-1 精确分段）

    inline float srgb_eotf(float v) {
        v = clamp(v, 0.0f, 1.0f);
        return (v <= 0.04045f) ? (v / 12.92f) : pow((v + 0.055f) * (1.0f / 1.055f), 2.4f);
    }

    // 扩展域编码：线性值可为负（模型输出/调色可能越界），保留符号便于定位问题
    inline float srgb_oetf(float v) {
        float s = (v < 0.0f) ? -1.0f : 1.0f;
        float a = fabs(v);
        float e = (a <= 0.0031308f) ? (a * 12.92f) : (1.055f * pow(a, 1.0f / 2.4f) - 0.055f);
        return s * e;
    }

    // BT.709 编码（ITU-R BT.709-6 §1.2 推荐 8bit 形式），同样保留符号走扩展域。
    // 与 sRGB 曲线的差别在两端：近黑段 4.5 斜率（sRGB 是 12.92）、亮段指数 0.45（sRGB 是 1/2.4）。
    // 二者不可互换——8bit 字节按哪个曲线编码，写码标签就必须声哪个传递函数。
    inline float bt709_oetf(float v) {
        float s = (v < 0.0f) ? -1.0f : 1.0f;
        float a = fabs(v);
        float e = (a < 0.018f) ? (a * 4.5f) : (1.099f * pow(a, 0.45f) - 0.099f);
        return s * e;
    }

    // BT.709 解码（oetf 的严格逆函数，ITU-R BT.709-6 8bit 形式）
    inline float bt709_eotf(float v) {
        float s = (v < 0.0f) ? -1.0f : 1.0f;
        float a = fabs(v);
        float e = (a < 0.081f) ? (a / 4.5f) : pow((a + 0.099f) / 1.099f, 1.0f / 0.45f);
        return s * e;
    }

    // MARK: - 1) 8bit sRGB → 线性光 fp16

    kernel void k_srgb8_to_linear(device const uchar4 *src [[buffer(0)]],
                                  texture2d<half, access::write> dst [[texture(0)]],
                                  constant uint2 &size [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.x || gid.y >= size.y) { return; }
        uchar4 p = src[gid.y * size.x + gid.x];
        float3 lin = float3(srgb_eotf(float(p.r) / 255.0f),
                            srgb_eotf(float(p.g) / 255.0f),
                            srgb_eotf(float(p.b) / 255.0f));
        dst.write(half4(half3(lin), half(1.0)), gid);
    }

    // 8bit BT.709（ITU-R BT.709-6 OETF）→ 线性光 fp16。
    // 与 sRGB 解码严格区分：码值按哪条曲线编码，这里就必须按哪条曲线解，误差 ~10% 量级。
    kernel void k_bt7098_to_linear(device const uchar4 *src [[buffer(0)]],
                                   texture2d<half, access::write> dst [[texture(0)]],
                                   constant uint2 &size [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.x || gid.y >= size.y) { return; }
        uchar4 p = src[gid.y * size.x + gid.x];
        float3 lin = float3(bt709_eotf(float(p.r) / 255.0f),
                            bt709_eotf(float(p.g) / 255.0f),
                            bt709_eotf(float(p.b) / 255.0f));
        dst.write(half4(half3(lin), half(1.0)), gid);
    }

    // MARK: - 2) 线性光重采样（Catmull-Rom，降采样按比例展宽支撑防混叠）

    inline float cr_weight(float t) {
        t = fabs(t);
        if (t < 1.0f) { return (1.5f * t - 2.5f) * t * t + 1.0f; }
        if (t < 2.0f) { return ((-0.5f * t + 2.5f) * t - 4.0f) * t + 2.0f; }
        return 0.0f;
    }

    kernel void k_resample_linear(texture2d<float, access::read> src [[texture(0)]],
                                  texture2d<half, access::write> dst [[texture(1)]],
                                  constant float2 &scale [[buffer(0)]],
                                  constant uint2 &size [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.x || gid.y >= size.y) { return; }
        int sw = int(src.get_width());
        int sh = int(src.get_height());
        float2 p = (float2(gid) + 0.5f) * scale - 0.5f;
        float span = max(1.0f, max(scale.x, scale.y));
        float support = 2.0f * span;
        int x0 = int(floor(p.x - support));
        int x1 = int(ceil(p.x + support));
        int y0 = int(floor(p.y - support));
        int y1 = int(ceil(p.y + support));
        float3 acc = float3(0.0f);
        float wsum = 0.0f;
        for (int y = y0; y <= y1; ++y) {
            float wy = cr_weight((p.y - float(y)) / span);
            int cy = clamp(y, 0, sh - 1);
            for (int x = x0; x <= x1; ++x) {
                float w = cr_weight((p.x - float(x)) / span) * wy;
                if (w == 0.0f) { continue; }
                int cx = clamp(x, 0, sw - 1);
                acc += w * src.read(uint2(uint(cx), uint(cy))).rgb;
                wsum += w;
            }
        }
        float3 lin = (wsum != 0.0f) ? (acc / wsum) : float3(0.0f);
        dst.write(half4(half3(lin), half(1.0)), gid);
    }

    // MARK: - 3) 模型域平面 → 线性光 fp16

    kernel void k_planes_to_linear(device const float *src [[buffer(0)]],
                                   texture2d<half, access::write> dst [[texture(0)]],
                                   constant uint2 &size [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.x || gid.y >= size.y) { return; }
        uint plane = size.x * size.y;
        uint idx = gid.y * size.x + gid.x;
        float r = src[idx] * 0.5f + 0.5f;
        float g = src[plane + idx] * 0.5f + 0.5f;
        float b = src[2u * plane + idx] * 0.5f + 0.5f;
        float3 lin = float3(srgb_eotf(r), srgb_eotf(g), srgb_eotf(b));
        dst.write(half4(half3(lin), half(1.0)), gid);
    }

    // MARK: - 4) 线性调色（曝光 / 对比度 / 饱和度，全部在峰值线性光上做）

    struct GradeParams {
        float ev;
        float contrast;
        float saturation;
    };

    kernel void k_grade_linear(texture2d<float, access::read> src [[texture(0)]],
                               texture2d<half, access::write> dst [[texture(1)]],
                               constant GradeParams &p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) { return; }
        float3 c = src.read(gid).rgb;
        if (p.ev != 0.0f) { c = max(c, 0.0f) * exp2(p.ev); }
        if (p.contrast != 0.0f) {
            float k = 1.0f + p.contrast;
            c = pow(max(c, 0.0f) / 0.18f, float3(k)) * 0.18f;
        }
        if (p.saturation != 1.0f) {
            float y = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
            c = mix(float3(y), c, p.saturation);
        }
        dst.write(half4(half3(c), half(1.0)), gid);
    }

    // MARK: - 5) 输出编码：sRGB 8bit（TPDF 抖动）

    inline float hash12(uint2 p) {
        uint h = p.x * 1973u + p.y * 9277u + 26699u;
        h = (h ^ (h >> 15)) * 2246822519u;
        h = (h ^ (h >> 13)) * 3266489917u;
        h ^= h >> 16;
        return float(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
    }

    kernel void k_linear_to_out8(texture2d<float, access::read> src [[texture(0)]],
                                 device uchar4 *dst [[buffer(0)]],
                                 constant uint2 &size [[buffer(1)]],
                                 constant uint &dither [[buffer(2)]],
                                 constant uint &transfer [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= size.x || gid.y >= size.y) { return; }
        float3 c = src.read(gid).rgb;
        // 编码曲线由调用方按写码标签选定：0=IEC 61966-2-1(sRGB)，1=ITU-R BT.709。
        // 字节编码与容器标签必须成对，否则播放器按标签反解就会整体偏亮/偏暗。
        float3 enc = (transfer == 1u)
            ? float3(bt709_oetf(c.r), bt709_oetf(c.g), bt709_oetf(c.b))
            : float3(srgb_oetf(c.r), srgb_oetf(c.g), srgb_oetf(c.b));
        enc = clamp(enc, 0.0f, 1.0f) * 255.0f;
        if (dither != 0u) {
            // TPDF：两组均匀噪声相减，幅度 ±1 LSB，把 8bit 量化台阶摊成噪声，消除渐变 banding
            float d0 = hash12(gid) - hash12(gid + uint2(1u, 7919u));
            float d1 = hash12(gid + uint2(104729u, 3u)) - hash12(gid + uint2(31u, 6151u));
            enc += float3(d0, d1, (d0 + d1) * 0.5f);
        }
        float3 q = clamp(floor(enc + 0.5f), 0.0f, 255.0f);
        dst[gid.y * size.x + gid.x] = uchar4(uchar(q.r), uchar(q.g), uchar(q.b), 255);
    }
    """
}
