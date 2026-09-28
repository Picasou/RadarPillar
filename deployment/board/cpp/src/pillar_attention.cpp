#include "pillar_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "thread_split.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#define PF_ATTN_NEON 1
#endif

namespace {
constexpr int KTILE = 64;   // key 分块: k/v 块驻 L1(b3 病根: 每行重读整张量)

inline float fast_expf(float x) {
    /* 快速 exp: x∈[-45,0] → 2^(x·log2e) 拆 阶码+尾段, 尾段6次多项式(chebfit, max_err 6.5e-9) */
    const float t = x * 1.442695041f;
    const float ft = std::floor(t);
    const float r = t - ft;
    const float p = (((((2.18712544e-04f * r + 1.23824100e-03f) * r + 9.68618799e-03f) * r
                      + 5.54789108e-02f) * r + 2.40231097e-01f) * r + 6.93146838e-01f) * r
                    + 1.00000001e+00f;
    const uint32_t bits = (uint32_t)((int32_t)ft + 127) << 23;   // x≤0 → 阶码∈[82,127] 无下溢
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// LayerNorm(eps, weight, bias) 就地: D 维, 方差用有偏估计(/D), 内部 double 保留原口径
void layer_norm(const float* x, float* out, int D, float eps,
                const float* w, const float* b) {
    double mean = 0.;
    for (int i = 0; i < D; ++i) mean += x[i];
    mean /= (double)D;
    double var = 0.;
    for (int i = 0; i < D; ++i) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= (double)D;
    float inv = 1.f / std::sqrt((float)var + eps);
    for (int i = 0; i < D; ++i) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// 转置权重 [out][in] → wt[in][out] (内层 out 连续, 供向量化累加)
void transpose_weight(const float* W, float* wt, int in, int out) {
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i)
            wt[(size_t)i * out + o] = W[(size_t)o * in + i];
}

// y = x·W^T + b (单行): y 先置 b, 再沿 in 维外积累加, 内层 out 连续无归约
void linear_row(const float* x, float* y, const float* wt, const float* b,
                int in, int out) {
    for (int o = 0; o < out; ++o) y[o] = b[o];
    for (int k = 0; k < in; ++k) {
        const float v = x[k];
        const float* wrow = wt + (size_t)k * out;
        for (int o = 0; o < out; ++o) y[o] += wrow[o] * v;
    }
}

#if PF_ATTN_NEON
// scores 单行微核: pi[j] = Σ_c q_row[c]·k_t[c][j], j 16 块 × c 8 段, 累加器驻 4 个 q 寄存器;
// 每元素加法序仍为 c 升序(段间/段内均升序), 与 b3 逐位同序; k_t 半段 15.5KB 驻 L1
void scores_row_neon(const float* q_row, const float* k_t, float* pi, int M, int D) {
    for (int c0 = 0; c0 < D; c0 += 8) {
        for (int jt = 0; jt < M; jt += 16) {
            const int je = std::min(jt + 16, M);
            if (je - jt < 16) {                     // 尾块标量, 序不变
                for (int c = c0; c < c0 + 8; ++c) {
                    const float qc = q_row[c];
                    const float* kr = k_t + (size_t)c * M;
                    for (int j = jt; j < je; ++j) pi[j] += kr[j] * qc;
                }
                continue;
            }
            float32x4_t a0, a1, a2, a3;
            if (c0 == 0) {
                a0 = a1 = a2 = a3 = vdupq_n_f32(0.f);
            } else {
                a0 = vld1q_f32(pi + jt);
                a1 = vld1q_f32(pi + jt + 4);
                a2 = vld1q_f32(pi + jt + 8);
                a3 = vld1q_f32(pi + jt + 12);
            }
            for (int c = c0; c < c0 + 8; ++c) {
                const float32x4_t qc = vdupq_n_f32(q_row[c]);
                const float* kr = k_t + (size_t)c * M;
                a0 = vfmaq_f32(a0, vld1q_f32(kr + jt), qc);
                a1 = vfmaq_f32(a1, vld1q_f32(kr + jt + 4), qc);
                a2 = vfmaq_f32(a2, vld1q_f32(kr + jt + 8), qc);
                a3 = vfmaq_f32(a3, vld1q_f32(kr + jt + 12), qc);
            }
            vst1q_f32(pi + jt, a0);
            vst1q_f32(pi + jt + 4, a1);
            vst1q_f32(pi + jt + 8, a2);
            vst1q_f32(pi + jt + 12, a3);
        }
    }
}

// AV 微核: attn[i][0:32] = Σ_j P[i][j]·vpack[j], 32 累加器驻 8 个 q 寄存器(跨全 j 不落内存);
// i 成对处理摊薄 v 读流量; 每元素仍 j 升序 fma, 与 b3 同序
void av_rows_neon(const float* P, const float* vpack, float* attn,
                  int M, int i0, int i1) {
    for (int i = i0; i + 1 < i1; i += 2) {
        const float* p0 = P + (size_t)i * M;
        const float* p1 = P + (size_t)(i + 1) * M;
        float32x4_t a[8], b[8];
        for (int k = 0; k < 8; ++k) { a[k] = b[k] = vdupq_n_f32(0.f); }
        for (int j = 0; j < M; ++j) {
            const float* vj = vpack + (size_t)j * 32;
            const float32x4_t pv0 = vdupq_n_f32(p0[j]);
            const float32x4_t pv1 = vdupq_n_f32(p1[j]);
            for (int k = 0; k < 8; ++k) {
                const float32x4_t v = vld1q_f32(vj + 4 * k);
                a[k] = vfmaq_f32(a[k], v, pv0);
                b[k] = vfmaq_f32(b[k], v, pv1);
            }
        }
        for (int k = 0; k < 8; ++k) {
            vst1q_f32(attn + (size_t)i * 32 + 4 * k, a[k]);
            vst1q_f32(attn + (size_t)(i + 1) * 32 + 4 * k, b[k]);
        }
    }
    if (i0 < i1 && (i1 - i0) % 2 == 1) {            // 尾行单行处理
        const int i = i1 - 1;
        const float* p0 = P + (size_t)i * M;
        float32x4_t a[8];
        for (int k = 0; k < 8; ++k) a[k] = vdupq_n_f32(0.f);
        for (int j = 0; j < M; ++j) {
            const float* vj = vpack + (size_t)j * 32;
            const float32x4_t pv0 = vdupq_n_f32(p0[j]);
            for (int k = 0; k < 8; ++k)
                a[k] = vfmaq_f32(a[k], vld1q_f32(vj + 4 * k), pv0);
        }
        for (int k = 0; k < 8; ++k) vst1q_f32(attn + (size_t)i * 32 + 4 * k, a[k]);
    }
}
#endif  // PF_ATTN_NEON
}  // namespace

void pillar_attention_inplace(float* feats, int M, const FrontWeights& w,
                              const DeployConfig& cfg) {
    /*
    attention 前向(tiling+NEON 微核版, true-N, B=1 无 padding):
    qkv 投影 → softmax(QK^T·scale)·V → out_proj → 残差+LN → FFN(GELU-erff) → 残差+LN
    板上诊断(M=485 串行): scores20.6/softmax12.7/av18.5ms → 微核寄存器阻塞 + fast_expf;
    数值口径: S[i][j] 仍 c 升序 fma、AV 仍 j 升序 fma(与 b3 同序), exp 换 fast_expf(≤3e-7 相对差)
    */
    const int D = cfg.attn_d_model;
    const int F = cfg.attn_ffn;
    const float scale = cfg.attn_scale;
    if (M <= 0) return;

    // 权重准备(每帧一次, KB 级): qkv 合并为 [D][3D] 转置, 其余同
    std::vector<float> wqkv_t((size_t)D * 3 * D), wout_t((size_t)D * D),
        wf1_t((size_t)D * F), wf2_t((size_t)F * D);
    {
        // in_proj_weight [3D][D] → 按 q/k/v 三段各自转置后列拼接为 [D][3D]
        std::vector<float> seg((size_t)D * D);
        for (int s = 0; s < 3; ++s) {
            transpose_weight(w.attn_in_w.data() + (size_t)s * D * D, seg.data(), D, D);
            for (int k = 0; k < D; ++k)
                std::copy(seg.begin() + (size_t)k * D,
                          seg.begin() + (size_t)(k + 1) * D,
                          wqkv_t.begin() + (size_t)k * 3 * D + (size_t)s * D);
        }
    }
    transpose_weight(w.attn_out_w.data(), wout_t.data(), D, D);
    transpose_weight(w.ffn1_w.data(), wf1_t.data(), D, F);
    transpose_weight(w.ffn2_w.data(), wf2_t.data(), F, D);

    std::vector<float> qkv((size_t)M * 3 * D), P((size_t)M * M), x1((size_t)M * D);

    // A. qkv 投影: 行独立
    run_two_threads(M, 128, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i)
            linear_row(feats + (size_t)i * D, &qkv[(size_t)i * 3 * D],
                       wqkv_t.data(), w.attn_in_b.data(), D, 3 * D);
    });
    // K 转置 [D][M](scale 折入) + V 连续 pack(AV 微核用)
    std::vector<float> k_t((size_t)D * M), vpack((size_t)M * D);
    for (int j = 0; j < M; ++j)
        for (int c = 0; c < D; ++c) {
            k_t[(size_t)c * M + j] = qkv[(size_t)j * 3 * D + D + c] * scale;
            vpack[(size_t)j * D + c] = qkv[(size_t)j * 3 * D + 2 * D + c];
        }

    // B. scores(NEON 微核/通用 tiling) + softmax: 本线程行段拼完整后就地归一
    run_two_threads(M, 128, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            float* pi = &P[(size_t)i * M];
            for (int j = 0; j < M; ++j) pi[j] = 0.f;
#if PF_ATTN_NEON
            scores_row_neon(&qkv[(size_t)i * 3 * D], k_t.data(), pi, M, D);
#else
            for (int jt = 0; jt < M; jt += KTILE) {
                const int je = std::min(jt + KTILE, M);
                for (int c = 0; c < D; ++c) {
                    const float qc = qkv[(size_t)i * 3 * D + c];
                    const float* krow = &k_t[(size_t)c * M];
                    for (int j = jt; j < je; ++j) pi[j] += krow[j] * qc;
                }
            }
#endif
            float mx = pi[0];
            for (int j = 1; j < M; ++j) if (pi[j] > mx) mx = pi[j];
            double sum = 0.;
            for (int j = 0; j < M; ++j) {
                const float e = fast_expf(pi[j] - mx);
                pi[j] = e;
                sum += (double)e;
            }
            const float inv = (float)(1.0 / sum);
            for (int j = 0; j < M; ++j) pi[j] *= inv;
        }
    });

    // C+D. AV(NEON 微核/通用 tiling) → out_proj → 残差+LN1 → FFN → 残差+LN2 → 写回 feats
    std::vector<float> attn((size_t)M * D);
    run_two_threads(M, 128, [&](int i0, int i1) {
#if PF_ATTN_NEON
        av_rows_neon(P.data(), vpack.data(), attn.data(), M, i0, i1);
#else
        for (int i = i0; i < i1; ++i)
            for (int c = 0; c < D; ++c) attn[(size_t)i * D + c] = 0.f;
        for (int jt = 0; jt < M; jt += KTILE) {
            const int je = std::min(jt + KTILE, M);
            for (int i = i0; i < i1; ++i) {
                const float* pi = &P[(size_t)i * M];
                float* ai = &attn[(size_t)i * D];
                for (int j = jt; j < je; ++j) {
                    const float pv = pi[j];
                    const float* vrow = &vpack[(size_t)j * D];
                    for (int c = 0; c < D; ++c) ai[c] += vrow[c] * pv;
                }
            }
        }
#endif
        std::vector<float> tmp(D), t(D), h(D), g(F);
        for (int i = i0; i < i1; ++i) {
            linear_row(&attn[(size_t)i * D], tmp.data(), wout_t.data(), w.attn_out_b.data(), D, D);
            for (int c = 0; c < D; ++c) t[c] = feats[(size_t)i * D + c] + tmp[c];
            layer_norm(t.data(), &x1[(size_t)i * D], D, cfg.attn_ln_eps,
                       w.ln1_w.data(), w.ln1_b.data());
        }
        for (int i = i0; i < i1; ++i) {
            linear_row(&x1[(size_t)i * D], h.data(), wf1_t.data(), w.ffn1_b.data(), D, F);
            for (int c = 0; c < F; ++c)
                g[c] = 0.5f * h[c] * (1.0f + std::erff(h[c] * 0.70710678118f));
            linear_row(g.data(), tmp.data(), wf2_t.data(), w.ffn2_b.data(), F, D);
            for (int c = 0; c < D; ++c) t[c] = x1[(size_t)i * D + c] + tmp[c];
            layer_norm(t.data(), &feats[(size_t)i * D], D, cfg.attn_ln_eps,
                       w.ln2_w.data(), w.ln2_b.data());
        }
    });
}
