#include "pillar_vfe.h"

#include <cmath>
#include <limits>
#include <vector>

#include "thread_split.h"

void pillar_vfe(const std::vector<Pillar>& pillars, const FrontWeights& w,
                const DeployConfig& cfg, float* out) {
    /*
    PFN 前向(float 批处理版): 每 pillar 拼装 12ch → BN折叠Linear → 点间max(含pad行) → ReLU
    数值等价变换: ReLU/max 对常数平移单调可交换, y=ReLU(max_p(dot_p, pad?0) + b'), pad 行 dot=0;
    f_cluster 均值仍 4-lane float 复刻 torch 求和序; 内积改 f32 累加(门禁: label/cd 对拍)
    */
    const int C = cfg.vfe_out_dim;   // 32
    const int IN = cfg.vfe_in_dim;   // 12
    // BN 折叠进 Linear: wt[k][c]=W[c][k]·s_c (转置布局, 内层 c 连续可向量化), b'[c]=b-mean·s
    std::vector<float> wt((size_t)IN * C), bf(C);
    for (int c = 0; c < C; ++c) {
        const float s = w.vfe_bn_w[c] / std::sqrt(w.vfe_bn_var[c] + cfg.vfe_bn_eps);
        bf[c] = w.vfe_bn_b[c] - w.vfe_bn_mean[c] * s;
        for (int k = 0; k < IN; ++k) wt[(size_t)k * C + c] = w.vfe_linear[(size_t)c * IN + k] * s;
    }
    const int T = cfg.max_points_per_voxel;

    auto range = [&](int m0, int m1) {
        std::vector<float> vmax(C), d(C), feat(IN);
        for (int m = m0; m < m1; ++m) {
            const Pillar& pl = pillars[m];
            const int n = pl.num_points;
            // f_cluster 均值仅对被保留的点; 4-lane 分 stride 归约复刻 torch 求和序
            float acc[4][3] = {};
            for (int i = 0; i < n; ++i) {
                const float* p = &pl.pts[(size_t)i * 6];
                float* a = acc[i & 3];
                a[0] += p[0]; a[1] += p[1]; a[2] += p[2];
            }
            float mx = ((acc[0][0] + acc[1][0]) + acc[2][0]) + acc[3][0];
            float my = ((acc[0][1] + acc[1][1]) + acc[2][1]) + acc[3][1];
            float mz = ((acc[0][2] + acc[1][2]) + acc[2][2]) + acc[3][2];
            mx /= (float)n; my /= (float)n; mz /= (float)n;
            const float cx = (float)pl.x * cfg.voxel_size[0] + cfg.vfe_center_offset[0];
            const float cy = (float)pl.y * cfg.voxel_size[1] + cfg.vfe_center_offset[1];
            const float cz = 0.f * cfg.voxel_size[2] + cfg.vfe_center_offset[2];

            // n<T 时 pad 行(dot=0)参与 max; n==T 用 -FLT_MAX 下界(-ffast-math 下禁用 inf)
            for (int c = 0; c < C; ++c)
                vmax[c] = (n < T) ? 0.f : -std::numeric_limits<float>::max();
            for (int i = 0; i < n; ++i) {
                const float* p = &pl.pts[(size_t)i * 6];
                feat[0] = p[0]; feat[1] = p[1]; feat[2] = p[2];
                feat[3] = p[3]; feat[4] = p[4]; feat[5] = p[5];
                feat[6] = p[0] - mx; feat[7] = p[1] - my; feat[8] = p[2] - mz;
                feat[9] = p[0] - cx; feat[10] = p[1] - cy; feat[11] = p[2] - cz;
                for (int c = 0; c < C; ++c) d[c] = 0.f;
                for (int k = 0; k < IN; ++k) {   // 内层 c 连续, 无归约可向量化
                    const float v = feat[k];
                    const float* wk = &wt[(size_t)k * C];
                    for (int c = 0; c < C; ++c) d[c] += wk[c] * v;
                }
                for (int c = 0; c < C; ++c) if (d[c] > vmax[c]) vmax[c] = d[c];
            }
            float* dst = out + (size_t)m * C;
            for (int c = 0; c < C; ++c) {
                const float y = vmax[c] + bf[c];
                dst[c] = y > 0.f ? y : 0.f;
            }
        }
    };
    run_two_threads((int)pillars.size(), 64, range);
}
