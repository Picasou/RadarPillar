#include "scatter.h"

#include <cstring>

void scatter_to_bev(const std::vector<Pillar>& pillars, const float* pillar_feats,
                    int C, int ny, int nx, float* spatial) {
    /*
    scatter: 清零后逐 pillar 写入 (c, y, x) 格
    */
    std::memset(spatial, 0, sizeof(float) * (size_t)C * ny * nx);
    const int plane = ny * nx;
    for (size_t m = 0; m < pillars.size(); ++m) {
        const size_t base = (size_t)pillars[m].y * nx + pillars[m].x;
        const float* src = pillar_feats + m * C;
        for (int c = 0; c < C; ++c) spatial[(size_t)c * plane + base] = src[c];
    }
}
