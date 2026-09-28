#include "pillarize.h"

#include <cmath>
#include <unordered_map>

std::vector<Pillar> pillarize(const float* points, int num_points,
                              int max_points_per_voxel,
                              const float pcr_min[3], const float pcr_max[3],
                              const float voxel_size[3], const int grid[3]) {
    /*
    pillar 化: 逐点算网格索引, 同格点取输入序前 N; 语义已对 golden 13464 pillar 精确验证
    (含 z∈[pcr_z_min, pcr_z_max) 半开区间过滤, 对应 spconv 出网格丢弃)
    */
    std::unordered_map<int32_t, int> cell_to_idx;  // key = y*grid_x + x
    cell_to_idx.reserve(1024);
    std::vector<Pillar> pillars;
    pillars.reserve(1024);
    for (int i = 0; i < num_points; ++i) {
        const float* p = points + (size_t)i * 6;
        // spconv 语义: 任一轴出网格即丢弃; x/y 数据侧已 mask, z 半开区间 [-10,10)
        bool in_z = p[2] >= pcr_min[2] && p[2] < pcr_max[2];
        bool in_x = p[0] >= pcr_min[0] && p[0] < pcr_max[0];
        bool in_y = p[1] >= pcr_min[1] && p[1] < pcr_max[1];
        if (!(in_x && in_y && in_z)) continue;
        int xi = (int)std::floor((p[0] - pcr_min[0]) / voxel_size[0]);
        int yi = (int)std::floor((p[1] - pcr_min[1]) / voxel_size[1]);
        if (xi < 0 || xi >= grid[0] || yi < 0 || yi >= grid[1]) continue;
        int32_t key = (int32_t)yi * grid[0] + xi;
        auto it = cell_to_idx.find(key);
        if (it == cell_to_idx.end()) {
            Pillar pl;
            pl.y = yi; pl.x = xi; pl.num_points = 0;
            cell_to_idx[key] = (int)pillars.size();
            pillars.push_back(std::move(pl));
            it = cell_to_idx.find(key);
        }
        Pillar& pl = pillars[it->second];
        if (pl.num_points < max_points_per_voxel) {  // 输入序前 N 点
            pl.pts.insert(pl.pts.end(), p, p + 6);
            ++pl.num_points;
        }
    }
    return pillars;
}
