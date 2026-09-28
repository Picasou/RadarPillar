#pragma once
#include <cstdint>
#include <vector>

// 单个非空 pillar(true-N, 点数≤max_points_per_voxel, 点序=输入序)
struct Pillar {
    int32_t y, x;        // 网格坐标
    int num_points;      // ≤ max_points_per_voxel
    std::vector<float> pts;  // [num_points][6] 平铺
};

// 点云 → pillar 化(spconv PointToVoxel 同语义: z 出网格点丢弃 + 输入序前 N 点 + 首现序 pillar)
// points: [N][6] 平铺; 返回 pillar 列表
std::vector<Pillar> pillarize(const float* points, int num_points,
                              int max_points_per_voxel,
                              const float pcr_min[3], const float pcr_max[3],
                              const float voxel_size[3], const int grid[3]);
