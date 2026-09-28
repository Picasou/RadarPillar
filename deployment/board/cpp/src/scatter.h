#pragma once
#include <vector>

#include "pillarize.h"

// coords → dense BEV [C][ny][nx](scatter 语义: 每 pillar 32 通道写入自身格)
void scatter_to_bev(const std::vector<Pillar>& pillars, const float* pillar_feats,
                    int C, int ny, int nx, float* spatial /*C*ny*nx, 需已清零或由本函数清零*/);
