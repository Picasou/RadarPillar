#pragma once
#include <vector>

#include "pillarize.h"
#include "weights_loader.h"

// VFE: pillars → pillar_features [M][C](12ch 拼装 + Linear + BN + ReLU + 点维 max)
// 注意: torch 的 max 含 padding 行(linear(0)=0 经 BN/ReLU 后非零), 以 pad 常量参与 max 复刻
void pillar_vfe(const std::vector<Pillar>& pillars, const FrontWeights& w,
                const DeployConfig& cfg, float* out);
