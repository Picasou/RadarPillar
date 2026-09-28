#pragma once
#include "weights_loader.h"

// true-N 单层 transformer block(原地): MHA(1 head) + post-LN + FFN(GELU=erf exact) + post-LN
void pillar_attention_inplace(float* feats /*[M][D]*/, int M,
                              const FrontWeights& w, const DeployConfig& cfg);
