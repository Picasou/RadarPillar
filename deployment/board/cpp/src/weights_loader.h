#pragma once
#include <string>
#include <vector>

#include "decode_nms.h"

// 部署常量(meta_flat.txt 生成口径, 与 meta.json 同源)
struct DeployConfig {
    float pcr_min[3], pcr_max[3];
    float voxel_size[3];
    int grid[3];                    // [nx, ny, nz]
    int max_points_per_voxel;
    int max_voxels;
    int vfe_in_dim, vfe_out_dim;
    float vfe_center_offset[3];
    float vfe_bn_eps;
    int attn_d_model, attn_heads, attn_ffn;
    float attn_ln_eps, attn_scale;
    int dpu_in[4], dpu_out[4];
    DecodeParams decode;
};

// CPU 前段权重(原始 f32 连续块, 顺序与 export_weights.py groups 一致)
struct FrontWeights {
    std::vector<float> vfe_linear;                        // [out*in] 行主序
    std::vector<float> vfe_bn_w, vfe_bn_b, vfe_bn_mean, vfe_bn_var;  // [out]
    std::vector<float> attn_in_w, attn_in_b;              // [96*d],[96]
    std::vector<float> attn_out_w, attn_out_b;            // [d*d],[d]
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;        // [d]
    std::vector<float> ffn1_w, ffn1_b, ffn2_w, ffn2_b;    // [d*d],[d]
};

DeployConfig load_config(const std::string& weights_dir);

FrontWeights load_front_weights(const std::string& weights_dir, const DeployConfig& cfg);
