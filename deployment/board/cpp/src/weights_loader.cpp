#include "weights_loader.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<float> read_bin(const std::string& path, size_t expect_floats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("权重文件打不开: " + path);
    std::vector<float> buf(expect_floats);
    f.read(reinterpret_cast<char*>(buf.data()), expect_floats * sizeof(float));
    if (f.gcount() != (std::streamsize)(expect_floats * sizeof(float)))
        throw std::runtime_error("权重字节数不符: " + path);
    return buf;
}
}  // namespace

DeployConfig load_config(const std::string& weights_dir) {
    /*
    配置读取: meta_flat.txt(key value 行) → DeployConfig; 缺 key 即抛错禁静默
    */
    std::ifstream f(weights_dir + "/meta_flat.txt");
    if (!f) throw std::runtime_error("meta_flat.txt 打不开: " + weights_dir);
    std::string key, line;
    std::vector<float> kv[64];
    std::string order[64];
    int n = 0;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        ss >> key;
        if (key.empty() || key[0] == '#') continue;
        order[n] = key;
        float v;
        while (ss >> v) kv[n].push_back(v);
        ++n;
    }
    auto get = [&](const std::string& k) -> const std::vector<float>& {
        for (int i = 0; i < n; ++i)
            if (order[i] == k) return kv[i];
        throw std::runtime_error("meta_flat.txt 缺 key: " + k);
    };

    DeployConfig c{};
    const auto& pcr = get("point_cloud_range");
    for (int i = 0; i < 3; ++i) { c.pcr_min[i] = pcr[i]; c.pcr_max[i] = pcr[i + 3]; }
    const auto& vs = get("voxel_size");
    for (int i = 0; i < 3; ++i) c.voxel_size[i] = vs[i];
    const auto& gs = get("grid_size");
    for (int i = 0; i < 3; ++i) c.grid[i] = (int)gs[i];
    c.max_points_per_voxel = (int)get("max_points_per_voxel")[0];
    c.max_voxels = (int)get("max_voxels_test")[0];
    c.vfe_in_dim = (int)get("vfe_in_dim")[0];
    c.vfe_out_dim = (int)get("vfe_out_dim")[0];
    const auto& off = get("vfe_center_offsets");
    for (int i = 0; i < 3; ++i) c.vfe_center_offset[i] = off[i];
    c.vfe_bn_eps = get("vfe_bn_eps")[0];
    c.attn_d_model = (int)get("attn_d_model")[0];
    c.attn_heads = (int)get("attn_num_heads")[0];
    c.attn_ffn = (int)get("attn_ffn_hidden")[0];
    c.attn_ln_eps = get("attn_ln_eps")[0];
    c.attn_scale = 1.0f / std::sqrt((float)c.attn_d_model);
    const auto& din = get("dpu_in"), dout = get("dpu_out");
    for (int i = 0; i < 4; ++i) { c.dpu_in[i] = (int)din[i]; c.dpu_out[i] = (int)dout[i]; }

    DecodeParams& d = c.decode;
    d.x_step = get("decode_x_step")[0];
    d.y_step = get("decode_y_step")[0];
    d.x_min = get("decode_x_min")[0];
    d.y_min = get("decode_y_min")[0];
    const auto& zt = get("decode_z_table");
    for (int i = 0; i < (int)zt.size() && i < 8; ++i) d.z_table[i] = zt[i];   // 长度随类数(v1 4类/msrv2 5类)
    d.score_thresh = get("decode_score_threshold")[0];
    const auto& lr = get("decode_limit_range");
    for (int i = 0; i < 3; ++i) { d.limit_min[i] = lr[i]; d.limit_max[i] = lr[i + 3]; }
    d.nms_iou_thresh = get("nms_iou_thresh")[0];
    d.nms_pre_max = (int)get("nms_pre_max")[0];
    d.nms_post_max = (int)get("nms_post_max")[0];
    d.hm_ch = (int)get("class_count")[0];
    d.H = c.dpu_out[2];
    d.W = c.dpu_out[3];
    return c;
}

FrontWeights load_front_weights(const std::string& weights_dir, const DeployConfig& cfg) {
    /*
    权重读取: 8 个 bin → FrontWeights(shape 校验防错位)
    */
    const int D = cfg.attn_d_model, V = cfg.vfe_out_dim, IN = cfg.vfe_in_dim;
    FrontWeights w;
    w.vfe_linear = read_bin(weights_dir + "/vfe_linear.bin", (size_t)V * IN);
    {   // vfe_bn.bin: [w | b | mean | var] 各 V 段
        auto all = read_bin(weights_dir + "/vfe_bn.bin", (size_t)4 * V);
        w.vfe_bn_w.assign(all.begin(), all.begin() + V);
        w.vfe_bn_b.assign(all.begin() + V, all.begin() + 2 * V);
        w.vfe_bn_mean.assign(all.begin() + 2 * V, all.begin() + 3 * V);
        w.vfe_bn_var.assign(all.begin() + 3 * V, all.end());
    }
    {   // attn_inproj.bin: [in_proj_weight 3D*D | in_proj_bias 3D]
        auto all = read_bin(weights_dir + "/attn_inproj.bin", (size_t)3 * D * D + 3 * D);
        w.attn_in_w.assign(all.begin(), all.begin() + (size_t)3 * D * D);
        w.attn_in_b.assign(all.begin() + (size_t)3 * D * D, all.end());
    }
    {
        std::vector<float> all = read_bin(weights_dir + "/attn_out.bin", (size_t)(D * D + D));
        w.attn_out_w.assign(all.begin(), all.begin() + D * D);
        w.attn_out_b.assign(all.begin() + D * D, all.end());
    }
    {
        std::vector<float> all = read_bin(weights_dir + "/attn_ln1.bin", (size_t)(2 * D));
        w.ln1_w.assign(all.begin(), all.begin() + D);
        w.ln1_b.assign(all.begin() + D, all.end());
    }
    {
        std::vector<float> all = read_bin(weights_dir + "/attn_ffn1.bin", (size_t)(D * D + D));
        w.ffn1_w.assign(all.begin(), all.begin() + D * D);
        w.ffn1_b.assign(all.begin() + D * D, all.end());
    }
    {
        std::vector<float> all = read_bin(weights_dir + "/attn_ffn2.bin", (size_t)(D * D + D));
        w.ffn2_w.assign(all.begin(), all.begin() + D * D);
        w.ffn2_b.assign(all.begin() + D * D, all.end());
    }
    {
        std::vector<float> all = read_bin(weights_dir + "/attn_ln2.bin", (size_t)(2 * D));
        w.ln2_w.assign(all.begin(), all.begin() + D);
        w.ln2_b.assign(all.begin() + D, all.end());
    }
    return w;
}
