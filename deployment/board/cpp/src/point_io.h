#pragma once
#include <string>
#include <vector>

// N×6 点特征读取([x,y,z,dop_x_gnd,dop_y_gnd,rcs] f32, 与 FPGA 接口契约同口径)
std::vector<float> read_points_bin(const std::string& path, int& num_points);

// f32 连续块落盘(spatial_features / logits 输出用)
void write_f32_bin(const std::string& path, const float* data, size_t n);

// f32 连续块读取
std::vector<float> read_f32_bin(const std::string& path, size_t expect_n);
