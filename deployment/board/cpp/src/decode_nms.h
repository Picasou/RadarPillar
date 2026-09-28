#pragma once
#include <vector>
struct DetBox { float x, y, z, dx, dy, dz, rot; float score; int label; };  // label 1..5
struct DecodeParams {
    float x_step = 2.0f, y_step = 1.0f, x_min = 0.0f, y_min = -20.0f;
    float score_thresh = 0.2f;
    float limit_min[3] = {0.f, -20.f, -10.f}, limit_max[3] = {200.f, 20.f, 10.f};
    float z_table[8] = {-0.75f, -0.85f, -0.85f, -1.5f, -0.85f, 0, 0, 0};   // 定长上限8类, 实际按 meta class_count 填充
    float nms_iou_thresh = 0.1f; int nms_pre_max = 4096, nms_post_max = 500;
    int hm_ch = 5, H = 40, W = 100;
};
// logits_chw: [11*40*100] 行主序 CHW; 返回框顺序 = 类 0→4 拼接、类内 score 降序
std::vector<DetBox> decode_nms(const float* logits_chw, const DecodeParams& p);
