/** decode_nms: RadarNeXtCenterHead2D(NoZ/Trunk) S3 段 C++ 复刻
 *  口径来源: radarnext_center_head_2d_noz.py predict / radarnext_center_head.py
 *  post_processing+_rotate_nms / iou3d_nms_utils.nms_gpu / iou3d_nms_kernel.cu
 *  全程 fp32，开闭区间/常量(EPS=1e-8, MARGIN=1e-2)与 PyTorch+CUDA 逐条对齐
 */
#include "decode_nms.h"

#include <algorithm>
#include <cmath>

namespace {

// ---------------- BEV 旋转 IoU：复刻 iou3d_nms_kernel.cu (fp32) ----------------

struct Vec2 { float x, y; };

inline Vec2 operator+(const Vec2& a, const Vec2& b) { return Vec2{a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2{a.x - b.x, a.y - b.y}; }

inline float cross2(const Vec2& a, const Vec2& b) {
    return a.x * b.y - a.y * b.x;
}

inline float cross3(const Vec2& p1, const Vec2& p2, const Vec2& p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline bool check_rect_cross(const Vec2& p1, const Vec2& p2, const Vec2& q1, const Vec2& q2) {
    return fminf(p1.x, p2.x) <= fmaxf(q1.x, q2.x) &&
           fminf(q1.x, q2.x) <= fmaxf(p1.x, p2.x) &&
           fminf(p1.y, p2.y) <= fmaxf(q1.y, q2.y) &&
           fminf(q1.y, q2.y) <= fmaxf(p1.y, p2.y);
}

inline bool check_in_box2d(const float* box, const Vec2& p) {
    // box: [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2f;
    float angle_cos = cosf(-box[6]), angle_sin = sinf(-box[6]);
    float rot_x = (p.x - box[0]) * angle_cos + (p.y - box[1]) * (-angle_sin);
    float rot_y = (p.x - box[0]) * angle_sin + (p.y - box[1]) * angle_cos;
    return fabsf(rot_x) < box[3] / 2 + MARGIN && fabsf(rot_y) < box[4] / 2 + MARGIN;
}

inline bool intersection(const Vec2& p1, const Vec2& p0, const Vec2& q1, const Vec2& q0, Vec2& ans) {
    const float EPS = 1e-8f;
    if (!check_rect_cross(p0, p1, q0, q1)) return false;
    float s1 = cross3(q0, p1, p0);
    float s2 = cross3(p1, q1, p0);
    float s3 = cross3(p0, q1, q0);
    float s4 = cross3(q1, p1, q0);
    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return false;
    float s5 = cross3(q1, p1, p0);
    if (fabsf(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        // 近平行：克莱姆法则 fallback（kernel 同款）
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;
        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }
    return true;
}

inline void rotate_around_center(const Vec2& center, float angle_cos, float angle_sin, Vec2& p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.x = new_x; p.y = new_y;
}

inline bool point_cmp(const Vec2& a, const Vec2& b, const Vec2& center) {
    return atan2f(a.y - center.y, a.x - center.x) > atan2f(b.y - center.y, b.x - center.x);
}

float box_overlap(const float* box_a, const float* box_b) {
    // box: [x, y, z, dx, dy, dz, heading]；角点序 (min,min)(max,min)(max,max)(min,max) 绕中心转 heading
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2;
    float a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    Vec2 center_a{box_a[0], box_a[1]}, center_b{box_b[0], box_b[1]};
    Vec2 a_corners[5] = {
        {box_a[0] - a_dx_half, box_a[1] - a_dy_half},
        {box_a[0] + a_dx_half, box_a[1] - a_dy_half},
        {box_a[0] + a_dx_half, box_a[1] + a_dy_half},
        {box_a[0] - a_dx_half, box_a[1] + a_dy_half},
        {},
    };
    Vec2 b_corners[5] = {
        {box_b[0] - b_dx_half, box_b[1] - b_dy_half},
        {box_b[0] + b_dx_half, box_b[1] - b_dy_half},
        {box_b[0] + b_dx_half, box_b[1] + b_dy_half},
        {box_b[0] - b_dx_half, box_b[1] + b_dy_half},
        {},
    };
    float a_cos = cosf(box_a[6]), a_sin = sinf(box_a[6]);
    float b_cos = cosf(box_b[6]), b_sin = sinf(box_b[6]);
    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_cos, a_sin, a_corners[k]);
        rotate_around_center(center_b, b_cos, b_sin, b_corners[k]);
    }
    a_corners[4] = a_corners[0];
    b_corners[4] = b_corners[0];

    // 4x4 边对线线交
    Vec2 cross_points[16];
    Vec2 poly_center{0.f, 0.f};
    int cnt = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (intersection(a_corners[i + 1], a_corners[i], b_corners[j + 1], b_corners[j],
                             cross_points[cnt])) {
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }
    // 角点包含：每 k 先 B 角点在 A 内、再 A 角点在 B 内（kernel 顺序）
    for (int k = 0; k < 4; k++) {
        if (check_in_box2d(box_a, b_corners[k])) {
            poly_center = poly_center + b_corners[k];
            cross_points[cnt++] = b_corners[k];
        }
        if (check_in_box2d(box_b, a_corners[k])) {
            poly_center = poly_center + a_corners[k];
            cross_points[cnt++] = a_corners[k];
        }
    }
    poly_center.x /= cnt;  // cnt=0 时 NaN，与 kernel 一致（后续循环不执行，面积恒 0）
    poly_center.y /= cnt;

    // 冒泡按 atan2 升序（kernel 同款）
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
                Vec2 tmp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = tmp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        area += cross2(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }
    return fabsf(area) / 2;
}

inline float iou_bev(const float* box_a, const float* box_b) {
    // box: [x, y, z, dx, dy, dz, heading]
    const float EPS = 1e-8f;
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

// ---------------- decode 候选 ----------------

struct Cand {
    float box[7];  // [x, y, z, dx, dy, dz, rot]，dz 恒 0（2D 头口径）
    float score;
    int cell;      // 原 cell idx，稳定排序次序用
};

}  // namespace

std::vector<DetBox> decode_nms(const float* logits_chw, const DecodeParams& p) {
    /**
     * S3 decode+NMS: DPU logits [11][H][W] 行主序 → DetBox 集（类 0→4 拼接、类内 score 降序）
     */
    const int HW = p.H * p.W;
    std::vector<std::vector<Cand>> per_class(p.hm_ch);

    for (int h = 0; h < p.H; h++) {
        for (int w = 0; w < p.W; w++) {
            const int idx = h * p.W + w;
            // hm 5 类 sigmoid，max/argmax（严格 > 取首个 max，同 torch.max 平局语义）
            float best = -1e30f;
            int best_c = 0;
            for (int c = 0; c < p.hm_ch; c++) {
                float s = 1.0f / (1.0f + expf(-logits_chw[(size_t)c * HW + idx]));
                if (s > best) {
                    best = s;
                    best_c = c;
                }
            }
            // reg/dim/rot 通道布局: [hm(5) reg(2) dim(2) rot(sin,cos)]
            float reg_x = logits_chw[(size_t)(p.hm_ch + 0) * HW + idx];
            float reg_y = logits_chw[(size_t)(p.hm_ch + 1) * HW + idx];
            float dim0 = logits_chw[(size_t)(p.hm_ch + 2) * HW + idx];
            float dim1 = logits_chw[(size_t)(p.hm_ch + 3) * HW + idx];
            float rot_s = logits_chw[(size_t)(p.hm_ch + 4) * HW + idx];
            float rot_c = logits_chw[(size_t)(p.hm_ch + 5) * HW + idx];
            // exp 不 clamp：保持与 torch 一致溢出行为
            float x = (w + reg_x) * p.x_step + p.x_min;
            float y = (h + reg_y) * p.y_step + p.y_min;
            float dx = expf(dim0), dy = expf(dim1);
            float rot = atan2f(rot_s, rot_c);
            float z = p.z_table[best_c];
            // score 严格大于 + range 双侧闭区间（NaN 比较为 false，同 torch）
            if (!(best > p.score_thresh)) continue;
            if (!(x >= p.limit_min[0] && y >= p.limit_min[1] && z >= p.limit_min[2] &&
                  x <= p.limit_max[0] && y <= p.limit_max[1] && z <= p.limit_max[2]))
                continue;
            Cand cd;
            cd.box[0] = x; cd.box[1] = y; cd.box[2] = z;
            cd.box[3] = dx; cd.box[4] = dy; cd.box[5] = 0.f; cd.box[6] = rot;
            cd.score = best;
            cd.cell = idx;
            per_class[best_c].push_back(cd);
        }
    }

    std::vector<DetBox> out;
    for (int c = 0; c < p.hm_ch; c++) {
        auto& cands = per_class[c];
        // 稳定排序 score 降序（等值保持 cell 升序，同 CUDA radix sort）→ 截前 pre_max
        std::stable_sort(cands.begin(), cands.end(),
                         [](const Cand& a, const Cand& b) { return a.score > b.score; });
        if ((int)cands.size() > p.nms_pre_max) cands.resize(p.nms_pre_max);
        // 贪心 NMS：与已保留框 iou 严格大于阈值即抑制；保留数达 post_max 提前结束
        std::vector<int> keep;
        for (size_t i = 0; i < cands.size(); i++) {
            bool suppressed = false;
            for (int k : keep) {
                if (iou_bev(cands[i].box, cands[k].box) > p.nms_iou_thresh) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                keep.push_back((int)i);
                if ((int)keep.size() >= p.nms_post_max) break;
            }
        }
        for (int k : keep) {
            const Cand& cd = cands[k];
            out.push_back(DetBox{cd.box[0], cd.box[1], cd.box[2], cd.box[3], cd.box[4], cd.box[5],
                                 cd.box[6], cd.score, c + 1});
        }
    }
    return out;
}
