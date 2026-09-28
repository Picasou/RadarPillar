/** test_decode: golden 对拍自测
 *  读 .tmp 下 dump 的 dpu_out .f32 + golden 框 txt → decode_nms → 逐帧对比
 *  数量/label 序列/score 差/中心距/尺寸与朝向差
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include "../src/decode_nms.h"

namespace {

struct GBox {
    float v[7];  // x y z dx dy dz rot
    float score;
    int label;
};

/**
 * 读 golden txt: 首行框数, 每行 7 框值 + score + label
 */
bool load_gt(const std::string& path, std::vector<GBox>& out) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    int n = 0;
    if (fscanf(f, "%d", &n) != 1 || n < 0) { fclose(f); return false; }
    out.clear();
    for (int i = 0; i < n; i++) {
        GBox b;
        if (fscanf(f, "%f %f %f %f %f %f %f %f %d", &b.v[0], &b.v[1], &b.v[2], &b.v[3], &b.v[4],
                   &b.v[5], &b.v[6], &b.score, &b.label) != 9) { fclose(f); return false; }
        out.push_back(b);
    }
    fclose(f);
    return true;
}

/**
 * 读整文件字节: 失败返回空
 */
std::vector<float> load_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> buf(sz / 4);
    size_t rd = fread(buf.data(), 4, buf.size(), f);
    fclose(f);
    if (rd != buf.size()) return {};
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1]
                               : "/home/admin/projects/RadarPillar/.tmp/2026-09-16/deploy-sim/decode_selftest";
    // 扫描 *_gt.txt 配对同名 .f32
    std::vector<std::string> stems;
    DIR* d = opendir(dir.c_str());
    if (!d) { printf("open dir fail: %s\n", dir.c_str()); return 1; }
    dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name.size() > 7 && name.substr(name.size() - 7) == "_gt.txt")
            stems.push_back(name.substr(0, name.size() - 7));
    }
    closedir(d);
    std::sort(stems.begin(), stems.end());
    if (stems.empty()) { printf("no *_gt.txt under %s\n", dir.c_str()); return 1; }

    int nfail = 0;
    for (const auto& stem : stems) {
        auto logits = load_f32(dir + "/" + stem + ".f32");
        std::vector<GBox> gt;
        if (logits.empty() || !load_gt(dir + "/" + stem + "_gt.txt", gt)) {
            printf("[%s] LOAD FAIL\n", stem.c_str());
            nfail++;
            continue;
        }
        DecodeParams p;  // 默认即 msrv2_pp64_ch2d_trunk 口径
        std::vector<DetBox> pred = decode_nms(logits.data(), p);

        const int np = (int)pred.size(), ng = (int)gt.size();
        bool cnt_ok = (np == ng);
        // label 序列逐位对比（仅数量一致时有意义）
        bool lab_ok = cnt_ok;
        int lab_bad = -1;
        if (cnt_ok) {
            for (int i = 0; i < np; i++) {
                if (pred[i].label != gt[i].label) { lab_ok = false; lab_bad = i; break; }
            }
        }
        // 同序框数值差
        float ds = 0.f, dc = 0.f, dz = 0.f, ddx = 0.f, ddy = 0.f, ddz = 0.f, dr = 0.f;
        int ncmp = 0;
        if (lab_ok) {
            for (int i = 0; i < np; i++) {
                const DetBox& a = pred[i];
                const GBox& b = gt[i];
                ds = fmaxf(ds, fabsf(a.score - b.score));
                dc = fmaxf(dc, sqrtf((a.x - b.v[0]) * (a.x - b.v[0]) + (a.y - b.v[1]) * (a.y - b.v[1])));
                dz = fmaxf(dz, fabsf(a.z - b.v[2]));
                ddx = fmaxf(ddx, fabsf(a.dx - b.v[3]));
                ddy = fmaxf(ddy, fabsf(a.dy - b.v[4]));
                ddz = fmaxf(ddz, fabsf(a.dz - b.v[5]));
                dr = fmaxf(dr, fabsf(a.rot - b.v[6]));
                ncmp++;
            }
        }
        bool pass = cnt_ok && lab_ok && ds <= 1e-5f && dc <= 1e-3f;
        if (!pass) nfail++;
        printf("[%s] pred=%d gt=%d labels=%s ncmp=%d | score=%.3e center=%.3e z=%.3e dx=%.3e dy=%.3e dz=%.3e rot=%.3e | %s\n",
               stem.c_str(), np, ng, lab_ok ? "OK" : (cnt_ok ? "MISMATCH" : "N/A"), ncmp,
               ds, dc, dz, ddx, ddy, ddz, dr, pass ? "PASS" : "FAIL");
        if (!cnt_ok) {
            printf("   pred labels:");
            for (const auto& a : pred) printf(" %d", a.label);
            printf("\n   gt   labels:");
            for (const auto& b : gt) printf(" %d", b.label);
            printf("\n");
        } else if (!lab_ok) {
            printf("   first label diff @%d: pred=%d gt=%d\n", lab_bad, pred[lab_bad].label,
                   gt[lab_bad].label);
        }
    }
    printf("== %zu frames, %d fail ==\n", stems.size(), nfail);
    return nfail == 0 ? 0 : 2;
}
