#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "decode_nms.h"
#include "dpu_backend.h"
#include "point_io.h"
#include "pillar_attention.h"
#include "pillar_vfe.h"
#include "pillarize.h"
#include "scatter.h"
#include "weights_loader.h"

namespace {
struct Timer {
    std::chrono::steady_clock::time_point t0;
    std::string tag;
    explicit Timer(std::string t) : t0(std::chrono::steady_clock::now()), tag(std::move(t)) {}
    ~Timer() {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[time] %s: %.2f ms\n", tag.c_str(), ms);
    }
};

int run_front(const std::string& pts_path, const std::string& wdir, const std::string& out_path) {
    /*
    仅前段: 点 bin → spatial_features bin(对拍 S2); 四段计时: preproc/voxel/pfn(vfe+attn)/scatter
    */
    int n = 0;
    std::vector<float> pts;
    DeployConfig cfg;
    FrontWeights w;
    {
        Timer t("preproc(io+cfg+weights)");
        pts = read_points_bin(pts_path, n);
        cfg = load_config(wdir);
        w = load_front_weights(wdir, cfg);
    }
    std::vector<float> spatial((size_t)cfg.vfe_out_dim * cfg.grid[1] * cfg.grid[0]);
    {
        Timer t("front(pillarize+vfe+attention+scatter)");
        std::vector<Pillar> pillars;
        {
            Timer t("voxel(pillarize)");
            pillars = pillarize(pts.data(), n, cfg.max_points_per_voxel,
                                cfg.pcr_min, cfg.pcr_max, cfg.voxel_size, cfg.grid);
        }
        std::vector<float> feats(pillars.size() * cfg.vfe_out_dim);
        {
            Timer t("pfn-vfe");
            pillar_vfe(pillars, w, cfg, feats.data());
        }
        {
            Timer t("pfn-attention");
            pillar_attention_inplace(feats.data(), (int)pillars.size(), w, cfg);
        }
        {
            Timer t("scatter");
            scatter_to_bev(pillars, feats.data(), cfg.vfe_out_dim, cfg.grid[1], cfg.grid[0],
                           spatial.data());
        }
        std::fprintf(stderr, "[front] N=%d pillars=%zu\n", n, pillars.size());
    }
    write_f32_bin(out_path, spatial.data(), spatial.size());
    return 0;
}

int run_back(const std::string& logits_path, const std::string& wdir, const std::string& out_path) {
    /*
    仅后段: logits bin → 框 txt(对拍 S3)
    */
    DeployConfig cfg = load_config(wdir);
    size_t elems = 1;
    for (int d : cfg.dpu_out) elems *= (size_t)d;
    std::vector<float> logits = read_f32_bin(logits_path, elems);
    std::vector<DetBox> boxes;
    {
        Timer t("back(decode+nms)");
        boxes = decode_nms(logits.data(), cfg.decode);
    }
    std::FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "写出失败 %s\n", out_path.c_str()); return 2; }
    std::fprintf(f, "%zu\n", boxes.size());
    for (const DetBox& b : boxes)
        std::fprintf(f, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d\n",
                     b.x, b.y, b.z, b.dx, b.dy, b.dz, b.rot, b.score, b.label);
    std::fclose(f);
    std::fprintf(stderr, "[back] boxes=%zu\n", boxes.size());
    return 0;
}

int run_full(const std::string& pts_path, const std::string& wdir, const std::string& onnx,
             const std::string& out_path) {
    /*
    全链: 点 bin → 前段 → DPU 后端 → decode/NMS → 框 txt(对拍 S4, 主目标)
    */
    int n = 0;
    std::vector<float> pts;
    DeployConfig cfg;
    FrontWeights w;
    {
        Timer t("preproc(io+cfg+weights)");
        pts = read_points_bin(pts_path, n);
        cfg = load_config(wdir);
        w = load_front_weights(wdir, cfg);
    }
    auto backend = create_dpu_backend(onnx);
    const int C = cfg.vfe_out_dim, NY = cfg.grid[1], NX = cfg.grid[0];
    std::vector<float> spatial((size_t)C * NY * NX);
    size_t logits_elems = 1;
    for (int d : cfg.dpu_out) logits_elems *= (size_t)d;
    std::vector<float> logits(logits_elems);
    std::vector<DetBox> boxes;
    {
        Timer t_total("full 端到端");
        std::vector<Pillar> pillars;
        std::vector<float> feats;
        {
            Timer t("front");
            {
                Timer t("voxel(pillarize)");
                pillars = pillarize(pts.data(), n, cfg.max_points_per_voxel,
                                    cfg.pcr_min, cfg.pcr_max, cfg.voxel_size, cfg.grid);
            }
            feats.resize(pillars.size() * C);
            {
                Timer t("pfn-vfe");
                pillar_vfe(pillars, w, cfg, feats.data());
            }
            {
                Timer t("pfn-attention");
                pillar_attention_inplace(feats.data(), (int)pillars.size(), w, cfg);
            }
            {
                Timer t("scatter");
                scatter_to_bev(pillars, feats.data(), C, NY, NX, spatial.data());
            }
        }
        {
            Timer t("dpu(" + std::string(backend->name()) + ")");
            backend->infer(spatial.data(), logits.data());
        }
        {
            Timer t("back");
            boxes = decode_nms(logits.data(), cfg.decode);
        }
    }
    std::FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) return 2;
    std::fprintf(f, "%zu\n", boxes.size());
    for (const DetBox& b : boxes)
        std::fprintf(f, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d\n",
                     b.x, b.y, b.z, b.dx, b.dy, b.dz, b.rot, b.score, b.label);
    std::fclose(f);
    std::fprintf(stderr, "[full] N=%d boxes=%zu backend=%s\n", n, boxes.size(), backend->name());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "front") == 0 && argc == 5)
        return run_front(argv[2], argv[3], argv[4]);
    if (argc >= 2 && std::strcmp(argv[1], "back") == 0 && argc == 5)
        return run_back(argv[2], argv[3], argv[4]);
    if (argc >= 2 && std::strcmp(argv[1], "full") == 0 && argc == 6)
        return run_full(argv[2], argv[3], argv[4], argv[5]);
    std::fprintf(stderr,
        "用法:\n"
        "  radar_pipeline front <points.bin> <weights_dir> <out_spatial.bin>\n"
        "  radar_pipeline back  <logits.bin> <weights_dir> <out_boxes.txt>\n"
        "  radar_pipeline full  <points.bin> <weights_dir> <onnx> <out_boxes.txt>\n");
    return 1;
}
