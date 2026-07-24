# note/radarpillar/ 历史归档说明（resbag 时代的兼容保留）

`note/radarpillar复现结论.md` 与本目录的资产（`loss_curve.png` / `tb_loss_curves.png` / `radarpillar_frames/`）是 **RadarPillar 基线训练（radarpillar_base）的历史产物**，对应 `output/train_log/vod/radarpillar_base/` 目录。

该基线训练 OUTPUT_ROOT 在后续清理中已不可用（best.pth / ckpt / 资产均缺失），无法走 resbag 落袋流程（resbag 要求自包含的 OUTPUT_ROOT）。为避免历史复现文档丢失，本目录**作为项目级归档保留**，不纳入 resbag 自动化。

新训练请走 model-train → resbag，自动产出：
- `<OUTPUT_ROOT>/resbag/`（自包含 8 类文件）
- `<OUTPUT_ROOT>/model_store.yaml`（单实验总览）
