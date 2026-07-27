# RPiN 阶段1 — 注意力 A 扫描统计（best-epoch，D13 主判 Car R40 moderate）

固定项 B1+C1+D1+E2(9维)+F3+AnchorHeadSingle（6-anchor 对齐 baseline）；变量 BACKBONE_3D。a0/a2/a3 取末20ep best(by Car R40 mod)；a1 复用 baseline best.pth(ep60)。

| model | attention | bestEp | Car R40 | Ped R40 | Cyc R40 | mAP R40 | mAP R11 | Params | FLOPs |
|---|---|---|---|---|---|---|---|---|---|
| a0 | None(直连) | 73 | 37.69 | 31.70 | 66.92 | 45.43 | 47.62 | 177,704 | N/A† |
| a3 | SEDWConv | 76 | 37.07 | 33.96 | 64.43 | 45.16 | 47.10 | 179,592 | N/A† |
| a2 | SEBlock | 80 | 36.56 | 31.84 | 65.26 | 44.55 | 46.83 | 178,216 | N/A† |
| a1 | PillarAttention(=baseline,复用) | 60 | 36.18 | 40.75 | 69.20 | 48.71 | 50.09 | 184,168 | N/A† |

† FLOPs 未计（同构 sweep 成本以 Params 为准；D13 成本 tie 用 Params）。

注：skill pickbest 因 eval 落 output/YAML/a*/a*/（cfg 派生路径）而非训练 OR 而失败；eval 数据完整，本表由自有解析器直选 best epoch 产出（详见 stage1 报告）。
