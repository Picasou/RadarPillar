# RPiN 阶段0 — 三 base 锚点统计（mAP R11/R40 + Params）

协议 T1：VoD 5f EAA，IoU Car=0.5 / Ped·Cyc=0.25，3D AP moderate；R11 为主、R40 参考（D13 主判用 R40）。

| model | Car R11/R40 | Ped R11/R40 | Cyc R11/R40 | mAP R11 | mAP R40 | Params | FLOPs |
|---|---|---|---|---|---|---|---|
| baseline_radarpillar | 39.27/36.18 | 42.34/40.75 | 68.65/69.20 | 50.09 | 48.71 | 184,168 | N/A† |
| baseline_radarnext_fpn | 32.76/31.00 | 36.75/34.23 | 66.87/69.04 | 45.46 | 44.76 | 1,105,549 | N/A† |
| baseline_radarnext_mdfen | 36.24/32.56 | 38.00/35.28 | 67.78/68.55 | 47.34 | 45.46 | 1,639,993 | N/A† |

† FLOPs 未计：三 base 架构异构（PointPillar / RadarNeXt-FPN / RadarNeXt-MDFEN），thop 对 dict-input 整模不适配；成本横比以 Params 为准。阶段1 同构 sweep 如需 D13 成本 tie-break 再补。

注：easy=moderate=hard 三难度等值系 VoD eval 难度过滤特性（所有 GT 过全部难度阈），moderate 即所用。
