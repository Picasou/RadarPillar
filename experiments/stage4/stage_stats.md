# 阶段4 — head-swap 扫描统计（2D-backbone × 检测头, VoD val R40 moderate）

**scope**: 11 个 head-swap 实验 = 4 种 2D backbone（`b1`=BaseBEV / `b8`=RepDWC / `n3`=PPMDFEN / `n6,n7`=RadarNeXtMDFEN）× 3 种检测头（`center`=RadarNeXtCenterHead 有 z / `2d`=RadarNeXtCenterHead2DNoZ 无 z / `rp_mask`=RadarPillar 自家 head）二维组合 + rp_mask 对照线。
**协议**: bs=8 ep=80, pickbest = ckpt 80 单点评估（无 median）, seed 666 FROZEN。
**评估口径**: 3D 头评 3D+BEV；2d 头无 z, 3D 恒 0（设计）, BEV 为唯一有效指标。IoU Car=0.5 / Ped·Cyc=0.25, EAA filter。
**数据源**: 主表 = 20260810_stage4_report.md（原 sweep 的 output OR 目录已清理, ckpt 不在盘上）；后续 run = 08-14/15 复跑与消融（OR 在盘, 路径见 manifest）。

## 1. 主表（2026-08-10 sweep, 11 tags）

| tag | backbone (2D) | head | 3d_mean | bev_mean | 3d (car/ped/cyc) | bev (car/ped/cyc) | Params | FLOPs |
|---|---|---|---:|---:|:---:|:---:|---:|---:|
| `head_b8_center` | RepDWC | center (有z) | **44.26** | 51.41 | 33.49/37.00/62.29 | 45.22/43.94/65.07 | 0.46M | 169.5G |
| `head_b1_center` | BaseBEV | center (有z) | 44.18 | **51.86** | 31.92/39.55/61.06 | 45.38/47.90/62.31 | 0.42M | 132.9G |
| `head_n3_center` | PPMDFEN | center (有z) | 42.56 | 50.87 | 33.21/33.08/61.38 | 45.28/41.83/65.50 | 1.22M | 131.8G |
| `head_n6_center` | RadarNeXtMDFEN | center (有z) | 39.95 | 46.79 | 29.67/32.71/57.46 | 38.39/40.03/61.95 | 1.17M | 128.4G |
| `rp_mask_center` | BaseBEV | rp center | 35.85 | 40.12 | 10.59/40.10/56.87 | 13.59/46.91/59.85 | 0.42M | 132.9G |
| `rp_mask` | BaseBEV | rp anchor | 31.02 | 34.34 | 13.51/29.00/50.55 | 14.28/35.41/53.32 | 0.18M | 34.3G |
| `head_b8_2d` | RepDWC | 2d (无z) | 0† | **50.23** | — | 44.66/41.00/65.04 | 0.42M | 153.9G |
| `head_n6_2d` | RadarNeXtMDFEN | 2d (无z) | 0† | 49.11 | — | 41.48/38.26/67.59 | 1.13M | 124.5G |
| `head_n3_2d` | PPMDFEN | 2d (无z) | 0† | 48.81 | — | 44.79/37.86/63.78 | 1.18M | 127.9G |
| `head_b1_2d` | BaseBEV | 2d (无z) | 0† | 48.14 | — | 44.66/40.74/59.04 | 0.39M | 117.2G |
| `head_n7_2d` | RadarNeXtMDFEN | 2d (无z) | — | — | — | — | — | — |

† 2d 头无 z, 3D 恒 0 是设计非 bug；BEV 为其有效指标。
⚠️ `head_n7_2d` eval 未产出（results.json 空）, ckpt_80/best.pth 当时在盘, 原始 OR 已清理 → 补跑需重训。

## 2. 后续 run（08-14/15, b1 底座复跑 + z/h 消融, OR 在盘）

| tag | 说明 | bestEp | 3d_mean | bev_mean | bev (car/ped/cyc) | Params | FLOPs | OR |
|---|---|---|---:|---:|:---:|---:|---:|---|
| `head_b1_center` 复跑 | 同 cfg 重训（单跑波动对照） | 72 | 42.94 | 52.59 | 48.15/43.06/66.56 | 0.42M | 132.9G | `202608141442_rpillar_head_b1_center_head_b1_center` |
| `head_b1_2d` 复跑 | 无 z 基线刷新 | 71 | 0† | 51.87 | 46.00/40.84/68.78 | 0.39M | 117.2G | `202608141140_rpillar_head_b1_2d_head_b1_2d` |
| `head_b1_2d_lossup` | 2d 头 loss 加权 | 72 | 0† | 51.84 | 44.61/42.27/68.63 | 0.39M | 117.2G | `202608141913_rpillar_head_b1_2d_lossup_head_b1_2d_lossup` |
| `head_b1_abl_z` | center 头去 z | 74 | 0† | 49.71 | 45.04/40.50/63.60 | 0.42M | 132.6G | `202608142232_rpillar_head_b1_abl_z_head_b1_abl_z` |
| `head_b1_abl_h` | center 头去 h | 70 | 3.89 | 50.52 | 44.14/40.89/66.53 | 0.39M | 117.4G | `202608150143_rpillar_head_b1_abl_h_head_b1_abl_h` |

消融读法（b1 底座, BEV mean）：center(z+h)=52.59 > 2d(去z+h)=51.87 > abl_h(留z去h)=50.52 > abl_z(去z留h)=49.71 —— BEV 增益来自 **z+h 联合监督**, 单独任一都不涨甚至掉；含 z/h 缺失的 head 3D AP 恒 0（见 memory zh-joint-supervision-synergy）。`lossup` BEV 51.84 ≈ 2d 51.87, loss 加权无增益。
复跑 vs 原 sweep：b1_center 3D 42.94 vs 44.18、b1_2d BEV 51.87 vs 48.14, 单跑波动 ~1-4pp, 交叉对比以同批 run 为准。

## 3. 裁决与关键发现（详见 20260810_stage4_report.md）

1. **3D 最强** `head_b8_center` (44.26)；**BEV 最强** `head_b1_center` (51.86)。RepDWC 换来 3D 小增益, BEV 反略降。
2. **2d 头 BEV ≈ center 头 BEV**（b8: 50.23 vs 51.41）→ 无 z 的 2D 头 BEV 不掉点, 合理轻量化路线。
3. **rp_mask 对照**: 自家轻量 head (0.18-0.42M) 全面弱于 RadarNeXt 头 ~10pp, Car 项尤差(<15)。

## 4. 缺口

- `head_n7_2d` eval 缺失（原始 ckpt 已清理, 需重训才能补齐第 11 个数据点）。
- 08-10 sweep 原始 OR（train log/ckpt/eval json）不在盘上, 数据以报告为准；08-14/15 后续 run 完整在盘（含 resbag）。
