# h11 Head-Swap 实验报告

**生成时间**: 2026-08-10（2026-08-13 修订：补 BEV 评估 + tag 命名说明）
**scope**: 11 个 head-swap 实验 = `<2D-backbone 变体> × <检测头>` 的二维组合 + rp_mask 对照线
**eval**: VoD val, R40 moderate, IoU=0.5(Car)/0.25(Ped/Cyc), EAA filter
**seed**: 666 (FROZEN)
**协议**: bs=8 ep=80, pickbest = ckpt 80 (单 ckpt 评估, 无 median)

---

## 0. tag 命名规则

实验 = **2D backbone (4 种) × 检测 head (3 种) 的 swap**。tag 形如 `head_<bb2d>_<head>`，`rp_mask` 线单独命名。每个 tag 实际配置：

| tag | backbone (2D) | head | head 类型 |
|-----|---------------|------|----------|
| `head_b1_center` | BaseBEVBackbone | RadarNeXtCenterHead | 3D |
| `head_b8_center` | RepDWCNoneBackbone | RadarNeXtCenterHead | 3D |
| `head_n3_center` | PPMDFENBackbone | RadarNeXtCenterHead | 3D |
| `head_n6_center` | RadarNeXtMDFENBackbone | RadarNeXtCenterHead | 3D |
| `head_b1_2d` | BaseBEVBackbone | RadarNeXtCenterHead2DNoZ | BEV-only |
| `head_b8_2d` | RepDWCNoneBackbone | RadarNeXtCenterHead2DNoZ | BEV-only |
| `head_n3_2d` | PPMDFENBackbone | RadarNeXtCenterHead2DNoZ | BEV-only |
| `head_n6_2d` | RadarNeXtMDFENBackbone | RadarNeXtCenterHead2DNoZ | BEV-only |
| `head_n7_2d` | RadarNeXtMDFENBackbone | RadarNeXtCenterHead2DNoZ | BEV-only |
| `rp_mask_center` | BaseBEVBackbone | RadarPillarCenterHead | 3D (RadarPillar 自家) |
| `rp_mask` | BaseBEVBackbone | RadarPillarAnchorHeadSingle | 3D anchor (RadarPillar 自家) |

backbone token 速查：`b1`=BaseBEV, `b8`=RepDWC, `n3`=PPMDFEN, `n6/n7`=RadarNeXtMDFEN。
head 三种：`center`=RadarNeXtCenterHead(有 z)、`2d`=RadarNeXtCenterHead2DNoZ(无 z)、`rp_mask`=RadarPillar 自家 head(center/anchor 两版)。

**评估口径**：3D 头(有 z) 评 3D+BEV；2d 头(无 z) 3D 恒 0，**BEV 为其唯一有效指标**。

---

## 1. 总表：3D + BEV (R40, moderate)

明细格式 `car/ped/cyc`；2d 头无 z，3D 恒 0（设计），BEV 为其有效指标。

| tag | 3d_mean | bev_mean | 3d (car/ped/cyc) | bev (car/ped/cyc) | params(M) | flops(G) |
|-----|--------:|---------:|:----------------:|:-----------------:|----------:|---------:|
| `head_b8_center` | **44.26** | **51.41** | 33.49/37.00/62.29 | 45.22/43.94/65.07 | 0.46 | 169.5 |
| `head_b1_center` | **44.18** | **51.86** | 31.92/39.55/61.06 | 45.38/47.90/62.31 | 0.42 | 132.9 |
| `head_n3_center` | **42.56** | **50.87** | 33.21/33.08/61.38 | 45.28/41.83/65.50 | 1.22 | 131.8 |
| `head_n6_center` | **39.95** | **46.79** | 29.67/32.71/57.46 | 38.39/40.03/61.95 | 1.17 | 128.4 |
| `rp_mask_center` | **35.85** | **40.12** | 10.59/40.10/56.87 | 13.59/46.91/59.85 | 0.42 | 132.9 |
| `rp_mask` | **31.02** | **34.34** | 13.51/29.00/50.55 | 14.28/35.41/53.32 | 0.18 | 34.3 |
| `head_b8_2d` | 0 | **50.23** | — | 44.66/41.00/65.04 | 0.42 | 153.9 |
| `head_n6_2d` | 0 | **49.11** | — | 41.48/38.26/67.59 | 1.13 | 124.5 |
| `head_n3_2d` | 0 | **48.81** | — | 44.79/37.86/63.78 | 1.18 | 127.9 |
| `head_b1_2d` | 0 | **48.14** | — | 44.66/40.74/59.04 | 0.39 | 117.2 |
| `head_n7_2d` | — | — | — | — | — | — |

---

## 2. key finding

- **3D 最强**: `head_b8_center` (44.26)；**BEV 最强**: `head_b1_center` (51.86)。RepDWC(b8) 换来 3D 小增益，BEV 反略降。
- **2d 头 BEV ≈ center 头 BEV**: `head_b8_2d` 50.23 ≈ `head_b8_center` 51.41。无 z 的 2D 头在 BEV 上不掉点，合理轻量化路线。
- **rp_mask 对照**: 自家轻量 head (0.18–0.42M) 全面弱于 RadarNeXt 头，3D/BEV 各落后 ~10 分，Car 项尤差(<15)。

---

## 3. issue / 偏差

- ⚠️ `head_n7_2d` **eval 未产出结果** (`results.json` 为空)；ckpt_80 / best.pth 均在，**补跑 eval 即可，无需重训**。
- ℹ️ 2d 头 3D 恒 0 是设计（无 z），非 bug；BEV 为其有效指标（见 §2）。
- ✅ 10/11 模型 3D+BEV 双评估数据有效。
- ✅ seed 666 FROZEN 在所有 yaml 中一致。

---

## 4. 下一步建议

1. 补跑 `head_n7_2d` eval (ckpt 已就绪)，补齐第 11 个 BEV 数据点。
2. 若需 backbone × head 交叉对比，按 §0 字典重排即可（如 b8 系: center vs 2d；n6 系: center vs 2d）。
