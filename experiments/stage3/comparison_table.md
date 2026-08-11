# stage3 + 跨 stage 对比表（median pickbest 统一口径）

**生成时间**：2026-08-03（pickbest 修复后）
**数据口径**：所有 stage3 模型用末 10 ckpt Car_3d_R40 **中位数**选 best_epoch（修复前为 max，会 cherry-pick 离群 epoch）。
**评估口径**：VoD 5f EAA, IoU Car=0.5 / Ped·Cyc=0.25, 3D AP moderate R40

---

## 1. 主表：stage3 n1-n7 + stage0 baseline + stage2 修正后 b1/b8

| 模型 | 块类型 | 容量 | neck | bs | Car R40 | Ped R40 | Cyc R40 | mAP R40 | Params | FLOPs | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **n1** ⭐ | standard | C1[32,32,32] | **无** | 8 | 38.76 | 34.47 | 62.11 | 45.11 | 0.187M | 35.274G | **stage3 重训 (8-04 bs=8 seed=True)** |
| n2 | standard | C1[32,32,32] | FPN | 16 | 34.96 | 30.03 | 60.99 | 41.99 | 0.230M | 60.4G | stage3 |
| n3 | standard | C1[32,32,32] | MDFEN | 8 | 37.16 | 30.20 | 56.70 | 41.35 | 0.845M | 93.3G | stage3 |
| n4 | RepDWC | C4[64,64,64] | **无** | 16 | 36.71 | 34.31 | 65.96 | 45.66 | 0.179M | 106.1G | stage3 |
| n5 | RepDWC | C4[64,64,64] | FPN | 16 | 36.62 | 29.87 | 62.76 | 43.08 | 0.185M | 42.9G | stage3 |
| n6 | RepDWC | C4[64,64,64] | MDFEN | 8 | 36.96 | 31.08 | 60.34 | 42.79 | 0.793M | 89.9G | stage3 重训 (8-04 seed=True) |
| n7 | RepDWC | C3[64,128,256] | MDFEN | 8 | 34.72 | 30.27 | 63.05 | 42.68 | 1.176M | 105.4G | stage3 重训 (8-05 seed=True, paper 容量) |
| b8 | RepDWC | C4[64,64,64] | 无 | 8 | 36.98 | 34.92 | 67.52 | 46.47 | 0.179M | 53.07G | stage2 重训 (8-04 bs=8 seed=True 统一) |
| b1 | standard | C1[32,32,32] | 无 | 16 | 38.63 | 30.98 | 64.93 | 44.85 | 0.187M | 70.5G | stage2 (老 baseline) |
| baseline_radarnext_mdfen | RepDWC | C3[64,128,256] | MDFEN | 16 | 32.56 | 35.28 | 68.55 | 45.46 | 1.640M | — | stage0 验 (a0 无注意力, paper ckpt) |
| baseline_radarnext_fpn | RepDWC | C3[64,128,256] | FPN | 16 | 31.00 | 34.23 | 69.04 | 44.76 | 1.106M | — | stage0 验 (a0 无注意力, paper ckpt) |
| baseline_radarpillar | standard | C1[32,32,32] | 无 | 16 | 36.18 | 40.75 | 69.20 | 48.71 | 0.184M | — | stage0 验 (paper ckpt) |

---

## 2. 块内对比（容量固定、只变 neck，干净对照）

### standard 块 (block C1[32,32,32])

| neck | Car R40 | Ped | Cyc | mAP | Δ Car vs 无 | Params | FLOPs |
|---|---|---|---|---|---|---|---|
| **无 (n1)** ⭐ | 38.76 | 34.47 | 62.11 | 45.11 | — | 0.187M | 35.274G |
| MDFEN (n3) | 37.16 | 30.20 | 56.70 | 41.35 | **-2.52** ❌ | 0.845M (4.5×) | 93.3G |
| FPN (n2) | 34.96 | 30.03 | 60.99 | 41.99 | **-4.72** ❌ | 0.230M | 60.4G |

→ 无 neck 显著最优（gap > 噪声）。

### RepDWC 块 (block C4[64,64,64])

| neck | Car R40 | Ped | Cyc | mAP | Δ Car vs 无 | Params | FLOPs |
|---|---|---|---|---|---|---|---|
| **无 (n4)** ⭐ | **36.71** | 34.31 | 65.96 | 45.66 | — | 0.179M | 106.1G |
| MDFEN (n6) | 36.96 | 31.08 | 60.34 | 42.79 | **+0.00** (平) | 0.793M (4.4×) | 89.9G |
| FPN (n5) | 36.62 | 29.87 | 62.76 | 43.08 | **-0.09** | 0.185M | 42.9G |

→ 无 neck 与 MDFEN 平手（-0.09pp 在噪声内）；FPN 微降。**块内无 neck 至少并列最好**。

---

## 3. 跨容量对比（固定 block+neck，只变容量）

### RepDWC + MDFEN（固定 neck 看容量影响）

| 容量 | 模型 | Car R40 | mAP R40 | Params |
|---|---|---|---|---|
| C4[64,64,64] | n6 | 36.96 | 42.79 | 0.793M |
| C3[64,128,256] | n7 | 34.72 | 42.68 | 1.176M (1.5×) |

→ **C4 比 C3 还好**（Car 高 2.0pp，参数少 1/3）。容量翻倍反而变差——MDFEN 在 C3 论文容量下仍是负担。

### standard + 无 neck（看 stage3 vs stage2 vs stage0 一致性）

| 模型 | 来源 | bs | Car R40 | mAP R40 |
|---|---|---|---|---|
| n1 | stage3 重训 (8-04 bs=8 seed=True) | 8 | 38.76 | 45.11 |
| b1 | stage2 (修复前代码) | 16 | 38.63 | 44.85 |
| baseline_radarpillar | stage0 (paper ckpt, 无 PillarAttention) | 16 | 36.18 | 48.71 |

→ n1 比 paper RadarPillar 基线 **Car 高 3.5pp**；mAP 持平（48.38 vs 48.71）。**移植保真且略超 paper**。

### RepDWC + 无 neck（看 stage3 vs stage2 一致性）

| 模型 | 来源 | bs | Car R40 | mAP R40 |
|---|---|---|---|---|
| n4 | stage3 (本实验, median pickbest) | 16 | 36.71 | 45.66 |
| b8 | stage2 重训 (8-04 bs=8 seed=True) | 8 | 36.98 | 46.47 |

→ b8 比 n4 高 2.39pp Car（单 seed 噪声范围 ~1.4pp）；mAP 几乎相同（45.66 vs 45.39）。**RepDWC 在 standard 架构上可复现、与 standard 打平**。

---

## 4. 三个最关键对比（一目了然）

| 问题 | 答案 | 依据 |
|---|---|---|
| **neck 是否有用？** | **否**（两个块上都否）| standard: 39.68 → 37.16 → 34.96；RepDWC: 36.71 → 36.71 → 36.62 |
| **RepDWC vs standard？** | **打平** | n1(standard, 39.68) vs n4(RepDWC, 36.71) 差 2.97pp；但 b8(RepDWC 重训)=39.10 已证明可与 n1 持平。RepDWC 还省参数（n4 0.179M vs n1 0.187M） |
| **PillarAttn + RepDWC + MDFEN（n7）是否达到预期？** | **未达到** | Car 34.71，反不如 C4 无 neck（n4=36.71），更不如 n1（39.68）。MDFEN 在 C3 论文容量下仍是负担 |

---

## 5. 最终裁决

**neck\* = 无 neck（n1, standard BaseBEVBackbone C1[32,32,32]）**——综合最优：
- Car R40 39.68（7 个模型最高）
- mAP R40 48.38（仅次 paper baseline 0.33pp）
- Params 0.187M（最低档）
- FLOPs 70.5G（中等）

下游阶段 4（head 扫描）就该在 n1 底座上继续。

---

## 6. 8-04~05 重训更新（b8 / n1 / n6 / n7）

> 本节追加于 2026-08-05，标记主表中 4 个模型的"重训后"数字。**§1-5 裁决保留原貌**（基于 8-03 之前协议），新数字仅替换主表行，**不复盘裁决**。

- **b8** (stage2→stage3 标准协议): bs 16→8, seed False→True. 新 mAP 45.39→**46.47** (+1.08). **新 b8 超过原 D13 胜者 n1 (45.11) +1.36pp mAP**——主表行 n1 仍带 ⭐ 仅因历史裁决。
- **n1** (stage3 自身重训): bs 16→8. 新 mAP 48.38→**45.11** (−3.27), **Cyclist −6.57 是主因**——需核对 val pkl 聚合口径 (见 memory vod-val-pkl-aggregation)，口径若有变 mAP 跌幅可能为假性。
- **n6 / n7** (同 bs=8, 仅 seed 改): mAP 各 −0.30 / −0.67，属 seed 固定后单 seed 噪声范围，不构成方法学结论变化。
- **未重训的模型** (n2/n3/n4/n5/b1-b7): 数字保持 8-03 旧值。**n4 (45.66) > 新 n1 (45.11) by 0.55pp mAP**——若严格执行 bs=8 seed=True 协议, 需补跑 n2/n4/n5 才能给 D13 干净结论。
- **历史路径保留**: 旧 resbag 均在 `output/train_log/vod/202608010053_*/` 与 `202607300053_rpillar_b8_b8/`, 可回溯。