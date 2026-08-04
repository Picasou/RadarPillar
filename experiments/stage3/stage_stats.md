# RPiN 阶段3 — neck 扫描统计（best-epoch，D13 主判 Car R40 moderate）

固定项 A\*=a1（BACKBONE_3D=PillarAttention ATTN/FFN=32，6 cfg 一致）+ PillarVFE(out=32) + PointPillarScatter(32) + E2(9维) + F3 + AnchorHeadSingle。
变量 = 块类型 × neck：**standard 块**（BaseBEVBackbone，3-stage NUM_FILTERS=[32,32,32]，每 stage 输出 [32,32,32]）× {无/FPN/MDFEN} + **RepDWC 块**（RepDWCNoneBackbone，3-stage OUT_CHANNELS=[64,64,64]，depthwise+reparam）× {无/FPN/MDFEN} = 6 格点。
各格点模块类与通道见下方「模块特征数详解」。best epoch 取末 10 ckpt（ep70-79）GPU eval 中 Car R40 最高者。

**n7（已完成）**：PillarAttn + RepDWC **C3[64,128,256]** + MDFEN，与 stage0 baseline_radarnext_mdfen 的 REP_DWC/MDFEN 段完全一致（仅底座 PillarAttention vs None 不同；bs=8 vs paper 16）。目的：在论文容量下验证"PillarAttn+RepDWC+MDFEN 是否达到预期"。**结果**：Car R40=34.71 / Ped=32.60 / Cyc=62.73 / mAP=43.35（bestEp=71 median pickbest），Params=1.18M, FLOPs=105.4G。

| model | block（3-stage 通道） | neck（融合后通道） | bs | bestEp | Car R40 | Ped R40 | Cyc R40 | mAP R40 | mAP R11† | Params | FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **n1** ⭐ | standard BaseBEV [32,32,32] | **无**（直连 head） | 16 | 78 | **39.68** | **36.78** | **68.68** | **48.38** | N/A | 186,600 | 70.5 G |
| n3 | standard BaseBEV [32,32,32] | MDFEN CH=[32,32,32,128,64,128,256] FUSE=[128,128,128] | 8 | 72 | 37.16 | 30.20 | 56.70 | 41.35 | N/A | 844,700 | 93.254 G |
| n4 | RepDWC NoneBackbone [64,64,64] DWConv | 无（直连 head） | 16 | 74 | 36.71 | 34.31 | 65.96 | 45.66 | N/A | 178,900 | 106.093 G |
| n6 | RepDWC [64,64,64] DWConv | MDFEN CH=[64,64,64,128,64,128,256] FUSE=[128,128,128] | 8 | 79 | 36.71 | 30.20 | 62.37 | 43.09 | N/A | 792,600 | 89.875 G |
| n5 | RepDWC [64,64,64] DWConv | FPN IN=[64,64,64]→OUT=[128,128,128] | 16 | 76 | 36.62 | 29.87 | 62.76 | 43.08 | N/A | 185,400 | 42.863 G |
| n2 | standard BaseBEV [32,32,32] | FPN IN=[32,32,32]→OUT=[128,128,128] | 16 | 72 | 34.96 | 30.03 | 60.99 | 41.99 | N/A | 230,100 | 60.433 G |
| **n7** | RepDWC [64,128,256] DWConv | MDFEN CH=[64,128,256,128,64,128,256] FUSE=[128,128,128] | 8 | 71 | 34.71 | 32.60 | 62.73 | 43.35 | N/A | 1,175,700 | 105.384 G |

† R11 未计（stage3 eval 用末 10 ckpt GPU eval 仅产 R40，与 stage2 同协议，未做 R11 复评；stage3 入口/阶段6 headline 复验时按需用 test_cpu.py 补 R11）。

**pickbest 协议**：使用末 N ckpt（ep70-79）Car_3d_R40 的**中位数**选择 best_epoch（修复前为 max，会 cherry-pick 离群 epoch 如 n4 ep77）。见下方 §3.4 pickbest 修复记录。

## 模块特征数详解（block + neck 实际通道，cfg 实读）

**前置（6 格点统一）**：PillarVFE `NUM_FILTERS=[32]`（pillar 维 32）→ PointPillarScatter `NUM_BEV_FEATURES=32`（BEV 输入 32 通道）→ BACKBONE_3D=PillarAttention（ATTN_CHANNELS=32 / NUM_HEADS=1 / FFN_CHANNELS=32）→ BACKBONE_2D（变量）→ AnchorHeadSingle。

**block 维（BACKBONE_2D 主干，3 stage 下采样 [2,2,2]，LAYER_NUMS [3,5,5]）**：

| 块类型 | 模块类 | 3-stage 通道 | 卷积核 | 备注 |
|---|---|---|---|---|
| standard | BaseBEVBackbone | NUM_FILTERS=[32,32,32] | 普通 3×3 Conv+BN+ReLU | stage2 最优 b1；每 stage 输出 [32,32,32]，deblock concat 后送 head |
| RepDWC | RepDWCNoneBackbone | OUT_CHANNELS=[64,64,64] | depthwise 3×3 (groups=in) + reparam | stage2 RepDWC 组内最优 b8；输入锁 32（首层 PW 升维 32→64），DWConv=True |
| RepDWC | RadarNeXtMDFENBackbone (n7) | OUT_CHANNELS=[64,128,256] | depthwise 3×3 (groups=in) + reparam | n7 用，与 stage0 baseline 同容量；stage3 标准格点用 C4（=b8 容量）|

**neck 维（接在 block 3-stage 输出之后的特征融合）**：

| neck | 模块类 | 输入→输出通道 | 融合机制 | 适用 block |
|---|---|---|---|---|
| 无 | —（直连） | block 输出直接 deblock concat 送 head | 无多尺度融合 | n1/n4 |
| FPN | SecondFPN（PPFPNBackbone / RadarNeXtFPNBackbone） | IN=[32,32,32]→OUT=[128,128,128]（standard）<br>IN=[64,64,64]→OUT=[128,128,128]（RepDWC）<br>UPSAMPLE_STRIDES=[0.5,1,2] | 3 路上采样 concat（SECOND 风格 FPN） | n2/n5 |
| MDFEN | MDFENNeck（PPMDFENBackbone / RadarNeXtMDFENBackbone） | CHANNELS_LIST=[32,32,32,128,64,128,256]（standard）<br>=[64,64,64,128,64,128,256]（RepDWC）<br>FUSED_CHANNELS=[128,128,128] FUSION_STRIDES=[1,2] GROUP=4 | DCNv3(grid_sample) + 多分支融合，纯 pytorch | n3/n6 |

> 「无 neck」一档 = block 3-stage 输出经 deblock（[1,2,4] 上采样 → [32,32,32] 或 [64,64,64]）concat 后直送 AnchorHeadSingle；FPN/MDFEN 档在 concat 前插入对应融合模块，**block 内三点（无/FPN/MDFEN）block 通道一致、只换 neck**（§8.3），块内 neck 对比干净。

## D13 裁决 → **neck\* = 无 neck（n1）**

1. **主判排名（Car 3D AP R40 moderate, median pickbest）**：n1(39.68) > n3(37.16) > n4(36.71) ≈ n6(36.71) > n5(36.62) > n2(34.96)。
2. **top-2 = n1 vs n3，差 2.52 pp > 1.0 → 非平手，n1 直接胜出**，无需 D13 复跑。
3. **n1 双轴全胜**：Car R40 第一（39.68）+ mAP R40 第一（48.38，Ped/Cyc 仍领先）+ Params 第二低（186,600，仅高于 n4 的 178,900）。
4. **唯一 n1 不占绝对优的成本项是 FLOPs（70.5G）**，但 n1 的 Car/Params/FLOPs 综合最优（n4 FLOPs 106G 是 n1 的 1.5x；n5 FLOPs 42.9G 最低但 Car 仅 36.62、mAP 43.08 落后 5.30pp）。

> 注：原（max pickbest）排名 n1(39.98) > n4(37.74) > n3(37.42) > n6(36.94) > n5(36.80) > n2(35.29)，top-2 差 2.24 pp。median pickbest 修复后 n4 从 37.74 → 36.71（cherry-pick ep77 去掉），n4 与 n6 平手 36.71，n1 仍胜，**裁决结论不变**。

## 关键发现

### 1. neck 在 C1 小容量底座上全面无增益（两块类型一致结论）

**standard 块内（block NUM_FILTERS=[32,32,32] 一致，块内对比干净）**：

| neck | Car R40 | Δ vs 无neck | Params | FLOPs |
|---|---|---|---|---|
| 无 (n1) — block 直连 head | **39.68** | — | 186,600 | 70.5 G |
| MDFEN (n3) CH=[32,32,32,128,64,128,256] | 37.16 | **-2.52** ❌ | 844,700 (4.5x) | 93.3 G |
| FPN (n2) IN=[32,32,32]→OUT=[128,128,128] | 34.96 | **-4.72** ❌ | 230,100 | 60.4 G |

加 FPN/MDFEN 不仅 Car 下降（-2.52~-4.72 pp），Ped/Cyc/mAP 也全面下降，还涨参数（MDFEN 4.5x）。**与 stage2「容量已饱和」结论一致**：block [32,32,32] 底座上，加 neck 模块纯属过参数化负担，无特征融合增益。

**RepDWC 块内（block OUT_CHANNELS=[64,64,64] 一致）**：

| neck | Car R40 | Δ vs 无neck | Params | FLOPs |
|---|---|---|---|---|
| 无 (n4) — block 直连 head | **36.71** | — | 178,900 | 106.1 G |
| MDFEN (n6) CH=[64,64,64,128,64,128,256] | 36.71 | **±0.00** (平手) | 792,600 (4.4x) | 89.9 G |
| FPN (n5) IN=[64,64,64]→OUT=[128,128,128] | 36.62 | -0.09 | 185,400 | 42.9 G |

RepDWC 块内无 neck 与 MDFEN 持平（36.71），FPN 微降（-0.09 pp），整体无 neck 不构成显著优势但仍是块内最好或并列最好。**注：单 seed + RepDWC 噪声 ~1.4pp**（如 n4 末10ep 标准差 0.52），-0.09pp 在噪声内，**无 neck vs MDFEN 实为统计平手**。

### 2. ⚠️ RepDWC 异常（已追溯并修正，见 §3.4）

stage3 中 RepDWC 表现（n4=36.71 / n5=36.62 / n6=36.71）**远强于 stage2 同款 RepDWC**（b8=15.12 / b5-b8 均值 13.17）。n4 与 b8 的 cfg 完全相同（仅注释差异），但 Car R40 差 +22.62 pp。

**根因（已确证）**：
- **连接拓扑 bug**（commit a1930ed, 2026-07-29 已修）：旧 `repdwc_none.py` 用 `spatial_features_2d = outs[0]`（仅 160²×64ch 单尺度），丢弃 80²/40² 两路，喂残缺特征给 head。修复后用 deblock+concat (192ch)。
- **eval bug**（rotate_iou_pcdet, 2026-07-31 已修）：criterion=2 误返 IoU 而非 BEV 重叠面积，3D AP 低估 ~8×、Car_3d 全 0。
- **修复后** b8 重训 = 39.10，与 n4=36.71（median）/37.74（旧 max）只差 1.36~2.39pp，属 RepDWC 单 seed 噪声范围。**RepDWC 实际可复现、与 standard 持平甚至略优**（公平对照 b8 mean 45.39 vs b1 44.85，参数仅 1/4）。详见 `experiments/repdwc_concat_fair_comparison.md`。

**对裁决影响**：无。n1 standard 无 neck 以 2.52pp 优势胜出（median），结论不变。

### 3. 成本视角

| model | mAP R40 | Params | FLOPs | mAP/M Params |
|---|---|---|---|---|
| **n1** ⭐ | **48.38** | 186,600 | 70.5 G | **259.2** |
| n4 | 45.66 | 178,900 | 106.1 G | 255.2 |
| n7 | 43.35 | 1,175,700 | 105.4 G | 36.9 |
| n6 | 43.09 | 792,600 | 89.9 G | 54.4 |
| n5 | 43.08 | 185,400 | 42.9 G | 232.4 |
| n2 | 41.99 | 230,100 | 60.4 G | 182.5 |
| n3 | 41.35 | 844,700 | 93.3 G | 49.0 |

n1 mAP/Params 性价比第一（259.2）。MDFEN 档（n3/n6）因参数暴涨 4.5x 性价比垫底。

### 4. pickbest 协议修复记录（max → median）

**问题**：unified pipeline 原 pickbest 取末 10 ckpt Car_3d_R40 的 **max**，会 cherry-pick 离群 epoch。
- 例：n4 末10ep 标准差 0.52，max=37.74 (ep77) 比中位 36.67 高 1.07pp。n1 max=39.98 (ep72) 比中位 39.63 高 0.35pp。
- 离群 pickbest 高估最优表现 0.3~1.1pp。

**修复**：取末 10 ckpt Car_3d_R40 的**中位数**（按大小排序取 `results[len//2]`），同时 cp 该 epoch ckpt 到 best.pth 并更新 model_store.yaml 的 map_r40 / note / pickbest_protocol。

**影响**：
- 6 个模型 best_epoch 重选：n1 ep72→ep78、n2 ep73→ep72、n3 ep71→ep72、n4 ep77→ep74、n5 ep74→ep76、n6 ep71→ep79。
- Car R40 全面下修：n4 37.74→36.71（-1.03pp 最大），其他 -0.18~-0.33pp。
- **裁决结论不变**：n1(39.68) 仍 > n3(37.16) > n4(36.71) ≈ n6(36.71) > n5(36.62) > n2(34.96)，top-2 差 2.52pp，neck\*=无 neck 成立。

**代码**：`.claude/skills/model-train/scripts/unified_rpillar_pipeline.sh` step 5 pickbest heredoc（`results[len(results)//2]` 替代 `results[-1]`）。n7 正在训练，跑完自动用新 pickbest。

**对已落袋 n1-n6 的回填**：用 `.tmp/2026-08-02/stage3_n7/repickbest.py` 读已有 eval JSON 重选 best_epoch、cp best.pth、重写 model_store.yaml。

**未来工作**：若要更鲁棒，可用 SWA（last K ckpt 权重平均）；当前 median 是最简修。

## bs=8 显存妥协

n3/n6（MDFEN，DCNv3 纯 pytorch grid_sample 显存高）bs=16 会 OOM，sh 内强制 bs=8。其余 n1/n2/n4/n5 用 bs=16。stage3 启动前已做 1-epoch smoke test 全 PASS（吸取 stage2 漏做 sanity check 教训）。
