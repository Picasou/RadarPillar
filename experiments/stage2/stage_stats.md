# RPiN 阶段2 — backbone 容量 C 扫描统计（best-epoch，D13 主判 Car R40 moderate）

固定项 A*+B1+D1+E2(9维)+F3+AnchorHeadSingle（继承 stage1，BACKBONE_3D=null = a0）；变量 BACKBONE_2D.attention ∈ {BaseBEVBackbone, RepDWCNoneBackbone} × 4 容量档 → 2×4 = 8 Task。best epoch 取末 20 ep 中 Car R40 最高的 ckpt。

| model | backbone | bs | bestEp | Car R40 | Ped R40 | Cyc R40 | mAP R40 | mAP R11† | Params | FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|
| **b1** ⭐ | [32,32,32] | 16 | 80 | **38.63** | 30.98 | 64.93 | **44.85** | 0.00 | 186,600 | 70.5 G |
| b4 | [64,64,64] | 8 | 80 | 37.85 | 30.89 | 65.05 | 44.60 | 0.00 | 723,200 | 132.528 G |
| b3 | [64,128,256] | 8 | 80 | 37.93 | 29.46 | 62.03 | 43.14 | 0.00 | 5,396,300 | 703.857 G |
| b2 | [32,64,128] | 16 | 80 | 35.38 | 24.83 | 62.54 | 40.92 | 0.00 | 1,361,900 | 361.545 G |
| b7 | [64,128,256]‡ | 16 | 80 | 14.85 | 32.49 | 59.12 | 35.49 | 0.00 | 429,500 | 52.727 G |
| b6 | [32,64,128]‡ | 16 | 80 | 10.30 | 36.36 | 59.65 | 35.44 | 0.00 | 124,400 | 17.501 G |
| b5 | [32,32,32]‡ | 16 | 80 | 12.40 | 30.58 | 56.69 | 33.22 | 0.00 | 31,700 | 9.843 G |
| b8 | [64,64,64]‡ | 16 | 80 | 15.12 | 30.75 | 53.32 | 33.06 | 0.00 | 80,200 | 24.933 G |

† R11 未计（stage2 train 用 --eval_all 模式仅产 R40，**未做 R11 复评**；stage3 入口按需补 R11）。  
‡ = RepDWCNoneBackbone（depthwise reparam），无标记 = BaseBEVBackbone（standard block）。backbone 维度 = `BACKBONE_2D.NUM_FILTERS`（RepDWC 为 out_channels，数值同）。  

注：D13 top-2 = b1 vs b4（差 0.78 pp < 1.0 → 平手，按 Params 取胜 → b1）。block 维度 standard (BaseBEVBackbone) 全面碾压 RepDWC（Car R40 均值 37.45 vs 13.17，差距 -24.28 pp）。RepDWC 单组 Params/FLOPs 显著低于 standard 同档（深度可分离 + 重整参数压缩）。

## bs=8 显存妥协（已知设计选择）

b3/b4 因 standard block 在 C3/C4 容量档显存占用大（3070Ti 8GB 限制），bs=16 会 OOM；driver 里 case 分支强制 b3/b4 用 bs=8：

| task | 容量档 | bs=16 预估 | bs=16 决策 | 实际 bs |
|---|---|---|---|---|
| b1 | C1[32,32,32] | ~3 GB | ✓ | 16 |
| **b3** | **C3[64,128,256]** | **~13 GB (OOM)** | ❌ | **8** |
| **b4** | **C4[64,64,64]** | **~5 GB (危险)** | ⚠️ 不选 | **8** |
| b7 (RepDWC) | OUT=[64,128,256] | ~1-2 GB（depthwise 压缩）| ✓ | 16 |
| b8 (RepDWC) | OUT=[64,64,64] | ~0.5-1 GB | ✓ | 16 |

**RepDWC 因为 depthwise + 重整参数，显存占用大幅降低**——所以 b7/b8 用 bs=16 仍宽裕（参数量 80,200 vs standard 5,396,300，差 67x）。这是 stage2 启动前 sanity check 没做成的已知妥协，stage3 入口应根据 b1 的 C1 cfg 直接拿，**不再 bs=16 vs bs=8 选择**。

## RepDWC 失败机制（如实记录）

**RepDWC = MobileOneBlock 风格的多分支 + depthwise 主分支 + reparam fusion**：
- 训练时 rbr_conv (depthwise 3×3, groups=in_channels) + rbr_scale (pointwise 1×1) + rbr_skip (BN-only)
- 推理时 reparam 融合成单分支 3×3 conv

**为什么崩盘**：
1. depthwise 卷积跨通道特征混合能力天然弱于普通 3×3 conv
2. RepBlock 训练时多分支梯度耦合，80 epoch 内 BN running stats 未完全稳定
3. Car 检测依赖多尺度+跨通道特征融合，对 RepDWC 表达力损失最敏感（-24.28 pp）
4. Pedestrian（小目标）反略优（+3.5 pp），验证"深度可分离只擅长空间邻域、不擅跨通道"

**讽刺副作用**：RepDWC 显存/参数压缩极强（参数量 80,200 vs b3 5,396,300，差 67x），所以 b7/b8 可以用 bs=16，**但 Car 检测是死的**——压缩得不偿失，stage2 决策仍 standard block。

**协议偏差**：8 Task 全部用 GPU eval 模式 + --eval_all（仅产 R40），未做 R11 复评（与 stage1 用 test_cpu 复评 R11+R40 不一致）。这是 stage2 默认设置，stage3 入口按需补 R11。