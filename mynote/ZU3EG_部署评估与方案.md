# ZU3EG 部署评估与方案

## 1. 硬件平台与算力约束

Xilinx Zynq UltraScale+ MPSoC **ZU3EG**：

- **APU**：4× ARM Cortex-A53 @~1.2GHz，但**仅两核可用**（其余被系统/既有功能占用）
- **RPU**：2× ARM Cortex-R5F
- **PL**：71K LUT / 142K FF / 360 DSP48 / 144 BRAM36，**无 URAM**

**关键约束**：两核软跑检测模型算力不够（量级见 §5）——CPU 软跑路线排除，**方向定为 FPGA 加速**。

## 2. 待部署模型（msr_centerhead3d_trunk）

- 参数量 **0.27M**，计算量 **5.49 GFLOPs**（计数器口径，≈2.75 GMACs；按 200×80 BEV + [3,5,5]×32ch 主干手算 1.2~1.5 GMACs 同量级）
- 结构：PillarVFE(32) → PillarAttention(32) → PointPillarScatter → BaseBEVBackbone（3 stage×32ch）→ RadarNeXtCenterHeadTrunk
- 输入：range [0,-20,-10, 200,20,10]，voxel 1.0×0.5（BEV 200×80）
- FP32 基线：mAP 88.13（R40 moderate）——**INT8 后需重测**

## 3. 资源现状（功能 A 已占用后剩）

- LUT：50K/71K 已用（71%），剩 ~20K
- FF：60K/142K 已用（42%），剩 ~82K
- DSP：198/360 已用（55%），剩 162
- BRAM：112/144 已用（78%），**仅剩 32 ← 唯一瓶颈**
- CPU：剩余两核

## 4. 候选方案可行性

- **APU 两核软跑**：**排除**。~0.9~1.4 FPS（§5），比 10~20 Hz 需求差一个数量级
- **DPU B1600**：需 ~60 BRAM > 剩 32 → **本板不可行**
- **DPU B800**：需 ~30 BRAM → 勉强，时序收敛风险高，功能 A 扩展可能冲突
- **DPU B400**：需 ~20 BRAM → 可行，性能弱
- **自定义 HLS**：可按需省 BRAM，开发难度大
- *DPU 精确资源未在线核实，上板前以 PG338 综合结果为准；BRAM 判据差距大（60 vs 32），不影响结论*

## 5. 软跑性能量级（两核，供决策不供复用）

两核 A53（无 SDOT，INT8 用 SMLAL，单核 ~4 MAC/cycle）→ 工程 2.4~3.8 GMACs/s → 2.75 GMACs/帧 → 单帧 0.7~1.1s → **0.9~1.4 FPS**。即便剪枝到 1 GMACs 也仅 ~3 FPS，够不到 10~20 Hz。**与既定 FPGA 方向一致，软跑不再作候选。**

## 6. 推荐路线

**公共前置（第 0 步，PC 端，任何 FPGA 方案都绕不开）——导出与算子适配 spike**

已知断点，都会卡死 DPU 编译链：
1. PillarAttention：forward 内 `.item()`/逐 batch Python 循环/动态 `max_pillars`（须 static：batch=1 + padding 到 MAX_PILLARS=3096，mask 静态化）
2. PointPillarScatter：动态 advanced indexing 无标准 ONNX 对应 → 拆图（VFE+attn CPU C++ scatter）+ 后段，或 custom op
3. 全整型算子覆盖：LayerNorm/GELU/MHA 的 DPU 支持（vai_c 编译 dry-run 出 CPU 回退层清单）

产出：ONNX 导通 + 与 PyTorch allclose + 算子兼容清单。

**主线分支（两核约束下）**：

1. **留在 3EG**：功能 A BRAM 占用分析（112 的构成：FIFO 深度/小 RAM→LUTRAM/位宽/时分复用/双缓冲改单缓冲）→ 降到 <80 → 释放 ≥30 BRAM → **B400（稳）或 B800（性能）**。功能 A 优化是硬件组活，lead time 长，**前置分析立即并行启动**，勿等任何失败后再排
2. **换 5EV 板**：BRAM 瓶颈消除（剩 176，另有 64 URAM），**DPU B1600 从容可行**、免功能 A 优化 → 见 [ZU5EV_部署评估与方案.md](ZU5EV_部署评估与方案.md)。两板取舍 = 5EV 硬件成本 vs 3EG 功能 A 改造工作量+风险

**判定标准（DPU 口径）**：
- 吞吐 ≥ 雷达帧率（10~20 Hz）
- 端到端单帧 p99 延迟 ≤ 1 帧周期（50~100 ms）；抖动用 CPU 亲和 + SCHED_FIFO 隔离
- **INT8 mAP 掉点 ≤ 1~2 点**（MSR eval 同 R40 moderate 口径对拍）。

## 7. 风险清单

- **INT8 精度**：attention+LayerNorm+GELU 对全整型量化敏感，RCS/doppler 动态范围宽；PTQ 掉点兜底 = per-channel 权重 → 敏感层保 FP16 → QAT（复用现有训练脚本，1~2 天/轮）；**无精度兜底不能验收**
- **算子适配**：§6 第 0 步三断点；DPU 侧 MHA 类算子可能回退 CPU 使加速打折——先 vai_c dry-run
- **功能 A 改造回归**：BRAM 优化涉及功能 A 硬件时序/带宽，需功能 A 单独验证回归（45° 收益与风险共担）
- **DDR/数据链路**：功能 A 持续占 PL DDR 带宽，功能 A→检测进程须零拷贝（UIO/dmabuf）
- **散热 sustained**：DPU 满载后板温/降频，压测 10 min 记录 p99
- **Vitis AI 栈版本**：DPU IP / VART / Petalinux 版本匹配是隐形工作量

## 8. 行动清单

1. 第 0 步导出 spike（PC：attention 静态化 → ONNX → allclose → vai_c 编译 dry-run）
2. **并行启动**功能 A BRAM 占用分析（决定留在 3EG 的可行性与 DPU 规格）
3. 定分支：留在 3EG → 功能 A 优化后 B400/B800 板测；换 5EV → B1600 板测
4. 每轮记录：延迟分布 + 对应 mAP + 模型/引擎版本哈希（挂 resbag 机制）