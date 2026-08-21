# ZU5EV 部署评估与方案

与 [ZU3EG_部署评估与方案.md](ZU3EG_部署评估与方案.md) 配套阅读（模型、软跑排除结论、公共前置见其 §2/§4/§6）。关键前提：**CPU 仅两核可用、软跑路线已排除，两板共用同一模型、同一 FPGA 加速目标**。功能 A 已迁移 5EV，占用绝对量相近（~50K LUT / ~198 DSP / ~112 BRAM）。

## 1. 硬件平台

Xilinx Zynq UltraScale+ MPSoC **ZU5EV**（DS891 官方口径）：

- **APU**：4× ARM Cortex-A53（同 3EG，仅两核可用；软跑在两板均被排除）
- **RPU**：2× ARM Cortex-R5F
- **PL**：117K LUT / 234K FF / 1,248 DSP48 / **288 BRAM36** / **64 URAM（288Kb/块）**
- **VCU**：集成 H.264/H.265 编解码（本项目用不上）

与 3EG 关键差异：**BRAM 翻倍（144→288）+ 新增 64 URAM**，片上存储瓶颈消除。

## 2. 待部署模型

同 3EG §2：msr_centerhead3d_trunk，0.27M 参数 / 5.49 GFLOPs / mAP 88.13（FP32）。

## 3. 资源现状（功能 A 占用后剩余）

- LUT：50K/117K（43%），剩 ~67K
- FF：60K/234K（26%），剩 ~174K
- DSP：198/1,248（16%），剩 **1,050**
- BRAM：112/288（39%），剩 **176**（3EG 仅剩 32）
- URAM：64 块全部空闲
- 结论：**无资源瓶颈**

## 4. 候选方案可行性

- **DPU B1600**：~60 BRAM << 剩 176 → **可行且从容**；DSP/LUT 余量大，为本板主力
- **B2304/更大规格**：BRAM 仍放得下，DSP 逼近上限需详核；仅 B1600 不达标时考虑
- **B800/B400**：可行但弱，仅在需给功能 A 预留大量扩展时有意义
- **APU 软跑**：两核 ~1 FPS 量级（见 3EG §5）→ **排除**
- **自定义 HLS**：URAM 可做片上缓冲；BRAM 已富余，无必要
- *DPU 精确资源未在线核实，以 PG338 综合结果为准；176 vs ~60 余量差数倍，结论对误差不敏感*

## 5. 推荐路线

**5EV 默认 = DPU B1600 硬件加速，免功能 A 优化、免模型压缩**：

- STRENGTH：BRAM 瓶颈不复存在，INT8 算力充足，支撑 tracker 10~20 Hz 与后续扩展
- 64 URAM 空闲可作 NMS/前处理 offload 落地空间
- 与 3EG 分支取舍（见 3EG §6）：5EV 硬件成本 vs 3EG 功能 A 改造工作量+风险

**判定标准同 3EG §6**：吞吐 ≥ 雷达帧率、端到端 p99 ≤ 1 帧周期、INT8 mAP 掉点 ≤ 1~2 点。

## 6. 风险与注意

- **INT8 精度**：与 3EG 同风险——PTQ 掉点先 per-channel/混合精度，再 QAT；掉点阈值为硬门槛
- **算子适配**：公共前置（attention 静态化 + 拆图 + allclose，见 3EG §6）先行；DPU 侧先 vai_c 编译 dry-run 出 CPU 回退层清单，MHA 类算子回退会打折
- **Vitis AI 栈版本**：DPU IP / VART / Petalinux 版本匹配是隐形工作量，立项即冻结版本
- **DDR/数据链路**：功能 A→检测进程零拷贝（UIO/dmabuf），影响确定性
- **散热 sustained**：DPU 满载压测 10 min 记录 p99

## 7. 行动清单

1. 公共前置：PC 导出 spike（attention 静态化 → ONNX → allclose → vai_c dry-run）
2. Vitis AI 编译 B1600 → 板端端到端 benchmark（pre+infer+post 全计入）+ sustained 压测
3. INT8 mAP 对拍（MSR eval 同口径 vs FP32 基线）
4. B1600 不达标 → 评估 B2304（DSP 详核）或回 3EG 备选链

## 参考来源

- [DS891: Zynq UltraScale+ MPSoC Data Sheet Overview](https://docs.amd.com/v/u/en-US/ds891-zynq-ultrascale-plus-overview)（ZU5EV 资源口径）
- PG338 (DPU IP) —— B1600 精确资源上板前复核