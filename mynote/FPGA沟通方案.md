# FPGA 部署沟通方案（ZU3EG / ZU5EV）

算法侧 → FPGA 工程师的沟通包。配套阅读 [ZU3EG_部署评估与方案.md](ZU3EG_部署评估与方案.md)、[ZU5EV_部署评估与方案.md](ZU5EV_部署评估与方案.md)、[部署先验.md](部署先验.md)。

两个核心需求：

1. **部署流程一键化**：从 PyTorch checkpoint 一键生成 FPGA 可执行文件
2. **任务时序**：确定 RSP → MODEL → TRACKER 的链路时序

---

## 1. 开场定调：两需求的业界说法

用这两句话开场，对方立刻知道要什么：

- 需求1 →「我要一条 **model deployment toolchain**（Vitis AI flow）：从 PyTorch checkpoint 一键编译成 DPU 可执行的 **.xmodel**」
- 需求2 →「我要定 **HW/SW partitioning**（软硬件划分）+ **dataflow pipeline** 的任务时序，签一份 **ICD**（Interface Control Document，软硬件接口契约文档）」

---

## 2. 需求1：部署流程一键化

### 2.1 标准链路（五步，前三步 PC 端 Docker 内完成，可全脚本化）

1. **静态化导出**
   - 模型包成 forward-only module，`torch.jit.trace` 通过，batch=1 固定 shape（本模型口径：pillars padding 到 3096×16）
   - **`vai_q_pytorch` 直接吃 .pth，不需要手动导 ONNX**（ONNX 仅为备选路径 vai_q_onnx）
2. **算子覆盖检查**
   - `vai_q_pytorch --inspect --target DPUCZDX8G_ISA1_B1600`
   - 产出 inspect 报告，列出哪些算子会掉到 CPU
   - 术语：**operator coverage / graph partitioning**（DPU subgraph + CPU subgraph 自动切分）
3. **量化（quantization）**
   - **PTQ**（post-training quantization）：`--quant_mode calib`，校准集 100~1000 帧，无需标注
   - host 端 `--quant_mode test` 出 INT8 mAP = 验收门槛
   - 掉点兜底链：`--fast_finetune` → **QAT** → per-layer mixed precision
4. **编译**
   - `vai_c_xir -x <model>_int.xmodel -a arch.json` → **.xmodel**（含 DPU/CPU 子图切分）
   - **arch.json 由对方 Vivado TRD 构建产出——唯一必须问对方要的输入**
5. **板端运行**
   - **VART**（Vitis AI Runtime）C++ API：`execute_async` / `wait`
   - 注意：Vitis AI Library 在 5.0 已停更，板端直接写 VART，不指望高层库

### 2.2 分工边界（当面讲清）

**算法侧（我）**：

- 静态化 wrapper、量化脚本 + 校准集
- INT8 精度对拍（host 端 test 口径）
- decode/NMS 的 CPU 侧 C++ 实现

**FPGA 侧（你）**：

- DPU IP 集成（Vivado **TRD**，`trd_prj.tcl` 配 DPU_ARCH / DPU_NUM）
- **arch.json** 交付
- PetaLinux + VART 移植
- **CMA**（连续物理内存）预算

### 2.3 "一键"的落点

- PC 端：Docker 镜像 `xilinx/vitis-ai-pytorch` 里 `make model` 串起 quant + compile，arch.json 作唯一外部输入
- 板端：`make run` 加载 .xmodel 跑 VART

### 2.4 验收口径（写进会议纪要）

- INT8 mAP 掉点 ≤ 1~2 pt（MSR eval 同口径）
- golden 张量对拍 allclose
- 板端 p99 latency，pre + infer + post 全计入

---

## 3. 需求2：任务时序 RSP → MODEL → TRACKER

RSP 按既有理解 = 功能 A（PL 上既有雷达信号处理）。

### 3.1 白板拓扑

```
RSP(PL) --AXI DMA/零拷贝--> 点云帧缓冲(DDR, ping-pong)
  → [A53] 前处理(pillar化+VFE+attn+scatter)
  → [DPU] CNN 推理(VART, .xmodel)
  → [A53] decode+NMS → tracker(C++) → 航迹输出
```

关键架构事实：DPU 是**访存型**加速器，输入必须经 DDR，所以 RSP→DPU 以共享 DDR 帧缓冲为界，用 **zero-copy（UIO/dmabuf）+ ping-pong double buffering**。

### 3.2 时序三句话（跟对方对齐的数学）

- 流水化后 **throughput = 1/max(级延迟)**（串行才是 1/Σ）——10 Hz 帧周期下，最慢一级 ≤ 100 ms
- **端到端 latency = Σ各级 + 排队**——验收用 **p99**，不用均值
- **frame ID/timestamp sideband** 全程随数据走（PL 侧 TUSER 侧带或 DMA 描述符字段），tracker 按帧号对齐——delay=1 帧 ego 对齐契约写进 ICD

### 3.3 ICD 必备段落（让对方出模板，算法侧填内容）

- **RSP→前处理**：点云帧格式（N×6 float：x, y, z, dop_x_gnd, dop_y_gnd, rcs）、缓冲物理地址/槽位、帧号位置、完成通知方式（interrupt，不用 polling）
- **模型输出→tracker**：检测列表 N×[x,y,z,l,w,h,heading,score,cls]；**阈值口径统一 0.2**（模型内已裁 0.2，tracker 配的 0.1 实际不生效，契约不留双阈值歧义）
- **时延预算表**：逐级 typ/max/p99，注明谁测量谁提供
- **丢帧策略**：有界队列满则丢旧保新（雷达要最新量测）+ 丢帧计数上报
- **A53 两核分配**：核1 前处理 + DPU 喂数，核2 decode+NMS+tracker；**isolcpus + SCHED_FIFO** 防调度抖动；RPU/R5 跑硬实时是后手，首版不提

---

## 4. 术语速查（开会前过一遍）

### 4.1 工具链

- **Vitis AI**：Xilinx 官方 DNN 部署工具链总称
- **vai_q_pytorch**：PyTorch 量化器（inspect / calib / test / deploy）
- **PTQ / QAT**：训练后量化 / 量化感知训练
- **calibration**：量化校准（100~1000 帧，无需标注）
- **INT8**：DPU 主精度
- **vai_c_xir**：DPU 编译器，产出 .xmodel
- **.xmodel**：DPU 可执行模型文件（含 DPU/CPU 子图切分）
- **arch.json**：目标 DPU 硬件配置描述（编译输入）
- **VART**：板端运行时（C++ API，execute_async/wait）
- **operator coverage / graph partitioning**：算子覆盖 / DPU-CPU 子图切分
- **jit trace**：torch.jit.trace，量化前置要求
- **TRD**：Vivado 参考设计（DPU 硬件集成脚本）
- **PetaLinux**：板端嵌入式 Linux
- **CMA**：Linux 连续物理内存区，DPU 部署硬需求
- **Docker `xilinx/vitis-ai-pytorch`**：官方量化编译环境镜像

### 4.2 芯片架构

- **PS / PL**：处理器侧（A53/RPU/外设）/ FPGA 逻辑侧
- **APU / RPU**：4× Cortex-A53 应用核簇 / 2× Cortex-R5 实时核
- **DPU（DPUCZDX8G）**：Zynq US+ 专用 CNN 推理 IP（微码处理器，换网络不重编硬件）
- **AXI DMA / AXI4-Stream**：内存↔流 桥 / 无地址流协议
- **BRAM / URAM**：片上存储资源（DPU 档位的主要瓶颈项）
- **bitstream**：FPGA 位流
- **Vivado / 综合（synthesis）**：硬件构建工具/流程

### 4.3 时序与接口

- **pipeline / dataflow**：帧级流水（不同级并行处理不同帧）
- **latency budget**：时延预算表
- **throughput vs latency**：吞吐 = 1/max(级)；端到端延迟 = Σ级 + 排队
- **p99 / WCET**：尾延迟 / 最坏执行时间
- **ping-pong（double buffering）**：双缓冲，传输与计算重叠
- **zero-copy（UIO/dmabuf）**：跨进程/软硬件零拷贝共享内存
- **interrupt vs polling**：中断驱动 vs 轮询
- **backpressure**：背压（满则丢旧保新）
- **frame ID sideband**：帧号/时间戳随数据同行，防跨级错位
- **ICD**：软硬件接口契约文档
- **NEON**：A53 SIMD 指令集（后处理/tracker C++ 优化用）

---

## 5. 算法侧带去的材料

- **model card 一页**（repo 口径，详见附录 A）
- **前后处理清单**：pillar 化 / decode / NMS 由算法侧 C++ 化（CPU 侧），CNN 段进 DPU
- **校准集**：100~1000 帧代表性 MSR 序列（无需标注）
- **golden 对拍数据**：固定输入 + FP32 输出
- **PC 端逐段计时数据**（P5 时延预算的现有测量）
- **tracker Python→C++ 移植**：算法侧工作量，主动认领

---

## 6. 当面问对方的六个问题

1. PL 剩余资源与 DPU 档位何时定？arch.json 什么时候给？Vivado 流还是 Vitis/xclbin 流？
2. 版本矩阵冻结：Vitis AI / DPU IP / VART / PetaLinux 配对版本（DPUCZDX8G IP 已 maturity 停更于 4.1，2023 后无 TRD/预编译镜像更新——立项即冻结）
3. RSP 输出点云怎么给：DDR 缓冲地址约定？zero-copy（UIO/dmabuf）可行性？帧号怎么带？
4. A53 两核怎么分？允许 isolcpus / RT 优先级吗？
5. CMA 预算多大？（`xdputil xmodel -l` + 官方公式核算）
6. 板端 benchmark 口径与 sustained 压测（10 min 记 p99）

---

## 7. 已知风险与坑（提前打预防针）

- **动态点云断点**：PillarAttention 的 `.item()`/动态 max_pillars 须静态化；PointPillarScatter 动态索引无标准算子对应（拆图走 CPU 或 custom op）——第 0 步导出 spike 先行
- **算子回退**：LayerNorm/GELU/MHA 可能掉 CPU，先 `--inspect` + `vai_c` dry-run 出回退清单；回退多则加速打折
- **量化精度**：RCS/doppler 动态范围宽，attention 系对全整型敏感；无精度兜底不验收
- **输出定标（fixpos）搞反**：坐标错几个量级且不报错——golden 对拍兜底
- **预处理不一致**：归一化/布局任何一处不匹配，精度静默崩——格式契约逐字段写死
- **多核 DPU 跑不满**：受 DDR 带宽上限折减，不能按单核线性乘
- **换网络 vs 换硬件**：换模型只换 .xmodel；换 DPU 档位/参数才动硬件，且模型须重编译
- **散热 sustained**：DPU 满载压测 10 min 记录 p99

---

## 8. 内部勘误（对既有笔记，会后跟对方核实）

- **B400 不存在**：官方 PG338 档位仅 B512/B800/B1024/B1152/B1600/B2304/B3136/B4096；ZU3EG 笔记中 "B400" 应为 B512 之误
- **官方资源数比笔记估算高**（PG338 单核口径）：
  - B512 ≈ 26.9K LUT / 72 BRAM36 / 118 DSP
  - B800 ≈ 29.7K LUT / 90 BRAM36 / 166 DSP
  - B1600 ≈ 38.4K LUT / 126 BRAM36 / 326 DSP
  - B2304 ≈ 42.1K LUT / 165 BRAM36 / 438 DSP（URAM 模式下 BRAM→0，占 60 URAM）
- **推论**：3EG 功能 A 占用后（剩 ~20K LUT / 32 BRAM）任何 DPU 档位都放不下，功能 A 的 LUT+BRAM 双释放是硬前提；5EV B1600 从容成立，B2304（URAM 模式 60/64）也可行——笔记中 "B2304 DSP 逼近上限" 不成立（438 << 剩 1050），真约束是 URAM 紧贴
- **导出环境限制**：vai_q_pytorch 支持 PyTorch ≤2.0 / Python ≤3.9（Docker env 为 1.13），导出环境与训练环境解耦

---

## 9. 参考来源

- [Vitis AI MPSoC quickstart](https://xilinx.github.io/Vitis-AI/5.0/html/docs/quickstart/mpsoc.html)（端到端流程：docker / quant / vai_c_xir / batch=1 / 校准集规模）
- [PG338: DPUCZDX8G Product Guide](https://docs.amd.com/r/en-US/pg338-dpu)（DPU 档位、资源表、官方口径）
- [vai_q_pytorch README](https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch)（量化 API、jit trace 要求、fast_finetune/QAT）
- [UG1414: Vitis AI User Guide](https://docs.amd.com/r/en-US/ug1414-vitis-ai)（VART 编程模型）
- [UG1393: Vitis Application Acceleration](https://docs.amd.com/r/en-US/ug1393-vitis-application-acceleration)（dataflow / double buffering / burst）
- [PG021: AXI DMA](https://docs.amd.com/r/en-US/pg021_axi_dma)（DMA 模式与中断）
- [UG1186: OpenAMP](https://docs.amd.com/r/en-US/ug1186-zynq-openamp-gsg)（R5/RPU AMP 备选）

---

## 附录 A：模型接口事实（repo 口径，2026-09 核对）

**输入**

- 每点 6 维特征：`[x, y, z, dop_x_gnd, dop_y_gnd, rcs]`（无归一化；dop 地面系补偿依赖 ego_speed/yaw_rate，属输入契约一部分）
- pillar 化：voxel 1.0×0.5 m，range [0,-20,-10, 200,20,10] → BEV 200×80
- 静态上限：3096 pillar × 16 点，batch=1

**输出**

- head：AnchorHeadSingle，4 类（轿车/行人/二轮车/卡车），32000 anchor
- 原始张量：cls (32000,4) + box (32000,7) + dir (32000,2)
- decode 链：ResidualCoder 解码 → 方向分类修正 → sigmoid → 多类 BEV 旋转 NMS（IoU 0.1）
- 最终每检测：`[x,y,z,l,w,h,heading] + score + label`；score 阈值 0.2（模型内已裁）
- 无速度回归，tracker 自产速度（history 差分 + α-β 平滑）

**时序假设**

- 帧周期 100 ms（10 Hz）；delay=1 帧 ego 量测滞后点云（100 ms，设计口径）
- 端到端须单帧内完成：吞吐 ≥ 雷达帧率，p99 ≤ 1 帧周期
