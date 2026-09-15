# FPGA 部署三断点详解（写给不熟悉部署的读者）

> 配套 `mynote/ZU3EG_部署评估与方案.md` §6 第 0 步。所有代码行号以当前仓库为准。

---

## 0. 先回答一个疑问：MAX_PILLARS 不是已经固定了吗？

**结论：配置里确实有固定值 3000，但它只是"每帧最多保留多少 pillar"的上限，不是 padding 长度。整条数据链没有任何一层把 tensor 补齐到固定长度。**

事实链（三步，每步都有代码依据）：

1. **配置层**：`tools/cfgs/dataset/msr_dataset.yaml:75` 设 `MAX_NUMBER_OF_VOXELS: 3000`。含义是"这一帧点云最多取前 3000 个非空 pillar，超了就截断"。**真实 pillar 数 ≤ 3000，且每帧不同**（这帧 800、下帧 1500）。
2. **组 batch 层**：`pcdet/datasets/dataset.py:159` 的 `collate_batch` 对 `voxels` 执行 `np.concatenate(val, axis=0)`——**按真实数量拼接，不做任何 padding**。
3. **VFE 层**：`pcdet/models/backbones_3d/vfe/pillar_vfe.py:293` 输出 `pillar_features`，形状 `(真实pillar数, C)`——**第一维每帧都在变**。

所以 PillarAttention 拿到的输入本身就是**动态长度**的；而它的 forward 内部又自己制造了更多动态性（见 §2）。

> 备注：ZU3EG 笔记里写的 "MAX_PILLARS=3096" 应更正为 3000（按当前 msr 配置）。

---

## 1. 预备知识：部署链路与"静态图"是什么

### 1.1 部署到 DPU 的完整链路

```
PyTorch 模型 → ① 导出 ONNX → ② 量化 INT8 (vai_q) → ③ 编译 (vai_c) → DPU 指令 → 板上运行
```

- **ONNX**：通用的模型中间格式，本质是一张"计算图"——一串标准算子（conv、matmul、relu…）节点，tensor 在节点间流动。
- **DPU**：Xilinx 预制的深度学习加速器 IP（一块设计好的电路），只执行编译后的整型指令。
- 每一步的产物都是下一步的输入，**第一步走不通，后面全部无从谈起**。

### 1.2 什么是"静态计算图"

训练时 PyTorch 是 **eager 模式**：Python 逐行执行，每帧的循环次数、tensor 形状可以随便变——这是它的灵活性。

部署编译器的要求正好相反：它需要**在编译期（PC 上）就确定整张图**——

- 所有 tensor 的**形状固定**（每一帧都一样）
- **没有依赖输入数据的控制流**（`for`/`if` 的次数不能等运行时看数据才知道）
- 每个操作都能对应到 **ONNX 标准算子**

这叫静态图。"静态"指的是**图的结构**编译期定死；数据本身（数值）当然每帧在变。

### 1.3 导出（tracing）是怎么工作的

`torch.onnx.export` 用 **tracing（跟踪）** 法：喂一个样例输入，跑一遍 forward，把执行到的每个算子按顺序录下来，串成图。

关键推论：**录下的只是"这一次执行走过的路径"**。如果代码里有 `.item()` 或数据决定的循环，那这次录到的具体数值/次数就被**烤死在图里**，换一帧输入图就错了——甚至导出时直接报错。

### 1.4 三条铁律（后面三个断点各撞一条）

1. 形状必须固定
2. 控制流不能依赖数据值
3. 算子必须是标准算子、且 DPU 支持

---

## 2. 断点 1：PillarAttention 的动态性（撞铁律 1 + 2）

代码：`pcdet/models/backbones_3d/pillar_attention.py:40-87`

### 2.1 三个毒点逐一指认

**毒点 A —— `.item()`（行 44、50）**

```python
batch_size = coords[:, 0].max().int().item() + 1        # 行 44
pillar_counts.append((coords[:, 0] == b).sum().item())  # 行 50
```

`.item()` 把 tensor 拉回 CPU 变成 **Python 整数**。后果：
- tracing 时这个数被烤死成常量（这帧 batch=4，图里就永远是 4）
- 它参与决定循环次数和 tensor 形状 → "图的结构由数据决定" → 静态图崩溃

**毒点 B —— 逐 batch 的 Python for 循环（行 49、63、83）**

```python
for b in range(batch_size):          # 行 49、63：次数 = .item() 的结果
for b in range(batch_size):          # 行 83：同上
    updated_features.append(x[b, :pillar_counts[b]])   # 切片长度每帧不同
```

ONNX 图里循环必须展开成固定的算子序列，**次数必须编译期已知**。这里的次数来自数据 → 违规。

**毒点 C —— 动态 max_pillars（行 52-55）**

```python
max_pillars = max(pillar_counts)    # 每帧真实 pillar 数的最大值，数据决定
padded_features = torch.zeros((batch_size, max_pillars, C))
```

即使输入形状固定，这里也**自己造出一个每帧不同的中间形状** `(B, max_pillars, C)` → 违反铁律 1。

### 2.2 解法：batch=1 + padding 到 3000 + mask 输入化

- **batch=1**（消 A、B）：上板推理单帧单帧跑，batch_size 恒为 1 → 所有 `for b in range(batch_size)` 循环天然消失，`.item()` 无存在必要
- **padding 到固定 3000**（消 C）：不管真实多少 pillar，一律补零到 `(1, 3000, C)`，形状恒定；白算一些空位，但 32 通道很便宜
- **mask 作为输入张量**（消 B 残余）：见下

**mask 静态化的具体含义**：attention 需要 `key_padding_mask`（形状 `(1, 3000)`，True=空位要忽略）。它的**形状**固定没问题，但**值**每帧不同（前 num_p 个 False，其余 True）。
合法做法：mask 在**图外**由 CPU 预处理生成，作为图的**输入**喂进来——形状不变、值每帧变，完全合法（输入本来就是每帧变的）。
非法做法：在图内用 `.item()`/动态循环现算 mask。

一句话总结改造思想：**把"数据决定结构"改成"数据只决定数值"**。

### 2.3 改造后的额外注意点

- padding 行的 `coords` 内容要处理（collate 的 padding/截断语义在导出链路里要保证确定性）
- 3000 是上限，若 p99 真实 pillar 数远小于 3000，可实测后 pad 到更紧的值省算力（如 2000），代价是极端帧截断
- 改完必须与原实现 **allclose 对拍**（同一帧输入，两边输出逐元素比对）——改造容易改错语义

---

## 3. 断点 2：PointPillarScatter 的动态索引（撞铁律 2 + 3）

代码：`pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py:27-31`

### 3.1 这层在干什么

把稀疏的 pillar 特征"摊"到稠密 BEV 网格画布上：

```
输入：pillar_features (N, C)   N = 真实 pillar 数，每帧不同
      coords (N, 4)            每个 pillar 落在网格的哪个格子
输出：spatial_features (B, C, 200, 80)   稠密画布，空格子为零
```

类比：N 个棋子，按坐标摆到一张固定的 200×80 棋盘上，没棋子的格子置零。

### 3.2 两个问题

```python
indices = this_coords[:,1] + this_coords[:,2]*self.nx + this_coords[:,3]  # 行 27：运行时算出的坐标
spatial_feature[:, indices] = pillars                                     # 行 31：按索引填值
```

1. **`tensor[:, indices] = value` 是 in-place 动态赋值**：ONNX 是函数式语义（每个节点是纯计算、输出新 tensor），这种"按可变长索引原地散射"没有标准算子能可靠表达（有 ScatterND 但对这种写法 tracing 导出极易出错/直接不支持）
2. **indices 的长度 = N，每帧不同** → 输入形状不固定，又撞铁律 1

### 3.3 解法：拆图（推荐）或 custom op

**拆图**：把模型从 scatter 处切成两段，各走各的硬件——

- **前段**（PillarVFE + PillarAttention + Scatter）：跑 **CPU C++**。理由：这三段本来就是"逐 pillar"的稀疏计算，DPU 不擅长；scatter 在 C++ 里就是 `canvas[y][x] = feat[i]` 的内存写入循环，没有任何数学难度
- **后段**（BaseBEVBackbone 的 2D 卷积 + CenterHead，占计算量绝大头）：单独导 ONNX → 进 DPU

两段之间以 `(1, C, 200, 80)` 的稠密 tensor 交接——形状固定，接口干净。

**custom op**（备选）：scatter 留在图里但标记为自定义算子，编译时挂自己写的 CPU 实现。可行但工具链版本敏感、维护成本高于拆图。

---

## 4. 断点 3：LayerNorm / GELU / MHA 的整型算子覆盖（撞铁律 3）

### 4.1 概念：DPU 只会"积木盒里有的"积木

DPU 是**纯 INT8 整数运算**的电路，内部预制了一组算子：conv、ReLU、element-wise 加减乘、concat、max-pool 等。**电路里没有的算子，永远跑不了**——不是慢，是没有。

量化成 INT8 ≠ 只把 float 换成 int：**每一层必须能映射到这组预制算子**。映射不上的层，vai_c 编译时标记为 **CPU 回退（CPU fallback）**——运行时数据从 DDR 搬进 CPU、用浮点算完、再搬回 DPU 继续。每多一层回退 = 多一次来回搬运 + 一段单核 CPU 浮点计算，加速比被一口口吃掉。

### 4.2 三个嫌疑犯各自为什么难

- **LayerNorm**（`pillar_attention.py:29,37`）：每个 token 沿通道"减均值、除标准差、再仿射"。需要沿轴归约统计 + 除法 + 开方，DPU 无此预制块 → 常见回退
- **GELU**（`pillar_attention.py:34`）：`x·Φ(x)`，含 `erf` 超越函数。电路只认 ReLU 这种"比较+选择"；erf 硬件没法直接算。新版工具链支持 tanh 近似，需验证
- **MHA**（`pillar_attention.py:73`）：核心是 `softmax(QKᵀ/√d)·V`——batched matmul + **softmax（指数运算）** + 每帧不同的 padding mask。三者里最可能**整体回退**

### 4.3 dry-run：先看清单再决定

**vai_c 编译 dry-run** = 不上板，PC 上把模型编译一遍，编译器输出**逐层清单**：哪些层进 DPU、哪些层回退 CPU。

为什么要先做：最坏情形是 MHA+LayerNorm+GELU 全回退——而 ZU3EG 只有**两核 A53** 可用，attention 整段回退 CPU ≈ 回到 ~1 FPS 软跑死局（见 ZU3EG §5）。清单拿到手，才知道加速比打折多少、要不要做算子替换（如 GELU→ReLU、LayerNorm 换 DPU 友好实现后重训微调）。

### 4.4 "会回退"这个断言的依据链（与边界声明）

三层依据，一层比一层接近事实：

1. **机制层（工具链的明文设计，非猜测）**：DPU 只执行电路里预制的算子清单（Vitis AI 官方 supported ops 列表，见 UG1414 / PG338）。vai_c 编译时**逐层对照清单**，映射不上的层按机制定义标记 CPU fallback——生成 CPU 浮点实现挂进运行时，数据从 DDR 搬进 CPU 算完再搬回 DPU。"回退"不是推测的行为，是编译器的明文设计。
2. **算术构成层（为什么这三者不在清单）**：本平台（ZU3EG/ZU5EV → DPUCZDX8G 系列）的预制块是 conv / eltwise 加减乘 / ReLU 系 / concat 这类"**乘加 + 比较选择**"积木。对照三者构成：
   - **LayerNorm**：per-token 沿通道 reduce（算 μ/σ）+ 除法 + 开方——通用 reduce、除法均无预制块
   - **GELU**：erf 超越函数——电路只认"比较+选择"级非线性
   - **MHA**：batched matmul + softmax（exp 指数）——exp 无电路
   - **直觉对比（同一 Norm 家族命运不同的分水岭）**：BatchNorm 能进 DPU，是因为推理期 μ/σ 是训练冻结的**常数**，可折叠成 conv 的 scale/shift；LayerNorm 的 μ/σ 每帧每 token **运行时现算**，折叠不了。分水岭 = 统计量是否运行时依赖输入。
3. **确定性边界（诚实声明）**：算子表随 Vitis AI 版本演进（新版对 transformer 类有增强）。本断言基于这代 DPUCZDX8G 的公开算子表，属**强预期而非对具体工具链版本的实证**——所以 §7 的 vai_c dry-run（逐层 DPU/CPU 归属清单）就是这个断言的**判决实验**，清单落地前不把话说死。

---

## 5. 前段不退 CPU 的替代路线（问：除了拆图还有什么路）

四条路，按可行性排序，每条给机制 / 代价 / 死穴：

1. **静态化硬塞 DPU 图**：pad 固定 N + mask 输入化消动态（§2.2）；scatter 改写成等价静态算子——one-hot matmul：`canvas = one_hot(N, 16000) @ pillars`（按实测 N≈700 量级约 360 MMAC，DPU 吃得下但纯浪费），或试 ScatterND。
   - 死穴：**绕一圈还是撞断点 3**——MHA 的 softmax、LayerNorm 的 reduce 照样没算子，静态化改造白做。
2. **算子替换 + 微调重训**：GELU→ReLU（近无损，常规操作）；LayerNorm→推理期可折叠形式（训练期就换 BN，推理期折进 conv 的 scale/shift）；MHA 是硬伤——softmax 的 exp 无电路，拆成 matmul 也进不去。
   - 边界：治得了 GELU/LN，**治不了 MHA**。
3. **蒸馏出无 attention 的部署版**：现模型当 teacher，student 去掉 PillarAttention。牺牲 AP 换全图 DPU 覆盖，常与路线 2 组合。
4. **前段走 HLS 自写 IP**：不退 CPU 也不进 DPU——VFE/scatter 这种稀疏内存操作用 HLS 写成自定义 IP 挂 AXI 总线，DPU 管后段。性能与确定性最好，工程量最大，需硬件组配合。

> 实测支撑（trainval 3000 帧，range/azi/elv 加 scale 转直角坐标、1.0×0.5 网格直算）：真实 pillar 数 **p50=356 / p99=619 / max=712**——pad 768 即可，无需 3000；同时此分布也是 §7 拆图方案 CPU 算力账的基础。

**结论**：MHA 无论怎么静态化都进不了这代 DPU → **attention 的归宿只有 CPU 或 FPGA 自定义逻辑二选一**。这是 §3.3 拆图被列为推荐路线的根本原因。

---

## 6. 三个断点各自卡在链路哪一步

- **断点 1（PillarAttention）**：卡在 ① 导出 ONNX——不改造就导不出图
- **断点 2（Scatter）**：卡在 ① 导出 ONNX——动态索引无标准算子对应
- **断点 3（整型覆盖）**：卡在 ③ vai_c 编译——能导出、能量化，但编译后可能"名为 DPU 加速、实为 CPU 主力"

## 7. spike 的执行顺序与每步验证什么

1. **attention 静态化**（消断点 1）：batch=1 + pad 3000 + mask 输入化 → 导出通过
2. **拆图导 ONNX**（消断点 2）：前段 C++ / 后段 ONNX → 两段各自导出通过
3. **allclose 对拍**（防改错）：同一帧输入，改造前后输出逐元素比对，容差内才算过
4. **vai_c dry-run**（探断点 3）：拿逐层回退清单 → 决定算子替换范围 → 才能定 3EG/5EV、B400/B1600

三个答案齐了，硬件选型决策才有依据；任何一步走不通，都是提前止损而非工程铺开后返工。
