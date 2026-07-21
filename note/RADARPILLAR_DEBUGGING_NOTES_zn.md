# RadarPillar + nuScenes 调试经验总览

> **适用对象**:在 RadarPillar 项目上做 nuScenes radar 检测训练/调试的同学。
> **文档定位**:**不是训练手册**(看 `NUSCENES_TRAINING_GUIDE_zn.md`),而是 **"调试工程链路时怎么思考、按什么顺序排查"** 的经验沉淀。
> **核心结论**:这类问题的本质不是"RadarPillar 模型不行",而是 nuScenes radar 数据适配、坐标变换、训练步数、后处理阈值共同构成的工程链路问题;**修复和验证必须按 pipeline 一层层做**。

---

## 一句话总结

> 这次问题不是单点 bug,而是一条完整链路问题。
> **环境 → 数据集 → 坐标系 → 模型 → 后处理 → 可视化 → 评估**,任何一环错了,最终都会表现成"GT 和点云不重合 / 模型不出框 / 1 batch overfit 失败 / 评估全 0"。
> **不能只看最终 mAP,也不能只看可视化,必须按链路逐层排查。**

---

## 目录

| 章 | 主题 | 你会得到什么 |
|---|---|---|
| 一 | 工程理解 | RadarPillar 的本质、nuScenes 适配最容易错的地方 |
| 二 | 环境问题 | import 阶段报错 → 优先怀疑环境,而非模型 |
| 三 | nuScenes Dataset 适配问题 | 4 个最常踩的坑:GT 坐标系 / 雷达变换 / sweep / feature 维度 |
| 四 | 1 Batch Overfit 的意义 | 为什么这是"最小闭环测试",它要证明什么 |
| 五 | Overfit 为什么一开始失败 | 3 个真实原因:epochs 不等于 step、阈值挡住了低分预测、官方 eval 不适合 debug |
| 六 | 为什么有时会输出 500 个框 | 不是没 NMS,而是后处理参数太宽 |
| 七 | Overfit 后图上为什么不完美 | 区分 raw GT / train GT;radar 稀疏 |
| 八 | 推荐的调试顺序 | **8 步逐层验证流程**(可直接照搬) |
| 九 | 推荐使用的命令 | 4 套可直接复用的命令 |
| 十 | 最终经验 | 10 条压缩成几句话的核心经验 |

---

## 一、工程理解

### 1.1 RadarPillar 的本质

RadarPillar 是基于 **OpenPCDet 风格**改出的 3D 检测工程。主要链路:

```
nuScenes radar 数据
  → dataset 读取和坐标转换
  → point feature 编码
  → pillarization
  → PillarVFE / PillarAttention
  → BEV backbone
  → anchor dense head
  → NMS 后处理
  → evaluation / visualization
```

主要变化是 **适配 nuScenes radar 数据结构**,模型主体仍是 PointPillar/RadarPillar 风格,**不是重新发明一个完全不同的检测框架**。

### 1.2 nuScenes 适配不是"简单读文件"

真正的难点集中在 dataset 这一层。**最容易错的地方**:

| # | 易错点 | 错的后果 |
|---|---|---|
| 1 | radar 点云坐标系 | 点云投到错误位置 |
| 2 | GT box 坐标系(可能默认是 global) | 点云和 GT 不重合 |
| 3 | ego / global / sensor 之间的变换 | 训练目标在错误空间 |
| 4 | 时间 sweep 的变换 | 点云拖影、目标位置不稳 |
| 5 | velocity 的坐标变换 | 速度 head 学不到正确目标 |
| 6 | 训练时过滤规则(range / class) | GT 被过滤光,loss 异常 |
| 7 | 评估时 split 是否匹配 | 报 `Samples in split doesn't match` |

**任一错一个,模型训练就会"看起来能跑",但学不到正确目标。**

---

## 二、环境问题

### 2.1 一个典型的非代码 bug

报错:
```
ValueError: All ufuncs must have type `numpy.ufunc`
```

**这不是 RadarPillar 代码逻辑问题**,而是 **numpy / scipy 版本冲突**。
判断依据:**程序还没进入训练,导入 `scipy.spatial` 就崩了**。

### 2.2 经验法则

| 报错阶段 | 优先怀疑 |
|---|---|
| **import 阶段** | 环境(numpy / scipy / torch / spconv / CUDA) |
| **forward / loss / eval 阶段** | 模型或数据 |

### 2.3 当前验证过的环境状态

```
conda env:    angle
numpy:        1.26.4
scipy:        1.15.3
scipy.spatial: ok
```

### 2.4 启动训练时进入环境的正确姿势

```bash
cd /home/dministrator1/RadarPillar
. /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
```

> 不要只依赖 `conda activate angle`,因为 **非交互 shell 里可能找不到 conda**。

---

## 三、nuScenes Dataset 适配问题

### 3.1 GT 坐标系问题(最关键)

nuScenes 的标注 box 默认可能处于 **global 坐标系**。
但模型训练需要的是 **ego / lidar / radar 对齐后的本车坐标系**。

**如果 GT 还在 global、而 radar 点云在 ego**,会出现:

- 点云和 GT 完全不重合
- 训练时 GT 被 range filter 过滤掉
- loss 异常
- overfit 不出目标
- 可视化看起来很离谱

> **经验**:只要点云和 GT 不重合,**第一优先级不是调模型,而是查坐标系**。

### 3.2 Radar 点云变换问题

之前发现 radar 的变换矩阵 **命名和实际含义不一致**:

| 变量名像 | 实际可能是 |
|---|---|
| `sensor -> ego` | `ego -> sensor` |

这会导致点云投到错误位置。

> **经验**:不要只相信变量名,**要用数值验证**。
> 例如 radar 前雷达的安装高度应该是正的,如果变换后 z 方向明显不合理,说明矩阵方向可能反了。

### 3.3 Sweep 变换问题

多帧 radar sweep **不能直接拼接**。每个 sweep 都要变换到当前帧坐标系。

如果 sweep 没有正确 transform,会造成:
- 点云拖影
- 目标位置不稳定
- 模型学到错误空间关系

### 3.4 Feature 维度问题

radar raw point feature 当前是 **7 维**:`x, y, z, rcs, vx, vy, time`。

训练 batch 中看到 `(N, 8)` 是正常的,因为 OpenPCDet 会额外加 **batch index**。

> **经验**:
> - raw point feature 是 **7 维**
> - batch 里的 points 是 **8 维**
> - **不要把 batch index 当成 radar feature**

---

## 四、1 Batch Overfit 的意义

### 4.1 它不是"测泛化"

**1 batch overfit 不是为了得到好泛化结果,而是为了验证工程链路是否闭环。**

### 4.2 它要证明

| # | 验证项 | 失败说明 |
|---|---|---|
| 1 | 这个样本能被读到 | dataset 路径/读取错 |
| 2 | GT 没有被过滤光 | range filter / class 错 |
| 3 | 模型能收到梯度 | 训练图(如 ddp / no_grad)错 |
| 4 | loss 能下降 | 数据 / label 错 |
| 5 | 分类分数能升高 | loss 设计错 |
| 6 | NMS 后能输出预测框 | 后处理 / anchor 错 |
| 7 | 预测框能靠近 GT | 回归头没学到 |

### 4.3 最小闭环测试

如果 1 batch 都 overfit 不起来,说明至少有一个基础环节有问题:

- 数据错
- 坐标错
- GT 错
- loss 错
- anchor 不匹配
- 训练没有真的发生
- 后处理阈值不合理

> **结论**:它是深度学习工程里的 **"最小闭环测试"**,通过 ≠ 模型好,不通过 = 工程链路有断点。

---

## 五、这次 Overfit 为什么一开始失败

### 5.1 `--epochs 100` ≠ 1200 次参数更新

对于 1 个样本:

```
1 epoch  ≈  1 step
100 epochs  ≈  100 次参数更新
```

而这次真正跑出效果的是 **1200 次参数更新**。

所以:

```bash
python tools/train.py --epochs 100    # 1 batch 场景下 = 100 step,太少
```

> 这就是为什么 `checkpoint_epoch_100` 不出目标。

### 5.2 后处理阈值挡住了低分预测

诊断结果:

```
raw_cls_prob_max = 0.0119
SCORE_THRESH     = 0.05
post_nms_boxes   = 0
```

**模型不是完全没有响应,而是分类分数太低,被阈值过滤掉了**。

#### 调试经验

| 现象 | 不要直接说 | 先查 |
|---|---|---|
| 没有预测框 | "模型没学到" | raw 分类分数是否存在 |
| 有 raw score 但低于阈值 | — | 训练不足 / 阈值设置问题 |
| raw score 完全异常 | — | 模型 / 数据问题 |

### 5.3 官方 eval 不适合直接判断 overfit1

overfit1 只预测 1 个样本,但 **nuScenes 官方 eval 期望完整 split**,所以可能报:

```
Samples in split doesn't match samples in predictions
```

**这不是模型训练失败**,而是 **debug dataset 和 official eval split 不匹配**。

#### overfit 阶段应该优先看

1. loss
2. raw score
3. `post_nms_boxes`
4. recall
5. 可视化

> **不要优先看官方 mAP / NDS。**

---

## 六、为什么有时会输出 500 个框

### 6.1 真相

**这不是没有 NMS,而是后处理参数太宽松**。当:

```
SCORE_THRESH      = 很低
NMS_POST_MAXSIZE  = 500
```

大量低质量候选框会通过阈值,NMS 最后保留到上限。

### 6.2 看到 500 个框时意味着

```
num_pred  = 500
score     = 很低甚至 raw score 为负
```

**不代表模型效果好,只能说明**:
- 模型开始有输出
- 但后处理太宽,质量还没控制住

### 6.3 推荐的调试设置

| 阶段 | SCORE_THRESH | NMS_POST_MAXSIZE | OUTPUT_RAW_SCORE |
|---|---|---|---|
| 看有没有输出(低阈值) | 0.05 或 0.1 | 80 | True |
| 看质量(正常调试) | 0.1 ~ 0.3 | 40 ~ 80 | False |
| 看真实表现(评估前) | 0.3 ~ 0.5 | 20 ~ 40 | False |

---

## 七、为什么 Overfit 后图上还不完美

### 7.1 现象

观察还有虚警和漏检,**这是合理的**,原因如下。

### 7.2 原因

| # | 原因 | 影响 |
|---|---|---|
| 1 | `raw_gt` 有 66 个,训练实际有效 gt 只有 51 个 | 没被训练的 GT 出现在图上 |
| 2 | radar 点云非常稀疏,很多 GT box 里没足够点支撑 | 模型看不到目标 |

### 7.3 1 batch overfit 的合理目标

> **不是"图上完全一模一样",而是**:

| 指标 | 期望 |
|---|---|
| loss | 明显下降(例 9.54 → 0.075) |
| 训练 GT recall | 接近满(`51 / 51`) |
| 预测框主体 | 贴近 GT |
| 预测分数 | 明显升高 |
| 虚警 | 可通过阈值/NMS 控制 |

### 7.4 这次 overfit 专用脚本结果

```
loss:            9.54  →  0.075
rcnn_0.3:        51 / 51
rcnn_0.5:        51 / 51
rcnn_0.7:        51 / 51
NMS 后预测:      80
top score:       约 0.96
```

**说明训练链路已经能闭环**,可视化质量和后处理还可以继续收紧。

---

## 八、推荐的调试顺序

> **遇到"不出目标""GT 不重合""loss 不收敛"时,严格按这个顺序查**。

```
Step 1  环境           ← import 错就先修环境
Step 2  数据长度       ← 区分 batch_size=1 和 dataset_len=1
Step 3  GT 数量        ← GT=0 不可能学到
Step 4  点云和 GT 重合  ← nuScenes radar 最关键
Step 5  单 step loss   ← loss 正常但梯度=0 查训练图
Step 6  长一点 overfit  ← 至少 800 ~ 1200 step
Step 7  后处理         ← 没预测框看 raw score;预测太多收紧阈值
Step 8  可视化         ← 区分 raw GT / train GT
```

### Step 1:环境

确认:
- `conda angle` 是否激活
- `torch` / `spconv` 是否能导入
- `numpy` / `scipy` 是否正常
- CUDA 是否可用

> import 就报错 → 先修环境。

### Step 2:数据长度

确认:
- `dataset_len` 是否符合预期
- overfit 是否真的是 1(普通训练是否不是 1)

**重点区分**:

| 看起来像 | 实际是 |
|---|---|
| `batch_size = 1` | dataset 仍有 N 个样本,每个 epoch 跑 N 个 step |
| `dataset_len = 1` | 真的只 1 个样本,每个 epoch = 1 step |

> **这两个完全不是一回事。**

### Step 3:GT 数量

确认:
- `raw_gt` 有多少
- 训练后 `valid_gt` 有多少
- 是否被 range filter 过滤光
- 类别 id 是否正确

> **GT = 0,模型不可能学到目标。**

### Step 4:点云和 GT 是否重合(★ nuScenes radar 最关键)

确认:
- 点云是否在合理范围
- GT 是否在点云附近
- x / y / z 方向是否合理
- 坐标系是否 ego 对齐

> 不重合 → 先修 dataset,不要调模型。

### Step 5:单 step loss 和梯度

确认:
- loss 是否为正常正数
- 梯度是否非零
- 参数是否更新

| 现象 | 优先查 |
|---|---|
| loss 有但梯度没有 | 训练图(是否 `no_grad` / DDP wrap 错) |
| loss 异常大或 NaN | 数据和 label |

### Step 6:长一点 overfit

不要只跑 100 step。建议:

| 场景 | 建议 iters |
|---|---|
| 最小验证 | 800 ~ 1200 |
| 标准 overfit | 1200 ~ 2000 |
| 充分 overfit | 3000 |

观察:
- loss 是否下降
- cls loss 是否下降
- loc loss 是否下降
- raw score 是否升高

### Step 7:后处理

确认:
- `SCORE_THRESH`
- `OUTPUT_RAW_SCORE`
- `NMS_THRESH`
- `NMS_POST_MAXSIZE`

| 现象 | 调法 |
|---|---|
| 没预测框 | 先看 raw score,降低阈值 |
| 预测框太多 | 提高 score threshold,降低 `post max size` |

### Step 8:可视化

> 最终才看图。图要区分:

| 颜色 / 类型 | 含义 |
|---|---|
| **raw GT** | 未参与训练的 GT(图上出现不算漏检) |
| **train GT** | 实际参与 loss 的 GT |
| **pred before NMS** | 网络输出原始预测 |
| **pred after NMS** | 后处理后最终输出 |

> 否则容易把"没参与训练的 GT"误判成漏检。

---

## 九、推荐使用的命令

### 9.1 标准进入环境

```bash
cd /home/dministrator1/RadarPillar
. /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
```

### 9.2 推荐的 1-batch overfit 命令

```bash
CUDA_VISIBLE_DEVICES=0 python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 1200 \
  --lr 0.003 \
  --score_thresh 0.1 \
  --nms_post_maxsize 80 \
  --out_dir output/overfit1_plot
```

### 9.3 更严格可视化版本

```bash
CUDA_VISIBLE_DEVICES=0 python /mnt/c/Users/Administrator/Documents/openDet/train_and_plot_overfit_one_batch.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --iters 3000 \
  --lr 0.002 \
  --score_thresh 0.3 \
  --nms_post_maxsize 40 \
  --out_dir output/overfit1_plot_tighter
```

### 9.4 如果坚持用 tools/train.py

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml \
  --batch_size 1 \
  --workers 0 \
  --epochs 1200 \
  --extra_tag overfit1_1200
```

> **不推荐**用它做 overfit 判断,**官方 eval 会受 split mismatch 干扰**。

### 9.5 命令选择建议

| 目的 | 推荐命令 |
|---|---|
| 验证工程链路闭环 | 9.2(专用 overfit 脚本,看 loss / raw score / 框数) |
| 收紧后处理看质量 | 9.3(高阈值 + 少框) |
| 与他人对比或回归 | 9.4(走标准 train.py) |

---

## 十、最终经验

> 把这次最重要的经验压缩成 10 条:

1. **不要一上来调模型**,先确认数据和坐标。
2. nuScenes 的 **global / ego / sensor 坐标是最容易错的地方**。
3. `batch_size = 1` **不等于** 1 batch overfit;`dataset_len = 1` 才是。
4. 1 epoch **不等于**充分训练;1 样本下 100 epoch 只有约 100 step。
5. 没有预测框时,先看 **raw score**,再看阈值。
6. 预测框很多时,**不是没 NMS**,而是阈值和 post max size 太宽。
7. **官方 eval 不适合直接判断** debug overfit。
8. 可视化必须区分 **raw GT** 和 **train GT**。
9. 1-batch overfit 是 **验证工程闭环**,**不是评价泛化性能**。
10. 工程调试要 **按链路逐层验证**,**不要凭最终结果猜原因**。

### 一句话总结(再次强调)

> 这次问题的本质不是"RadarPillar 模型不行",而是 **nuScenes radar 数据适配、坐标变换、训练步数、后处理阈值**共同造成的工程链路问题;**修复和验证都必须按 pipeline 一层层做**。

---

## 附录:本项目内可对照的产物

| 文档 / 文件 | 角色 |
|---|---|
| `docs/NUSCENES_TRAINING_GUIDE_zn.md` | 训练手册(how-to) |
| `docs/RADARPILLAR_DEBUGGING_NOTES_zn.md` | 本文档(why & debug 经验) |
| `tools/cfgs/nuscenes_models/radarpillar_nuscenes_overfit1.yaml` | 1-batch overfit 配置 |
| `output/overfit1_plot_recheck/` | 本次 overfit 验证产物(losses.npy / .png / .pth) |

> 两者互补:训练手册教你"怎么跑",本文档教你"出问题怎么想"。