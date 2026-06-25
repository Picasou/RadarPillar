# 多目标跟踪（MOT）评估指标详解

> 面向 2D/3D 多目标跟踪的性能评估指标体系梳理，按评估维度与演进顺序组织。
> 适用于阅读 KITTI / nuScenes / Waymo 等 3D MOT 论文时对照查阅。

---

## 〇、速览表（名字 + 含义）

| 缩写 | 全称 | 一句话含义 |
|------|------|-----------|
| **TP** | True Positive | 正确匹配上的检测数（真阳性） |
| **FP** | False Positive | 误检 / 多余检测（假阳性） |
| **FN** | False Negative | 漏检（假阴性） |
| **IDS** | ID Switch | 同一目标 ID 发生切换的次数 |
| **FRAG** | Fragmentation | 轨迹中断次数 |
| **MOTA** | Multi-Object Tracking Accuracy | 综合漏检/误检/ID切换的跟踪准确率 |
| **MOTP** | Multi-Object Tracking Precision | 匹配框的平均定位误差（定位精度） |
| **IDF1** | ID-based F1 | 基于 ID 的 F1 分数，偏关联质量 |
| **DetA** | Detection Accuracy | HOTA 中的检测准确度 |
| **AssA** | Association Accuracy | HOTA 中的关联准确度 |
| **HOTA** | Higher Order Tracking Accuracy | DetA 与 AssA 的几何平均，统一指标 |
| **LocA** | Localization Accuracy | HOTA 的定位精度 |
| **AMOTA** | Average MOTA | 对置信度阈值积分平均的 MOTA（3D 专用） |
| **AMOTP** | Average MOTP | 对阈值平均的 MOTP |
| **sAMOTA** | scaled AMOTA | nuScenes 缩放版 AMOTA |
| **VAE** | Velocity Angle Error | 速度方向（角度）误差 |
| **VNE** | Velocity Norm Error | 速度大小（幅值）误差 |
| **VAIE** | Velocity Angle Inverse Error | 速度方向反向的严重程度 |
| **VIR** | Velocity Inversion Ratio | 速度方向反向的帧数比例 |
| **VSE** | Velocity Smoothness Error | 速度曲线平滑度误差 |
| **VDE** | Velocity Delay Error | 速度信号相对真值的延迟 |

---

## 一、计数类指标（最基础，CLEAR [Bernardin 2008]）

| 指标 | 含义 | 说明 |
|------|------|------|
| **TP** (True Positive) | 正确匹配的检测数 | 越多越好 |
| **FP** (False Positive) | 误检 / 多余检测 | 越少越好 |
| **FN** (False Negative) | 漏检 | 越少越好 |
| **MOTA** | $1 - \dfrac{\sum(FN+FP+IDS)}{\sum GT}$ | 综合漏检、误检、ID切换的**检测+关联**质量，是 2D/3D MOT 最常用主指标 |
| **MOTP** | 匹配框的平均定位误差 | 反映定位精度，2D 用 IoU，3D 用 3D-IoU 或中心点距离 |

**MOTA 局限**：被检测质量（FP/FN）主导，对 ID 关联质量不敏感——一个检测很准但 ID 乱跳的方法仍可能 MOTA 很高。

---

## 二、轨迹连续性指标

| 指标 | 含义 | 说明 |
|------|------|------|
| **IDS** (ID Switch) | 同一目标 ID 发生切换的次数 | 衡量关联稳定性，越少越好 |
| **FRAG** (Fragmentation) | 轨迹中断次数 | 反映轨迹完整度 |
| **IDF1** [Ristani 2016] | 基于 ID 的 F1-score | 比 MOTA 更关注 ID 关联正确性 |

**IDF1 vs MOTA**：当 ID 关联很差但检测很好时，IDF1 会显著下降，而 MOTA 可能无明显变化。诊断关联质量时优先看 IDF1。

---

## 三、HOTA [Luiten 2021] —— 当前 MOT 主流统一指标

将跟踪拆解为**检测**与**关联**两个独立维度：

```
HOTA = √(DetA × AssA)
            检测准确度 × 关联准确度
```

| 子指标 | 含义 |
|--------|------|
| **DetA** (Detection Accuracy) | 检测质量（TP / FP / FN） |
| **AssA** (Association Accuracy) | 关联质量（ID 是否一致） |
| **LocA** (Localization Accuracy) | 定位精度 |

> **HOTA 的优势**：能分别诊断是"检测不行"还是"关联不行"。
> **KITTI 3D MOT 主指标就是 HOTA**（如 MCTrack 在 KITTI 的 82.56% 即此）。

---

## 四、3D MOT 专用指标

| 指标 | 含义 | 说明 |
|------|------|------|
| **AMOTA / AMOTP** [Weng 2020, AB3DMOT] | 对置信度阈值平均的 MOTA / MOTP | 解决 3D 检测有 score、单一阈值不公平的问题；**nuScenes / Waymo 3D MOT 主指标** |
| **sAMOTA** | 缩放后的 AMOTA | nuScenes 官方采用 |
| **3D-IoU / 中心点距离** | 3D 框匹配质量 | 替代 2D IoU，常用 2 m 中心距或 IoU > 0.3 / 0.5 / 0.7 |

> **为何需要 AMOTA**：3D 检测器输出带 score，传统 MOTA 只在单一阈值下计算，对 score 阈值选择敏感。AMOTA 对不同置信度阈值下的 MOTA 积分平均，更公平地反映整体性能。

---

## 五、运动状态指标（MCTrack 提出，填补空白）

传统指标只看 **ID 对不对**，不评价跟踪输出的**速度 / 加速度**对不对——而下游规划模块正依赖这些运动信息。MCTrack 提出 6 个：

| 指标 | 全称 | 评估对象 |
|------|------|----------|
| **VAE** | Velocity Angle Error | 速度方向（角度）误差 |
| **VNE** | Velocity Norm Error | 速度大小（幅值）误差 |
| **VAIE** | Velocity Angle Inverse Error | 速度方向反向的严重程度（超过 π/2 才计入） |
| **VIR** | Velocity Inversion Ratio | 速度方向反向的帧数比例 |
| **VSE** | Velocity Smoothness Error | 速度曲线平滑度（Savitzky-Golay 滤波） |
| **VDE** | Velocity Delay Error | 速度信号相对真值的**延迟**（对自动驾驶安全最关键） |

> **VDE 的工程意义**：以两车 100 km/h 行驶、车距 100 m 为例，若前车急刹减速而跟踪模块有 VDE 延迟，自车会误判前车仍在高速行驶，导致安全距离被悄悄压缩——运动信息的及时性直接关系自动驾驶安全。

---

## 六、指标选用速查表

| 场景 | 主指标 | 辅助指标 |
|------|--------|----------|
| **KITTI 3D MOT** | HOTA | MOTA, MOTP, IDS, FRAG, AssA |
| **nuScenes 3D MOT** | AMOTA | AMOTP, IDS, TP / FP / FN |
| **Waymo 3D MOT** | MOTA (L1/L2) | MOTP, IDS |
| **通用关联质量诊断** | IDF1, AssA | IDS, FRAG |
| **下游规划 / 运动质量** | VAE, VNE, VDE | VSE, VIR |

---

## 七、数学定义（公式与距离表达）

### 7.1 计数类（MOTA / MOTP / IDF1）

设第 $t$ 帧的真值框集合 $G_t$、跟踪框集合 $D_t$，匹配阈值 $\tau$（IoU 或距离）下匹配对集合 $\mathcal{M}_t \subseteq G_t \times D_t$。

**TP / FP / FN（逐帧累计）**：

$$
\text{TP}=\sum_t |\mathcal{M}_t|,\quad
\text{FN}=\sum_t (|G_t|-|\mathcal{M}_t|),\quad
\text{FP}=\sum_t (|D_t|-|\mathcal{M}_t|)
$$

**MOTA**（$\text{GT}_{tot}=\sum_t|G_t|$，IDS 为 ID 切换总数）：

$$
\text{MOTA}=1-\frac{\text{FN}+\text{FP}+\text{IDS}}{\text{GT}_{tot}}
$$

**MOTP（定位误差）**：2D/3D-IoU 用法取 $(1-\overline{\text{IoU}})$；中心点距离用法取平均欧氏距：

$$
\text{MOTP}=\frac{1}{\text{TP}}\sum_{(g,d)\in\cup\mathcal{M}_t}\text{dist}(g,d),\qquad
\text{dist}(g,d)=\lVert c_g-c_d\rVert_2
$$

**IDF1（基于 ID 的精确率/召回率的调和平均）**：

$$
\text{IDP}=\frac{\text{IDTP}}{\text{IDTP}+\text{FP}},\quad
\text{IDR}=\frac{\text{IDTP}}{\text{IDTP}+\text{FN}},\quad
\text{IDF1}=\frac{2\,\text{IDP}\cdot\text{IDR}}{\text{IDP}+\text{IDR}}
$$

### 7.2 HOTA（检测 × 关联）

$$
\text{DetA}=\frac{|\text{TP}|}{|\text{TP}|+|\text{FP}|+|\text{FN}|},\qquad
\text{AssA}=\frac{1}{|\text{TP}|}\sum_{c\in\text{TP}}\frac{\text{TPA}(c)}{\text{TPA}(c)+\text{FPA}(c)+\text{FNA}(c)}
$$

$$
\text{HOTA}=\sqrt{\text{DetA}\cdot\text{AssA}}
$$

其中 $\text{TPA}(c)$ 为同一真值 ID $c$ 在各真阳匹配上的帧数，FPA/FNA 为关联假阳/假阴。

### 7.3 AMOTA（对置信度阈值积分平均）

对置信度阈值集合 $\{s_1,\dots,s_K\}$（如 nuScenes 取 40 个），逐阈值计算 $\text{MOTA}(s_k)$：

$$
\text{AMOTA}=\frac{1}{40}\sum_{k=1}^{40}\text{MOTA}(s_k)\quad(\text{或对 PR 曲线积分})
$$

### 7.4 距离 / 重叠度量（匹配门限 $\tau$ 用）

| 度量 | 数学表达 | 适用 |
|------|---------|------|
| **2D IoU** | $\dfrac{\|B_g\cap B_d\|}{\|B_g\cup B_d\|}$（面积交并） | 2D 框 |
| **3D IoU** | $\dfrac{\|B_g\cap B_d\|}{\|B_g\cup B_d\|}$（体积交并） | 3D 框，阈值常 0.3/0.5/0.7 |
| **中心点欧氏距** | $\lVert c_g-c_d\rVert_2=\sqrt{(x_g{-}x_d)^2+(y_g{-}y_d)^2+(z_g{-}z_d)^2}$ | 3D，常取 2 m |
| **BEV 中心距** | $\sqrt{(x_g{-}x_d)^2+(y_g{-}y_d)^2}$ | 只用 x,y |
| **航向角差** | $\min(|\theta_g{-}\theta_d|,\ 2\pi-|\theta_g{-}\theta_d|)$ | 朝向一致性 |

> **本项目的匹配门限**：BEV 中心距（见 [match.py](../../../match.py)），阈值由 `cfg.ini` 的 `match_thresh` 给出。

### 7.5 运动状态指标（速度误差）

设真值速度 $\mathbf v_g=(v_{g,x},v_{g,y})$、跟踪输出速度 $\mathbf v_d=(v_{d,x},v_{d,y})$，幅值 $|\cdot|$、方向角 $\angle(\cdot)=\arctan2(\cdot)$。

| 指标 | 数学表达 |
|------|---------|
| **VAE** | $\angle\mathbf v_d-\angle\mathbf v_g$（速度方向角之差，取绝对值） |
| **VNE** | $\big\lvert|\mathbf v_d|-|\mathbf v_g|\big\rvert$（速度幅值之差） |
| **VAIE** | $\mathbb 1[\lvert\Delta\angle\rvert>\pi/2]\cdot\lvert\Delta\angle\rvert$（仅反向时计入） |
| **VIR** | $\dfrac{1}{T}\sum_t\mathbb 1\!\left[\angle(\mathbf v_{d,t})-\angle(\mathbf v_{g,t})>\pi/2\right]$（反向帧占比） |
| **VSE** | $\big\lVert\,\text{SG-filter}(\mathbf v_d)-\mathbf v_d\,\big\rVert$（SG 滤波后的残差，越小越平滑） |
| **VDE** | $\arg\max_\delta \text{corr}\big(\mathbf v_d(t),\,\mathbf v_g(t-\delta)\big)$（使相关最大的时移，即延迟） |

---

## 八、核心记忆点

- **MOTA** 偏检测质量（被 FP/FN 主导）
- **IDF1 / HOTA 的 AssA** 偏关联质量（ID 一致性）
- **AMOTA** 是 3D 特化版本（对置信度阈值平均）
- **Motion Metrics** 评估运动输出质量（速度/加速度），面向下游规划

> 现代论文通常同时报告 **HOTA + MOTA + AMOTA + IDS**，以覆盖检测、关联、3D 特化、稳定性四个维度。

---

## 九、各指标的提出文献

| 指标族 | 文献 | 年份 |
|--------|------|------|
| CLEAR (MOTA/MOTP/TP/FP/FN) | Bernardin & Stiefelhagen, *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics* | 2008 |
| IDF1 | Ristani et al., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking* | 2016 |
| HOTA | Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking* | 2021 |
| AMOTA / sAMOTA | Weng et al., *AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics* | 2020 (IROS) |
| Motion Metrics (VAE/VNE/VDE/...) | Wang et al., *MCTrack: A Unified 3D Multi-Object Tracking Framework* | 2024 (arXiv 2409.16149) |
