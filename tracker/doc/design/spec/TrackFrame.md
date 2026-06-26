# Tracker 链路设计

> RadarPillar + Tracker：以 RadarPillar 为检测模型，后接 tracker 全链路，仿真验证把 RadarPillar 引入 Tracker 的性能与可行性。
>
> 主链路：`loader → preprocess → detect → match → filter → manager`
>
> 流程编排由 `tracker.py` 统一驱动（逐帧串联全链路）。

```
tracker.py ->
`loader → preprocess → detect → match → filter → manager`
```

一帧的编排顺序（`tracker.py` 内）：

1. `preprocess.run(frame)`
2. `detect.run(frame)` → `Objs`
3. `filter.predict_all(manager.tracks)` — 预测必须先于关联
4. `match.associate(manager.tracks, objs)`
5. `filter.update_all(matches)`
6. `manager.update(matches, objs, frame)`
7. 记录轨迹（供 performance）

## 各模块设计

### Public 结构体声明

**数据结构（VDS/VDD/FRAME 系列）：**

| 类 | 作用 |
|---|---|
| `VDS` | 静态参数 - 轴距、传感器安装位置等（对齐 RUN.vds） |
| `VDD` | 动态参数 - 车速、档位等 |
| `PT` | 单点检测 - 对齐车载雷达 Det_t（距离/角度/多普勒/RCS 等） |
| `PTs` | 点云 - num + list[PT] |
| `GT` | 真值 - 单条 [x/y/z/vx/vy/length/width/height/heading/type/isghost/ispassable] |
| `GTs` | 真值集 - num + list[GT] |
| `FRAME` | 帧数据 - GTs + PTs + VDD |
| `FRAMEs` | 帧序列 - num + list[FRAME] |

**目标/航迹/匹配：**

| 类 | 作用 |
|---|---|
| `Objs` | 目标 - [x/y/vx/vy/length/width/heading/type/isghost/ispassable] |
| `Trks` | 航迹 - 对齐车载雷达 Trk_t（运动学/尺寸/标准差/分类状态全字段） |
| `Matches` | 匹配 - matched + 未匹配 Trks + 未匹配 Objs |

**配置（Cfg 系列，镜像 cfg.yaml）：**

| 类 | 作用 |
|---|---|
| `Cfg` | 配置容器，聚合 8 大组 |
| `CfgRun` / `CfgVds` | 运行配置（mode/overlap/delay/vds） |
| `CfgData` | 数据配置（paths） |
| `CfgModel` | 模型配置（cfg/ckpt/score_thresh） |
| `CfgFilter` / `CfgFilterPara` / `CfgFilterParaKf` | 滤波配置（type/para） |
| `CfgMatch` | 关联配置（gap_type/gap_dim/gap_weight/thresh） |
| `CfgVisualize` | 可视化配置（enable/show/metrics） |
| `CfgEvaluate` | 性能评估配置（type/report/template） |
| `CfgManager` | 航迹管理配置（heat/dt/history_horizon/adapter） |

---
### .yaml文件
- **数据配置**
  1.数据绝对地址（可有多个）

- **模型配置**
  1.模型配置文件或结构：主要针对前向传播的结构
  2.模型权重地址

- **滤波配置**
  1.种类：
    1-α-β;2-kf;3-ekf;4-imm
  2.滤波参数:
    α-β
      α和β  ： 代码自动归一化
    KF
      维度 ： 2维度(x/y)/4维度(x/y/vx/vy)
      QR   :
    EKF:
      ...

- **关联配置**
  1.距离种类：1-欧氏距离 ;2-马氏距离
  2.距离维度：2-x/y; 3-x/y/z; 4-x/y/dpl_gnd
  3.距离权重：[a,b,c] :按照以上维度自适应归一化

- **可视化配置**
  1. 形式：0-on ;1-off
  2. 类别：
      点云： 0 - disable/ 1 - enable
      Trks： 0 - disable/ 1 - enable
      Objs： 0 - disable/ 1 - enable
      GTs： 0 - disable/ 1 - enable
  3. 性能指标：0-on ;1-off
  4. 配置： 
    - **TP**  :0 - disable/ 1 - enable
    - **FP**  :0 - disable/ 1 - enable
    - **FN**  :0 - disable/ 1 - enable
    - **IDS** :0 - disable/ 1 - enable
    - **FRAG**:0 - disable/ 1 - enable
    - **MOTA**:0 - disable/ 1 - enable
    - **MOTP**:0 - disable/ 1 - enable
    - **IDF1**:0 - disable/ 1 - enable
    - **DetA**:0 - disable/ 1 - enable
    - **AssA**:0 - disable/ 1 - enable
    - **HOTA**:0 - disable/ 1 - enable
    - **LocA**:0 - disable/ 1 - enable
    - **AMOTA**:0 - disable/ 1 - enable
    - **AMOTP**:0 - disable/ 1 - enable
    - **sAMOTA**:0 - disable/ 1 - enable
    - **VAE** :0 - disable/ 1 - enable
    - **VNE** :0 - disable/ 1 - enable
    - **VAIE**:0 - disable/ 1 - enable
    - **VIR** :0 - disable/ 1 - enable
    - **VSE** :0 - disable/ 1 - enable
    - **VDE** :0 - disable/ 1 - enable


- **性能评估配置**
  1. 种类：0:off; 1：online; 2.offline
  2. 报告：0:off; 1: on
  3. 模版：

- **航迹管理配置**
  

- **运行模式配置**
  mode: 0:display;1:normal;2:regress
  overlap : 0:不覆盖；1:覆盖

---

### tracker.py — 编排入口（新增）

- **描述**：流程编排器，逐帧驱动全链路；本身不写算法。
- **输入**：cfg 文件路径（各个配置参数在 cfg 内）。
- **输出**：无对外返回。
- **内部**：算法 SIL 入口——init 解析 cfg 路径并构造各模块；run() 逐帧串联
           preprocess→detect→predict→match→update→manage→evaluate

---

### loader.py — 加载与切片

- **描述**：解析cfg文件，通过文件初始化各个模块；
- **输入**：`Cfg`。
- **输出**：迭代产出 `Frame`。
- **pipeline**:
  cfg = load_cfg(cfg_filepath)  # cfg 是 class Cfg:
  if !cfg_isvaild(cfg)
    抛出异常
  datalist = get_datalist(cfg.DATA.paths)
  for path in datalist:
    data = load_data(path)
    frames = parser(data)



---

### preprocessor.py — 预处理

- **描述**：单帧点云特征工程 + 体素化，对齐 `pcdet/datasets/processor`，无增强/无 batch。
- **输入**：`Frame`。
- **输出**：规整后的 `Frame`。
- **类**：`Preprocessor` — 复用 `PointFeatureEncoder` + `DataProcessor`，关闭训练增强。
- **功能** ：
  1. 根据VFE来，做输入模型的准备工作

---

### detector.py — 检测推理

- **描述**：调用 RadarPillar 推理输出 3D 框，对齐 `tools/demo.py`。
- **输入**：`Frame`。
- **输出**：`Objs` 列表。
- **类**：`Detector` — 加载权重/建模型、逐帧推理、按 score_thresh 过滤。
  1. 根据cfg进行模型前向传播配置 model = build_model(cfg.model)
  2. 根据cfg对模型进行权重设置   model = set_weiget(weight_file)

---

### matcher.py — 检测与轨迹关联

- **描述**：将预测后轨迹与当前检测关联，3 步级联。
- **输入**：预测后 `Track` 列表 + `Objs` 列表。
- **输出**：`Matches`。
- **3 步级联**：① cls 合并候选 → ② 无冲突 1:1 匹配 → ③ 冲突用 KM/PDA。
- **类**：`Matcher` — 关联器，门限 = `match_thresh`（BEV 中心距，m）。

---

### filter.py — 卡尔曼滤波

- **描述**：BEV 4 维 `[x,y,vx,vy]` 常速度 KF（init/predict/update）；z/heading/lwh 不在此。
- **输入**：`Objs`。
- **输出**：`Track`（更新 `[x,y,vx,vy]`）。
- **类**：`KalmanFilter` — 滤波器，提供 init/predict/update 及批量 predict_all/update_all。
- **待调**：Q/R 整定值；是否升级 EKF。

---

### manager.py — 轨迹集合管理

- **描述**：维护活跃轨迹集，串联目标管理、历史维护、adapter 输出。
- **输入**：`Matches` + `Objs` 列表 + `Frame`。
- **输出**：当前帧确认的活跃 `Track` 列表。
- **类**：`TrackManager` — 轨迹集合管理器，只持有并暴露**已预测**的轨迹供 match 读取（predict 动作由 filter 完成）。
- **四项职责**：
  1. 目标管理 birth/death（heat 逻辑）；
  2. 维护 4s 隐藏历史（x/y/z/l/w/heading/vx/vy/type/ghost/passable）；
  3. adapter：连续量历史平滑、type/ghost/passable 马尔科夫链输出；
  4. 属性补充：置信度等规则粗略输出（非重点）。

---

### evaluator.py — 性能评估与可视化

- **描述**：用 GT 评估跟踪结果，指标可配（见 `MTT_评估指标.md`）；离线保存与可视化。
- **输入**：对应帧 `TRKs`+ `GT`。
- **输出**：评估报告 + 落盘文件。
- **类**：`Evaluator` — 提供 evaluate/save/visualize，可视化适配 fastseer 或 autosil，报告存 `tracker/doc`。

---

### utils.py — 工具函数

- **描述**：自车补偿、多帧叠加等纯函数。
- **函数**：`compensate()` — 用 ego 增量（dx/dy/dθ）把历史补偿到当前 ego 坐标系。

---
