# nuScenes + RadarPillar 训练操作手册（小白版 · 总分结构）

> 适用对象：第一次接触 OpenPCDet / RadarPillar / nuScenes 的同学。
> 目标：在自己的机器上把 **RadarPillar（radar-only）** 在 **nuScenes v1.0-mini（或 trainval）** 上跑通训练 + 评估，并理解每一步在做什么、为什么这么做。
>
> 文档结构：**先"总"后"分"**。第 1 章给你一张完整鸟瞰图，之后每章先给该章的"数据流小图"，再展开操作细节。

---

# 总：第 1 章 鸟瞰图

## 1.1 一句话目标

> 把 nuScenes 的 5 个雷达通道的稀疏点云 + 标注框，喂给 RadarPillar，让它学会在 BEV（鸟瞰图）上输出 10 类目标的 3D 框（带速度）。

## 1.2 任务清单（你最终要交付什么）

| # | 交付物 | 文件位置 | 章节 |
|---|---|---|---|
| 1 | 一个能激活的 conda 环境 `angle` | — | 第 2 章 |
| 2 | nuScenes 数据集（mini 或 trainval）按约定目录放好 | `data/nuscenes/v1.0-*/` | 第 3 章 |
| 3 | 一份"数据画像报告"（点云分布 / 类别分布 / 速度分布 / BEV 热力图） | `reports/` | 第 4 章 |
| 4 | 两份 infos pkl（train / val）+ 数据集类已能正确取样 | `data/nuscenes/v1.0-*/nuscenes_infos_radar_*.pkl` | 第 5 章 |
| 5 | 训练好的 checkpoint | `output/cfgs/nuscenes_models/radarpillar_nuscenes/<run>/ckpt/checkpoint_*.pth` | 第 6 章 |
| 6 | 评估报告（NDS / mAP / 各类 AP） | `output/.../<run>/eval/metrics_summary.json` | 第 7 章 |
| 7 | BEV 可视化图（GT 绿 vs 预测红） | `reports/figures/bev_pred/` | 第 8 章 |
| 8 | 你能回答"为什么 mAP 不高 / Loss 为什么不降"这种问题 | — | 第 9 章 |

## 1.3 高层数据流（端到端）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          A.  离线一次性准备                                │
└──────────────────────────────────────────────────────────────────────────┘
  nuScenes 原始数据                              infos pkl
  ┌────────────────────┐   fill_radar_infos    ┌──────────────────────────┐
  │ samples/RADAR_*/   │ ────────────────────► │ nuscenes_infos_radar_    │
  │   *.pcd (18维)     │   （只跑一次）         │   1sweeps_train.pkl      │
  │ v1.0-*/            │                       │ nuscenes_infos_radar_    │
  │   *.json (元数据)  │                       │   1sweeps_val.pkl        │
  └────────────────────┘                       └──────────────────────────┘
                                                          │  （持久化）
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          B.  在线训练（每个 epoch 重复）                  │
└──────────────────────────────────────────────────────────────────────────┘
                                                          ┌──────────────────────────┐
                                                          │ NuScenesRadarDataset     │
                                                          │  (PyTorch Dataset)       │
                                                          │  - get_radar_with_sweeps │
                                                          │  - prepare_data          │
                                                          └──────────────────────────┘
                                                                    │  __getitem__(idx)
                                                                    ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ batch_dict                                                  │
              │   points     : List[Tensor(N_i, 7+1)]   ←── 5 通道 + time    │
              │   voxels     : Tensor(B, 20, 7)         ←── PillarVFE 之前   │
              │   voxel_coords: Tensor(B, 4)            ←── 哪批/哪帧/哪格   │
              │   gt_boxes   : Tensor(B, M, 9)         ←── x,y,z,l,w,h,yaw,vx,vy│
              │   gt_names   : List[array]                                │
              └─────────────────────────────────────────────────────────────┘
                                                                    │  DataLoader
                                                                    ▼
                                                          ┌──────────────────────────┐
                                                          │  PointPillar.forward     │
                                                          │  ┌────────────────────┐  │
                                                          │  │ PillarVFE          │  │  (N,7) → (P,9) pillar 特征
                                                          │  │ PillarAttention    │  │  pillar 内部 self-attention
                                                          │  │ PointPillarScatter │  │  → BEV 320×320×32
                                                          │  │ BaseBEVBackbone    │  │  2D CNN 上采样
                                                          │  │ AnchorHeadSingle   │  │  → cls / box / dir
                                                          │  └────────────────────┘  │
                                                          └──────────────────────────┘
                                                                    │  pred_dicts
                                                                    ▼
              ┌─────────────────────────────────────────────────────────────┐
              │ loss = cls_loss + 2.0 * box_loss + 0.2 * dir_loss            │
              │       backward → Adam OneCycle                                │
              └─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          C.  评估（train 之后跑一次）                      │
└──────────────────────────────────────────────────────────────────────────┘
                                  预测框 ──► NuScenesEval（官方）
                                            ──► NDS / mAP / 5 个 error 分解
                                            ──► metrics_summary.json
```

## 1.4 全局概念对照表（先认脸，再操作）

| 术语 | 英文 | 在本项目中是什么 |
|---|---|---|
| 关键帧 | keyframe / sample | nuScenes 中 2 Hz 标注的雷达帧；404（mini）/ 28k（trainval）个 |
| sweep | sweep | 同雷达通道比当前帧更早的某一帧（用于时序聚合） |
| 雷达通道 | RADAR channel | RADAR_FRONT / _LEFT / _RIGHT / _BACK_LEFT / _BACK_RIGHT 共 5 个 |
| PCD | Point Cloud Data | 18 维 ASCII 头 + 二进制点（x,y,z,rcs,vx,vy,...） |
| infos pkl | infos pickle | OpenPCDet 约定的"数据集索引"——每个关键帧一行元信息 |
| GT | ground truth | 标注的 3D 框；nuScenes 23 类 → 本项目 10 类 |
| BEV | Bird's-Eye View | 从上往下看的栅格图，本项目 320×320×32 |
| pillar | voxel column | z 方向不分割、xy 方向 0.2 m 的"柱子"，多个点 → 1 个 pillar |
| NDS | NuScenes Detection Score | nuScenes 官方主指标；mAP + 5 个 error 加权和 |
| mAP | mean Average Precision | 各类 AP 的平均，越高越好 |
| mATE/ASE/AOE/AVE/AAE | 5 个 error | 平移/尺度/朝向/速度/属性 误差；越低越好 |

## 1.5 阅读路径建议

| 你的时间 | 看哪些章节 |
|---|---|
| 1 小时跑通 | 第 1 → 2 → 3 → 5.3 → 6.2 → 7.2 |
| 半天理解原理 | 加上第 4（统计）、第 5.4（历史 bug 复盘）、第 8（可视化） |
| 1 天调优 | 再加上第 9 章 + 附录 |

---

# 分：从第 2 章开始，每章自带数据流小图

> **约定**：从这一章起，每章开头都先画一张"本章数据流小图"，标出本章负责的环节在整体 pipeline 里的位置，再讲操作细节。

---

## 2. 环境准备

### 2.1 本章数据流

```
[第 1 章:你已经看过鸟瞰图]
        │
        ▼ 本章目标
   激活 angle 环境 + 装好 4 类依赖
   ┌──────────────┐
   │ PyTorch      │ ← 深度学习框架
   │ spconv       │ ← 稀疏 3D 卷积（点云体素化）
   │ nuscenes-devkit│ ← nuScenes 官方工具包
   │ pcdet (本项目)│ ← OpenPCDet 检测框架
   └──────────────┘
        │
        ▼ 下一章:下载数据
```

### 2.2 进入 conda 环境 `angle`

```bash
conda activate angle
python -V          # 期望 >= 3.8 且 < 3.11（devkit 限制）
pip -V             # 期望 >= 22
```

如果 `angle` 不存在：

```bash
conda create -n angle python=3.9 -y   # 推荐 3.9，兼容 nuscenes-devkit 1.0.5
conda activate angle
```

### 2.3 安装 PyTorch（CUDA 12.x）

```bash
# 视你的 CUDA 版本调整（CUDA 12.1 示例）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

> 如果你只有 CPU 或无 GPU：训练会非常慢（mini 数据集一个 epoch 也要数小时）。本文档主要面向有 NVIDIA GPU 的情况。
> 如果 GPU 是 RTX 40 系（Ampere 之后），用 `cu121` 或 `cu118` 都行，关键是 PyTorch ≥ 2.1。

### 2.4 安装项目 + nuscenes-devkit

```bash
# 在项目根目录
cd /home/dministrator1/RadarPillar
pip install -r requirements.txt
pip install nuscenes-devkit==1.0.5    # 关键：devkit 版本必须 == 1.0.5
python setup.py develop
```

> ❗ **nuscenes-devkit 版本必须锁死 1.0.5**。1.1.x 改了表结构（`sample_annotation` 直接带 `category_name` 不再要查表），你按本文档的脚本会读不到字段。
> 本文档里的脚本已经按 1.0.5 写好，1.1.x 也能跑通（取字段时优先用 `category_name`），但官方训练脚本（`create_nuscenes_radar_info`）只能跑 1.0.5。

### 2.5 安装 spconv（雷达点云体素化）

```bash
# CUDA 12.x + PyTorch 2.4 对应 spconv 2.3.6+
pip install spconv-cu120==2.3.6
```

> 如果你 PyTorch 是 `cu118`，对应 `spconv-cu118==2.3.6`。

### 2.6 验证安装

```bash
python -c "
import torch, spconv, nuscenes, pcdet
print('torch       :', torch.__version__)
print('cuda        :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('spconv      :', spconv.__version__)
print('nuscenes    : OK (', nuscenes.__file__, ')')
print('pcdet       :', pcdet.__file__)
"
```

期望输出类似：
```
torch       : 2.4.0
cuda        : True NVIDIA GeForce RTX 4090
spconv      : 2.3.6
nuscenes    : 1.0.5
pcdet ok
```

> ❗ 如果 `nuscenes.__version__` 不是 `1.0.5`，立刻 `pip install nuscenes-devkit==1.0.5 --force-reinstall`，否则后面 `nuscenes.nuscenes.NuScenes` 加载会报 schema 错误。

### 2.7 常见环境问题速查

| 现象 | 原因 | 修复 |
|---|---|---|
| `ImportError: No module named 'spconv'` | 没装 spconv | 2.5 节 |
| `RuntimeError: ...CUDA driver version...` | PyTorch CUDA 版本与系统不匹配 | 2.3 节换版本 |
| `nuscenes-devkit` 报 schema 错误 | 版本不是 1.0.5 | 2.4 节强制重装 |
| `ImportError: No module named 'av'` | 缺 PyAV（视频解码，tracker 可能用到） | `pip install av` |
| `numba.cuda ... segmentation fault` | numba 0.58 + WSL2 已知问题 | 详见 10 章 FAQ |

---

## 3. 数据下载与组织

### 3.1 本章数据流

```
[nuScenes 官网]                          [你的硬盘]
   │                                        │
   │  注册 + 下载                            │
   │  v1.0-mini.tgz (4 GB)                 │
   │  v1.0-trainval01/02/03_blobs.tgz       │
   │       (~40 GB)                         │
   ▼                                        ▼
   tar -xzf                            解压得到
                                  ┌─────────────────────┐
                                  │ samples/            │ ← RADAR_*/LIDAR_TOP/CAM_*
                                  │ sweeps/             │ ← 历史 sweep（mini 没有，trainval 有）
                                  │ maps/               │ ← 高精地图
                                  │ v1.0-mini/          │ ← 元数据 JSON
                                  │   scene.json       │
                                  │   sample.json      │
                                  │   sample_data.json │
                                  │   sample_annotation│
                                  │   calibrated_sensor│
                                  │   ego_pose.json    │
                                  │   ...              │
                                  └─────────────────────┘
                                           │
                                           ▼ 软链到约定路径
                                  RadarPillar/
                                  └── data/
                                      └── nuscenes/
                                          └── v1.0-mini/  ──► 实际目录的软链
```

### 3.2 数据集大小

| 版本 | 大小 | 帧数（关键帧） | 适用场景 |
|---|---|---|---|
| v1.0-mini | 4 GB | 404 | 跑通 pipeline / 调试 |
| v1.0-trainval | ~40 GB | 28,130 | 真正训练 |
| v1.0-test | ~30 GB | 6,008 | 提交排行榜 |

**小白建议：先用 mini 跑通。**

### 3.3 下载地址

到 [https://www.nuscenes.org/download](https://www.nuscenes.org/download) 注册账号 → 下载：
- `v1.0-mini.tgz` （mini）
- 或 `v1.0-trainval01_blobs.tgz` + `v1.0-trainval02_blobs.tgz` + `v1.0-trainval_meta.tgz`（trainval，注意是 3 个文件）

下载完后用 `tar -xzf *.tgf` 解压。

### 3.4 组织目录

本项目已为你把 mini 软链到 `data/nuscenes/v1.0-mini`：

```
RadarPillar/
└── data/
    └── nuscenes/
        └── v1.0-mini/         ← 软链到 /mnt/d/DATASET_PART
            ├── samples/        ← CAM_*、RADAR_*、LIDAR_TOP/*.pcd|.jpg
            ├── sweeps/         ← 时序帧缓存（nuScenes 内置）
            ├── maps/
            └── v1.0-mini/      ← 元数据 json
                ├── scene.json
                ├── sample.json
                ├── sample_data.json
                ├── sample_annotation.json
                ├── ego_pose.json
                ├── calibrated_sensor.json
                └── ...
```

> ⚠️ 软链目前在 `/mnt/d/DATASET_PART`（即 Windows 盘 D 下的 `DATASET_PART`）。如果你换机器或挂载路径变了，需要重新链接：
>
> ```bash
> ln -s /path/to/your/nuscenes/v1.0-mini data/nuscenes/v1.0-mini
> ```

### 3.5 验证目录结构

```bash
ls data/nuscenes/v1.0-mini/samples/RADAR_FRONT/ | head -3
ls data/nuscenes/v1.0-mini/v1.0-mini/ | head
# 期望：能看到 .pcd 文件 + 元数据 json
```

### 3.6 元数据 JSON 之间的关联图

```
scene.json          sample.json          sample_data.json       sample_annotation.json
┌──────────┐       ┌──────────┐        ┌──────────────────┐    ┌─────────────────────┐
│ scene    │ 1───* │ sample   │ 1───*  │ sample_data      │    │ sample_annotation   │
│ token    │       │ token    │        │ token            │    │ token               │
│          │       │ scene_tok│        │ sample_token     │    │ sample_token        │
│          │       │ data:    │        │ prev/next        │    │ instance_token      │
│          │       │  RADAR_* │        │ ego_pose_token   │    │ category_name       │
│          │       │  LIDAR_* │        │ calib_sensor_tok │    │ translation (x,y,z) │
│          │       │  CAM_*   │        │ filename (.pcd)  │    │ size (w,l,h)        │
│          │       │ anns:[]  │        │ timestamp        │    │ rotation (quat)     │
└──────────┘       └──────────┘        └──────────────────┘    │ velocity (vx,vy)    │
                              │                │               │ num_lidar_pts       │
                              │                │               │ num_radar_pts       │
                              ▼                ▼               └─────────────────────┘
                          关键帧 ──► 雷达 .pcd  ──► ego_pose ──► 标注框
                                       (18 维点云)   (位姿)       (10 类)
```

> 理解这张图就够了。后面的 `fill_radar_infos` 就是按 `sample` 为主键，遍历 `data` 拿到 .pcd 路径，再去 `sample_annotation` 拿标注。

### 3.7 5 个雷达通道的物理布局

```
                ego 车
                  ▲ x (前)
                  │
       ┌──────────┼──────────┐
       │  FL      │      FR  │   FL = RADAR_FRONT_LEFT
       │          │          │   FR = RADAR_FRONT_RIGHT
       │          │          │   F  = RADAR_FRONT
       │     [F]──┼          │   BL = RADAR_BACK_LEFT
       │          │          │   BR = RADAR_BACK_RIGHT
       └──────────┼──────────┘
                  │
                  │ ◄── y (左)
                  ▼

       ┌──────────────────────┐
       │  BL          BR       │
       │   ┌────────┐         │
       │   │        │         │   5 个雷达水平 FOV ≈ 360° 覆盖一圈
       │   │  ego   │         │   但角度分辨率低 + 点云稀疏
       │   │        │         │
       │   └────────┘         │
       │  FL          FR       │
       └──────────────────────┘
```

> 重要含义：**只用 1 个通道会丢掉 4/5 的覆盖**。早期版本 loader 确实只用 RADAR_FRONT（已修复，详见 5.4）。

---

## 4. 数据统计分析

### 4.1 本章数据流

```
[第 3 章:数据已就位]
        │
        ▼ 本章目标:用脚本给数据"拍 X 光"
   ┌─────────────────────────────────────────────────┐
   │ 脚本 1  元信息: 多少 scene / sample / anno      │ → reports/basic_stats.json
   │ 脚本 2  点云: 每帧多少点 / 距离 / RCS / 速度     │ → reports/figures/radar_*.png
   │ 脚本 3  类别: 类别直方图 / 距离 / 速度 / 尺寸     │ → reports/figures/gt_*.png
   │ 脚本 4  几何: BEV 热力图                        │ → reports/figures/bev_heatmap.png
   └─────────────────────────────────────────────────┘
        │
        ▼ 下一章:用这些统计决定 anchor / 范围 / 增强
```

### 4.2 创建脚本目录

```bash
mkdir -p tools/nuscenes_analysis reports
```

### 4.3 脚本 1：基础元信息（场景 / 样本 / 标注）

完整代码见附录 A.1。保存为 `tools/nuscenes_analysis/01_basic_stats.py`。

跑：

```bash
python tools/nuscenes_analysis/01_basic_stats.py
```

**期望输出**（mini 数据集）：

```json
{
  "version": "v1.0-mini",
  "n_scene": 10,
  "n_sample": 404,
  "n_sample_data": 31206,
  "n_annotation": 18538,
  "n_radar_files_present": 2020,
  "n_radar_files_missing": 0,
  "avg_anno_per_frame": 45.89,
  "category_raw_dist": {
    "vehicle.car": 7619,
    "human.pedestrian.adult": 4765,
    "movable_object.barrier": 2323,
    ...
  }
}
```

**怎么读这份报告**：

| 字段 | 你要关心什么 | 含义 / 启示 |
|---|---|---|
| `n_radar_files_present` | 是否 = `5 × n_sample` | mini 应是 5×404=2020，缺了说明数据下载不全 |
| `avg_anno_per_frame` | 是否 ≥ 10 | mini 上 45.89 是正常的 |
| `category_raw_dist` | 哪些类多哪些类少 | car 占 41%、barrier+cone 占 20% → 类别严重不均衡 |

### 4.4 脚本 2：雷达点云统计（点数 / 距离 / RCS / 速度）

完整代码见附录 A.2。保存为 `tools/nuscenes_analysis/02_radar_point_stats.py`。

跑：

```bash
python tools/nuscenes_analysis/02_radar_point_stats.py
# trainval 数据集（更大）建议抽 200 帧即可：
python tools/nuscenes_analysis/02_radar_point_stats.py --max_frames 200
```

**期望输出**：

```json
{
  "n_frames_sampled": 200,
  "range_m": {"mean": 50.7, "p50": 44.0, "p95": 111.2, "max": 257.6},
  "rcs_dbsm": {"mean": 8.1, "p50": 7.0, "p95": 22.0, "max": 50.0},
  "vmag_ms":  {"mean": 5.1, "p50": 5.4, "p95": 10.5, "max": 41.1},
  "points_per_chan_mean": {
    "RADAR_FRONT": 64.27,
    "RADAR_FRONT_LEFT": 15.10,
    "RADAR_FRONT_RIGHT": 16.63,
    "RADAR_BACK_LEFT": 52.52,
    "RADAR_BACK_RIGHT": 54.97
  }
}
```

**怎么读这份报告**：

| 字段 | 启示 |
|---|---|
| `range_m.p95 = 111` | 95% 的点都在 111 m 内；`POINT_CLOUD_RANGE` 设到 ±50 m 就够用（远处点稀疏、噪声大） |
| `rcs_dbsm.p50 = 7` | 雷达反射强度集中在 7 dBsm 附近；过小的 RCS 多是噪声，可考虑滤掉 `< 0` 的点 |
| `points_per_chan_mean.FRONT ≈ 64` | 前向通道点最多（FOV 集中）；其他 4 个通道显著少 → 这就是为什么"5 通道 concat"能让训练更稳 |
| `vmag_ms.max = 41` | 极值来自多普勒模糊的伪速度；建议做 sanity 滤波或 clamp |

**生成的图**：
- `reports/figures/radar_points_per_channel.png` —— 每通道点数箱线图
- `reports/figures/radar_range_m.png` —— 距离直方图
- `reports/figures/radar_rcs_dbsm.png` —— RCS 直方图
- `reports/figures/radar_vmag_ms.png` —— 速度幅值直方图

### 4.5 脚本 3：GT 类别 / 距离 / 速度 分布（10 类 detection 标签）

完整代码见附录 A.3。保存为 `tools/nuscenes_analysis/03_gt_distribution.py`。

跑：

```bash
python tools/nuscenes_analysis/03_gt_distribution.py
```

**怎么读这份报告**：

- `gt_class_dist.png` —— 看到 car 一柱擎天、trailer 几乎为 0 → 类别不均衡是 nuScenes 的固有特性
- `gt_range_by_class.png` —— 看每类目标出现的距离范围 → 决定 `anchor_sizes` 和 `POINT_CLOUD_RANGE`
- `gt_speed_by_class.png` —— 看每类目标的速度分布 → 决定是否训练时预测速度

**生成的图 + json**：
- `reports/figures/gt_class_dist.png`
- `reports/figures/gt_range_by_class.png`
- `reports/figures/gt_speed_by_class.png`
- `reports/gt_distribution.json`（含每类尺寸均值，可用来调 `anchor_sizes`）

### 4.6 脚本 4：BEV 占用热力图

完整代码见附录 A.4。保存为 `tools/nuscenes_analysis/04_bev_heatmap.py`。

跑：

```bash
python tools/nuscenes_analysis/04_bev_heatmap.py
```

**怎么读**：

- 如果热力图沿 x 轴（前向）有清晰密度梯度 → 说明车主要朝前开，anchor 朝向 `[0, π/2]` 合理
- 如果热力图在 y 轴（侧向）也有明显分布 → 你可能需要更多朝向的 anchor
- 如果大部分目标集中在 ±30 m 内 → 可以把 `POINT_CLOUD_RANGE` 收窄到 ±30 m 加速训练

### 4.7 跑完后你会得到

```
reports/
├── basic_stats.json
├── radar_point_stats.json
├── gt_distribution.json
└── figures/
    ├── radar_points_per_channel.png
    ├── radar_range_m.png
    ├── radar_rcs_dbsm.png
    ├── radar_vmag_ms.png
    ├── gt_class_dist.png
    ├── gt_range_by_class.png
    ├── gt_speed_by_class.png
    └── bev_heatmap.png
```

### 4.8 重点看（训练前的"必看图"清单）

| 图 | 关键观察 | 决策 |
|---|---|---|
| `radar_points_per_channel.png` | 不同雷达天线点数差很多（前向多、侧向少） | 决定是否用 CBGS / 加权 loss |
| `radar_range_m.png` | 是否大量点 >50 m | 决定 `POINT_CLOUD_RANGE` 怎么设 |
| `gt_class_dist.png` | 10 类是否严重不均衡 | 决定是否要 `BALANCED_RESAMPLING` |
| `bev_heatmap.png` | 场景几何覆盖 | 决定 anchor 朝向 / `WORLD_ROT_ANGLE` 范围 |

---

## 5. 数据集构建（生成 infos pkl）

### 5.1 本章数据流

```
[第 4 章:统计已做]
        │
        ▼ 本章目标:把"裸数据 + 元数据"打包成 infos pkl
   ┌─────────────────────────────────────────────────────────┐
   │ create_nuscenes_radar_info (在 nuscenes_radar_dataset.py) │
   │  1) nusc = NuScenes(...)                   加载元数据    │
   │  2) get_available_radar_scenes             过滤有效场景  │
   │  3) fill_radar_infos                      遍历每个 sample│
   │     - 读 ref_chan (RADAR_FRONT) 元数据                   │
   │     - 计算 ref_from_car / car_from_global                │
   │     - 沿 prev 指针回溯 max_sweeps-1 个 sweep             │
   │     - 读 sample_annotation → gt_boxes (10 类)            │
   │  4) pickle.dump → pkl                                     │
   └─────────────────────────────────────────────────────────┘
        │
        ▼ 输出
   data/nuscenes/v1.0-mini/nuscenes_infos_radar_1sweeps_train.pkl
   data/nuscenes/v1.0-mini/nuscenes_infos_radar_1sweeps_val.pkl
        │
        ▼ 下一章:dataset 加载器会读这两个 pkl
```

### 5.2 infos pkl 的 schema（每个 info 包含什么）

```python
{
    'radar_path'        : 'samples/RADAR_FRONT/n015-...pcd',  # 主通道相对路径
    'radar_channels'    : {                                    # 5 通道相对路径
        'RADAR_FRONT'         : 'samples/RADAR_FRONT/...pcd',
        'RADAR_FRONT_LEFT'    : 'samples/RADAR_FRONT_LEFT/...pcd',
        'RADAR_FRONT_RIGHT'   : 'samples/RADAR_FRONT_RIGHT/...pcd',
        'RADAR_BACK_LEFT'     : 'samples/RADAR_BACK_LEFT/...pcd',
        'RADAR_BACK_RIGHT'    : 'samples/RADAR_BACK_RIGHT/...pcd',
    },
    'token'             : 'ca9a282c9e77460f8360f564131a8af5',  # nuScenes sample token
    'sweeps'            : [                                     # 历史 sweep 列表
        {
            'radar_path'      : 'samples/RADAR_FRONT/...pcd',
            'transform_matrix': 4x4 ndarray (从该帧到当前帧的变换),
            'time_lag'        : 0.052,  # 秒
        },
        ...
    ],
    'ref_from_car'      : 4x4 ndarray,   # 雷达坐标系 → 车体
    'car_from_global'   : 4x4 ndarray,   # global → 车体
    'timestamp'         : 1532402927.664,  # 当前帧时间戳（秒）
    'ref_chan'          : 'RADAR_FRONT',
    'gt_boxes'          : (M, 9) ndarray, # x,y,z,l,w,h,yaw,vx,vy
    'gt_names'          : array of str,    # 长度 M，对应 10 类
    'gt_boxes_velocity' : (M, 3) ndarray,
    'gt_boxes_token'    : array of str,
    'num_lidar_pts'     : (M,) int array,
    'num_radar_pts'     : (M,) int array,
}
```

### 5.3 已有脚本与调用

`pcdet/datasets/nuscenes/nuscenes_radar_dataset.py` 已经写好了 `create_nuscenes_radar_info` 函数。直接调用即可：

```bash
# 在 angle 环境 + 项目根目录
conda activate angle
cd /home/dministrator1/RadarPillar

python -m pcdet.datasets.nuscenes.nuscenes_radar_dataset \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_radar_dataset.yaml \
    --func create_nuscenes_radar_info \
    --version v1.0-mini
```

> 跑命令时 `--func` 显式传 `create_nuscenes_radar_info`（单数），与代码中 `line 358` 的 default 和 `line 362` 的 dispatch 一致。

### 5.4 已修复的坑（早期版本遗留）

#### 坑：loader 只用 1 个雷达通道（RADAR_FRONT），丢 4/5 覆盖 ✅ 已修复

- **症状**：`nuscenes_radar_dataset.py` 旧版 `get_radar_with_sweeps` 只读 `info['radar_path']`（RADAR_FRONT 单通道），完全忽略 `info['radar_channels']`（5 通道全路径）
- **影响**：5 个雷达各看一个扇区，合起来才是 360° 覆盖；只用 FRONT 直接丢 4/5 点云，**mAP 显著低于论文**
- **修复**：把当前帧的加载改为遍历 `info['radar_channels']` 读 5 个 .pcd 并 concat
- **验证**（mini 数据集前 50 帧）：
  | 指标 | fix 前 | fix 后 |
  |---|---|---|
  | 每帧平均点数 | 65.3 | **217.9** |
  | 放大倍数 | 1x | **3.34x** |
  | 输出 shape | (N, 8) | (N, 8) ✓ |

代码改动只在 `get_radar_with_sweeps` 的"当前帧"部分，sweep 部分保持单通道（当前 `MAX_SWEEPS=1` 走不到 sweep 逻辑，不受影响）。**无需重新生成 pkl**——`radar_channels` 字段在 `fill_radar_infos` 早就有，旧 pkl 直接兼容。

### 5.5 成功标志

脚本跑完你会看到类似：

```
场景总数: 10
存在雷达数据的场景数: 10
v1.0-mini: 训练场景(8), 验证场景(2)
生成 radar infos: 100%|██████████████████████████| 404/404
训练样本数: 323, 验证样本数: 81
```

并且在 `data/nuscenes/v1.0-mini/` 下生成：

```
nuscenes_infos_radar_1sweeps_train.pkl
nuscenes_infos_radar_1sweeps_val.pkl
```

> 注：本项目目前的默认版本是 mini，pkl 已存在。如果你要用 trainval，把 yaml 的 `VERSION` 改成 `v1.0-trainval` 再跑一次（耗时 1–2 小时）。

### 5.6 验证 pkl

```python
import pickle
infos = pickle.load(open('data/nuscenes/v1.0-mini/nuscenes_infos_radar_1sweeps_train.pkl','rb'))
print(len(infos), 'train samples')
print(infos[0].keys())
print('radar_channels keys:', list(infos[0]['radar_channels'].keys()))
```

### 5.7 配置 yaml 文件与 pkl 的关联

```
tools/cfgs/dataset_configs/nuscenes_radar_dataset.yaml
│
├── DATASET:        'NuScenesRadarDataset'   → 选 dataset 类
├── DATA_PATH:      'data/nuscenes'           → 根目录
├── VERSION:        'v1.0-mini'               → 子目录
├── MAX_SWEEPS:     1                         → dataset 加载时用
├── POINT_CLOUD_RANGE: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]   → 体素化范围
├── INFO_PATH: { 'train': [...train.pkl], 'test': [...val.pkl] }   → 第 5 章产物
├── POINT_FEATURE_ENCODING: { 7 字段 }       → 用 PCD 的哪几列
└── DATA_PROCESSOR: [mask + shuffle + voxel] → 数据预处理
```

---

## 6. 训练

### 6.1 本章数据流

```
[第 5 章:infos pkl]
        │
        ▼ 本章目标:启动一次训练
   ┌──────────────────────────────────────────────┐
   │ tools/train.py                                │
   │  1) 加载 cfg_file → 合并 model + dataset cfg  │
   │  2) build_dataloader(NuScenesRadarDataset)    │
   │     → 每个 iter 出一个 batch_dict              │
   │  3) model_fn_decorator:                       │
   │     - load_data_to_gpu (numpy → torch.cuda)  │
   │     - ret, tb, disp = model(batch)            │
   │     - loss = ret['loss'].mean()               │
   │  4) loss.backward() + Adam OneCycle step      │
   │  5) 每 epoch 评估 → checkpoint_best.pth       │
   └──────────────────────────────────────────────┘
        │
        ▼ 输出
   output/cfgs/nuscenes_models/radarpillar_nuscenes/<run>/
   ├── ckpt/
   │   ├── checkpoint.pth              # 最新
   │   ├── checkpoint_epoch_5.pth
   │   ├── checkpoint_epoch_10.pth
   │   └── checkpoint_best.pth         # val 指标最优
   ├── tensorboard/
   │   └── events.out.tfevents.*       # tensorboard 日志
   ├── log.txt                          # 训练过程打印
   └── eval/
       └── epoch_*/                     # 每次评估的 json 结果
```

### 6.2 单卡启动（最常用）

```bash
cd /home/dministrator1/RadarPillar
conda activate angle

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --batch_size 2 \
    --workers 2 \
    --extra_tag radarpillar_nuscenes_mini_run01
```

### 6.3 关键参数说明

| 参数 | 含义 | 推荐值（mini） |
|---|---|---|
| `--cfg_file` | 模型 + 数据集 yaml | `tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml` |
| `--batch_size` | 总 batch（单卡就是单卡 batch） | 2（mini），4–8（trainval） |
| `--workers` | DataLoader worker 数 | 2（mini），4–8（trainval） |
| `--extra_tag` | 输出目录后缀 | 自取，区分实验 |
| `--ckpt` | 断点续训时传 | 见 6.6 |
| `--epochs` | 覆盖 yaml 里的 `NUM_EPOCHS` | 默认 20 |
| `--eval_all` | 训练后立刻 eval 所有 ckpt | 调试时偶尔用 |

### 6.4 看 tensorboard

```bash
# 新开一个终端
conda activate angle
tensorboard --logdir output/cfgs/nuscenes_models/radarpillar_nuscenes/ --port 6006
# 浏览器开 http://localhost:6006
```

可关注曲线：
- `train/cls_loss`、`train/box_loss`、`train/dir_loss` → 是否同步下降
- `val/mAP` 或 `val/NDS` → 是否稳步上升
- `lr` → OneCycle 曲线是否正常

### 6.5 训练时数据流（一个 batch 的内部旅程）

```
DataLoader 取 batch
    │
    ▼
batch_dict  (CPU, numpy)
    │
    ▼  load_data_to_gpu()  ← model_fn_decorator
batch_dict  (CUDA, torch)
    │
    ▼  PointPillar.forward
    │
    ├── PillarVFE.forward(batch_dict)
    │   inputs:
    │     voxels (B, 20, 7)        ←──  DataProcessor 已把点云体素化
    │     voxel_coords (B, 4)      ←──  (batch_idx, z, y, x)
    │     voxel_num_points (B,)
    │   outputs:
    │     voxel_features (P, 9)    ←──  每个 pillar 一个 9 维特征 (x_c, y_c, z_c, x_p, y_p, z_p, x_p-x_c, y_p-y_c, z_p-z_c) + rcs/vx/vy/time
    │     voxel_coords
    │
    ├── PillarAttention.forward(voxel_features)
    │   用 key-padding mask 做 pillar 间 self-attention
    │   outputs:
    │     pillar_features (P, C=32)
    │
    ├── PointPillarScatter.forward(pillar_features, voxel_coords)
    │   outputs:
    │     spatial_features (B, 32, 320, 320)   ←── BEV 伪图
    │
    ├── BaseBEVBackbone.forward(spatial_features)
    │   3 个 2D CNN 块 + 上采样
    │   outputs:
    │     spatial_features_2d (B, 32, 320, 320)
    │
    ├── AnchorHeadSingle.forward(spatial_features_2d)
    │   outputs:
    │     cls_preds (B, num_anchors, num_classes)
    │     box_preds (B, num_anchors, 7)
    │     dir_preds (B, num_anchors, 2)
    │
    ▼
pred_dicts  +  ret_dict (含 loss)
    │
    ▼  loss = cls + 2.0*box + 0.2*dir  反向传播
```

### 6.6 断点续训

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --batch_size 2 --workers 2 \
    --extra_tag radarpillar_nuscenes_mini_run01 \
    --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/radarpillar_nuscenes_mini_run01/ckpt/checkpoint.pth \
    --pretrained_model <same_path>
```

> `--ckpt` 加载权重 + optimizer + epoch；`--pretrained_model` 只加载权重。两者都传即可无缝续训。

### 6.7 多卡训练（可选）

```bash
# 假设 2 卡
bash tools/scripts/dist_train.sh 2 \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --batch_size 4 --workers 4
```

> 多卡时 `--batch_size` 是 **单卡** batch，全局 batch = `2 × batch_size`。

### 6.8 ⚠️ mini vs trainval 的差异（必读）

| 项 | mini | trainval |
|---|---|---|
| 训练帧 | 323 | ~28k |
| 1 epoch | ~2 min（batch=2） | ~2 h（batch=8, 4 卡） |
| 推荐 epochs | 20（先跑通） | 20–40 |
| `MAX_SWEEPS` 建议 | 1（mini 没多少 sweep） | 1（默认） 或 2 |
| `BALANCED_RESAMPLING` | True | True（nuScenes 类别严重不平衡） |

> 想跑 trainval 时，把 `tools/cfgs/dataset_configs/nuscenes_radar_dataset.yaml` 的 `VERSION` 改成 `v1.0-trainval`，然后重新跑 5.3。

---

## 7. 评估与验证

### 7.1 本章数据流

```
[第 6 章:checkpoint_best.pth]
        │
        ▼ 本章目标:跑官方评估 + 解读指标
   ┌──────────────────────────────────────────────┐
   │ tools/test.py                                 │
   │  1) 加载 cfg + checkpoint                     │
   │  2) 遍历 val_loader,跑模型 → pred_dicts       │
   │  3) NuScenesRadarDataset.evaluation           │
   │     - 转 pred_dicts → nusc_annos 格式        │
   │     - 写 results_nusc_radar.json              │
   │     - 调 NuScenesEval.main(...)                │
   │     - 写 metrics_summary.json                  │
   │  4) format_nuscene_results → 终端打印         │
   └──────────────────────────────────────────────┘
        │
        ▼ 输出
   output/cfgs/nuscenes_models/radarpillar_nuscenes/<run>/eval/
   ├── results_nusc_radar.json     # 模型预测转 nuScenes 格式
   ├── metrics_summary.json        # 全部指标
   └── ...                         # 各类分布图、TP/FP 表
```

### 7.2 跑官方 nuScenes 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/<your_run>/ckpt/checkpoint_best.pth \
    --batch_size 1 --workers 2
```

评估会调用 nuScenes 官方 `NuScenesEval`，输出到：

```
output/cfgs/nuscenes_models/radarpillar_nuscenes/<your_run>/eval/
├── results_nusc_radar.json     # 模型预测转 nuScenes 格式
├── metrics_summary.json        # 全部指标
└── ...                         # 各类分布图、TP/FP 表
```

### 7.3 关键指标解读

nuScenes 官方指标（取自 `metrics_summary.json`）：

| 指标 | 含义 | 目标 |
|---|---|---|
| NDS | NuScenes Detection Score，主指标 | 越高越好 |
| mAP | mean Average Precision（各类平均） | 越高越好 |
| mATE / mASE / mAOE / mAVE / mAAE | 5 个 error 分解 | 越低越好 |

**公式**：
```
NDS = 1/10 × [5 × mAP
            + (1 − min(mATE, 1))
            + (1 − min(mASE, 1))
            + (1 − min(mAOE, 1))
            + (1 − min(mAVE, 1))
            + (1 − min(mAAE, 1))]
```

| 指标 | 含义 | radar 表现 |
|---|---|---|
| **mATE** | translation error (米) | 弱（点云稀疏） |
| **mASE** | scale error | 中 |
| **mAOE** | orientation error (弧度) | 弱（角分辨率差） |
| **mAVE** | velocity error (m/s) | **强**（多普勒直接给） |
| **mAAE** | attribute error | 中 |

> 雷达的"优势指标"是 mAVE；"短板指标"是 mAOE 和 mATE。这能解释为什么 radar-only 模型 mAP 不如 lidar，但 mAVE 反而更好。

### 7.4 验证训练结果"是否合理"的 sanity check 清单

| 检查项 | 怎么验证 | 期望 |
|---|---|---|
| Loss 在下降 | 看 tensorboard | 训练 5 个 epoch 内 cls/box 都在掉 |
| Val mAP > 0 | log 里每 epoch 评估 | mini 上 20 epoch 后约 0.05–0.20 |
| 类别 mAP 不全为 0 | metrics_summary | 至少有 car 类别 mAP > 0 |
| 没有 NaN | log 文本搜索 `nan` | 不应出现 |
| GT 速度标注正确 | 跑 4.4 脚本 | speed 直方图覆盖 0–25 m/s |
| mAVE 显著低于 mATE | metrics_summary | mAVE 应该明显比 mATE 小（雷达测速优势） |

> ⚠️ mini 上 mAP 不会高（数据太少 + 类别不平衡）。如果 mini 上 mAP > 0.2、car 类 > 0.4，就算 pipeline 通了。下一步上 trainval 才有意义。

---

## 8. 可视化与诊断

### 8.1 本章数据流

```
[第 6 章:checkpoint_best.pth]
        │
        ▼ 本章目标:打开"黑盒",看模型实际预测了什么
   ┌──────────────────────────────────────────────┐
   │ 脚本 5  BEV 可视化: GT 框(绿) + 预测框(红)    │ → reports/figures/bev_pred/frame_*.png
   │ tools/scripts/plot_loss.py                    │ → loss_curve.png
   │ tools/visualize_anchors.py                    │ → anchor_visualization.png
   └──────────────────────────────────────────────┘
```

### 8.2 BEV 真值 + 预测可视化

完整代码见附录 A.5。保存为 `tools/nuscenes_analysis/05_visualize_pred.py`。

跑：

```bash
python tools/nuscenes_analysis/05_visualize_pred.py \
    --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/<your_run>/ckpt/checkpoint_best.pth \
    --num 8
```

**怎么读图**：

| 颜色 | 含义 |
|---|---|
| 灰色散点 | 雷达点云投影到 BEV |
| **绿色框** | GT（ground truth） |
| **红色框** | 模型预测 |

观察要点：
- 绿框有但红框没 → 漏检（recall 低）
- 红框有但绿框没 → 误检（precision 低）
- 红框比绿框大/小很多 → box 回归不准
- 红框朝向不对 → dir head 没学好

### 8.3 画 loss 曲线

`tools/scripts/plot_loss.py` 已存在。直接：

```bash
python tools/scripts/plot_loss.py \
    --log-dir output/cfgs/nuscenes_models/radarpillar_nuscenes/<your_run>
```

### 8.4 Anchor 可视化（确认 anchor 大小和位置匹配 GT 尺寸分布）

```bash
python tools/visualize_anchors.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml
# 输出 tools/cfgs/nuscenes_models/anchor_visualization.png
```

> 如果 anchor 看着明显比 GT 大/小，回到 yaml 调 `anchor_sizes`（参考第 4.5 步的 GT 尺寸均值）。

---

## 9. 常见问题 FAQ

### Q1：`pip install nuscenes-devkit==1.0.5` 装不上

A：先确认 Python 版本 ≥ 3.8 且 < 3.11（devkit 1.0.5 不支持 3.11+）。换 Python：

```bash
conda install python=3.9 -y
pip install nuscenes-devkit==1.0.5
```

### Q2：训练时 `KeyError: 'radar_channels'`

A：infos pkl 是旧版本生成的，跑第 5 章重新生成一次。

### Q3：`NuScenesEval` 报 `results_nusc_radar.json not found`

A：先跑评估让模型产生预测 json；如果模型 checkpoint 路径错也会不写。检查 `--ckpt` 路径是否正确。

### Q4：Loss = nan

A：常见原因：
- 学习率太大（yaml 里 `LR: 0.0015` 改成 `0.0005` 试试）
- NaN 速度（yaml 已设 `SET_NAN_VELOCITY_TO_ZEROS: True`，确认下）
- 点云全空（`FILTER_MIN_POINTS_IN_GT` 设的过严）

### Q5：训练 1 epoch 比 mini 还慢

A：检查 `Workers` 参数；mini 数据集小，worker 不要开太多（4 足够）。还有 `BALANCED_RESAMPLING` 会让 mini 训练集放大 ~10x，batch_size 不够时会卡 IO。

### Q6：mAP 一直是 0

A：通常是 anchor 没对齐类别 / 类别不匹配：
- 检查 yaml `CLASS_NAMES` 是否等于数据集的 10 类（小写：`car`, `pedestrian`, ...）
- `map_name_from_general_to_detection` 把 nuScenes 23 类映射到这 10 类，确认你的 yaml 类名能 match
- 如果只是 truck/bus 这类 mAP = 0 正常（mini 样本太少）

### Q7：能不能直接用论文预训练权重？

A：RadarPillar 论文没公开权重。可以参考 View-of-Delft 的同结构 ckpt 做迁移：

```bash
# 加载 vod 训练好的权重（注意 voxel size / class_names 不一致，需要 --pretrained_model 不带 --ckpt）
python tools/train.py --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --pretrained_model output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth \
    --extra_tag nu_from_vod
```

> 注意：View-of-Delft 只有 3 类，nuScenes 有 10 类，最后一层 cls head 维度对不上，需要让 PCDet 自动忽略不匹配（OpenPCDet 默认行为是跳过）。

### Q8：硬盘不够

A：mini 4 GB、trainval 40 GB。mini 阶段只跑 mini 即可。

### Q9：`numba.cuda ... segmentation fault`（训练启动就崩）

A：`tools/train.py` 第 6 行 `from test import repeat_eval_ckpt` 会触发 `numba` 的 CUDA JIT 编译。在某些环境（特别是 WSL2 + 新 numba）会段错误。

**临时绕过**：删掉或注释掉 `train.py:6` 的 import（如果你的训练流程不需要 eval 复用）：

```python
# tools/train.py
# from test import repeat_eval_ckpt   ← 注释这一行
from eval_utils.eval_utils import eval_one_epoch
```

**根本修复**（二选一）：
- 降级 numba：`pip install "numba<0.58"`
- 或升级 CUDA driver 到 ≥ 535

### Q10：`pointpillar` / `spconv` import 报 `undefined symbol`

A：spconv 和 PyTorch / CUDA 版本不匹配。回到 2.5 节按 PyTorch 版本选对 `spconv-cuXXX`。

---

## 附录 A：所有脚本的完整代码

> 所有脚本独立成文件、自己负责 import，不依赖项目里其他 pcdet 代码的间接修改。

### A.1 `tools/nuscenes_analysis/01_basic_stats.py`

```python
"""nuScenes 雷达数据集基础元信息统计。

输入：nuScenes 数据集目录 + version
输出：终端报告 + reports/basic_stats.json

用法：
    python tools/nuscenes_analysis/01_basic_stats.py
    python tools/nuscenes_analysis/01_basic_stats.py --version v1.0-trainval
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from nuscenes.nuscenes import NuScenes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--out', default='reports/basic_stats.json')
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # 1) 顶层表数量
    n_scene = len(nusc.scene)
    n_sample = len(nusc.sample)
    n_sample_data = len(nusc.sample_data)
    n_anno = len(nusc.sample_annotation)
    n_instance = len(nusc.instance)

    # 2) 23 个原始类别出现频次（devkit 1.0.5 直接给出 category_name）
    name_counter = Counter(a['category_name'] for a in nusc.sample_annotation)

    # 3) 雷达文件计数：每个 sample 都应有 5 个 RADAR_*
    radar_chan = [
        'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
    ]
    n_radar_files = 0
    n_radar_missing = 0
    for s in nusc.sample:
        for ch in radar_chan:
            if ch not in s['data']:
                n_radar_missing += 1
                continue
            sd_tok = s['data'][ch]
            path, _, _ = nusc.get_sample_data(sd_tok)
            if Path(path).exists():
                n_radar_files += 1
            else:
                n_radar_missing += 1

    # 4) 每个 sample 的平均标注数
    avg_anno = round(n_anno / max(n_sample, 1), 2)

    report = {
        'version': args.version,
        'n_scene': n_scene,
        'n_sample': n_sample,
        'n_sample_data': n_sample_data,
        'n_annotation': n_anno,
        'n_instance': n_instance,
        'n_radar_files_present': n_radar_files,
        'n_radar_files_missing': n_radar_missing,
        'avg_anno_per_frame': avg_anno,
        'category_raw_dist': dict(name_counter.most_common()),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\n[ok] report saved -> {out}')


if __name__ == '__main__':
    main()
```

### A.2 `tools/nuscenes_analysis/02_radar_point_stats.py`

```python
"""雷达点云统计：点数 / 距离 / RCS / 速度。

对每个 RADAR_* 通道统计每帧点数，并汇总所有点的距离、RCS、|v| 分布。
输出图到 reports/figures/。

用法：
    python tools/nuscenes_analysis/02_radar_point_stats.py
    python tools/nuscenes_analysis/02_radar_point_stats.py --max_frames 1000
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # 无 GUI 也能画图
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


RADAR_CHANNELS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]
# 18 维 PCD 中需要用到的列
COL_X, COL_Y, COL_Z = 0, 1, 2
COL_RCS = 5
COL_VX, COL_VY = 6, 7


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--max_frames', type=int, default=200,
                    help='统计帧数上限（mini 全跑也很快，trainval 抽 200 帧即可）')
    ap.add_argument('--out', default='reports/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    n_per_chan = {ch: [] for ch in RADAR_CHANNELS}
    rng_all, rcs_all, vmag_all = [], [], []

    print(f'正在遍历前 {args.max_frames} 个关键帧 ...')
    for i, sample in enumerate(nusc.sample):
        if i >= args.max_frames:
            break
        for ch in RADAR_CHANNELS:
            sd_tok = sample['data'][ch]
            path, _, _ = nusc.get_sample_data(sd_tok)
            if not Path(path).exists():
                continue
            pc = RadarPointCloud.from_file(path)            # (18, N)
            n = pc.points.shape[1]
            n_per_chan[ch].append(n)

            xyz = pc.points[[COL_X, COL_Y, COL_Z], :]
            rng = np.linalg.norm(xyz, axis=0)
            rcs = pc.points[COL_RCS, :]
            vxy = pc.points[[COL_VX, COL_VY], :]
            vmag = np.linalg.norm(vxy, axis=0)

            rng_all.append(rng)
            rcs_all.append(rcs)
            vmag_all.append(vmag)

    rng_all = np.concatenate(rng_all)
    rcs_all = np.concatenate(rcs_all)
    vmag_all = np.concatenate(vmag_all)

    # 1) 每通道点数箱线图
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [n_per_chan[ch] for ch in RADAR_CHANNELS]
    ax.boxplot(
        data,
        tick_labels=[ch.replace('RADAR_', '') for ch in RADAR_CHANNELS],
    )
    ax.set_ylabel('points per frame')
    ax.set_title(f'Radar points per channel ({args.max_frames} frames sampled)')
    fig.tight_layout()
    fig.savefig(out_dir / 'radar_points_per_channel.png', dpi=120)
    plt.close(fig)

    # 2) 距离 / RCS / 速度 直方图
    for arr, name, xlabel in [
        (rng_all, 'range_m', 'range (m)'),
        (rcs_all, 'rcs_dbsm', 'RCS (dBsm)'),
        (vmag_all, 'vmag_ms', '|v| (m/s)'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(arr, bins=80, color='steelblue', edgecolor='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.set_title(f'{name} distribution (total {len(arr):,} pts)')
        fig.tight_layout()
        fig.savefig(out_dir / f'radar_{name}.png', dpi=120)
        plt.close(fig)

    summary = {
        'n_frames_sampled': args.max_frames,
        'range_m': {
            'mean': float(rng_all.mean()),
            'p50': float(np.percentile(rng_all, 50)),
            'p95': float(np.percentile(rng_all, 95)),
            'max': float(rng_all.max()),
        },
        'rcs_dbsm': {
            'mean': float(rcs_all.mean()),
            'p50': float(np.percentile(rcs_all, 50)),
            'p95': float(np.percentile(rcs_all, 95)),
            'max': float(rcs_all.max()),
        },
        'vmag_ms': {
            'mean': float(vmag_all.mean()),
            'p50': float(np.percentile(vmag_all, 50)),
            'p95': float(np.percentile(vmag_all, 95)),
            'max': float(vmag_all.max()),
        },
        'points_per_chan_mean': {
            ch: float(np.mean(v)) if v else 0.0
            for ch, v in n_per_chan.items()
        },
    }
    print('---- Radar point statistics ----')
    print(json.dumps(summary, indent=2))

    (out_dir.parent / 'radar_point_stats.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f'\n[ok] figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
```

### A.3 `tools/nuscenes_analysis/03_gt_distribution.py`

```python
"""GT 类别 / 距离 / 速度 / 尺寸分布（10 类 detection 标签）。

用法：
    python tools/nuscenes_analysis/03_gt_distribution.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--out', default='reports/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cls_dist: dict[str, int] = defaultdict(int)
    range_by_cls: dict[str, list[float]] = defaultdict(list)
    speed_by_cls: dict[str, list[float]] = defaultdict(list)
    size_by_cls: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for sample in nusc.sample:
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            raw_name = ann['category_name']  # devkit 1.0.5 直接给
            name = map_name_from_general_to_detection.get(raw_name, 'ignore')
            if name == 'ignore':
                continue
            cls_dist[name] += 1

            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            range_by_cls[name].append(float(np.linalg.norm(box.center[:2])))
            vel = nusc.box_velocity(ann_tok)
            if vel is not None and not np.isnan(vel).any():
                speed_by_cls[name].append(float(np.linalg.norm(vel[:2])))
            size_by_cls[name].append(tuple(box.wlh))  # (w, l, h)

    # 1) 类别直方图
    names = list(cls_dist.keys())
    counts = [cls_dist[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color='seagreen')
    ax.set_ylabel('count')
    ax.set_title('GT class distribution')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_class_dist.png', dpi=120)
    plt.close(fig)

    # 2) 每个类别的距离直方图（叠在一起）
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(range_by_cls.keys()):
        arr = np.array(range_by_cls[n])
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=50, alpha=0.45, label=f'{n}({len(arr)})')
    ax.set_xlabel('range (m)')
    ax.set_ylabel('count')
    ax.set_title('GT range by class')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_range_by_class.png', dpi=120)
    plt.close(fig)

    # 3) 每个类别的速度分布
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(speed_by_cls.keys()):
        arr = np.array(speed_by_cls[n])
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=50, alpha=0.45, label=f'{n}({len(arr)})')
    ax.set_xlabel('|v| (m/s)')
    ax.set_ylabel('count')
    ax.set_title('GT speed by class')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_speed_by_class.png', dpi=120)
    plt.close(fig)

    summary = {
        'class_count': dict(sorted(cls_dist.items(), key=lambda x: -x[1])),
        'range_m_mean': {
            n: float(np.mean(range_by_cls[n]))
            for n in names if range_by_cls[n]
        },
        'speed_mps_mean': {
            n: float(np.mean(speed_by_cls[n]))
            for n in names if speed_by_cls[n]
        },
        'size_lwh_mean': {
            n: np.mean(size_by_cls[n], axis=0).tolist()  # wlh 顺序
            for n in names if size_by_cls[n]
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    (out_dir.parent / 'gt_distribution.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f'\n[ok] figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
```

### A.4 `tools/nuscenes_analysis/04_bev_heatmap.py`

```python
"""BEV 占用热力图：把所有帧的 GT 中心画在 BEV 栅格里。

用法：
    python tools/nuscenes_analysis/04_bev_heatmap.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes

from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--range', type=float, default=50.0,
                    help='BEV 半边长 (米)')
    ap.add_argument('--out', default='reports/figures/bev_heatmap.png')
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    centers = []
    for sample in nusc.sample:
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            raw = ann['category_name']  # devkit 1.0.5 直接给
            if map_name_from_general_to_detection.get(raw, 'ignore') == 'ignore':
                continue
            centers.append(ann['translation'][:2])  # x, y
    centers = np.array(centers)

    R = args.range
    H, _, _ = np.histogram2d(
        centers[:, 0], centers[:, 1],
        bins=[100, 100],
        range=[[-R, R], [-R, R]],
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        H.T, origin='lower', extent=[-R, R, -R, R],
        cmap='hot', interpolation='bilinear',
    )
    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('y (m, left)')
    ax.set_title(f'GT center BEV heatmap ({len(centers)} boxes, range=±{R}m)')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[ok] heatmap saved -> {out}')


if __name__ == '__main__':
    main()
```


### A.5 `tools/nuscenes_analysis/05_visualize_pred.py`

```python
"""可视化验证集预测：BEV 上画 GT 框 (绿) 和预测框 (红)。

依赖：训练完的 checkpoint_best.pth
用法：
    python tools/nuscenes_analysis/05_visualize_pred.py \
        --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/<run>/ckpt/checkpoint_best.pth \
        --num 8
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def box_corners_to_bev(box):
    """box: (x,y,z,l,w,h,yaw) -> BEV 矩形的 (cx,cy,w,h,yaw)。"""
    x, y = float(box[0]), float(box[1])
    l, w = float(box[3]), float(box[4])
    yaw = float(box[6])
    return x, y, w, l, yaw


def draw_box(ax, box, color, lw=1.2):
    x, y, w, l, yaw = box_corners_to_bev(box)
    rect = Rectangle((-l / 2, -w / 2), l, w, fill=False, edgecolor=color, linewidth=lw)
    t = Affine2D().rotate_around(0, 0, yaw) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    ax.plot([x], [y], '+', color=color, markersize=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg_file', default='tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num', type=int, default=4)
    ap.add_argument('--out', default='reports/figures/bev_pred')
    ap.add_argument('--score_thresh', type=float, default=0.1)
    ap.add_argument('--range_limit', type=float, default=50.0)
    args = ap.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    logger.info(f'Loaded config: {args.cfg_file}')

    _, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=val_loader.dataset,
    )
    model.load_params_from_file(args.ckpt, logger=logger)
    model.cuda()
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = args.range_limit
    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if saved >= args.num:
                break
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda(non_blocking=True)
            batch['points'] = [b.cuda() for b in batch['points']]

            pred_dicts, _ = model(batch)

            for j in range(len(pred_dicts)):
                if saved >= args.num:
                    break

                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_xlim(-R, R)
                ax.set_ylim(-R, R)
                ax.set_aspect('equal')
                ax.set_xlabel('x (m, forward)')
                ax.set_ylabel('y (m, left)')

                # 雷达点云（投影到 BEV）
                pts = batch['points'][j][:, :3].cpu().numpy()
                ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c='gray', alpha=0.5)

                # GT
                gt_boxes = batch['gt_boxes'][j].cpu().numpy()
                gt_names = batch['gt_names'][j]
                for b, n in zip(gt_boxes, gt_names):
                    draw_box(ax, b, color='green', lw=1.5)

                # Pred
                pb = pred_dicts[j]['pred_boxes'].cpu().numpy()
                ps = pred_dicts[j]['pred_scores'].cpu().numpy()
                pl = pred_dicts[j]['pred_labels'].cpu().numpy()
                for b, s, l_idx in zip(pb, ps, pl):
                    if s < args.score_thresh:
                        continue
                    draw_box(ax, b, color='red', lw=1.0)
                    cx = float(b[0])
                    cy = float(b[1])
                    name = cfg.CLASS_NAMES[int(l_idx) - 1]
                    ax.text(cx, cy, f'{name}\n{s:.2f}', color='red', fontsize=6)

                token = batch['metadata'][j].get('token', f'idx_{i}')
                ax.set_title(f'frame {saved}  token={token[:8]}  GT(green) vs Pred(red)')
                fig.tight_layout()
                fig.savefig(out_dir / f'frame_{saved:03d}.png', dpi=120)
                plt.close(fig)
                saved += 1

    print(f'[ok] saved {saved} frames -> {out_dir}/')


if __name__ == '__main__':
    main()
```

---

## 附录 B：完整命令速查

```bash
# === 一次性准备 ===
conda activate angle
pip install -r requirements.txt
pip install nuscenes-devkit==1.0.5 spconv-cu120==2.3.6
python setup.py develop

# === 验证数据 ===
ls -la data/nuscenes/v1.0-mini/samples/RADAR_FRONT/ | head -3

# === 数据分析（可选但强烈建议） ===
python tools/nuscenes_analysis/01_basic_stats.py
python tools/nuscenes_analysis/02_radar_point_stats.py
python tools/nuscenes_analysis/03_gt_distribution.py
python tools/nuscenes_analysis/04_bev_heatmap.py

# === 修 bug（早期版本必须，当前代码已修复，详见 5.4） ===
# 1) nuscenes_radar_dataset.py --func 默认值拼写（已修复）
# 2) nuscenes_radar_dataset.py:155-201 改用 5 通道（已修复，pkl 兼容）

# === 生成 infos ===
python -m pcdet.datasets.nuscenes.nuscenes_radar_dataset \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_radar_dataset.yaml \
    --func create_nuscenes_radar_info \
    --version v1.0-mini

# === 训练 ===
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --batch_size 2 --workers 2 --extra_tag run01

# === 看 tensorboard ===
tensorboard --logdir output/cfgs/nuscenes_models/radarpillar_nuscenes/run01 --port 6006

# === 评估 ===
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/run01/ckpt/checkpoint_best.pth

# === 可视化预测 ===
python tools/nuscenes_analysis/05_visualize_pred.py \
    --ckpt output/cfgs/nuscenes_models/radarpillar_nuscenes/run01/ckpt/checkpoint_best.pth \
    --num 8
```

---

## 附录 C：文件清单（这次操作涉及/新增的文件）

| 路径 | 状态 | 说明 |
|---|---|---|
| `tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml` | 已存在 | 模型 + 训练配置 |
| `tools/cfgs/dataset_configs/nuscenes_radar_dataset.yaml` | 已存在 | 数据集配置 |
| `pcdet/datasets/nuscenes/nuscenes_radar_dataset.py` | 已存在 | 数据集类（5 通道 loader 已修复，详见 5.4） |
| `pcdet/datasets/nuscenes/nuscenes_radar_utils.py` | 已存在 | infos 生成 |
| `tools/nuscenes_analysis/01_basic_stats.py` | **待新建** | 基础元信息统计（代码见附录 A.1） |
| `tools/nuscenes_analysis/02_radar_point_stats.py` | **待新建** | 雷达点云统计 + 画图（代码见附录 A.2） |
| `tools/nuscenes_analysis/03_gt_distribution.py` | **待新建** | GT 类别 / 距离 / 速度（代码见附录 A.3） |
| `tools/nuscenes_analysis/04_bev_heatmap.py` | **待新建** | BEV 占用热力图（代码见附录 A.4） |
| `tools/nuscenes_analysis/05_visualize_pred.py` | **待新建** | BEV GT vs Pred（代码见附录 A.5） |
| `reports/` | **待新建目录** | 统计输出（json + png） |

---

## 附录 D：接下来要做什么（建议你按这个顺序推进）

1. **跑通 mini**（半天）：第 2 → 5.3 → 6.2 → 7.2，跑通即胜利
2. **做数据统计分析**（半天）：第 4 节，画图、看分布
3. ~~**修 bug + 5 通道**（半天）：第 5.4 节~~（已修复，可跳过）
4. **调 anchor / 数据增强**（1 天）：参考 `experiments/RESULTS.md`
5. **上 trainval**（2 天）：改 yaml 的 VERSION 即可
6. **接 tracker**（之后）：用 `tracker/` 接检测结果做完整 pipeline

祝训练顺利 🚀