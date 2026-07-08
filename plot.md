# 训练后可视化实现指导

## 需求概述

`train.py` 训练完成后，输出可视化窗口，包含三部分：

| 区域 | 内容 |
|------|------|
| 左侧 | BEV 俯视图：雷达点云 + GT boxes（实线）+ 预测 boxes（虚线），不同类别不同颜色 |
| 右侧 | 6 个相机图像：3D box 投影到图像，按前/后/左/右方位排列 |
| 附加 | Loss 曲线图（训练过程中各 loss 分量随 epoch 变化） |

---

## 一、文件规划

```
tools/
  train.py                          # 修改：训练结束后调用可视化
  utils/visual_utils/
    visualize_results.py            # 新增：主可视化脚本（BEV + 图像）
    visualize_loss.py               # 已有：loss 曲线（直接复用）
    visualize_utils.py              # 已有：boxes_to_corners_3d 等工具函数
```

**新增 1 个文件**：`tools/utils/visual_utils/visualize_results.py`
**修改 1 个文件**：`tools/train.py`

---

## 二、train.py 修改

在 `repeat_eval_ckpt` 调用结束后（line 273 之后），加一段：

```python
# ---- 训练后可视化 ----
if not args.skip_visualize:
    from utils.visual_utils.visualize_results import visualize_res
    visualize_res(
        cfg=cfg,
        model=model.module if dist_train else model,
        output_dir=output_dir,
        dataroot=str(cfg.DATA_CONFIG.DATA_PATH),
        version=cfg.DATA_CONFIG.VERSION,
        class_names=cfg.CLASS_NAMES,
        score_thresh=0.1,
        max_frames=args.vis_num,
        logger=logger,
    )
```

新增命令行参数：

```python
parser.add_argument('--skip_visualize', action='store_true', help='skip post-training visualization')
parser.add_argument('--vis_num', type=int, default=50, help='number of frames to visualize')
```

> **注意**：可视化放在 eval 之后，这样 `eval_output_dir` 已确定；传入 `model.module if dist_train else model` 解包 DDP。
> 不再绑定 `skip_eval` 条件——即使跳过 eval 也可单独跑可视化。

---

## 三、visualize_results.py 实现

### 3.1 脚本入口

```python
"""
训练后可视化：BEV 点云 + GT/Pred boxes + 多相机图像 + loss 曲线
"""
import os
import glob
import numpy as np
import matplotlib
# 不设 Agg，保持默认交互式后端（TkAgg / Qt5Agg），plt.show() 才能弹窗
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

import torch
from pcdet.datasets import build_dataloader, NuScenesRadarDataset
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.box_utils import boxes_to_corners_3d

# 根据类别列表自动生成颜色（通用方案，支持任意数据集）
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2',  # 蓝, 橙, 紫, 棕, 粉
    '#17becf', '#bcbd22', '#d62728', '#ecef1a', '#2ca02c',  # 青, 黄绿, 红, 黄, 绿
    '#ff9896', '#aec7e8', '#ffbb78', '#c5b0d5', '#c49c94',  # 更多颜色
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#ff7f0e',
]


def _generate_class_colors(class_names):
    """根据类别列表自动生成颜色映射"""
    return {name: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, name in enumerate(class_names)}


def visualize_res(cfg, model, output_dir, dataroot, version, class_names,
                  score_thresh=0.1, max_frames=50, logger=None):
    """
    train.py 训练完成后调用此函数

    Args:
        cfg: EasyDict 配置对象
        model: 已加载权重的模型（非 DDP 包装）
        output_dir: Path, 训练输出目录（含 ckpt/ 子目录）
        dataroot: str, nuScenes 数据根目录
        version: str, 如 'v1.0-trainval'
        class_names: list[str], 类别名称列表
        score_thresh: float, 预测框过滤阈值
        max_frames: int, 最多可视化帧数
        logger: Logger
    """
    if logger:
        logger.info('==== 开始训练后可视化 ====')

    # ---- 根据类别列表生成颜色映射 ----
    class_colors = _generate_class_colors(class_names)

    # ---- 动态判断数据集类型并初始化 nuScenes devkit ----
    nusc = None
    if version and 'nuscenes' in version.lower():
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # ---- 构建 test dataloader ----
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=class_names,
        batch_size=1,           # 可视化逐帧展示，batch=1 即可
        dist=False, workers=2, logger=logger, training=False
    )

    # ---- 加载 best checkpoint ----
    best_ckpt = str(output_dir / 'ckpt' / 'checkpoint_best.pth')
    if not os.path.isfile(best_ckpt):
        # 回退：按修改时间取最新的 checkpoint
        ckpt_list = glob.glob(str(output_dir / 'ckpt' / '*checkpoint_epoch_*.pth'))
        if ckpt_list:
            ckpt_list.sort(key=os.path.getmtime)
            best_ckpt = ckpt_list[-1]
            if logger:
                logger.info(f'checkpoint_best.pth 不存在，回退使用 {best_ckpt}')
        else:
            if logger:
                logger.warning('未找到任何 checkpoint，跳过可视化')
            return

    model.load_params_from_file(filename=best_ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # ---- 先画 loss 曲线（单独 figure，与 BEV+相机窗口同时显示） ----
    from utils.visual_utils.visualize_loss import parse_log, visualize_loss
    log_files = glob.glob(str(output_dir / 'log_train_*.txt'))
    if log_files:
        log_path = Path(sorted(log_files)[-1])  # 取最新的 log
        steps, epoch_sorted = parse_log(log_path)
        visualize_loss(steps, epoch_sorted,
                       out=output_dir / 'loss_curve.png')
        # loss 曲线已保存为 PNG；如需弹窗显示可取消下面的注释
        # plt.figure('Loss Curve')
        # ... 重新绘制交互式版本
    else:
        if logger:
            logger.warning('未找到训练日志，跳过 loss 曲线')

    # ---- 逐帧推理 + 可视化（BEV + 相机窗口） ----
    with torch.no_grad():
        for i, batch_dict in enumerate(test_loader):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            draw_frame(batch_dict, pred_dicts, nusc, dataroot, class_names, class_colors, score_thresh)
            if i >= max_frames - 1:
                break

    plt.show()  # 保持所有 figure 窗口打开
    if logger:
        logger.info('==== 可视化结束 ====')
```

### 3.2 核心绘图 `draw_frame()`

```python
def draw_frame(batch_dict, pred_dicts, nusc, dataroot, class_names, class_colors, score_thresh=0.1):
    """
    绘制单帧：左侧 BEV + 右侧相机图像

    Args:
        batch_dict: dataloader 输出的单 batch（batch_size=1）
        pred_dicts: list[dict], 模型预测结果，长度 = batch_size
        nusc: NuScenes 对象，None 时跳过相机
        dataroot: str, 数据根目录
        class_names: list[str]
        class_colors: dict[str, str]
        score_thresh: float
    """
    # ---- 取数据 ----
    # points: collate 后 shape (total_N, 8)，第一列为 batch_idx
    points = batch_dict['points'].cpu().numpy()
    pts_single = points[points[:, 0] == 0, 1:]  # 取 batch 0 的点，去掉 batch_idx 列 → (N, 7)

    # gt_boxes: shape (B, max_M, C)，C 取决于 PRED_VELOCITY
    #   PRED_VELOCITY=True  → C=10: [x,y,z,dx,dy,dz,yaw,vx,vy,cls_id]
    #   PRED_VELOCITY=False → C=8:  [x,y,z,dx,dy,dz,yaw,cls_id]
    gt_boxes_raw = batch_dict['gt_boxes'][0].cpu().numpy()   # (max_M, C)
    # 过滤全零行（padding）
    nonzero_mask = np.any(gt_boxes_raw != 0, axis=1)
    gt_boxes_raw = gt_boxes_raw[nonzero_mask]

    # 提取 box 7 维 + cls_id（始终在最后一列）
    gt_boxes_7 = gt_boxes_raw[:, :7]    # [x,y,z,dx,dy,dz,yaw]
    gt_cls_ids = gt_boxes_raw[:, -1]    # cls_id 在最后一列

    # pred_dicts[0] 包含 'pred_boxes'(M,7), 'pred_scores'(M,), 'pred_labels'(M,)
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()    # (M, 7)
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()  # (M,)
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # (M,), 1-indexed

    # metadata: collate 后为 numpy array of dicts
    meta = batch_dict['metadata'][0]
    token = meta['token'] if isinstance(meta, dict) else meta.item()['token']

    # ---- 创建 figure ----
    fig = plt.figure(figsize=(28, 9))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.05)

    # ==== 左侧：BEV ====
    ax_bev = fig.add_subplot(outer[0])
    draw_bev(ax_bev, pts_single, gt_boxes_7, gt_cls_ids,
             pred_boxes, pred_scores, pred_labels,
             class_names, class_colors, score_thresh)

    # ==== 右侧：6 个相机 ====
    if nusc is not None:
        cam_channels = [
            ['CAM_FRONT_LEFT',  'CAM_FRONT',       'CAM_FRONT_RIGHT'],
            ['CAM_BACK_LEFT',   'CAM_BACK',        'CAM_BACK_RIGHT'],
        ]
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[1], hspace=0.1, wspace=0.05)
        for row_idx, row in enumerate(cam_channels):
            for col_idx, ch in enumerate(row):
                ax_cam = fig.add_subplot(gs_right[row_idx, col_idx])
                draw_camera(ax_cam, ch, token, nusc, dataroot,
                            gt_boxes_7, gt_cls_ids,
                            pred_boxes, pred_scores, pred_labels,
                            class_names, class_colors, score_thresh)
    else:
        ax_right = fig.add_subplot(outer[1])
        ax_right.text(0.5, 0.5, 'nuScenes devkit not available\n(no camera images)',
                      ha='center', va='center', fontsize=14)
        ax_right.axis('off')

    plt.suptitle(f'Sample: {token[:8]}...', fontsize=12)
    plt.waitforbuttonpress(timeout=60)  # 按键翻页，60s 超时自动下一帧
    plt.close(fig)
```

### 3.3 BEV 绘图 `draw_bev()`

```python
def draw_bev(ax, points, gt_boxes, gt_cls_ids,
             pred_boxes, pred_scores, pred_labels,
             class_names, class_colors, score_thresh):
    """
    Args:
        points: (N, 7) radar 点云
        gt_boxes: (M, 7) [x,y,z,dx,dy,dz,yaw]
        gt_cls_ids: (M,) int, 1-indexed class id
        pred_boxes: (K, 7) [x,y,z,dx,dy,dz,yaw]
        pred_scores: (K,) float
        pred_labels: (K,) int, 1-indexed class id
        class_names: list[str]
        class_colors: dict[str, str]
        score_thresh: float
    """
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('BEV View')

    # 画雷达点云
    ax.scatter(points[:, 0], points[:, 1], s=1, c='gray', alpha=0.3)

    # 收集所有 x/y 用于自适应范围
    all_x = list(points[:, 0])
    all_y = list(points[:, 1])

    # 画 GT boxes（实线）
    for box, cls_id in zip(gt_boxes, gt_cls_ids):
        cls_name = class_names[int(cls_id) - 1]  # label 是 1-indexed
        color = class_colors.get(cls_name, 'white')
        corners = boxes_to_corners_3d(box[None, :7])  # (1, 8, 3)
        _draw_bev_box(ax, corners[0], color, linestyle='-')
        all_x.extend(corners[0, :4, 0].tolist())
        all_y.extend(corners[0, :4, 1].tolist())

    # 画 Pred boxes（虚线），过滤低分
    for box, score, cls_id in zip(pred_boxes, pred_scores, pred_labels):
        if score < score_thresh:
            continue
        cls_name = class_names[int(cls_id) - 1]
        color = class_colors.get(cls_name, 'white')
        corners = boxes_to_corners_3d(box[None, :7])
        _draw_bev_box(ax, corners[0], color, linestyle='--')
        all_x.extend(corners[0, :4, 0].tolist())
        all_y.extend(corners[0, :4, 1].tolist())

    # 图例
    legend_elements = [
        Line2D([0], [0], color='white', linestyle='-', label='GT'),
        Line2D([0], [0], color='white', linestyle='--', label='Pred'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 自适应范围
    margin = 5
    if all_x:
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)


def _draw_bev_box(ax, corners, color, linestyle='-'):
    """
    corners: (8, 3) 3D 角点 → BEV 底面 4 点连线（已包含 heading 旋转）
    角点顺序（boxes_to_corners_3d）:
      0: (+x, +y, -z)  1: (+x, -y, -z)  2: (-x, -y, -z)  3: (-x, +y, -z)  ← 底面
      4: (+x, +y, +z)  5: (+x, -y, +z)  6: (-x, -y, +z)  7: (-x, +y, +z)  ← 顶面
    """
    box_2d = corners[:4, :2]  # 底面 4 角点的 x, y
    box_2d = np.vstack([box_2d, box_2d[0]])  # 闭合
    ax.plot(box_2d[:, 0], box_2d[:, 1], color=color, linestyle=linestyle, linewidth=1.5)
```

### 3.4 相机图像 + 3D Box 投影 `draw_camera()`

```python
def draw_camera(ax, channel, token, nusc, dataroot,
                gt_boxes, gt_cls_ids,
                pred_boxes, pred_scores, pred_labels,
                class_names, class_colors, score_thresh):
    """在 ax 上画相机图像 + 投影的 3D box

    Args:
        gt_boxes: (M, 7) [x,y,z,dx,dy,dz,yaw]  ego 坐标系
        gt_cls_ids: (M,) 1-indexed
        pred_boxes: (K, 7) ego 坐标系
        pred_scores: (K,)
        pred_labels: (K,) 1-indexed
    """
    sample = nusc.get('sample', token)
    cam_token = sample['data'][channel]
    cam_data = nusc.get('sample_data', cam_token)

    # 加载图像
    img_path = os.path.join(dataroot, cam_data['filename'])
    if not os.path.isfile(img_path):
        ax.text(0.5, 0.5, f'{channel}\nimage not found', ha='center', va='center')
        ax.axis('off')
        return
    img = plt.imread(img_path)

    # 获取标定参数
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    intrinsic = np.array(cs_record['camera_intrinsic'])  # (3, 3)

    # ego → sensor 变换矩阵
    sensor_from_ego = _get_sensor_from_ego(cs_record)

    ax.imshow(img)
    ax.set_title(channel.replace('CAM_', ''), fontsize=8)
    ax.axis('off')

    # 投影 GT boxes（实线）
    for box, cls_id in zip(gt_boxes, gt_cls_ids):
        cls_name = class_names[int(cls_id) - 1]
        color = class_colors.get(cls_name, 'white')
        corners_2d, depths = _project_box(box[:7], intrinsic, sensor_from_ego)
        _draw_box_on_image(ax, corners_2d, depths, color, linestyle='-')

    # 投影 Pred boxes（虚线）
    for box, score, cls_id in zip(pred_boxes, pred_scores, pred_labels):
        if score < score_thresh:
            continue
        cls_name = class_names[int(cls_id) - 1]
        color = class_colors.get(cls_name, 'white')
        corners_2d, depths = _project_box(box[:7], intrinsic, sensor_from_ego)
        _draw_box_on_image(ax, corners_2d, depths, color, linestyle='--')
```

### 3.5 坐标变换函数

```python
def _get_sensor_from_ego(cs_record):
    """
    计算 ego 坐标系 → sensor 坐标系的 4x4 变换矩阵

    仅使用 calibrated_sensor 记录，因为：
    - GT/Pred boxes 已在 ego 坐标系（nuscenes_radar_utils.py 中 _box_global_to_ego 已完成转换）
    - sensor_to_ego = (R_sensor_to_ego | t_sensor_to_ego) 是固定的传感器安装参数
    - ego → sensor = inv(sensor → ego)，不涉及 ego_pose

    注意：此变换假设 box 的 ego 时间戳与相机 ego 时间戳一致。
    nuScenes 中不同传感器的 ego_pose 时间戳可能有毫秒级差异，
    对于静止/慢速场景影响可忽略。若需精确对齐，需额外做 ego-motion 补偿。
    """
    from pyquaternion import Quaternion

    # 构建 sensor → ego 的 4x4 刚体变换
    rot = Quaternion(cs_record['rotation']).rotation_matrix          # (3, 3)
    trans = np.array(cs_record['translation'])                       # (3,)
    sensor_to_ego = np.eye(4)
    sensor_to_ego[:3, :3] = rot
    sensor_to_ego[:3, 3] = trans

    # ego → sensor = inverse(sensor → ego)
    sensor_from_ego = np.linalg.inv(sensor_to_ego)
    return sensor_from_ego


def _project_box(box_3d, intrinsic, sensor_from_ego):
    """
    将 ego 坐标系下的 3D box 投影到相机图像

    Args:
        box_3d: (7,) [x, y, z, dx, dy, dz, heading]，ego 坐标系
        intrinsic: (3, 3) 相机内参矩阵
        sensor_from_ego: (4, 4) ego → sensor 变换矩阵

    Returns:
        corners_2d: (8, 2) 图像像素坐标
        depths: (8,) 各角点在相机坐标系下的 z 值（深度）
    """
    corners_3d = boxes_to_corners_3d(box_3d[None])  # (1, 8, 3)
    corners_3d = corners_3d[0]  # (8, 3)

    # ego → sensor
    corners_homo = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)
    corners_cam = (sensor_from_ego @ corners_homo.T).T       # (8, 4)

    depths = corners_cam[:, 2].copy()

    # 过滤 z <= 0 的点（在相机后方），避免投影异常
    valid = corners_cam[:, 2] > 0.1
    corners_cam[~valid, 2] = 1e-6  # 避免除零

    # 投影到图像：u = fx * Xc / Zc + cx,  v = fy * Yc / Zc + cy
    corners_2d = (intrinsic @ corners_cam[:, :3].T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    return corners_2d, depths
```

### 3.6 图像上画 3D 框

```python
def _draw_box_on_image(ax, corners_2d, depths, color, linestyle='-'):
    """
    在图像上画 3D 投影框

    Args:
        corners_2d: (8, 2) 图像像素坐标
        depths: (8,) 深度，用于判断是否在相机前方

    角点顺序（boxes_to_corners_3d）:
      底面: 0-1-2-3
      顶面: 4-5-6-7
      竖边: 0-4, 1-5, 2-6, 3-7
    """
    # nuScenes 相机图像尺寸：1600×900，用于裁剪超出图像范围的线段
    img_w, img_h = 1600, 900

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
    ]
    for i, j in lines:
        if depths[i] > 0.1 and depths[j] > 0.1:  # 两端都在相机前方
            ax.plot([corners_2d[i, 0], corners_2d[j, 0]],
                    [corners_2d[i, 1], corners_2d[j, 1]],
                    color=color, linestyle=linestyle, linewidth=1.5)
```

---

## 四、Loss 曲线

项目已有 `tools/utils/visual_utils/visualize_loss.py`，直接复用：

```python
from utils.visual_utils.visualize_loss import parse_log, visualize_loss

log_path = Path(sorted(glob.glob(str(output_dir / 'log_train_*.txt')))[-1])
steps, epoch_sorted = parse_log(log_path)
visualize_loss(steps, epoch_sorted, out=output_dir / 'loss_curve.png')
```

**注意事项**：
- `visualize_loss.py` 内部使用了 `matplotlib.use("Agg")`，调用 `visualize_loss` 后当前进程的
  matplotlib 后端已切换为 Agg（非交互式）。后续 BEV/相机窗口需要交互式后端。
- 解决方案：在 `visualize_results.py` 中先 import matplotlib 并**不设 Agg**，
  然后在调用 `visualize_loss` 前临时切换后端，调用后切回交互式：

  ```python
  import matplotlib
  backend_prev = matplotlib.get_backend()
  matplotlib.use('Agg')             # visualize_loss 需要 Agg 来 savefig
  from utils.visual_utils.visualize_loss import parse_log, visualize_loss
  # ... 调用 parse_log / visualize_loss ...
  matplotlib.use(backend_prev)      # 切回交互式后端
  import matplotlib.pyplot as plt   # 重新 import 以生效
  ```

  或更简单的方案：直接复用 `parse_log` 提取数据，自己画 loss 曲线（不调用 `visualize_loss`），
  避免 Agg 后端冲突。

如需嵌入同一窗口，改为 3 列布局（`width_ratios=[1, 1.4, 0.8]`），右侧画 loss。

---

## 五、调用方式

### 方式 A：集成到 train.py

```bash
python tools/train.py --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml
# 训练 + eval 结束后自动弹出可视化窗口 + 保存 loss 曲线
```

跳过可视化：
```bash
python tools/train.py --cfg_file ... --skip_visualize
```

### 方式 B：独立脚本

```bash
python tools/utils/visual_utils/visualize_results.py \
    --cfg_file tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml \
    --ckpt output/.../ckpt/checkpoint_best.pth \
    --data_path data/nuscenes/v1.0-trainval \
    --version v1.0-trainval \
    --vis_num 50
```

独立脚本需自行处理模型加载、dataloader 构建、nuScenes devkit 初始化。
参考 `tools/demo.py` 的写法。

---

## 六、关键依赖与复用

| 依赖 | 位置 | 说明 |
|------|------|------|
| `boxes_to_corners_3d()` | `pcdet/utils/box_utils.py:27` | 3D box → 8 角点，BEV 和图像投影都要用 |
| `build_dataloader()` | `pcdet/datasets/__init__.py` | 构建 dataloader，签名：`(dataset_cfg, class_names, batch_size, dist, workers, logger, training, ...)` |
| `load_data_to_gpu()` | `pcdet/models/__init__.py` | 将 batch_dict 中的 tensor 搬到 GPU |
| `NuScenesRadarDataset` | `pcdet/datasets/nuscenes/nuscenes_radar_dataset.py` | 数据加载 |
| `parse_log() / visualize_loss()` | `tools/utils/visual_utils/visualize_loss.py` | Loss 曲线；`parse_log(path: Path) → (steps, epoch_sorted)` |
| `NuScenes` | `nuscenes.nuscenes` | 获取相机数据、标定参数 |
| `Quaternion` | `pyquaternion` | 构建旋转变换矩阵 |

---

## 七、注意事项

### 7.1 radar info 不存相机路径

`nuscenes_radar_utils.py:fill_radar_infos()` 只存了 radar 通道，没有相机信息。
**必须运行时通过 nuScenes devkit 动态获取**：

```python
sample = nusc.get('sample', token)
cam_token = sample['data']['CAM_FRONT']  # 6 个通道都可用
```

### 7.2 ego → camera 变换

info 中的 `ref_from_car` 是 ego → radar sensor，不是 ego → camera。
相机变换必须从 devkit 的 `calibrated_sensor` 重建：

```python
# calibrated_sensor 记录定义了 sensor → ego 的刚性安装参数
sensor_to_ego = np.eye(4)
sensor_to_ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
sensor_to_ego[:3, 3] = cs_record['translation']
sensor_from_ego = np.linalg.inv(sensor_to_ego)
```

**不需要 `ego_pose`**：因为 GT/Pred boxes 已在 ego 坐标系，
而 `calibrated_sensor` 定义了 sensor↔ego 的固定变换（传感器安装参数），
`ego_pose` 仅用于 global↔ego 转换。

### 7.3 gt_boxes 格式

gt_boxes 的列数取决于 `PRED_VELOCITY` 配置：
- `PRED_VELOCITY=True`（默认）：`(M, 10)` → `[x,y,z,dx,dy,dz,yaw,vx,vy,cls_id]`，cls_id 在 index 9
- `PRED_VELOCITY=False`：`(M, 8)` → `[x,y,z,dx,dy,dz,yaw,cls_id]`，cls_id 在 index 7

**cls_id 始终在最后一列（`gt_boxes[:, -1]`）**，box 7 维始终在 `gt_boxes[:, :7]`。
不要硬编码 `box[7]` 作为 cls_id。

### 7.4 boxes_to_corners_3d 角点顺序

返回 (N, 8, 3)：
- 0-3: 底面四角（z = center_z - dz/2）
- 4-7: 顶面四角（z = center_z + dz/2）
- BEV 用 `corners[:4, :2]`
- 图像投影用全部 8 点

角点详细排列（旋转前）：
```
    7 -------- 4
   /|         /|
  6 -------- 5 .
  | |        | |
  . 3 -------- 0
  |/         |/
  2 -------- 1
```
0: (+x,+y,-z), 1: (+x,-y,-z), 2: (-x,-y,-z), 3: (-x,+y,-z)

### 7.5 points 的 batch_idx

collate 后 `batch_dict['points']` shape 为 `(total_N, 8)`，
第一列是 batch_idx。取单样本点云时需过滤：
```python
pts = batch_dict['points'].cpu().numpy()
pts_single = pts[pts[:, 0] == 0, 1:]  # (N, 7)
```

### 7.6 matplotlib 后端冲突

`visualize_loss.py` 内部 `matplotlib.use("Agg")` 会将全局后端切换为非交互式。
若在同一进程中既调用 `visualize_loss` 又需要 `plt.show()` 弹窗，
必须在调用 `visualize_loss` 后手动切换回交互式后端，
或直接用 `parse_log` 提取数据、自行绘制 loss 曲线。

### 7.7 可选：用 nuScenes devkit 自带投影

`nusc.get_sample_data()` 已内置 box 投影逻辑，可直接参考或调用：

```python
from nuscenes.utils.data_classes import Box
boxes = [Box(center, size, Quaternion(yaw)) for ...]
_, boxes_in_cam, _ = nusc.get_sample_data(cam_token, boxes=boxes)
```

### 7.8 交互浏览

```python
# 按键翻页
plt.waitforbuttonpress(timeout=60)  # 60s 超时自动下一帧

# 或用 pause
plt.pause(0.5)
```

---

## 八、验证清单

- [ ] BEV 图：点云散点可见，范围自适应
- [ ] GT boxes 实线、Pred boxes 虚线，颜色区分正确
- [ ] 6 个相机图像按方位排列（上前左/前/前右，下后左/后/后右）
- [ ] 3D box 投影到图像上位置合理
- [ ] 相机后方的 box 不被绘制（depths > 0 过滤生效）
- [ ] gt_boxes 的 cls_id 取自最后一列（`[:, -1]`），兼容 PRED_VELOCITY 开关
- [ ] points 取单样本时正确过滤 batch_idx 列
- [ ] Loss 曲线正常生成
- [ ] 键盘翻页可逐帧浏览
- [ ] `model.eval()` + `torch.no_grad()` 已包裹推理循环
- [ ] `load_data_to_gpu()` 在推理前调用
- [ ] matplotlib 后端不冲突（交互式 + Agg 共存处理）
