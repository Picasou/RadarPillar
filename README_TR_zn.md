<div align="center">

# RadarPillars：View-of-Delft 数据集上的复现（土耳其语版译本）

**仅使用雷达（radar-only）的 3D 目标检测 —— 基于 OpenPCDet 对 [Gillen 等人, IROS 2024](https://arxiv.org/abs/2408.05020) 工作的复现**

</div>

> 本文档为 `README_TR.md`（土耳其语原文）的中文译本。译文遵循技术文档风格：英文专业术语（mAP、BEV、checkpoint、anchor、augmentation 等）保留原文，仅在叙述性内容上译为中文。

---

## 核心结果一览

| 方法 | 汽车 | 行人 | 骑车人 | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **本工作 — 最优随机种子** | **41.6** | **44.8** | 71.3 | **52.56** |
| 本工作 — 3 种子均值 | 41.0 | 43.2 | 70.1 | 51.43 ± 0.99 |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| **RadarPillars（原文）** | 41.1 | 38.6 | 72.6 | **50.70** |
| CenterPoint 基线 | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars 基线 | 37.9 | 31.2 | 65.7 | 45.0 |

VoD 验证集上的 3D AP (R11)，IoU 阈值：汽车 = 0.50，行人 / 骑车人 = 0.25。

最优权重：`output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth`
完整的消融实验、按种子的训练日志、超参数表见 [`experiments/RESULTS.md`](experiments/RESULTS.md)。

---

## 整体架构

```
雷达点云 (N,7)
  → PillarVFE（体素化 + 多普勒分解：vx, vy 通过 atan2 还原）
  → PillarAttention（带掩码的 Self-Attention，C=E=32）
  → PointPillarScatter（320×320×32 BEV 鸟瞰图）
  → BaseBEVBackbone（3 个 2D CNN 块，统一通道数 C=32）
  → AnchorHeadSingle（汽车 / 行人 / 骑车人）
```

关键实现细节：

- **VFE 中的速度分解**：`vx = v_r_comp·cos(φ)`，`vy = v_r_comp·sin(φ)`，其中 `φ = atan2(y, x)`
- **物理一致的数据增强**：速度向量随点坐标一起旋转 / 翻转（修复了 OpenPCDet 假设 nuScenes 列布局的 bug）
- **PillarAttention** 使用 key-padding mask，使空 pillar 不会污染注意力分数
- **`FFN_CHANNELS` 改为配置驱动**（`pillar_attention.py`），之前硬编码为 `*2`

---

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

环境要求：Python 3.8+、PyTorch 2.4+、CUDA 12.x、spconv 2.3.6。

---

## 数据

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

生成 info pkl 与 GT 数据库：

```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset/vod_dataset_radar.yaml
```

---

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <run_name> --workers 4
```

3 种子多轮训练（与上方核心结果一致）：

```bash
bash experiments/chain_scripts/multiseed_v2.sh
```

---

## 评估

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth
```

---

## 配置文件

| 文件 | 用途 |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | 与论文第 IV 节一致的基线（无旋转增强） |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **加入旋转增强的变体 —— 即产生上方核心结果的配置** |

---

## 引用

```bibtex
@inproceedings{gillen2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Gillen, Julius and Bieder, Manuel and Stiller, Christoph},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  url    = {https://github.com/open-mmlab/OpenPCDet}
}
```

---

## 许可证

以 Apache 2.0 许可证发布 —— 详见 [LICENSE](LICENSE)。本项目基于 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 构建，后者同样为 Apache 2.0 许可证。
