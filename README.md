<div align="center">

# RadarPillars: Reproduction on View-of-Delft

**Radar-only 3D object detection — OpenPCDet-based reproduction of [Gillen et al., IROS 2024](https://arxiv.org/abs/2408.05020)**

</div>

---

## Headline

| Method | Car | Ped | Cyc | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **Ours — best seed** | **41.6** | **44.8** | 71.3 | **52.56** |
| Ours — 3-seed mean | 41.0 | 43.2 | 70.1 | 51.43 ± 0.99 |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| **RadarPillars (paper)** | 41.1 | 38.6 | 72.6 | **50.70** |
| CenterPoint baseline | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars baseline | 37.9 | 31.2 | 65.7 | 45.0 |

Best checkpoint: `output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth`
Full ablation, per-seed logs, hyperparameter tables → [`experiments/RESULTS.md`](experiments/RESULTS.md).

---

## Architecture

```
Radar pcd (N,7)
  → PillarVFE (voxelize + Doppler decomp: vx, vy via atan2)
  → PillarAttention (masked self-attention, C=E=32)
  → PointPillarScatter (320×320×32 BEV)
  → BaseBEVBackbone (3-block 2D CNN, uniform C=32)
  → AnchorHeadSingle (Car / Pedestrian / Cyclist)
```

Key implementation details:
- **Velocity decomposition** in VFE: `vx = v_r_comp·cos(φ)`, `vy = v_r_comp·sin(φ)`, `φ = atan2(y, x)`
- **Physics-consistent augmentation**: velocity vectors rotated/flipped with point coordinates (fixes a bug in OpenPCDet that assumed nuScenes column layout)
- **PillarAttention** with key-padding mask so empty pillars don't poison attention scores
- **`FFN_CHANNELS` config-driven** in `pillar_attention.py` (was hardcoded `*2` before)

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

Requirements: Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6.

---

## Data

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

Generate info pkl + GT db:
```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

---

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <run_name> --workers 4
```

3-seed multi-run (matches the headline number):
```bash
bash experiments/chain_scripts/multiseed_v2.sh
```

---

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth
```

---

## Configs

| File | Purpose |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | paper-faithful baseline (no rotation) |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **rotation-augmented variant — produced the headline result** |

---

## Citation

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

## License

Released under the Apache 2.0 License — see [LICENSE](LICENSE). This project is built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is itself Apache 2.0 licensed.
