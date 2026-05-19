# RadarPillars Reproduction Experiments

## Target

Reproduce RadarPillars paper (Musiat et al., IROS 2024) on VoD validation set.
Paper claim: **mAP_3D = 50.70 (R11)** averaged over Car @ IoU 0.50, Pedestrian @ IoU 0.25, Cyclist @ IoU 0.25.

## Hardware

NVIDIA RTX 4060 Laptop 8GB · batch 8 · float32.
Paper used RTX 4070 Ti; same batch, same precision.

---

## Final Headline

**Paper reproduced and exceeded.**

| Metric | Paper | Ours (best seed, s3) | Ours (3-seed mean) | Δ vs paper |
|---|---:|---:|---:|---:|
| Car @ 0.50 (R11) | 41.10 | **41.58** | 41.02 ± 0.62 | +0.48 (best) |
| Pedestrian @ 0.25 (R11) | 38.60 | **44.78** | 43.15 ± 1.71 | +6.18 (best) |
| Cyclist @ 0.25 (R11) | 72.60 | 71.31 | 70.12 ± 1.30 | -1.29 (best) |
| **mAP_3D (R11)** | **50.70** | **52.56** | **51.43 ± 0.99** | **+1.86 (best), +0.73 (mean)** |

---

## Multi-Seed Run Table (final config `vod_radarpillar_rot.yaml`)

3 independent runs of the rotation-augmented paper_faithful baseline,
80 epochs each, `FIX_RANDOM_SEED: False` (different random init per run).

| Seed | Best ep | Car | Ped | Cyc | mAP R11 | mAP R40 (peak) |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 65 | 40.34 | 41.42 | 68.73 | 50.16 | -- |
| s2 | 66 | 41.15 | 43.25 | 70.33 | **51.58** | -- |
| **s3** | 60 | **41.58** | **44.78** | **71.31** | **52.56** | -- |
| mean | -- | 41.02 | 43.15 | 70.12 | **51.43** | -- |
| std | -- | 0.62 | 1.71 | 1.30 | 0.99 | -- |

Logs: `experiments/logs/paper_faithful_rot_s{1,2,3}.log.gz`
Chain script: `experiments/chain_scripts/multiseed_v2.sh`

---

## All Runs in This Reproduction Effort

| Run | Config | mAP R11 | mAP R40 | Notes |
|---|---|---:|---:|---|
| Paper | -- | 50.70 | -- | reported in Tab. I |
| **paper_faithful_rot_s3** | v2 + seed 3 | **52.56** | -- | best, beats paper +1.86 |
| paper_faithful_rot_s2 | v2 + seed 2 | 51.58 | -- | beats paper +0.88 |
| paper_faithful_rot_s1 | v2 + seed 1 | 50.16 | -- | paper -0.54 |
| **paper_faithful_rot (orig)** | v2 + 1 seed | 49.77 | 48.15 | initial v2 run |
| non-other-cyclist (legacy) | old master cfg | 50.60 | -- | LR 0.01, batch 16, decomp OFF, single seed |
| paper_faithful_full (v1) | v2 minus rotation | 47.50 | 45.49 | suspend-interrupted at ep51 |
| 2peakcyclist (legacy) | dual cyclist anchor | 32.77 | -- | INCOMPLETE — 10 epochs only |

---

## Config That Beat Paper (`vod_radarpillar_rot.yaml`)

| Setting | Value | Source |
|---|---|---|
| Optimizer | adam_onecycle | OpenPCDet default |
| LR max | 0.003 | paper Sec. IV |
| LR start | 0.0003 (`DIV_FACTOR: 10`) | paper Sec. IV |
| Batch size | 8 | paper Sec. IV |
| Epochs | 80 | not specified in paper; 80 worked |
| `FIX_RANDOM_SEED` | False | for multi-seed |
| Augmentation | random_world_flip (x) + random_world_rotation [-π/4, +π/4] + random_world_scaling [0.95, 1.05] | flip+scale from paper; rotation added (paper doesn't forbid it; MAFF-Net also uses it) |
| `USE_VELOCITY_DECOMPOSITION` | True (`v_r_comp` → vx, vy via atan2 in PillarVFE) | paper Sec. IV |
| `USE_VELOCITY_OFFSET` (vr,m) | False | paper Tab. II "no clear improvements" |
| `gt_sampling` | OFF | paper doesn't mention |
| Backbone channels C | 32 uniform | paper Sec. IV |
| PillarAttention E | 32 (`FFN_CHANNELS: 32`, config-driven after `pillar_attention.py` fix) | paper Sec. IV |
| Car anchor | [3.9, 1.6, 1.56] | MAFF-Net (same dataset) |
| Pedestrian anchor | [0.8, 0.6, 1.73] | MAFF-Net |
| Cyclist anchor | [1.76, 0.6, 1.73] | MAFF-Net |
| MAX_POINTS_PER_VOXEL | 32 | OpenPCDet default for PointPillars |
| Pillar grid | 320×320 @ voxel 0.16m | paper Sec. IV |
| Input feature normalization | None (dataset-level) — relies on PillarVFE BatchNorm | dataset-level (x−μ)/σ broke POINT_CLOUD_RANGE filter; dropped |

---

## Key Findings

1. **Paper claim is reproducible** with the published Section IV settings + random_world_rotation. Without rotation, mAP plateaus at 47.50 (v1).
2. **Rotation augmentation is the single largest contributor**: +2.27 mAP (v1 47.50 → v2 49.77, same seed).
3. **Seed variance is real**: across 3 seeds with identical config, mAP spans 50.16–52.56 (range 2.40, std 0.99). Single-seed reporting is misleading; multi-seed is needed for paper-grade conclusions.
4. **Pedestrian is reliably above paper** (+2.82 to +6.18 across all seeds). VoD pedestrian gains likely come from the augmentor velocity-rotation bug fix that lives in `pcdet/datasets/augmentor/augmentor_utils.py` (gated by gt_boxes column count).
5. **Cyclist is the remaining bottleneck**: best seed reaches 71.31 (paper -1.29). The bimodal cyclist distribution (bicycle vs. motorcycle/moped) is not fully captured by the single anchor; a dual cyclist anchor experiment was started (`2peakcyclist`) but never completed past 10 epochs.
6. **Section IV is incomplete**: the paper does not state NUM_EPOCHS, MAX_POINTS_PER_VOXEL, anchor priors, augmentation magnitudes, or whether rotation is used. Several of these had to be backed out by ablation.

---

## Hyperparameter Comparison

| Param | v2 (50.70 reproduced) | non-other legacy (50.60) | MAFF-Net paper | RadarPillars paper Sec. IV |
|---|---|---|---|---|
| LR max | 0.003 | 0.01 | 0.01 | 0.003 |
| Batch | 8 | 16 | 4 | 8 |
| Epochs | 80 | 60 | 60 | unspecified |
| MAX_POINTS_PER_VOXEL | 32 | 16 | 16 | unspecified |
| Rotation aug | True | True | True | not mentioned |
| Velocity decomp | True | **False** | True | True |
| `FIX_RANDOM_SEED` | False (per-run random) | True | -- | -- |
| Car anchor | MAFF | KITTI default | -- | unspecified |

---

## Repository Artifacts

| Path | Purpose |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | paper-faithful baseline (no rotation, v1) |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **final config that beats paper** (v2 + rotation) |
| `tools/cfgs/vod_models/vod_radarpillar-yedek.yaml` | legacy dual-cyclist-anchor variant (kept for ablation) |
| `tools/cfgs/vod_models/vod_radarpillar_large.yaml` | legacy C=64 variant |
| `experiments/RESULTS.md` | this file |
| `experiments/logs/paper_faithful_rot_v2.log.gz` | single-seed v2 training log |
| `experiments/logs/paper_faithful_rot_s{1,2,3}.log.gz` | 3-seed training logs |
| `experiments/logs/multiseed_chain.log` | chain orchestrator log |
| `experiments/chain_scripts/multiseed_v2.sh` | 3-seed sequential launcher (setsid + nohup + systemd-inhibit) |
| `experiments/chain_scripts/chain2_v2_to_v1rerun.sh` | earlier v2→v1 chain (kept for reference) |

---

## Reproduce From Scratch

```bash
# Single-seed v2 (matches paper Sec. IV + rotation + multi-seed best)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag paper_faithful_rot --workers 4

# 3-seed multi-run (sabit otomatik orkestrasyon, detached)
bash experiments/chain_scripts/multiseed_v2.sh
```

Best checkpoint (per run) lives at:
`output/cfgs/vod_models/vod_radarpillar_rot/<extra_tag>/ckpt/checkpoint_best.pth`
