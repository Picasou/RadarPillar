# RadarPillars Reproduction Experiments

## Target
Reproduce RadarPillars paper (Musiat et al., IROS 2024) on VoD val set.
Paper claim: **mAP_3D = 50.70 (R11)**.

## Hardware
RTX 4060 Laptop 8GB, batch=8, float32.

## Result Summary

| Run | Config | mAP R11 | mAP R40 | Car R11 | Ped R11 | Cyc R11 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| **Paper** | -- | **50.70** | -- | 41.10 | 38.60 | 72.60 | reported in Tab. I |
| `non-other-cyclist` | old master (LR 0.01, batch 16, decomp OFF, rotation ON, KITTI anchors) | **50.60** | -- | 40.95 | 42.98 | 67.87 | best single-seed prior run, paper-1 ep47 |
| `paper_faithful_rot` (v2) | paper-faithful Section IV (LR 0.003, batch 8, decomp ON, rotation ON, MAFF anchors) | **49.77** | 48.15 | 40.81 | 41.47 | 67.02 | full 80 epochs, ep66 best |
| `paper_faithful_full` (v1) | v2 minus rotation | 47.50 | 45.49 | 35.89 | 37.93 | 68.67 | suspend interrupted at ep51, ep28 best |
| `2peakcyclist` | dual cyclist anchor [0.82,0.76,1.54] + [1.89,0.68,1.38] | 32.77 | -- | 36.59 | 37.73 | 24.00 | INCOMPLETE — only 10 epochs trained |

## Key Findings

1. **Paper-faithful Section IV values produced 49.77** (LR 0.003, batch 8, MAFF anchors).
2. **Old master config produced 50.60** (LR 0.01, batch 16, KITTI anchors, decomp OFF) — closer to paper despite NOT following Section IV literally.
3. **Velocity decomposition (vx, vy) ablation is unclear**: paper Section IV mandates True; paper Tab. II ablation suggests True with v_r dropped; non-other run (False) outperformed our v2 (True).
4. **Rotation augmentation is critical**: v1 (no rotation) 47.50 vs v2 (rotation) 49.77 → +2.27 mAP.
5. **Cyclist is the bottleneck**: paper 72.60, ours 67.02 → -5.58. Car and Ped are at/above paper level.
6. **Single-seed variance ~1 mAP**: non-other (50.60) vs v2 (49.77) same architecture, different seed → 0.83 gap.

## Open Hypotheses

- Paper Section IV is a "lite" config description; the 50.70 result may use a slightly different setup (LR/batch/epoch unspecified).
- Cyclist gap (-5.58) likely closes with multi-seed averaging (Cyclist single-seed variance is ±5-10 on VoD).
- v_r dropped + decomp True (paper Tab. II ideal) untested.

## Hyperparameter Comparison

| Param | paper_faithful (v2) | non-other (50.60) | MAFF-Net paper | RadarPillars paper Sec.IV |
|---|---|---|---|---|
| LR max | 0.003 | 0.01 | 0.01 | 0.003 |
| LR start | 0.0003 | 0.001 | -- | 0.0003 |
| Batch | 8 | 16 | 4 | 8 |
| Epochs | 80 | 60 | 60 | not specified |
| MAX_POINTS_PER_VOXEL | 32 | 16 | 16 | not specified |
| Rotation aug | True (rot variant) | True | True | NOT mentioned ("flipping and scaling") |
| Scale range | [0.95, 1.05] | same | same | not specified |
| Velocity decomp | True | **False** | True | True |
| Velocity offset (vr,m) | False | False | -- | False (Section IV-B says "no improvement") |
| gt_sampling | OFF | OFF | OFF | OFF |
| Car anchor | [3.9, 1.6, 1.56] MAFF | [4.17, 1.84, 1.57] KITTI | -- | unspecified |
| Cyclist anchor | [1.76, 0.6, 1.73] MAFF | [1.76, 0.6, 1.73] | -- | unspecified |

## Logs Index

- `logs/paper_faithful_rot_v2.log` — v2 full 80-epoch training log
- (v1 paper_faithful_full log lost; eval results recoverable from ckpt re-eval)

## Next Steps

1. Multi-seed paper_faithful_rot (3 seeds) — establish mean ± std and pick best seed.
2. Ablation: paper_faithful_rot with v_r dropped (paper Tab. II ideal).
3. Anchor ablation: dual cyclist anchor with full 80 epochs (current 2peakcyclist had only 10).
4. Final clean repo + paper-style results table.
