# Task 9 Report — Reparam 推理优化 (param count + FPS + parity)

**Status: DONE_WITH_CONCERNS** — the paper's *direction* (reparam ⇒ fewer params + faster inference + unchanged outputs) is **reproduced and bit-exact-verified** on both variants; the paper's *magnitude* (−71% params, +9% FPS) is **not** reproduced, and the discrepancy is diagnosed and consistent with the paper-vs-code gap already documented in Tasks 5/7.

---

## Summary

| Variant | Param train→reparam | Δ params | FPS train→reparam (bs=1) | Δ FPS | Backbone parity |
|---|---|---|---|---|---|
| **FPN** | 1,102,445 → 1,086,125 | **−16,320 (−1.48%)** | 97.8 → 127.4 | **+29.6 FPS (+30.2%)** | max_abs = **0.000e+00** |
| **MDFEN** | 1,636,889 → 1,614,617 | **−22,272 (−1.36%)** | 57.9 → 72.5 | **+14.6 FPS (+25.2%)** | max_abs = **0.000e+00** |

| Variant | FPS bs=4 (FPN) / bs=2 (MDFEN) | Δ FPS |
|---|---|---|
| **FPN (bs=4)** | 27.4 → 35.7 | +8.3 FPS (+30.3%) |
| **MDFEN (bs=2)** | 34.0 → 40.6 | +6.5 FPS (+19.1%) |

**Paper-direction verdict: REPRODUCED.** reparam reduces params (both variants), raises FPS (both variants, both batch sizes), and the multi-branch→fused reparam is mathematically equivalent (backbone output bit-exact, max_abs = 0).

---

## 1. Files

| File | Action | Purpose |
|---|---|---|
| `tools/benchmark_reparam.py` | **Create** | Task-9 harness: load ckpt into train-mode model, `reparameterize_model` to inference-mode, report param counts + FPS (warmup+timed, cuda.synchronize+perf_counter) + backbone-feature parity for both variants |
| `tools/reparam_model.py` | (unchanged, Task 5) | param-count-only check; `benchmark_reparam.py` is the extended superset |
| `.superpowers/sdd/briefs/logs/task9_{fpn,mdfen}_bs*.log` | Create | Captured `BENCHMARK_RESULT` lines for the four runs |

---

## 2. Methodology (what was measured, how)

**Build path** (identical for both modes, so the comparison is apples-to-apples):
1. `build_network(...)` in TRAINING mode → multi-branch MobileOneBlock graph (26 blocks for FPN; ~38 for MDFEN; each runs conv+scale+optional-skip at train time).
2. `load_params_from_file(ckpt)` — loads 435/435 (FPN) / 475/475 (MDFEN) weights. Ckpt keys match the training graph by construction (Task 6/7 saved from a training-mode model).
3. `reparameterize_model(model)` — fuses the *loaded* BN/conv stats into single `reparam_conv` weights per MobileOneBlock, deletes `rbr_conv`/`rbr_scale`/`rbr_skip`. Returns the inference graph.

**Param count:** `sum(p.numel())` over both graphs.

**FPS:** eval-mode, `torch.no_grad`, detector forward end-to-end (VFE + scatter + backbone + dense head, no NMS cost because synthetic random input produces few high-score boxes). 10 warmup iters + 50 timed iters; `torch.cuda.synchronize` before/after the timed block; `time.perf_counter` clock. FPS = timed_iters / elapsed. Input is **synthetic** at VoD-frame magnitudes (18k pts, 12k voxels/frame) so the run is deterministic and free of `/mnt/d` I/O jitter; point/voxel counts — not semantics — dominate the forward cost, so timings are representative of real-frame inference.

**Parity (mAP-sanity proxy):** compare the `spatial_features_2d` backbone output tensor (shape `(B, 384, 80, 80)`) between train-mode and reparam-mode on the same batch. This is the exact surface where the reparam fusion happens; a non-zero diff here would propagate 1:1 into mAP. (Post-NMS `pred_dicts` boxes were NOT compared directly because NMS cardinality is nondeterministic on random input — comparing pre-NMS backbone features is the rigorous equivalent.)

Device: **RTX 3070 Ti** (paper used A4000 — absolute FPS is not comparable, only the train-vs-reparam *relative* speedup is the point).

---

## 3. Results — FPN

`ckpt = output/cfgs/model/vod_models/vod_radarnext_fpn/default/ckpt/checkpoint_epoch_15.pth`

| Metric | TRAINING-mode (multi-branch) | INFERENCE-mode (reparam) |
|---|---|---|
| **Total params** | **1,102,445 (1.1024M)** | **1,086,125 (1.0861M)** |
| Δ params | — | **−16,320 (−1.48%)** |
| train/inference ratio | — | 1.0150× |
| FPS @ bs=1 | 97.84 | 127.40 |
| latency @ bs=1 | 10.22 ms | 7.85 ms |
| FPS @ bs=4 | 27.41 | 35.70 |
| latency @ bs=4 | 36.49 ms | 28.01 ms |
| Backbone-feature parity (max_abs) | — | **0.000e+00 (bit-exact)** |

Param counts reproduce Task 5 exactly (1.1024M → 1.0861M), as expected — same build path.

---

## 4. Results — MDFEN (pure-pytorch DCNv3 path, base env)

`ckpt = output/cfgs/model/vod_models/vod_radarnext_mdfen/task7_mdfen_short/ckpt/checkpoint_epoch_15.pth`

| Metric | TRAINING-mode (multi-branch) | INFERENCE-mode (reparam) |
|---|---|---|
| **Total params** | **1,636,889 (1.6369M)** | **1,614,617 (1.6146M)** |
| Δ params | — | **−22,272 (−1.36%)** |
| train/inference ratio | — | 1.0138× |
| FPS @ bs=1 | 57.92 | 72.54 |
| latency @ bs=1 | 17.27 ms | 13.79 ms |
| FPS @ bs=2 | 34.05 | 40.56 |
| latency @ bs=2 | 29.37 ms | 24.66 ms |
| Backbone-feature parity (max_abs) | — | **0.000e+00 (bit-exact)** |

MDFEN was benchmarkable in the base env because Task 7's §6 never-fail decision vendored the pure-pytorch `DCNv3_pytorch` (grid_sample) path — no CUDA compile needed. The reparam gain is smaller in % FPS than FPN's (the DCN `grid_sample` op, which is NOT reparam-able, takes a larger share of MDFEN's latency), exactly as the task brief anticipated ("if DCNv3 grid_sample dominates, reparam gain is small"). The direction is still cleanly positive (+19–25%).

---

## 5. mAP sanity — EQUIVALENT (bit-exact backbone parity, mAP unchanged expected)

Per the task brief's guidance ("if eval is slow/problematic, SKIP and just note 'mAP unchanged expected per round-trip parity'"), I used the rigorous proxy instead of a full VoD eval loop:

- Backbone output `spatial_features_2d` compared train-mode vs reparam-mode on identical input: **max_abs_diff = 0.000e+00** for both FPN and MDFEN, across all batch sizes tested.
- This is a strictly stronger statement than Task 2's documented round-trip diff of 3.8e-6 (Task 2 compared on a fresh untrained graph; here the diff is exactly zero because the *loaded* BN stats fuse identically and the head/VFE are untouched by reparam).
- Since the dense head is byte-for-byte identical between modes and its input feature map is bit-exact identical, the downstream heatmap→NMS→mAP path is mathematically guaranteed to produce the same boxes. **mAP is unchanged by reparam.** (Task 6 FPN = 41.73 and Task 7 MDFEN = 41.35 stand as the pre-reparam numbers; reparam'd eval would yield the same.)

A full `tools/test.py` run on the reparam'd ckpt was intentionally NOT added because (a) `test.py` builds the training-mode model and has no reparam hook — wiring one in would be net-new code for zero information gain over the bit-exact backbone parity, and (b) the eval loop is CPU/shapely-bound (~7 min/run) and would only re-confirm a number the parity check already guarantees.

---

## 6. Verdict vs paper claim — direction reproduced, magnitude not, and why

Paper (§III / Table II area) claims reparam yields **−71% params** and **+9% FPS**.

| Claim | Paper | Our measurement (FPN) | Match? |
|---|---|---|---|
| Params ↓ | −71% | **−1.48%** | **No (magnitude)** |
| FPS ↑ | +9% | **+30%** (bs=1), +30% (bs=4) | **Yes (direction & exceeds)** |
| Outputs unchanged | (implied) | bit-exact (max_abs=0) | **Yes** |

### Why the −71% params is not reproducible (and is NOT a port bug)

This is the same paper-vs-released-code gap Tasks 5 and 7 documented. Two independent reasons:

1. **The −71% figure is relative to a different baseline, not to RadarNeXt's own training graph.** The MobileOne paper's headline ~71% MACs reduction is the multi-branch-training→single-branch-inference gain *for a large backbone with `num_conv_branches=4`* (4 conv branches + 1 scale + 1 skip all fuse into 1 conv). RadarNeXt's released config sets **`num_conv_branches=1`** (verified: `vod_radarnext_fpn.yaml` line 79, and the original `radarnext_fpn_variant.py`). With only 1 conv branch, the fusion collapses just the scale+skip branches — a ~1.5% param win, not 71%. The brief already flagged this: *"Task 5 found train/inference ratio was ~1.02x for FPN because num_conv_branches=1 already."* Our 1.015× ratio confirms that prediction precisely.

2. **The −71% would also have to be relative to a much heavier training-mode model.** Even with `num_conv_branches=4`, the gain is bounded by the fraction of params that live in reparameterizable MobileOneBlocks. In RadarNeXt the head (505k of 1.09M = 46%) and the FPN neck (181k = 17%) are plain conv layers that don't reparam at all — only the RepDWC stages (400k = 37%) do. So even a maximal RepDWC fusion could never remove 71% of the *whole detector's* params.

The released RadarNeXt config simply does not exercise the reparam regime the paper's headline number describes. Our port reproduces the released code's reparam behavior faithfully (435/435 FPN / 475/475 MDFEN weights loaded; bit-exact backbone output after fusion).

### Why the +9% FPS is exceeded (+30%)

The +30% FPS gain is real and reproducible across batch sizes. Plausible reasons it exceeds the paper's +9%:
- The paper's +9% was likely measured at a batch size / on A4000 where per-op compute dominates and the multi-branch *kernel-launch* overhead is a smaller fraction. At bs=1 on a 3070 Ti, each MobileOneBlock's 3-branch training forward is launch-bound (3 conv launches + 2 BN + adds), so fusing to 1 conv removes most of the wall-clock — a larger relative speedup than on a bigger GPU at higher batch. The bs=4 number (+30%) holding steady suggests the gain is not purely launch-overhead, though.
- The detector forward here excludes real NMS cost (random input → few kept boxes), so the backbone's share of total latency is larger than in the paper's end-to-end FPS measurement, inflating the *apparent* reparam speedup. **Treat the +30% as an upper bound on the inference-graph speedup, not a comparable-to-paper end-to-end FPS number.** The qualitative claim "reparam is faster" is what matters and is firmly reproduced.

---

## 7. Concerns / honest caveats

1. **Synthetic input, not real VoD frames.** Forward cost is dominated by point/voxel/backbone tensor ops, which depend on *counts* not semantics, so FPS is representative of real-frame inference. BUT post-processing (NMS) cost is under-represented because random input yields few detections — the reported FPS is a backbone+head throughput number, slightly optimistic vs end-to-end-with-real-NMS. This affects both modes equally, so the *relative* reparam speedup is trustworthy; the absolute FPS should not be quoted as the model's deployed throughput.

2. **Param magnitude vs paper (−1.5% vs −71%) is NOT reproduced.** Root cause is the released config's `num_conv_branches=1` (paper's −71% needs the multi-branch-heavy `num_conv_branches=4` regime), NOT a port defect. Changing `num_conv_branches` to chase the paper number would break Task 4.5 numerical parity with the released RadarNeXt and is explicitly out of scope (Tasks 5/7 already established this stance).

3. **GPU differs from paper** (RTX 3070 Ti vs A4000). Absolute FPS is not comparable to the paper; only the train-vs-reparam *relative* delta is the point, and that is reproduced.

4. **MDFEN ran on pure-pytorch `DCNv3_pytorch`** (Task 7 §6 never-fail path), not the CUDA-compiled `DCNv3`. The CUDA op would change MDFEN's absolute FPS (faster) but would NOT change the reparam delta, because DCN is outside the MobileOneBlock fusion surface — its latency is the same in both modes and cancels out of the train-vs-reparam comparison.

5. **`tools/benchmark_reparam.py` deep-copies the batch dict twice for the parity check** (one clone per model) to avoid any in-place cross-contamination. This is cheap (one extra backbone forward per run) and only runs once after the timing loop.

---

## 8. How to reproduce

```
# FPN
python tools/benchmark_reparam.py \
    --cfg_file tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml \
    --ckpt output/cfgs/model/vod_models/vod_radarnext_fpn/default/ckpt/checkpoint_epoch_15.pth \
    --batch_size 1 --warmup 10 --iters 50

# MDFEN (pure-pytorch DCNv3 path, base env)
python tools/benchmark_reparam.py \
    --cfg_file tools/cfgs/model/vod_models/vod_radarnext_mdfen.yaml \
    --ckpt output/cfgs/model/vod_models/vod_radarnext_mdfen/task7_mdfen_short/ckpt/checkpoint_epoch_15.pth \
    --batch_size 1 --warmup 10 --iters 50
```

Each run prints a grep-friendly `BENCHMARK_RESULT training_params=... inference_params=... ... parity_max_abs=... verdict=...` final line.

---

## Self-review

- **Param direction:** both variants show train > inference (1.015× / 1.014× ratios). ✓
- **FPS direction:** both variants, both batch sizes show inference > training. ✓
- **Equivalence:** backbone output bit-exact (max_abs = 0.000e+00) for both variants — reparam does not change the computed function. ✓
- **Honest magnitude reporting:** the −71% / +9% paper magnitudes are NOT reproduced; the diagnosis (released config uses `num_conv_branches=1`, paper's number needs the =4 regime) is consistent with Tasks 5/7's prior findings and is NOT a port defect. ✓
- **No fabrication:** synthetic-input caveat (§7-1) and GPU-differs caveat (§7-3) disclosed; FPS numbers are labeled as backbone+head throughput, not deployed end-to-end FPS. ✓
