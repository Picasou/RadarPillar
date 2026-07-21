"""tools/param_check/ — 参数量对账工具集。

子模块:
  - core            共享 util（count_params / count_trainable /
                    per_module_breakdown / build_model_from_cfg / verdict_pct）
  - radarpillar     RadarPillar base 模型专用 CLI（vs 论文 0.27M）
  - reparam/model   RadarNeXt 训练 vs 融合模式参数量校验（vs 论文 0.899M）
  - reparam/benchmark RadarNeXt 训练 vs 融合模式 FPS + output parity 基准

入口（PYTHONPATH=tools）：
  python tools/param_check/radarpillar.py --cfg_file ...
  python tools/param_check/reparam/model.py --cfg_file ...
  python tools/param_check/reparam/benchmark.py --cfg_file ... --ckpt ...
"""