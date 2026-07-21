"""Shared param-counting utilities for tools/param_check/ scripts.

Three call sites do almost the same thing:
  - tools/param_check/radarpillar.py       (RadarPillar base, trainable only)
  - tools/param_check/reparam/model.py     (RadarNeXt, train vs inference)
  - tools/param_check/reparam/benchmark.py (RadarNeXt, train vs inference + FPS)

Each of them used to inline its own copy of ``count_params`` and its own
``build_dataloader → build_network`` boilerplate. This module centralises
those pieces so the three scripts agree on what "parameter count" means
and so per-module breakdowns stay comparable across scripts.

What lives here:
  - count_params(model)               total numel (all params)
  - count_trainable(model)            numel of requires_grad params only
  - per_module_breakdown(model, ...)  (name, numel) for every own-param submodule
  - build_model_from_cfg(cfg_file)    cfg + dataset + model, returns dataset/model
  - verdict_pct(actual, target, ...)  'PASS'|'WARN'|'FAIL' against a target

Usage (PYTHONPATH=tools):
    from param_check.core import count_params, build_model_from_cfg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

# Repo-root bootstrap: importable as ``param_check.core`` when PYTHONPATH=tools.
# Putting repo root at sys.path[0] is what the original count_params.py
# already did — keep the same convention so dataset/model imports resolve to
# the local pcdet, not a site-packages copy.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
import sys as _sys
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader        # noqa: E402
from pcdet.models import build_network             # noqa: E402
from pcdet.utils import common_utils               # noqa: E402


# -------- param counting --------

def count_params(model) -> int:
    """Total parameter count (all params, regardless of requires_grad)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable(model) -> int:
    """Trainable (requires_grad) parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def per_module_breakdown(model, prefix: str = '') -> Iterable[Tuple[str, int]]:
    """Yield (qualified_name, numel) for every module that owns parameters
    directly (i.e. ``parameters(recurse=False)`` > 0). Walks ``named_children``
    recursively. The root is reported as ``<root>`` when it owns params."""
    own = sum(p.numel() for p in model.parameters(recurse=False))
    if own > 0:
        rows = [(prefix if prefix else '<root>', own)]
    else:
        rows = []
    for name, child in model.named_children():
        child_prefix = f'{prefix}.{name}' if prefix else name
        rows.extend(per_module_breakdown(child, child_prefix))
    return rows


# -------- model construction --------

def build_model_from_cfg(
    cfg_file: str,
    *,
    training: bool = True,
    batch_size: int = 1,
    workers: int = 0,
    logger: Optional[logging.Logger] = None,
):
    """Load cfg + build a dataset (only used for module_topology inputs)
    + build_network. Returns (dataset, model, cfg).

    ``training`` only affects the dataset split; the model itself is always
    built in its default mode (callers that need train→eval transitions do
    them after this returns)."""
    if logger is None:
        logger = common_utils.create_logger(rank=0)
        for h in logger.handlers:
            h.setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)

    cfg_from_yaml_file(cfg_file, cfg)
    dataset, _, _ = build_dataloader(
        cfg.DATA_CONFIG,
        cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=training,
        total_epochs=1 if training else 0,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    return dataset, model, cfg


# -------- verdict --------

def verdict_pct(actual: int, target: int, pass_pct: float = 2.0, warn_pct: float = 5.0) -> str:
    """PASS if |Δ| <= pass_pct; WARN if <= warn_pct; else FAIL."""
    if target <= 0:
        return 'FAIL'
    abs_pct = abs(actual - target) / target * 100.0
    if abs_pct <= pass_pct:
        return 'PASS'
    if abs_pct <= warn_pct:
        return 'WARN'
    return 'FAIL'