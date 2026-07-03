"""Integration test for the top-level post-train entry point.

This is Task 4: ``run_post_train_artifacts`` is the function that
``tools/train.py`` calls right after eval finishes (or after the
``--skip_eval`` early-return). The first-milestone contract here is
the disable path: when the user does not enable ``TRAIN_POSTPROCESS``
the call must return ``{"status": "skipped", "reason": "disabled"}``
without touching the filesystem.

Subsequent milestones (loss curve creation, sample visualization
gating, etc.) are covered by ``test_post_train_visualization.py``.
"""
from pathlib import Path

from tools.postprocess.post_train_visualization import run_post_train_artifacts


def test_run_post_train_artifacts_skips_cleanly_when_disabled(tmp_path):
    result = run_post_train_artifacts(
        cfg=type("Cfg", (), {})(),
        train_set=None,
        output_dir=tmp_path,
        log_file=tmp_path / "log_train.txt",
        logger=None,
        model=None,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "disabled"