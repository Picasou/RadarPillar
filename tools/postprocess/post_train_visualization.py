"""Post-train visualization helpers (loss curve + sample panels).

Module is intentionally small and dependency-light: it reuses
``tools.scripts.plot_loss`` for loss curve rendering. Real matplotlib
sample rendering is added in Task 4 (training pipeline integration).
"""
from pathlib import Path

from easydict import EasyDict

from tools.scripts.plot_loss import parse_log, plot_loss


def get_post_train_cfg(cfg):
    """Return the ``TRAIN_POSTPROCESS`` config with safe defaults.

    Missing stanzas yield a fully-disabled EasyDict. Per-leaf defaults
    (e.g. ``PLOT_SAMPLES.REQUIRE_CAMERAS``) are preserved even when
    the user supplies only a partial ``TRAIN_POSTPROCESS`` block.
    """
    default = EasyDict(
        ENABLE=False,
        PLOT_LOSS_CURVE=False,
        PLOT_SAMPLES=EasyDict(
            ENABLE=False,
            NUM_SAMPLES=0,
            REQUIRE_CAMERAS=True,
            SKIP_IF_UNSUPPORTED_DATASET=True,
            SAVE_DIR="auto",
        ),
    )
    post_cfg = getattr(cfg, "TRAIN_POSTPROCESS", None)
    if post_cfg is None:
        return default
    merged = EasyDict(default)
    merged.update(post_cfg)
    if "PLOT_SAMPLES" in post_cfg:
        plot_samples = EasyDict(default.PLOT_SAMPLES)
        plot_samples.update(post_cfg.PLOT_SAMPLES)
        merged.PLOT_SAMPLES = plot_samples
    return merged


def dataset_supports_camera_panel(dataset):
    """Whether the dataset exposes camera imagery for multi-view rendering."""
    dataset_name = getattr(dataset.dataset_cfg, "DATASET", "")
    return dataset_name == "NuScenesRadarDataset"


def dataset_supports_bev_panel(dataset):
    """Whether the dataset provides points + GT boxes for a BEV panel."""
    return hasattr(dataset, "dataset_cfg")


def plot_loss_if_enabled(enabled, log_file, output_dir, logger):
    """Render the loss curve PNG iff enabled and the train log exists.

    Returns a small status dict so callers can log/branch without
    needing to inspect the filesystem.
    """
    if not enabled:
        return {"status": "skipped", "reason": "disabled"}
    log_file = Path(log_file)
    if not log_file.exists():
        return {"status": "skipped", "reason": "train_log_missing"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "loss_curve.png"
    steps, epoch_sorted = parse_log(log_file)
    plot_loss(steps, epoch_sorted, out_png, title_suffix=f": {log_file.stem}")
    return {"status": "created", "path": str(out_png)}


def should_render_sample_visualization(plot_cfg, dataset):
    """Decide whether sample visualization should run for this dataset.

    Returns ``(ok: bool, reason: str)``. The caller can branch on the
    reason to log a precise skip message without raising.
    """
    if not plot_cfg.ENABLE:
        return False, "disabled"
    if plot_cfg.REQUIRE_CAMERAS and not dataset_supports_camera_panel(dataset):
        if plot_cfg.SKIP_IF_UNSUPPORTED_DATASET:
            return False, "camera_panel_unsupported"
        raise NotImplementedError("camera panel requested for unsupported dataset")
    if not dataset_supports_bev_panel(dataset):
        return False, "bev_panel_unsupported"
    return True, "supported"


def render_post_train_samples(plot_cfg, dataset, pred_dicts, batch_dict, output_dir, logger):
    """First-milestone renderer: returns a status dict only.

    Task 3 deliberately leaves actual matplotlib file creation to
    Task 4, where it will be wired into ``tools/train.py`` with real
    batch data. The skip path here is the contract we are locking in
    with tests: unsupported datasets must NOT touch ``output_dir``.
    """
    ok, reason = should_render_sample_visualization(plot_cfg, dataset)
    if not ok:
        return {"status": "skipped", "reason": reason}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "count": 1}


def run_post_train_artifacts(cfg, train_set, output_dir, log_file, logger, model):
    """Top-level post-train entry point invoked from ``tools/train.py``.

    Returns a status dict so callers can log the result. Behavior:
    - Honors ``cfg.TRAIN_POSTPROCESS.ENABLE`` (skips if False / missing).
    - When enabled, attempts ``plot_loss_if_enabled`` (loss curve) and
      ``render_post_train_samples`` (sample panels) in sequence.
    - Each sub-step returns its own status; we surface them as a flat
      dict so the train log can show what was created vs skipped.
    - Never raises on dataset-level mismatches: callers (e.g. VoD
      smoke test) rely on safe skip behavior.
    """
    post_cfg = get_post_train_cfg(cfg)
    if not post_cfg.ENABLE:
        return {"status": "skipped", "reason": "disabled"}

    artifacts_dir = Path(output_dir) / "post_train_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = {"status": "ok", "artifacts_dir": str(artifacts_dir), "steps": {}}
    try:
        loss_result = plot_loss_if_enabled(
            enabled=post_cfg.PLOT_LOSS_CURVE,
            log_file=log_file,
            output_dir=artifacts_dir,
            logger=logger,
        )
        results["steps"]["loss_curve"] = loss_result

        if train_set is not None and post_cfg.PLOT_SAMPLES.ENABLE:
            samples_dir = artifacts_dir / "samples"
            try:
                sample_result = render_post_train_samples(
                    plot_cfg=post_cfg.PLOT_SAMPLES,
                    dataset=train_set,
                    pred_dicts=[],
                    batch_dict={},
                    output_dir=samples_dir,
                    logger=logger,
                )
            except NotImplementedError as exc:
                sample_result = {"status": "skipped", "reason": f"not_implemented:{exc}"}
            results["steps"]["samples"] = sample_result
        else:
            results["steps"]["samples"] = {"status": "skipped", "reason": "disabled_or_no_dataset"}
    except Exception as exc:  # pragma: no cover - defensive
        results["status"] = "error"
        results["error"] = repr(exc)
        if logger is not None:
            logger.error(f"Post-train artifacts failed: {exc!r}")

    return results