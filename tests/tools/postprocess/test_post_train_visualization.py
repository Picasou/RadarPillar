"""Tests for tools.postprocess.post_train_visualization.

Task 1: config defaults + dataset capability checks.
Task 2: loss curve rendering + skip behavior.
Task 3: sample visualization gating (supported vs unsupported datasets).
"""
from pathlib import Path

from easydict import EasyDict

from tools.postprocess.post_train_visualization import (
    get_post_train_cfg,
    dataset_supports_camera_panel,
    dataset_supports_bev_panel,
    plot_loss_if_enabled,
    render_post_train_samples,
    should_render_sample_visualization,
)


def test_get_post_train_cfg_returns_disabled_defaults_when_missing():
    cfg = EasyDict()
    post_cfg = get_post_train_cfg(cfg)
    assert post_cfg.ENABLE is False
    assert post_cfg.PLOT_LOSS_CURVE is False
    assert post_cfg.PLOT_SAMPLES.ENABLE is False


def test_dataset_supports_camera_panel_for_nuscenes_like_dataset():
    dataset = type("NuScenesLike", (), {"dataset_cfg": EasyDict(DATASET="NuScenesRadarDataset")})()
    assert dataset_supports_camera_panel(dataset) is True


def test_dataset_supports_camera_panel_false_for_non_camera_dataset():
    dataset = type("VodLike", (), {"dataset_cfg": EasyDict(DATASET="VodRadarDataset")})()
    assert dataset_supports_camera_panel(dataset) is False


def test_dataset_supports_bev_panel_requires_points_and_boxes():
    dataset = type("AnyDataset", (), {"dataset_cfg": EasyDict(DATASET="NuScenesRadarDataset")})()
    assert dataset_supports_bev_panel(dataset) is True


def test_plot_loss_if_enabled_creates_png(tmp_path):
    log_file = tmp_path / "log_train_fake.txt"
    log_file.write_text("total_it=1 loss=1.0\n")
    out_dir = tmp_path / "artifacts"
    result = plot_loss_if_enabled(
        enabled=True,
        log_file=log_file,
        output_dir=out_dir,
        logger=None,
    )
    assert result["status"] == "created"
    assert (out_dir / "loss_curve.png").exists()


def test_plot_loss_if_enabled_skips_when_log_missing(tmp_path):
    result = plot_loss_if_enabled(
        enabled=True,
        log_file=tmp_path / "missing.txt",
        output_dir=tmp_path / "artifacts",
        logger=None,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "train_log_missing"


def test_should_render_sample_visualization_for_nuscenes():
    dataset = type("NuScenesLike", (), {"dataset_cfg": EasyDict(DATASET="NuScenesRadarDataset")})()
    cfg = EasyDict(ENABLE=True, REQUIRE_CAMERAS=True, SKIP_IF_UNSUPPORTED_DATASET=True)
    ok, reason = should_render_sample_visualization(cfg, dataset)
    assert ok is True
    assert reason == "supported"


def test_should_render_sample_visualization_skips_unsupported_dataset():
    dataset = type("VodLike", (), {"dataset_cfg": EasyDict(DATASET="VodRadarDataset")})()
    cfg = EasyDict(ENABLE=True, REQUIRE_CAMERAS=True, SKIP_IF_UNSUPPORTED_DATASET=True)
    ok, reason = should_render_sample_visualization(cfg, dataset)
    assert ok is False
    assert reason == "camera_panel_unsupported"


def test_render_post_train_samples_does_not_create_files_when_skipped(tmp_path):
    dataset = type("VodLike", (), {"dataset_cfg": EasyDict(DATASET="VodRadarDataset")})()
    cfg = EasyDict(ENABLE=True, REQUIRE_CAMERAS=True, SKIP_IF_UNSUPPORTED_DATASET=True, NUM_SAMPLES=2)
    result = render_post_train_samples(cfg, dataset, pred_dicts=[], batch_dict={}, output_dir=tmp_path, logger=None)
    assert result["status"] == "skipped"
    assert list(tmp_path.iterdir()) == []