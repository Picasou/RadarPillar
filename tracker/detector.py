from __future__ import annotations
from dataclasses import fields
from typing import List
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg as pcdet_global_cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.msr.msr_utils import build_msr_features
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.utils import common_utils

from .utils import cpu_patch
from .schemas import Cfg, FRAME, Obj, PT, PTs


class _PcdetDataset(DatasetTemplate):
    """DatasetTemplate 子类: 点云来自内存, 复用 prepare_data 跑完整预处理管线 (training=False)."""

    def __init__(self, model_cfg, class_names, logger=None):
        super().__init__(
            dataset_cfg=model_cfg.DATA_CONFIG, class_names=class_names,
            training=False, root_path=Path('.'), logger=logger,
        )


def _pts_to_raw(pts: PTs) -> np.ndarray:
    """
    PT 列表转原始字段数组: dataclass 全字段直填, 供 build_msr_features 消费
    """
    assert pts.num == len(pts.Lst), f"pts.num({pts.num}) != len(Lst)({len(pts.Lst)})"
    names = [f.name for f in fields(PT)]
    raw = np.zeros(pts.num, dtype=[(n, '<f8') for n in names])
    for n in names:
        raw[n] = [getattr(p, n) for p in pts.Lst]
    return raw


class Detector:
    """
    in: frame.pts (PTs 原始字段, 经 build_msr_features 转训练口径);
    out: list[Obj] (vx/vy=0).
    """

    def __init__(self, cfg: Cfg) -> None:
        self.score_thresh = cfg.MODEL.score_thresh
        self.device = cfg.MODEL.device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            cpu_patch.apply_cpu_patch()
        self.logger = common_utils.create_logger()

        pcdet_cfg = cfg_from_yaml_file(cfg.MODEL.cfg, pcdet_global_cfg)
        self.class_names = pcdet_cfg.CLASS_NAMES  # type: ignore[attr-defined]
        self.point_cloud_range = pcdet_cfg.DATA_CONFIG.POINT_CLOUD_RANGE  # type: ignore[index]

        self.dataset = _PcdetDataset(pcdet_cfg, self.class_names, self.logger)
        self.model: Detector3DTemplate = build_network(model_cfg=pcdet_cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset)  # type: ignore[attr-defined]
        self.model.load_params_from_file(filename=cfg.MODEL.ckpt, logger=self.logger, to_cpu=(self.device == 'cpu'))
        if self.device == 'cuda':
            self.model.cuda()
        self.model.eval()

    def run(self, frame: FRAME) -> List[Obj]:
        if frame.pts.num == 0 or not frame.pts.Lst:
            return []
        raw = _pts_to_raw(frame.pts)
        ego = frame.vdd.speed_ms if frame.vdd is not None else 0.0
        yr = frame.vdd.yaw_rate if frame.vdd is not None else 0.0
        feats = build_msr_features(raw, list(raw.dtype.names), ego_speed=ego, yaw_rate=yr)
        # ROI 判空: 全点在 point_cloud_range 外时提前返回, 防空 voxel 前向崩溃
        # (feats 前三列恒 x/y/z, 与 mask_points_by_boxes 同口径)
        pcr = self.point_cloud_range
        in_roi = ((feats[:, 0] >= pcr[0]) & (feats[:, 0] <= pcr[3]) &
                  (feats[:, 1] >= pcr[1]) & (feats[:, 1] <= pcr[4]) &
                  (feats[:, 2] >= pcr[2]) & (feats[:, 2] <= pcr[5]))
        if not in_roi.any():
            return []
        data_dict = self._prepare(feats)
        pred_dicts = self._infer(data_dict)
        return self._to_objs(pred_dicts[0])

    def _prepare(self, points: np.ndarray) -> dict:
        # prepare_data 跑完整 DATA_PROCESSOR 管线 (mask/shuffle/feature_encoding/voxelize),
        # collate_batch 再做 batch collate
        data_dict = self.dataset.prepare_data({'points': points, 'frame_id': 0})
        return self.dataset.collate_batch([data_dict])

    def _infer(self, data_dict: dict) -> list:
        if self.device == 'cuda':
            load_data_to_gpu(data_dict)
        else:
            cpu_patch.load_data_to_cpu(data_dict)
        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)  # type: ignore[call-arg]
        return pred_dicts

    def _to_objs(self, pred: dict) -> List[Obj]:
        boxes  = pred['pred_boxes'].cpu().numpy()
        scores = pred['pred_scores'].cpu().numpy()
        labels = pred['pred_labels'].cpu().numpy()

        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        objs: List[Obj] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            objs.append(Obj(
                id=i,                                # 帧内临时 id, 真正航迹 id 由 manager 赋
                x=box[0], y=box[1], z=box[2],        # [x,y,z] → Obj.x/y/z
                length=box[3], width=box[4], height=box[5],   # [dx,dy,dz] → length/width/height
                heading=float(np.degrees(box[6])),    # rad→deg: Obj.heading 契约为度(与 loader 同口径)
                type=int(label),                     # label: 1-based class index
                score=float(score),                  # 检测置信度
                vx=0.0, vy=0.0,                      # 模型不回归速度, 留给 filter
                isghost=0, ispassable=0,             # 需时序/场景判定, 留给 manager/决策层
            ))
        return objs
