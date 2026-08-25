"""RadarPillar 检测: (N,7) 点云 → list[Obj]. 耦合 pcdet, 仿 tools/demo.py."""
from __future__ import annotations
from typing import List
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg as pcdet_global_cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.utils import common_utils

from .utils import cpu_patch
from .schemas import Cfg, FRAME, Obj


class _PcdetDataset(DatasetTemplate):
    """DatasetTemplate 子类: 点云来自内存, 复用 prepare_data 跑完整预处理管线 (training=False)."""

    def __init__(self, model_cfg, class_names, logger=None):
        super().__init__(
            dataset_cfg=model_cfg.DATA_CONFIG, class_names=class_names,
            training=False, root_path=Path('.'), logger=logger,
        )


class Detector:
    """
    in: frame.proc.points (N,7) [x,y,z,rcs,v_r,v_r_comp,time]; 
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

        self.dataset = _PcdetDataset(pcdet_cfg, self.class_names, self.logger)
        self.model: Detector3DTemplate = build_network(model_cfg=pcdet_cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset)  # type: ignore[attr-defined]
        self.model.load_params_from_file(filename=cfg.MODEL.ckpt, logger=self.logger, to_cpu=(self.device == 'cpu'))
        if self.device == 'cuda':
            self.model.cuda()
        self.model.eval()

    def run(self, frame: FRAME) -> List[Obj]:
        points = frame.proc.points
        if points is None or points.shape[0] == 0:
            return []
        data_dict = self._prepare(self._to_src_points(points, frame.vdd))
        pred_dicts = self._infer(data_dict)
        return self._to_objs(pred_dicts[0])

    def _to_src_points(self, points: np.ndarray, vdd) -> np.ndarray:
        """
        点云转训练口径: tracker (N,7)[x,y,z,rcs,v_r,v_r_comp,time] → src_feature_list 18 列
        (used 列 dop_x_gnd/dop_y_gnd 按 build_msr_features 公式重构, azi=atan2(y,x) 精确还原)
        """
        enc = self.dataset.point_feature_encoder
        idx = {n: i for i, n in enumerate(enc.src_feature_list)}
        ego = float(vdd.speed_ms) if vdd is not None else 0.0
        yr = float(vdd.yaw_rate) if vdd is not None else 0.0
        x, y, dop = points[:, 0], points[:, 1], points[:, 4]
        azi = np.arctan2(y, x)
        dop_x, dop_y = dop * np.cos(azi), dop * np.sin(azi)
        cols = {
            'x': x, 'y': y, 'z': points[:, 2], 'rcs': points[:, 3],
            'dop_x': dop_x, 'dop_y': dop_y,
            'dop_x_gnd': dop_x - ego, 'dop_y_gnd': dop_y + ego * np.tan(yr),
        }
        out = np.zeros((points.shape[0], len(enc.src_feature_list)), dtype=np.float32)
        for name, arr in cols.items():
            out[:, idx[name]] = arr
        return out

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
                x=box[0], y=box[1],                  # [x,y,z] → Obj.x/y (z 丢弃)
                length=box[3], width=box[4],         # [dx,dy,dz] → length/width (height 丢弃)
                heading=float(np.degrees(box[6])),    # rad→deg: Obj.heading 契约为度(与 loader 同口径)
                type=int(label),                     # label: 1-based class index
                score=float(score),                  # 检测置信度
                vx=0.0, vy=0.0,                      # 模型不回归速度, 留给 filter
                isghost=0, ispassable=0,             # 需时序/场景判定, 留给 manager/决策层
            ))
        return objs
