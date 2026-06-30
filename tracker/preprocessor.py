from __future__ import annotations
import os
import numpy as np
import yaml
from easydict import EasyDict

from schemas import Cfg, VDS, FRAME

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CFG = os.path.normpath(os.path.join(
    THIS_DIR, '..', 'tools', 'cfgs', 'dataset_configs', 'astyx_dataset_radar.yaml'
))
NUM_FEATURES = 5  # x, y, z, rcs, vr


class Preprocessor:
    """单帧预处理: 极坐标→直角 + ego 补偿 + 范围裁剪 + 体素化 (numpy 手写)"""

    def __init__(self, cfg: Cfg, data_cfg_path: str = DEFAULT_DATA_CFG):
        self.cfg = cfg
        with open(data_cfg_path, 'r', encoding='utf-8') as f:
            self.data_cfg = EasyDict(yaml.safe_load(f))
        self.point_cloud_range    = self.data_cfg.POINT_CLOUD_RANGE
        voxel_cfg = next(p for p in self.data_cfg.DATA_PROCESSOR if p.NAME == 'transform_points_to_voxels')
        self.voxel_size           = voxel_cfg.VOXEL_SIZE
        self.max_points_per_voxel = voxel_cfg.MAX_POINTS_PER_VOXEL
        self.max_voxels           = voxel_cfg.MAX_NUMBER_OF_VOXELS.get('test', 40000)
        self.grid_size = [
            int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]),
            int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]),
            int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]),
        ]

    # ---- 公共接口 ----

    def run(self, frame: FRAME, vds: VDS) -> FRAME:
        pts = frame.pts.Lst
        if not pts:
            return self._empty_frame(frame)

        # 1. 极坐标 → 直角坐标
        r     = np.array([p.range_m for p in pts], dtype=np.float32)
        theta = np.array([p.ang_rad for p in pts], dtype=np.float32)
        phi   = np.array([p.elv_rad for p in pts], dtype=np.float32)
        rcs   = np.array([p.rcs for p in pts], dtype=np.float32)
        vr    = np.array([p.doppler_mps for p in pts], dtype=np.float32)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi) + vds.z_pos_m

        # 2. ego 补偿
        x, y = self._compensate_ego(x, y, frame.vdd.speed_ms, frame.vdd.yaw_rate, vds.cycle_s)

        # 3. 拼 points (N, 5) — 列序 [x, y, z, rcs, vr]
        points = np.stack([x, y, z, rcs, vr], axis=1)

        # 4. 6 维范围裁剪
        pcr = self.point_cloud_range
        keep = (
            (points[:, 0] >= pcr[0]) & (points[:, 0] <= pcr[3]) &
            (points[:, 1] >= pcr[1]) & (points[:, 1] <= pcr[4]) &
            (points[:, 2] >= pcr[2]) & (points[:, 2] <= pcr[5])
        )
        points = points[keep]

        # 5. 体素化
        voxels, voxel_coords, voxel_num_points = self._voxelize_numpy(points)

        # 6. 字段透传
        frame.voxels           = voxels
        frame.voxel_coords     = voxel_coords
        frame.voxel_num_points = voxel_num_points
        frame.use_lead_xyz     = True
        frame.frame_id         = str(pts[0].frame)
        return frame

    # ---- 内部方法 ----

    def _compensate_ego(self, x, y, hostv, yaw_rate, cycle_s):
        if cycle_s <= 0:
            return x, y
        dx = hostv * cycle_s
        wt = yaw_rate * cycle_s
        if abs(yaw_rate) < 1e-4:
            return x - dx, y
        cos_wt, sin_wt = np.cos(wt), np.sin(wt)
        dx_pos = x - dx
        dy_pos = y
        x_new = dx_pos * cos_wt + dy_pos * sin_wt
        y_new = dx_pos * (-sin_wt) + dy_pos * cos_wt
        return x_new, y_new

    def _voxelize_numpy(self, points):
        if len(points) == 0:
            return (np.zeros((0, self.max_points_per_voxel, NUM_FEATURES), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        pcr = self.point_cloud_range
        vs  = self.voxel_size
        gs  = self.grid_size
        origin = np.array(pcr[:3], dtype=np.float32)
        voxel  = np.array(vs, dtype=np.float32)

        coors = np.floor((points[:, :3] - origin) / voxel).astype(np.int32)
        keep = (
            (coors[:, 0] >= 0) & (coors[:, 0] <= gs[0] - 1) &
            (coors[:, 1] >= 0) & (coors[:, 1] <= gs[1] - 1) &
            (coors[:, 2] >= 0) & (coors[:, 2] <= gs[2] - 1)
        )
        points    = points[keep]
        coors     = coors[keep]

        voxel_ids = coors[:, 0] + coors[:, 1] * gs[0] + coors[:, 2] * gs[0] * gs[1]
        sort_idx  = np.argsort(voxel_ids, kind='stable')
        points    = points[sort_idx]
        coors     = coors[sort_idx]
        voxel_ids = voxel_ids[sort_idx]

        unique_ids, group_starts = np.unique(voxel_ids, return_index=True)
        n_voxels = len(unique_ids)
        if n_voxels > self.max_voxels:
            unique_ids = unique_ids[:self.max_voxels]
            n_voxels = self.max_voxels

        voxels          = np.zeros((n_voxels, self.max_points_per_voxel, NUM_FEATURES), dtype=np.float32)
        voxel_num_points = np.zeros(n_voxels, dtype=np.int32)

        n_unique = len(group_starts)
        for v in range(n_voxels):
            start = group_starts[v]
            end   = group_starts[v + 1] if v + 1 < n_unique else len(voxel_ids)
            n     = min(end - start, self.max_points_per_voxel)
            voxels[v, :n] = points[start:start + n]
            voxel_num_points[v] = n

        voxel_coords = np.zeros((n_voxels, 3), dtype=np.int32)
        voxel_coords[:, 0] =  unique_ids % gs[0]
        voxel_coords[:, 1] = (unique_ids // gs[0]) % gs[1]
        voxel_coords[:, 2] =  unique_ids // (gs[0] * gs[1])

        return voxels, voxel_coords, voxel_num_points

    def _empty_frame(self, frame: FRAME) -> FRAME:
        frame.voxels           = np.zeros((0, self.max_points_per_voxel, NUM_FEATURES), dtype=np.float32)
        frame.voxel_coords     = np.zeros((0, 3), dtype=np.int32)
        frame.voxel_num_points = np.zeros((0,), dtype=np.int32)
        frame.use_lead_xyz     = True
        frame.frame_id         = ''
        return frame