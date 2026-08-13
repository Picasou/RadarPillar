"""MSR (MC_Single_Radar) dataset.

纯雷达 BEV,无 calib / camera / FOV 裁剪。仿 VoD 结构精简。
- POINTS / LABELS / PARAMS 各自的 struct.json 运行时解析(dtype 不写死)。
- POINTS scale 字段可信(range/doppler/azi/elv),由 get_radar 在调 build_msr_features 之前应用
  (build 接收物理量);rcs/snr 等无 scale 的列不动。
- LABELS 几何量强制 ×0.01(parse_label_boxes 内部已做,这里不改)。
- ego_speed/yaw_rate 由 get_dynamic_param 从 PARAMS 读出 scale 后还原为物理量再传给 build。
"""

import json
import pickle
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from .msr_utils import (
    MSR_FEATURE_ORDER,
    build_msr_features,
    load_struct_dtype,
    parse_label_boxes,
)

# POINTS 中带 scale 的字段集合(get_radar 在 build 之前对它们应用 json scale)。
# 这些字段对应 MSR_FEATURE_ORDER 的前 4 列(range/doppler/azi/elv)。
_POINTS_SCALABLE = {'range_m', 'doppler_mps', 'ang_rad', 'elv_rad'}


class MsrDataset(DatasetTemplate):
    """MSR dataset 类。CLASS_NAMES=['1','4','5'],无 calib,ghost 不过滤,z=0 BEV。"""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,
            root_path=root_path, logger=logger,
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # MSR 无 training/testing 二级目录,数据直挂在 DATA_PATH 下
        self.root_split_path = self.root_path

        split_dir = self.root_path / 'IMAGESETS' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.msr_infos = []
        self.include_msr_data(self.mode)

        # PointFeatureEncoder 用 src_feature_list = MSR_FEATURE_ORDER(18 列)
        self.radar_feature_order = MSR_FEATURE_ORDER
        self.selected_feature_list = list(self.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
        self.selected_feature_idx = [self.radar_feature_order.index(x) for x in self.selected_feature_list]

        norm_cfg = self.dataset_cfg.get('POINT_FEATURE_NORMALIZATION', None)
        self.use_feature_norm = bool(norm_cfg and norm_cfg.get('USE_NORM', False))
        if self.use_feature_norm:
            mean = np.array(norm_cfg.get('MEAN', []), dtype=np.float32)
            std = np.array(norm_cfg.get('STD', []), dtype=np.float32)
            if mean.shape[0] != len(self.selected_feature_list):
                raise ValueError('POINT_FEATURE_NORMALIZATION.MEAN length must match used_feature_list')
            if std.shape[0] != len(self.selected_feature_list):
                raise ValueError('POINT_FEATURE_NORMALIZATION.STD length must match used_feature_list')
            self.feature_mean = mean
            self.feature_std = std
        else:
            self.feature_mean = None
            self.feature_std = None

        # 地面系速度补偿开关(默认 True);为 False 时 ego_speed/yaw_rate 一律传 0
        self.use_gnd_velocity = bool(self.dataset_cfg.get('USE_GND_VELOCITY', True))

        # 预加载 struct.json 路径 + 动态构造 dtype(运行时解析,schema 变更自动适配)
        self.points_json = self.root_path / 'POINTS' / 'struct.json'
        self.labels_json = self.root_path / 'LABELS' / 'struct.json'
        self.params_json = self.root_path / 'PARAMS' / 'radar_dynamic.struct.json'
        self._points_dtype, _, self._points_names = load_struct_dtype(self.points_json)
        self._labels_dtype, _, self._labels_names = load_struct_dtype(self.labels_json)
        self._params_dtype, _, self._params_names = load_struct_dtype(self.params_json)

        # POINTS 各字段 scale(output_name -> scale;无 scale 字段不入表)
        spec = json.load(open(self.params_json))
        # PARAMS 用到的字段 scale(hostVelocity scale=1.0,vehicleYawRate scale=57.29577... 即弧度→度)
        self._params_scale = {}
        for fld in spec['structures'][0]['fields']:
            if 'scale' in fld:
                self._params_scale[fld.get('output_name', fld['name'])] = float(fld['scale'])

        points_spec = json.load(open(self.points_json))
        self._points_scale = {}
        for fld in points_spec['structures'][0]['fields']:
            if 'scale' in fld:
                self._points_scale[fld.get('output_name', fld['name'])] = float(fld['scale'])

    # ---------- info pkl 加载 ----------

    def include_msr_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading MSR dataset')
        msr_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                msr_infos.extend(infos)
        self.msr_infos.extend(msr_infos)
        if self.logger is not None:
            self.logger.info('Total samples for MSR dataset: %d' % (len(msr_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path
        split_dir = self.root_path / 'IMAGESETS' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    # ---------- reader ----------

    def get_dynamic_param(self, idx):
        """读 PARAMS/{idx}.bin + radar_dynamic.struct.json → {'ego_speed': float, 'yaw_rate': float}。

        PARAMS 是单条结构体(84B/帧)。json 的 scale 字段:
            hostVelocity_mps: scale=1.0(m/s 直接读)
            vehicleYawRate_radps: scale=57.2957...(即 raw 是度/s,json scale=弧度→度,所以 raw/scale=弧度/s)
        读失败或 USE_GND_VELOCITY=False 时由调用处决定是否传 0。
        """
        param_file = self.root_split_path / 'PARAMS' / ('%s.bin' % idx)
        if not param_file.exists():
            return {'ego_speed': 0.0, 'yaw_rate': 0.0}
        raw = np.fromfile(str(param_file), dtype=self._params_dtype)
        if raw.shape[0] == 0:
            return {'ego_speed': 0.0, 'yaw_rate': 0.0}
        rec = raw[0]
        ego_speed = float(rec['ego_speed']) / self._params_scale.get('ego_speed', 1.0)
        yaw_rate = float(rec['ego_yawrate']) / self._params_scale.get('ego_yawrate', 1.0)
        return {'ego_speed': ego_speed, 'yaw_rate': yaw_rate}

    def get_radar(self, idx):
        """读 POINTS/{idx}.bin + struct.json → (N, used_feature_list 对应列数)。

        流程:frombuffer 解原始整数 → 对 range/doppler/azi/elv 乘 json scale(物理量)
        → build_msr_features(ego_speed, yaw_rate) 产 18 列 → 选 used_feature_list 列 → 可选归一化。
        """
        points_file = self.root_split_path / 'POINTS' / ('%s.bin' % idx)
        assert points_file.exists(), 'POINTS file missing: %s' % points_file
        raw = np.fromfile(str(points_file), dtype=self._points_dtype)
        if raw.shape[0] == 0:
            # 空帧兜底:返回 0 行 used_feature_list 列
            return np.zeros((0, len(self.selected_feature_idx)), dtype=np.float32)

        # 拿 ego_speed / yaw_rate 给 gnd 速度补偿;关开关或读失败传 0
        if self.use_gnd_velocity:
            param = self.get_dynamic_param(idx)
            ego_speed = param['ego_speed']
            yaw_rate = param['yaw_rate']
        else:
            ego_speed = 0.0
            yaw_rate = 0.0

        # 对带 scale 的列应用 json scale(物理量)。原 raw 是 int dtype,直接赋值会被截断,
        # 所以这里按 output_names 顺序构建 float64 dtype 的 structured 数组承载物理量。
        float_dtype = np.dtype([(n, '<f8') for n in self._points_names])
        scaled = np.zeros(raw.shape[0], dtype=float_dtype)
        for n in self._points_names:
            col = raw[n].astype(np.float64)
            if n in _POINTS_SCALABLE:
                col = col * self._points_scale.get(n, 1.0)
            scaled[n] = col

        feats = build_msr_features(scaled, self._points_names, ego_speed=ego_speed, yaw_rate=yaw_rate)
        feats = feats[:, self.selected_feature_idx]
        if self.use_feature_norm:
            feats = (feats - self.feature_mean) / self.feature_std
        return feats

    def get_label(self, idx):
        """读 LABELS/{idx}.bin + struct.json → (gt_boxes(N,7), gt_names(N))。

        parse_label_boxes 内部已对几何量 ×0.01(cm→m)+ heading 转 rad。
        """
        label_file = self.root_split_path / 'LABELS' / ('%s.bin' % idx)
        assert label_file.exists(), 'LABELS file missing: %s' % label_file
        raw = np.fromfile(str(label_file), dtype=self._labels_dtype)
        if raw.shape[0] == 0:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=np.str_)
        return parse_label_boxes(raw, self._labels_names)

    # ---------- DatasetTemplate 契约 ----------

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.msr_infos) * self.total_epochs
        return len(self.msr_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.msr_infos)

        info = self.msr_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_radar(sample_idx)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes = annos['gt_boxes_lidar'] if gt_names.shape[0] > 0 else np.zeros((0, 7), dtype=np.float32)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes,
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = info.get('image', {}).get('image_shape', np.array([0, 0], dtype=np.int32))
        return data_dict
