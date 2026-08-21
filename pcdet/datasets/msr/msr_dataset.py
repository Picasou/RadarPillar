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
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from ..dataset import DatasetTemplate
from .msr_utils import (
    MSR_FEATURE_ORDER,
    boxes_lidar_to_pseudo_camera,
    build_msr_features,
    load_struct_dtype,
    parse_label_boxes,
)

# POINTS 中带 scale 的字段集合(get_radar 在 build 之前对它们应用 json scale)。
# 这些字段对应 MSR_FEATURE_ORDER 的前 4 列(range/doppler/azi/elv)。
_POINTS_SCALABLE = {'range_m', 'doppler_mps', 'ang_rad', 'elv_rad'}

# type id → 语义类名(eval anno 用;与 visualize_msr.py 的 CLASS_LABEL 一致)。
# kitti eval 的 clean_data 按语义名匹配,anno 必须写语义名而非 '1'/'2'/'4'/'5'。
MSR_CLASS_LABEL = {'1': 'Car', '2': 'Pedestrian', '4': 'Cyclist', '5': 'Truck'}


class MsrDataset(DatasetTemplate):
    """MSR dataset 类。CLASS_NAMES=['1','2','4','5'],无 calib,ghost 不过滤,z=0 BEV。"""

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
        # selected_* 仅诊断/文档用途(check_msr 打印、可视化取列):表达 used_feature_list
        # 重排后的最终选列。实际选列由基类 PointFeatureEncoder.forward 完成(见 get_radar 注释)。
        # xyz 强制前 3(基类断言 + voxel 取 points[:,0:3]),其余按 used_feature_list 原顺序。
        used = list(self.dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
        xyz = [f for f in used if f in ('x', 'y', 'z')]
        missing = {'x', 'y', 'z'} - set(xyz)
        if missing:
            raise ValueError(f'used_feature_list 必须包含 x/y/z,缺: {missing}')
        rest = [f for f in used if f not in ('x', 'y', 'z')]
        self.selected_feature_list = ['x', 'y', 'z'] + rest
        self.selected_feature_idx = [self.radar_feature_order.index(x) for x in self.selected_feature_list]

        norm_cfg = self.dataset_cfg.get('POINT_FEATURE_NORMALIZATION', None)
        self.use_feature_norm = bool(norm_cfg and norm_cfg.get('USE_NORM', False))
        if self.use_feature_norm:
            mean = np.array(norm_cfg.get('MEAN', []), dtype=np.float32)
            std = np.array(norm_cfg.get('STD', []), dtype=np.float32)
            if mean.shape[0] != len(self.radar_feature_order):
                raise ValueError('POINT_FEATURE_NORMALIZATION.MEAN length must match MSR_FEATURE_ORDER(18 列)')
            if std.shape[0] != len(self.radar_feature_order):
                raise ValueError('POINT_FEATURE_NORMALIZATION.STD length must match MSR_FEATURE_ORDER(18 列)')
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
        # PARAMS 目录兼容:旧布局单目录 PARAMS/;2026-08 MSR 包拆为 PARAMS_DYNAMIC/+PARAMS_FIXED/,
        # 动态参数(radar_dynamic.struct.json + 每帧 bin)在 PARAMS_DYNAMIC/ 下,schema 不变
        self.params_dir = 'PARAMS_DYNAMIC' if (self.root_path / 'PARAMS_DYNAMIC').is_dir() else 'PARAMS'
        self.params_json = self.root_path / self.params_dir / 'radar_dynamic.struct.json'
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
        param_file = self.root_split_path / self.params_dir / ('%s.bin' % idx)
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
        """读 POINTS/{idx}.bin + struct.json → (N, 18) MSR_FEATURE_ORDER 全列。

        流程:frombuffer 解原始整数 → 对 range/doppler/azi/elv 乘 json scale(物理量)
        → build_msr_features(ego_speed, yaw_rate) 产 18 列 → 可选归一化。

        返回 src 全列而非预选列:选列统一由 DatasetTemplate.prepare_data →
        PointFeatureEncoder.forward 按 used_feature_list 完成(基类按 src_feature_list
        索引二次选列,若此处预选,越界切片会产生空列,特征只剩 xyz —— 已踩过)。
        """
        points_file = self.root_split_path / 'POINTS' / ('%s.bin' % idx)
        assert points_file.exists(), 'POINTS file missing: %s' % points_file
        raw = np.fromfile(str(points_file), dtype=self._points_dtype)
        if raw.shape[0] == 0:
            # 空帧兜底:返回 0 行 18 列
            return np.zeros((0, len(self.radar_feature_order)), dtype=np.float32)

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
        if self.use_feature_norm:
            # 归一化作用于 18 列全列(此时未选列),MEAN/STD 需按 MSR_FEATURE_ORDER 配置
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

    # ---------- image / info pkl 生成 ----------

    def get_image_shape(self, idx):
        """读 IMAGES/{idx}.png shape[:2];无图返回 [0,0]。MSR 无相机,仅占位/可视化用。"""
        img_file = self.root_split_path / 'IMAGES' / ('%s.png' % idx)
        if img_file.exists():
            return np.array(io.imread(str(img_file)).shape[:2], dtype=np.int32)
        return np.array([0, 0], dtype=np.int32)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        """装 info dict 列表(仿 VodDataset.get_infos,精简无 calib)。

        每条 info 包含:
          - point_cloud: {num_features, lidar_idx(=sample_idx)}
          - image:       {image_idx, image_shape}
          - param:       {ego_speed, yaw_rate}(get_dynamic_param)
          - annos(若 has_label): {name, gt_boxes_lidar, score(全1), [num_points_in_gt]}
        """
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            num_features = self.point_feature_encoder.num_point_features
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            param = self.get_dynamic_param(sample_idx)
            info['param'] = {'ego_speed': param['ego_speed'], 'yaw_rate': param['yaw_rate']}

            if has_label:
                gt_boxes, gt_names = self.get_label(sample_idx)
                annotations = {}
                if gt_boxes.shape[0] == 0:
                    annotations['name'] = np.array([], dtype=np.str_)
                    annotations['gt_boxes_lidar'] = np.zeros((0, 7), dtype=np.float32)
                    annotations['score'] = np.array([], dtype=np.float32)
                    info['annos'] = annotations
                    return info

                annotations['name'] = gt_names
                annotations['gt_boxes_lidar'] = gt_boxes  # 已是 lidar/雷达系 xyzwlh+heading(rad)
                annotations['score'] = np.ones((gt_boxes.shape[0],), dtype=np.float32)
                info['annos'] = annotations

                if count_inside_pts:
                    # MSR 无 FOV 裁剪,直接用全部点(已选列,前 3 列强制为 x/y/z)
                    points = self.get_radar(sample_idx)
                    num_gt = gt_boxes.shape[0]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        def _safe_process(sample_idx):
            """单帧缺文件容忍:9p 挂载盘偶发丢文件(实测 MSRv1 val 缺 00001307.bin),
            跳过并告警,不让单文件缺失毁掉整次生成。"""
            points_file = self.root_split_path / 'POINTS' / ('%s.bin' % sample_idx)
            if not points_file.exists():
                print('WARN: POINTS file missing, skip %s: %s' % (self.split, sample_idx))
                return None
            return process_single_scene(sample_idx)

        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            infos = executor.map(_safe_process, sample_id_list)
        return [i for i in infos if i is not None]

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        """读 train pkl → 每个 gt box 抠点存 gt_database/{idx}_{name}_{i}.bin → 收集 dbinfos pkl。

        仿 VodDataset.create_groundtruth_database。点前 3 列(xyz)减去 box 中心。
        """
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('msr_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_radar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (num_obj, num_points) in/out 标志

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points = gt_points.copy()
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'wb') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_info = {
                        'name': names[i],
                        'path': str(filepath.relative_to(self.root_path)),
                        'image_idx': sample_idx, 'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0],
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    # ---------- DatasetTemplate 契约 ----------

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        pred 张量 → KITTI 风格 anno 字典列表: 每帧一份 {name, score, boxes_lidar, location, dimensions, rotation_y, ...}

        camera 字段经 boxes_lidar_to_pseudo_camera 免 calib 轴变换填入(供 kitti eval 内核);
        alpha=-10 关 AOS,bbox=0(无图像,不适用);output_path 给定时逐帧落盘 pk。
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'alpha': np.full(num_samples, -10.),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            location, dimensions, rotation_y = boxes_lidar_to_pseudo_camera(pred_boxes)
            pred_dict['name'] = np.array([MSR_CLASS_LABEL[n] for n in class_names])[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['location'] = location
            pred_dict['dimensions'] = dimensions
            pred_dict['rotation_y'] = rotation_y
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if output_path is not None:
                cur_out_file = output_path / ('%s.pkl' % batch_dict['frame_id'][index])
                with open(cur_out_file, 'wb') as f:
                    pickle.dump(single_pred_dict, f)
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """
        全集聚合 det_annos vs GT annos → AP/mAP: 逐类 BEV/3D IoU 匹配,返回 (result_str, result_dict)

        GT 侧从 msr_infos annos 的 gt_boxes_lidar 轴变换成 camera 格式,det 同变换,
        调 kitti_object_eval_python.get_msr_eval_result(阈值 Car/Truck BEV0.5·3D0.25, Ped/Cyc 0.25/0.25)。
        """
        if 'annos' not in self.msr_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        import copy

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        for info in self.msr_infos:
            gt_boxes = info['annos']['gt_boxes_lidar']
            gt_names = np.array([MSR_CLASS_LABEL.get(n, n) for n in info['annos']['name']])
            gt_annos = {
                'name': gt_names,
                'alpha': np.full(len(info['annos']['name']), -10.),
                'bbox': np.zeros([len(info['annos']['name']), 4]),
                'score': np.ones(len(info['annos']['name'])),
            }
            if len(gt_boxes) > 0:
                location, dimensions, rotation_y = boxes_lidar_to_pseudo_camera(gt_boxes)
                gt_annos['location'] = location
                gt_annos['dimensions'] = dimensions
                gt_annos['rotation_y'] = rotation_y
            else:
                gt_annos['location'] = np.zeros([0, 3])
                gt_annos['dimensions'] = np.zeros([0, 3])
                gt_annos['rotation_y'] = np.zeros(0)
            eval_gt_annos.append(gt_annos)

        ap_result_str, ap_dict = kitti_eval.get_msr_eval_result(eval_gt_annos, eval_det_annos, class_names)
        return ap_result_str, ap_dict

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


def create_msr_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    """生成 MSR info pkls + gt_database(仿 create_vod_infos,精简无 calib/FOV)。

    流程:
      train_split='training', val_split='val'
      set_split(train) → get_infos(has_label=True, count_inside_pts=True) → dump msr_infos_training.pkl
      set_split(val)   → 同上 → msr_infos_val.pkl
      合并 → msr_infos_trainval.pkl
      set_split('testing') → get_infos(has_label=False) → msr_infos_testing.pkl
      最后 set_split(train) + create_groundtruth_database(msr_dbinfos_training.pkl + gt_database/)

    MSR 无 training/testing 二级目录,split 名直接对应 IMAGESETS/{split}.txt。
    """
    dataset = MsrDataset(dataset_cfg=dataset_cfg, class_names=class_names,
                         root_path=data_path, training=False)
    train_split, val_split = 'training', 'val'

    train_filename = save_path / ('msr_infos_%s.pkl' % train_split)
    val_filename = save_path / ('msr_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'msr_infos_trainval.pkl'
    test_filename = save_path / 'msr_infos_testing.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    msr_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(msr_infos_train, f)
    print('MSR info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    msr_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(msr_infos_val, f)
    print('MSR info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(msr_infos_train + msr_infos_val, f)
    print('MSR info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('testing')
    msr_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(msr_infos_test, f)
    print('MSR info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(info_path=train_filename, used_classes=class_names, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'create_msr_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.full_load(open(sys.argv[2])))
        create_msr_infos(
            dataset_cfg=dataset_cfg,
            class_names=['1', '2', '4', '5'],
            data_path=Path('/mnt/d/DataSet/MSRv1'),
            save_path=Path('/mnt/d/DataSet/MSRv1'),
        )
