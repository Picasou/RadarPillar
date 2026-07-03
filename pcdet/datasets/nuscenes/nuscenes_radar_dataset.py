import pickle
from pathlib import Path

import numpy as np

from ...utils import common_utils
from ..dataset import DatasetTemplate


# nuScenes 雷达 .pcd 共 18 维，本项目只用其中 7 维：
#   0/1/2  -> x/y/z       (ego 坐标系, 米)
#   5      -> rcs        (dBsm)
#   8/9    -> vx/vy_comp (自车运动补偿后, 世界系)
#   17     -> time       (相对当前关键帧的时间偏移, 秒)
# 注：原始 vx/vy (6/7) 是 sensor frame 多普勒，不使用。
RADAR_USED_FIELDS = ['x', 'y', 'z', 'rcs', 'vx', 'vy', 'time']
RADAR_SRC_FIELDS = ['x', 'y', 'z', 'rcs', 'vx', 'vy', 'time']
# 18 维 PCD 中训练用字段的列下标。
RADAR_FIELD_TO_INDEX = {
    'x': 0, 'y': 1, 'z': 2,
    'rcs': 5, 'vx': 8, 'vy': 9,
    'time': 17,
}
RADAR_NUM_FEATURES = 7


class NuScenesRadarDataset(DatasetTemplate):
    """nuScenes 雷达数据集，加载 5 通道雷达点云并叠加历史 sweep，输出 ego 系合并点云。"""

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        if root_path is None:
            root_path = (Path(dataset_cfg.DATA_PATH) / dataset_cfg.VERSION).resolve()
        else:
            root_path = Path(root_path).resolve() / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=training, root_path=root_path, logger=logger
        )
        # 父类会用 DATA_PATH 覆盖 root_path，强制保留绝对路径
        self.root_path = root_path
        self.infos = []
        self.include_nuscenes_radar_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        debug_num_samples = self.dataset_cfg.get('DEBUG_NUM_SAMPLES', None)
        if debug_num_samples is not None:
            self.infos = self.infos[:int(debug_num_samples)]

    def include_nuscenes_radar_data(self, mode):
        self.logger.info('正在加载 nuScenes 雷达数据集')
        radar_infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                radar_infos.extend(infos)
        self.infos.extend(radar_infos)
        self.logger.info('nuScenes 雷达数据集总样本数: %d' % len(radar_infos))

    def balanced_infos_resampling(self, infos):
        """CBGS（Class-Balanced Grouping and Sampling），平衡罕见类采样比例。"""
        if self.class_names is None:
            return infos
        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)
        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}
        sampled_infos = []
        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]
        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('CBGS 均衡采样后总样本数: %s' % len(sampled_infos))
        return sampled_infos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = self.infos[index]
        points = self.get_radar_with_sweeps(info, max_sweeps=self.dataset_cfg.get('MAX_SWEEPS', 1))

        input_dict = {
            'points': points,
            'frame_id': Path(info['radar_path']).stem,
            'metadata': {'token': info['token']},
        }
        if 'gt_boxes' in info:
            input_dict.update({
                'gt_names': info['gt_names'],
                'gt_boxes': info['gt_boxes'],
            })
        data_dict = self.prepare_data(data_dict=input_dict)
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes
        # 不预测速度时，gt_boxes 去掉两列速度 (idx 7, 8)，保留 yaw(-1)
        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        return data_dict

    @staticmethod
    def _ensure_sensor_to_ego(T_ego_sensor):
        if T_ego_sensor is None:
            return None
        T = np.asarray(T_ego_sensor, dtype=np.float32)
        # 旧 pkl 把 ego->sensor 错存成 sensor->ego；radar 安装 z>0，若 z 平移为负说明方向反了
        if T.shape == (4, 4) and T[2, 3] < 0:
            T = np.linalg.inv(T).astype(np.float32)
        return T

    @staticmethod
    def load_one_radar_pcd(pcd_path, used_indices, T_ego_sensor=None):
        """读取单个 nuScenes 雷达 .pcd，按 used_indices 切片为 (N, 7) float32；可选 xyz 投到 ego 系。

        Args:
            pcd_path: 绝对路径。
            used_indices: 18 维 PCD 中保留的列下标。
            T_ego_sensor: sensor→ego 4x4 矩阵；None 时 xyz 保留 sensor frame（兼容旧 pkl）。
        """
        # nuScenes 雷达 .pcd 带 ASCII 头，必须用 devkit 解析
        from nuscenes.utils.data_classes import RadarPointCloud
        pc = RadarPointCloud.from_file(pcd_path)
        cols = pc.points[used_indices, :].T.astype(np.float32)

        if T_ego_sensor is None:
            return cols

        # 仅 xyz 投到 ego；rcs/vx/vy/time 不参与 sensor→ego 变换
        T = NuScenesRadarDataset._ensure_sensor_to_ego(T_ego_sensor)
        xyz = cols[:, :3]
        homo = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
        )
        xyz_ego = homo @ T.T
        cols = np.concatenate([xyz_ego[:, :3], cols[:, 3:]], axis=1)
        return cols

    def get_radar_with_sweeps(self, info, max_sweeps=1):
        """合并当前关键帧 5 通道雷达点云与过去 (max_sweeps-1) 个 sweep，输出 (N, 7) ego 系点云。"""
        used_indices = [RADAR_FIELD_TO_INDEX[f] for f in RADAR_USED_FIELDS]
        T_per_ch = info.get('radar_T_ego_sensor', {})

        # 5 个雷达各覆盖一个扇区，合起来才是 nuScenes 的完整 360°；只用 RADAR_FRONT 会丢 4/5 点云
        channel_points = []
        for ch, rel_path in info.get('radar_channels', {}).items():
            p = self.root_path / rel_path
            if p.exists():
                channel_points.append(
                    self.load_one_radar_pcd(
                        str(p), used_indices, T_per_ch.get(ch)
                    )
                )

        if len(channel_points) == 0:
            # 旧版 pkl 没有 radar_channels 字段，回退到单通道
            fallback = self.root_path / info['radar_path']
            channel_points.append(self.load_one_radar_pcd(str(fallback), used_indices, None))

        current_points = np.concatenate(channel_points, axis=0)

        # 最后一列存 time（当前帧置 0），保持 7 维不变
        current_points[:, -1] = 0.0

        point_list = [current_points]

        # sweep 在 pkl 中只存 ref_chan 单通道；当前 MAX_SWEEPS=1 不进此分支
        sweeps = info.get('sweeps', [])
        if max_sweeps > 1 and len(sweeps) > 0:
            n_pick = min(max_sweeps - 1, len(sweeps))
            pick_idx = np.random.choice(len(sweeps), n_pick, replace=False)
            for k in pick_idx:
                sweep = sweeps[k]
                sweep_path = self.root_path / sweep['radar_path']
                sweep_points = self.load_one_radar_pcd(
                    str(sweep_path), used_indices, sweep.get('transform_matrix')
                )
                # 复用最后一列存 time_lag，避免特征维度从 7 变成 8
                sweep_points[:, -1] = float(sweep['time_lag'])
                point_list.append(sweep_points)

        points = np.concatenate(point_list, axis=0)
        return points

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """把模型输出 tensor 转成 nuScenes 评估器要求的 dict 格式。"""
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples),
                'metadata': [{} for _ in range(num_samples)],
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        """格式化预测结果并跑 nuScenes 官方评估；box 格式与 lidar 版 utils 兼容（7 维 ego frame）。"""
        total_pred = sum(len(anno.get('name', [])) for anno in det_annos)
        if total_pred == 0:
            output_path = Path(kwargs['output_path'])
            output_path.mkdir(exist_ok=True, parents=True)
            result_str = 'No predictions for nuScenes radar evaluation. mAP/NDS are 0.0.\n'
            result_dict = {
                'nusc/mAP': 0.0,
                'nusc/NDS': 0.0,
            }
            for class_name in class_names:
                result_dict[f'nusc/{class_name}_AP'] = 0.0
            self.logger.info(result_str.strip())
            return result_str, result_dict

        import json
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
        from . import nuscenes_utils

        nusc = NuScenes(version=self.dataset_cfg.VERSION,
                        dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        # 标记为 radar-only 评估
        nusc_annos['meta'] = {
            'use_camera': False, 'use_lidar': False,
            'use_radar': True, 'use_map': False, 'use_external': False,
        }
        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc_radar.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)
        self.logger.info(f'预测结果已保存到 {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return '测试集无标注，无法评估', {}

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test',
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
            nusc_eval = NuScenesEval(
                nusc, config=eval_config, result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path), verbose=True,
            )
            metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)
            with open(output_path / 'metrics_summary.json', 'r') as f:
                metrics = json.load(f)
            result_str, result_dict = nuscenes_utils.format_nuscene_results(
                metrics, self.class_names, version=eval_version
            )
            return result_str, result_dict
        except Exception as e:
            self.logger.error(f'评估失败: {e}')
            return '', {}


def create_nuscenes_radar_info(version, data_path, save_path, max_sweeps=10):
    """为 nuScenes 生成雷达 infos pkl；元数据走 devkit，路径和变换矩阵走雷达专属逻辑。"""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_radar_utils

    data_path = data_path / version
    save_path = save_path / version
    save_path.mkdir(parents=True, exist_ok=True)

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes, val_scenes = splits.train, splits.val
    elif version == 'v1.0-test':
        train_scenes, val_scenes = splits.test, []
    elif version == 'v1.0-mini':
        train_scenes, val_scenes = splits.mini_train, splits.mini_val

    nusc = NuScenes(version=version, dataroot=str(data_path), verbose=True)
    # 仅保留雷达数据完整的场景
    available_scenes = nuscenes_radar_utils.get_available_radar_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
    print('%s: 训练场景(%d), 验证场景(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_radar_utils.fill_radar_infos(
        data_path=data_path, nusc=nusc,
        train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps,
    )

    if version == 'v1.0-test':
        print('测试样本数: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_radar_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('训练样本数: %d, 验证样本数: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_radar_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_radar_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='nuScenes 雷达数据集 infos 生成脚本')
    parser.add_argument('--cfg_file', type=str, default=None, help='数据集配置文件路径')
    parser.add_argument('--func', type=str, default='create_nuscenes_radar_info', help='要执行的函数名')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='数据集版本')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_radar_info':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_radar_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )
        radar_dataset = NuScenesRadarDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True,
        )
        if hasattr(radar_dataset, 'create_groundtruth_database'):
            radar_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
