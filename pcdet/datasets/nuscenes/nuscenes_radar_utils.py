from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from .nuscenes_utils import map_name_from_general_to_detection, quaternion_yaw


RADAR_CHANNELS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


def get_available_radar_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_rec = nusc.get("scene", scene["token"])
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_token = sample_rec["data"]["RADAR_FRONT"]
        radar_path = nusc.get_sample_data_path(sd_token)
        if Path(radar_path).exists():
            available_scenes.append(scene)
    print("exist radar scene num:", len(available_scenes))
    return available_scenes


def _rel_path(path, data_path):
    return Path(path).relative_to(data_path).__str__()


def _sensor_to_ego_matrix(calibrated_sensor_rec):
    return transform_matrix(
        calibrated_sensor_rec["translation"],
        Quaternion(calibrated_sensor_rec["rotation"]),
        inverse=False,
    ).astype(np.float32)


def _box_global_to_ego(box, ego_pose_rec):
    box.translate(-np.asarray(ego_pose_rec["translation"]))
    box.rotate(Quaternion(ego_pose_rec["rotation"]).inverse)

    vel = np.asarray(box.velocity)
    if vel.shape[0] == 3 and not np.any(np.isnan(vel)):
        box.velocity = Quaternion(ego_pose_rec["rotation"]).inverse.rotate(vel)
    return box


def fill_radar_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc="create radar infos", dynamic_ncols=True)

    ref_chan = "RADAR_FRONT"

    for sample in nusc.sample:
        progress_bar.update()

        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]
        ref_radar_path = nusc.get_sample_data_path(ref_sd_token)

        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        ).astype(np.float32)
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"],
            Quaternion(ref_cs_rec["rotation"]),
            inverse=True,
        ).astype(np.float32)

        radar_channels = {}
        radar_t_ego_sensor = {}
        for ch in RADAR_CHANNELS:
            sd_rec = nusc.get("sample_data", sample["data"][ch])
            cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            radar_channels[ch] = _rel_path(nusc.get_sample_data_path(sd_rec["token"]), data_path)
            radar_t_ego_sensor[ch] = _sensor_to_ego_matrix(cs_rec)

        info = {
            "radar_path": _rel_path(ref_radar_path, data_path),
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
            "ref_chan": ref_chan,
            "radar_channels": radar_channels,
            "radar_T_ego_sensor": radar_t_ego_sensor,
        }

        curr_sd_rec = ref_sd_rec
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweeps.append({
                        "radar_path": _rel_path(ref_radar_path, data_path),
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": radar_t_ego_sensor[ref_chan],
                        "time_lag": 0.0,
                    })
                else:
                    sweeps.append(sweeps[-1])
                continue

            curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])
            curr_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
            curr_cs_rec = nusc.get("calibrated_sensor", curr_sd_rec["calibrated_sensor_token"])

            global_from_curr_car = transform_matrix(
                curr_pose_rec["translation"],
                Quaternion(curr_pose_rec["rotation"]),
                inverse=False,
            )
            curr_car_from_sensor = transform_matrix(
                curr_cs_rec["translation"],
                Quaternion(curr_cs_rec["rotation"]),
                inverse=False,
            )
            sensor_to_ref_ego = reduce(np.dot, [car_from_global, global_from_curr_car, curr_car_from_sensor])
            time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]
            sweeps.append({
                "radar_path": _rel_path(nusc.get_sample_data_path(curr_sd_rec["token"]), data_path),
                "sample_data_token": curr_sd_rec["token"],
                "transform_matrix": sensor_to_ref_ego.astype(np.float32),
                "time_lag": time_lag,
            })
        info["sweeps"] = sweeps

        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            num_lidar_pts = np.array([anno["num_lidar_pts"] for anno in annotations])
            num_radar_pts = np.array([anno["num_radar_pts"] for anno in annotations])
            mask = num_lidar_pts + num_radar_pts > 0

            ref_boxes = []
            for anno in annotations:
                box = nusc.get_box(anno["token"])
                box.velocity = nusc.box_velocity(anno["token"])
                ref_boxes.append(_box_global_to_ego(box, ref_pose_rec))

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            info["gt_boxes"] = gt_boxes[mask, :]
            info["gt_boxes_velocity"] = velocity[mask, :]
            info["gt_names"] = np.array([map_name_from_general_to_detection[n] for n in names])[mask]
            info["gt_boxes_token"] = tokens[mask]
            info["num_lidar_pts"] = num_lidar_pts[mask]
            info["num_radar_pts"] = num_radar_pts[mask]

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos
