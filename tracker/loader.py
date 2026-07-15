from __future__ import annotations
import os
import warnings
import numpy as np

from .schemas import (Cfg, VDS, VDD, PT, PTs, GT, GTs,
                      Obj, Objs, FRAME, FRAMEs)
from .utils.rw_struct import (struct_read,
                              Raw_DetHead, Raw_Det,
                              Raw_TrkHead, Raw_Trk,
                              Raw_Vdd, Raw_Vds)
from .utils.common import compute_cycle_s, compensate_state



class Loader:
    """
    in : 数据目录 path (含 radar.default/0100|0101|0200|0201|2021|2031 bin)
    out: FRAMEs (逐帧 gts/pts/vdd/objs), VDS (静态参数)
    """

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.relpath = 'radar.default'

    # ---- 公共接口 ----

    def getframes(self, path: str) -> FRAMEs:
        data_path = os.path.join(path, self.relpath)
        vds = self.getvds(path)
        pts_list = self._load_PTs(data_path, vds)
        gts_list = self._load_GTs(data_path)
        vdd_list = self._load_VDD(data_path)
        objs_list = self._load_objs(data_path)

        n = min(len(pts_list), len(vdd_list), len(objs_list))
        if n != len(pts_list) or n != len(vdd_list) or n != len(objs_list):
            warnings.warn(
                f"[loader] 帧级长度不一致, 已截断到 {n} "
                f"(pts={len(pts_list)}, vdd={len(vdd_list)}, objs={len(objs_list)})",
                RuntimeWarning, stacklevel=2
            )

        frames = []
        for i in range(n):
            frame = FRAME(
                gts=gts_list[i] if i < len(gts_list) else GTs(num=0, Lst=[]),
                pts=pts_list[i],
                vdd=vdd_list[i],
                objs=objs_list[i]
            )
            frames.append(frame)
        return FRAMEs(num=len(frames), Lst=frames)

    def getvds(self, path: str) -> VDS:
        data_path = os.path.join(path, self.relpath)
        vds_file = self._find_vds_file(data_path)
        if vds_file is not None:
            return self._load_vds(vds_file)
        return self._vds_from_cfg()

    # ---- 内部方法 ----

    def _load_PTs(self, path: str, vds: VDS) -> list[PTs]:
        file_0100 = os.path.join(path, '0100.00000.bin')
        file_0101 = os.path.join(path, '0101.00000.bin')
        if not os.path.exists(file_0100) or not os.path.exists(file_0101):
            return []

        det_head_list = struct_read(file_0100, Raw_DetHead)
        det_list = struct_read(file_0101, Raw_Det)

        z_pos = vds.z_pos_m
        pts_list = []
        offset, limit = 0, len(det_list)
        for head in det_head_list:
            num = head.det_num
            if offset + num > limit:
                num = limit - offset
            if num < 0:
                break
            pts = []
            for j in range(num):
                d = det_list[offset + j]
                r     = d.range / 100
                theta = np.radians(d.azimuth / 100)
                phi   = np.radians(d.elevation / 100)
                cos_phi = np.cos(phi)
                pt = PT(
                    beam=d.beam,
                    extra_cnt=d.extra_cnt,
                    exist_confidence=d.exist_confi,
                    doppler_anti_amb_confi=d.doppler_anti_amb_confi,
                    id=d.id,
                    flags=d.flags,
                    rcs=d.rcs,
                    snr=d.snr,
                    frame=d.frame,
                    range_m=r,
                    ang_rad=theta,
                    elv_rad=phi,
                    doppler_mps=d.doppler / 100,
                    x_m=r * cos_phi * np.cos(theta),
                    y_m=r * cos_phi * np.sin(theta),
                    z_m=r * np.sin(phi) + z_pos,
                )
                pts.append(pt)
            pts_list.append(PTs(num=len(pts), Lst=pts))
            offset += head.det_num
        return pts_list

    def _load_objs(self, path: str) -> list[Objs]:
        file_0200 = os.path.join(path, '0200.00000.bin')
        file_0201 = os.path.join(path, '0201.00000.bin')
        if not os.path.exists(file_0200) or not os.path.exists(file_0201):
            return []

        trk_head_list = struct_read(file_0200, Raw_TrkHead)
        trk_list = struct_read(file_0201, Raw_Trk)

        vdd_path = os.path.join(path, '2021.00000.bin')
        vdd_raw_list = struct_read(vdd_path, Raw_Vdd) if os.path.exists(vdd_path) else []
        cycle_s_list = compute_cycle_s(vdd_raw_list)

        objs_list = []
        offset, limit = 0, len(trk_list)
        for frame_i, head in enumerate(trk_head_list):
            num = head.trk_num
            if offset + num > limit:
                num = limit - offset
            if num < 0:
                break

            cycle_s = cycle_s_list[frame_i] if frame_i < len(cycle_s_list) else 0.05
            v_raw = vdd_raw_list[frame_i] if frame_i < len(vdd_raw_list) else None
            v = VDD(
                speed_ms=v_raw.hostVelocity_mps,
                yaw_rate=v_raw.vehicleYawRate_radps,
                gear=v_raw.driveGearEngaged,
            ) if v_raw is not None else None

            objs = []
            for j in range(num):
                t = trk_list[offset + j]
                x, y, vx, vy = compensate_state(
                    t.x / 100.0, t.y / 100.0,
                    t.vx / 100.0, t.vy / 100.0,
                    v, cycle_s
                )
                objs.append(Obj(
                    id=t.id,
                    x=x, y=y, vx=vx, vy=vy,
                    length=t.length / 100.0,
                    width=t.width / 100.0,
                    heading=t.heading / 100.0,
                    type=t.classification,
                    isghost=0,
                    ispassable=0,
                ))
            objs_list.append(Objs(num=len(objs), Lst=objs))
            offset += head.trk_num
        return objs_list

    def _load_GTs(self, path: str) -> list[GTs]:
        return []

    def _load_VDD(self, path: str) -> list[VDD]:
        file_2021 = os.path.join(path, '2021.00000.bin')
        if not os.path.exists(file_2021):
            return []
        vdd_raw_list = struct_read(file_2021, Raw_Vdd)
        return [VDD(speed_ms=b.hostVelocity_mps,
                    yaw_rate=b.vehicleYawRate_radps,
                    gear=b.driveGearEngaged)
                for b in vdd_raw_list]

    def _find_vds_file(self, path: str):
        vds_file = os.path.join(path, '2031.00000.bin')
        return vds_file if os.path.exists(vds_file) else None

    def _load_vds(self, vds_file) -> VDS:
        vds_raw_list = struct_read(vds_file, Raw_Vds, 1)
        if not vds_raw_list:
            return self._vds_from_cfg()
        b = vds_raw_list[0]
        return VDS(
            wheelbase_m=b.wheelbase_m,
            x_pos_m=b.xpos,
            y_pos_m=b.ypos,
            z_pos_m=b.zpos,
            rotation_rad=b.rotation,
            cycle_s=0.1,
            oritation=b.orientation
        )

    def _vds_from_cfg(self) -> VDS:
        v = self.cfg.RUN.vds
        return VDS(
            wheelbase_m=v.wheelbase_m,
            x_pos_m=v.x_pos_m,
            y_pos_m=v.y_pos_m,
            z_pos_m=v.z_pos_m,
            rotation_rad=0.0,
            cycle_s=v.cycle_s,
            oritation=0,
        )