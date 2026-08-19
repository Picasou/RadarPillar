"""BEV 可视化: 点云 + GT/检测/航迹按开关叠加, PNG/GIF 落盘, 由 VISUAL 段统一配置."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from .schemas import FRAME, Trk
from .utils.common import load_data_cfg

# 类色避开点云 doppler diverging(蓝↔灰↔红)色域: 黄/品红/草绿, 与 tools 侧可视化规范一致
DEFAULT_CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_COLORS = ['#FFC53D', '#E85CA8', '#7AC943']
FALLBACK_COLOR = '#9e9e9e'
DIV_BLUE, DIV_GRAY, DIV_RED = '#2a78d6', '#f0efec', '#e34948'
DOPPLER_CMAP = LinearSegmentedColormap.from_list('trk_div', [DIV_BLUE, DIV_GRAY, DIV_RED])

# 线型语义(对齐可视化规范): GT=细实线无填充 / 检测=中虚线 / 航迹=粗虚线+半透明填充+ID
GT_LW, OBJ_LW, TRK_LW = 1.2, 1.8, 2.8
TRK_FILL_ALPHA = 0.25
OUT_ROOT = Path('output/tracker_viz')


def _box_corners(x: float, y: float, length: float, width: float, heading: float) -> np.ndarray:
    """
    框角点: 中心(x,y) 长×宽 朝向 heading(rad) → (4,2) 车规系 [x 前, y 左]
    """
    dx = np.array([length / 2, length / 2, -length / 2, -length / 2])
    dy = np.array([width / 2, -width / 2, -width / 2, width / 2])
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([x + dx * c - dy * s, y + dx * s + dy * c], axis=1)


class Visualizer:
    """
    逐帧 BEV 出图: Tracker step7 调 run, 序列结束调 on_seq_end 收 GIF
    """
    def __init__(self, cfg, class_names: list = None) -> None:
        v = cfg.VISUAL
        self.enable = (v.enable == 1)
        self.save = v.save
        self.label = (v.label == 1)
        self.show = v.show
        self.cycle_s = cfg.RUN.vds.cycle_s
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self.pcr = load_data_cfg().POINT_CLOUD_RANGE
        self._cur_seq = 'seq'
        self._gif_frames: list[np.ndarray] = []

    # ---- 对外入口 ----
    def begin_seq(self, seq_name: str) -> None:
        """
        序列切换: 记录当前序列名(PNG 目录/GIF 文件名), 清 GIF 缓冲
        """
        self._cur_seq = seq_name
        self._gif_frames = []

    def run(self, frame: FRAME, trks: list[Trk]) -> None:
        """
        单帧出图: 画布画完按 save 落 PNG/攒 GIF 帧
        """
        if not self.enable:
            return
        fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
        self._draw_points(ax, frame)
        n_gt = self._draw_gts(ax, frame) if self.show.get('gts', 1) else 0
        n_obj = self._draw_objs(ax, frame) if self.show.get('objs', 1) else 0
        n_trk = self._draw_tracks(ax, trks) if self.show.get('tracks', 1) else 0

        # 车规 BEV: x 前(纵轴) y 左(横轴), range 固定便于跨帧对比
        ax.set_xlim(self.pcr[1], self.pcr[4])
        ax.set_ylim(self.pcr[3], self.pcr[0])     # x 前朝上 → 纵轴反转
        ax.set_aspect('equal')
        ax.set_xlabel('y (m)')
        ax.set_ylabel('x (m)')
        ax.set_title('%s  gt=%d det=%d trk=%d' % (frame.frame_id, n_gt, n_obj, n_trk))
        self._draw_legend(ax, frame)

        out_dir = OUT_ROOT / self._cur_seq
        if self.save in (2, 3):
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / ('%s.png' % frame.frame_id), dpi=110,
                        facecolor='white', bbox_inches='tight')
        if self.save in (1, 3):
            fig.canvas.draw()
            self._gif_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    def on_seq_end(self) -> None:
        """
        序列收尾: 攒的帧合成一个 GIF 落盘
        """
        if not (self.enable and self.save in (1, 3) and self._gif_frames):
            self._gif_frames = []
            return
        out = OUT_ROOT / ('%s.gif' % self._cur_seq)
        out.parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(arr) for arr in self._gif_frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(self.cycle_s * 1000), loop=0)
        print('  [visualizer] %d frames -> %s' % (len(imgs), out))
        self._gif_frames = []

    # ---- 绘制件 ----
    def _draw_points(self, ax, frame: FRAME) -> None:
        if not self.show.get('points', 1):
            return
        pts = frame.proc.points
        if pts is None or pts.shape[0] == 0:
            return
        c = pts[:, 5]                                     # v_r_comp 对地多普勒
        vmax = max(float(np.abs(c).max()), 1e-3)
        ax.scatter(pts[:, 1], pts[:, 0], c=c, cmap=DOPPLER_CMAP,
                   norm=Normalize(vmin=-vmax, vmax=vmax), s=4, linewidths=0, alpha=0.8)

    def _draw_gts(self, ax, frame: FRAME) -> int:
        for g in frame.gts.Lst:
            if getattr(g, 'isghost', 0):
                continue
            self._patch(ax, g.x, g.y, g.length, g.width, g.heading,
                        self._color(g.type), lw=GT_LW, ls='-', fill=False)
        return len(frame.gts.Lst)

    def _draw_objs(self, ax, frame: FRAME) -> int:
        for o in frame.objs.Lst:
            self._patch(ax, o.x, o.y, o.length, o.width, o.heading,
                        self._color(o.type), lw=OBJ_LW, ls='--', fill=False)
        return len(frame.objs.Lst)

    def _draw_tracks(self, ax, trks: list[Trk]) -> int:
        n = 0
        for t in trks:
            if not t.obstacle_prob:                       # 只画上桌航迹, 与输出层口径一致
                continue
            h = np.radians(t.heading_deg)                 # Trk 契约: heading_deg 为度
            self._patch(ax, t.x_m, t.y_m, t.length_m, t.width_m, h,
                        self._color(t.type), lw=TRK_LW, ls='--', fill=True)
            if self.label:
                fx = t.x_m + t.length_m / 2 * np.cos(h)    # ID 放框前端中点
                fy = t.y_m + t.length_m / 2 * np.sin(h)
                ax.text(fy, fx, str(t.id), fontsize=8, ha='center', va='bottom',
                        color='black', bbox=dict(boxstyle='round,pad=0.15',
                                                 fc='white', ec='none', alpha=0.7))
            n += 1
        return n

    def _patch(self, ax, x, y, length, width, heading, color, lw, ls, fill) -> None:
        """
        单框绘制: Polygon 角点法, 车规朝向(plot 横=y 纵=x)
        """
        if length <= 0 or width <= 0:
            return
        corners = _box_corners(x, y, length, width, heading)
        ax.add_patch(MplPolygon(corners[:, [1, 0]], closed=True, edgecolor=color,
                                facecolor=(color if fill else 'none'),
                                linewidth=lw, linestyle=ls,
                                alpha=(TRK_FILL_ALPHA if fill else 1.0)))

    def _draw_legend(self, ax, frame: FRAME) -> None:
        handles = [Line2D([0], [0], color=c, lw=2, label=self._class_name(i))
                   for i, c in enumerate(CLASS_COLORS[:len(self.class_names)])]
        if self.show.get('gts', 1):
            handles.append(Line2D([0], [0], color='gray', lw=GT_LW, ls='-', label='GT'))
        if self.show.get('objs', 1):
            handles.append(Line2D([0], [0], color='gray', lw=OBJ_LW, ls='--', label='Det'))
        if self.show.get('tracks', 1):
            handles.append(Line2D([0], [0], color='gray', lw=TRK_LW, ls='--', label='Trk'))
        if self.show.get('points', 1):
            handles.append(Line2D([0], [0], marker='o', color='none',
                                  markerfacecolor=DIV_BLUE, markersize=5, label='pts(dpl)'))
        ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.85)

    # ---- 小工具 ----
    def _color(self, type_idx: int) -> str:
        if 0 <= type_idx - 1 < len(CLASS_COLORS):
            return CLASS_COLORS[type_idx - 1]             # detector label 为 1-based
        return FALLBACK_COLOR

    def _class_name(self, i: int) -> str:
        return self.class_names[i] if i < len(self.class_names) else 'cls%d' % (i + 1)
