import numpy as np
from collections import defaultdict, deque
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
import cv2


class Track3DPlotter:
    def __init__(
        self,
        figsize=(4, 3),
        dpi=150,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(0.0, 2.0),
        elev=90,           
        azim=-90,      
        window_frames=100,
        maxlen=100,
        y_down=True,
        title="3D Tracks (cam=0)",
        legend_max=16
    ):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.title = title
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)
        self.zlim = tuple(zlim)
        self.elev = elev
        self.azim = azim
        self.y_down = bool(y_down)

        self.legend_max = int(legend_max)
        self.window_frames = int(window_frames) if window_frames and window_frames > 0 else None
        self.track_points = defaultdict(lambda: deque(maxlen=int(maxlen) if maxlen and maxlen > 0 else None))
        self.latest_frame = -1 
        self.id2color = {}

        # 초기 세팅
        self._apply_view_and_axes()

    def _apply_view_and_axes(self):
        ax = self.ax
        ax.cla()
        ax.set_title("3D Tracks", pad=0.1, fontsize=8, weight='bold')
        ax.set_xlabel("X", labelpad=0.1, fontsize=6)
        ax.set_ylabel("Y", labelpad=0.1, fontsize=6)
        ax.set_zlabel("Z", labelpad=0.1, fontsize=6)
        self.ax.tick_params(axis='x', which='major', pad=0.0, labelsize=5)
        self.ax.tick_params(axis='y', which='major', pad=0.0, labelsize=5)
        self.ax.tick_params(axis='z', which='major', pad=0.0, labelsize=5)

        ax.view_init(elev=self.elev, azim=self.azim)

        ax.set_xlim(*self.xlim)
        if self.y_down:
            ax.set_ylim(self.ylim[1], self.ylim[0])
        else:
            ax.set_ylim(*self.ylim)

        ax.set_zlim(self.zlim[1], self.zlim[0])

        xr = self.xlim[1] - self.xlim[0]
        yr = self.ylim[1] - self.ylim[0]
        zr = self.zlim[1] - self.zlim[0]
        ax.set_box_aspect([xr, yr, zr])

    def _color_for_id(self, tid: int):
        if tid not in self.id2color:
            rng = np.random.default_rng(seed=int(tid))
            self.id2color[tid] = rng.random(3,)
        return self.id2color[tid]

    def add_points_from_tracking(self, tracking_result, cam_select=0):
        for res in tracking_result:
            if int(res.get("cam", -1)) != cam_select:
                continue
            x, y, z = res.get("xyz", [np.nan, np.nan, np.nan])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            tid = int(res.get("trk_id"))
            frame = int(res.get("frame", -1))
            if frame >= 0:
                if frame > self.latest_frame:
                    self.latest_frame = frame
                self.track_points[tid].append((frame, float(x), float(y), float(z)))
            else:
                pseudo_frame = self.latest_frame + 1
                self.latest_frame = pseudo_frame
                self.track_points[tid].append((pseudo_frame, float(x), float(y), float(z)))

    def _filtered_points(self, seq):
        if not seq:
            return None

        if self.window_frames is None or self.latest_frame < 0:
            return np.asarray([(f, x, y, z) for (f, x, y, z) in seq], dtype=float)

        start_frame = self.latest_frame - self.window_frames + 1
        return np.asarray(
            [(f, x, y, z) for (f, x, y, z) in seq if f >= start_frame],
            dtype=float
        )

    def _draw_tracks(self):
        ax = self.ax
        legend_handles = []

        tids = list(self.track_points.keys())
        tids.sort()

        show_tids = tids[-self.legend_max:] if len(tids) > self.legend_max else tids

        for tid in tids:
            seq = self.track_points[tid]
            filt = self._filtered_points(seq)
            if filt is None or len(filt) == 0:
                continue
            pts = filt[:, 1:4]
            c = self._color_for_id(tid)

            if len(pts) > 1:
                ax.scatter(pts[:-1, 0], pts[:-1, 1], pts[:-1, 2], s=1, c=[c], depthshade=False)
            ax.scatter(pts[-1:, 0], pts[-1:, 1], pts[-1:, 2], s=2, c=[c], depthshade=True, marker='.')

            if tid in show_tids:
                legend_handles.append(mpatches.Patch(color=c, label=f"ID {tid}"))

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=5, framealpha=0.8)

    def render_bgr(self) -> np.ndarray:
        self._apply_view_and_axes()
        self._draw_tracks()

        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb = buf[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def render_bgr_resized_to_height(self, target_h: int) -> np.ndarray:
        bgr = self.render_bgr()
        if target_h is None or target_h <= 0:
            return bgr
        scale = target_h / bgr.shape[0]
        new_w = max(1, int(round(bgr.shape[1] * scale)))
        return cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
