from __future__ import annotations
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional
from queue import Empty

import os
import numpy as np

# Force software rendering for viewer-only processes (helps on WSLg)
# os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "0")

import pyvista as pv

# Quell noisy destructor AttributeErrors on some PyVista builds
try:
    from pyvista.plotting import plotter as _pv_plotter_mod  # type: ignore
    _pv_plotter_mod.BasePlotter.__del__ = lambda self: None  # noqa: E731
except Exception:
    pass

# Report OpenGL backend details for diagnostics
def _report_gl_backend(pl):
    try:
        rw = getattr(pl, "ren_win", None) or getattr(pl, "render_window", None)
        caps = rw.ReportCapabilities()
        for line in caps.splitlines():
            if ("OpenGL vendor" in line) or ("OpenGL renderer" in line) or ("OpenGL version" in line):
                print("[viz-sim]", line.strip())
    except Exception as e:
        print("[viz-sim] GL backend probe failed:", repr(e))

# Cap for GT points drawn (performance guardrail)
MAX_GT_PTS = 50000

# --- Quaternion helpers (NumPy only; no JAX here) ---
def quat_to_R_numpy(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from quaternion [w,x,y,z]."""
    w, x, y, z = [float(v) for v in q]
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)],
    ], dtype=np.float32)

def quat_to_forward_numpy(q: np.ndarray) -> np.ndarray:
    """Return body +X axis in world frame (camera forward)."""
    R = quat_to_R_numpy(q)
    return (R @ np.array([1.0, 0.0, 0.0], np.float32)).astype(np.float32)


def make_point_cloud_polydata(pts: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData()
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"make_point_cloud_polydata expected (N,3), got {pts.shape}")
    n = pts.shape[0]
    poly.points = pts
    verts = np.empty(n * 2, dtype=np.int64)
    verts[0::2] = 1
    verts[1::2] = np.arange(n, dtype=np.int64)
    poly.verts = verts
    return poly


@dataclass
class Playback:
    p_seq: Optional[np.ndarray] = None
    q_seq: Optional[np.ndarray] = None
    dt: float = 0.05
    T: int = 0
    N: int = 0
    idx: int = 0
    playing: bool = True
    speed: float = 1.0
    last_tick: float = 0.0

    def set_traj(self, p_seq: np.ndarray, q_seq: Optional[np.ndarray], dt: float):
        p_seq = np.asarray(p_seq, np.float32)
        self.p_seq = p_seq
        self.q_seq = None if q_seq is None else np.asarray(q_seq, np.float32)
        self.dt = float(dt)
        self.T, self.N = (p_seq.shape[0], p_seq.shape[1])
        self.idx = 0
        self.last_tick = 0.0

    def step_if_due(self) -> bool:
        if not self.playing or self.p_seq is None or self.T == 0:
            return False
        now = time.perf_counter()
        if self.last_tick == 0.0:
            self.last_tick = now
            return False
        need = (1.0 / max(1e-6, self.speed)) * self.dt
        if (now - self.last_tick) >= need:
            self.idx = min(self.idx + 1, self.T - 1)
            self.last_tick = now
            return True
        return False


class BasicTrainViewer:
    def __init__(self):
        pv.set_plot_theme("document")
        try:
            self.pl = pv.Plotter(shape=(1, 3), window_size=(1400, 720), enable_keybindings=False)
        except TypeError:
            self.pl = pv.Plotter(shape=(1, 3), window_size=(1400, 720))
            for attr in ("enable_keybindings", "enable_key_bindings"):
                if hasattr(self.pl, attr):
                    try:
                        setattr(self.pl, attr, False)
                    except Exception:
                        pass
        self.pl.add_text("live_mvp :: training viewer (GT + drones + live map + loss)", font_size=10)

        self.gt_actor = None
        self.gt_poly: Optional[pv.PolyData] = None

        self.goal_poly: Optional[pv.PolyData] = None
        self.goal_actor = None

        self.drone_poly: Optional[pv.PolyData] = None
        self.drone_actor = None  # legacy single-actor handle (unused)
        self.drone_actor_left = None
        self.drone_actor_mid = None

        self.lb = np.array([-6.0, -6.0, 0.0], np.float32)
        self.ub = np.array([+6.0, +6.0, 4.0], np.float32)

        self.pb = Playback()

        self.hud_actor = None
        self._last_hud_update = 0.0
        self.arrow_actor = None  # legacy arrow glyphs (unused after optimization)
        self.arrow_lines_poly = None  # persistent polydata for arrow line segments
        self.arrow_lines_actor = None  # persistent actor for arrow line segments

        self.speed_slider = None
        self._add_speed_slider()

        self._bind_keys()

        # --- Panes ---
        # Left: GT + drones
        try:
            self.pl.subplot(0, 0)
            self.pl.show_axes()
        except Exception:
            pass

        # Middle: Live reconstruction (lazy init on first 'init' containing live_pts)
        self.live_poly: Optional[pv.PolyData] = None
        self.live_actor = None
        self.live_lo_hint = 0.0
        self.live_hi_hint = 1.0
        self.live_auto_clim = True
        self.live_thresh_alpha = 1.0  # show points with value <= this threshold
        self.live_field_last: Optional[np.ndarray] = None
        self._live_thresh_user_set = False
        try:
            self.pl.subplot(0, 1)
            self.pl.show_axes()
        except Exception:
            pass

        # Right: Metrics chart (loss over time)
        self.metric_series = ["loss"]
        self.metric_window = 5000
        self.metric_x = deque(maxlen=self.metric_window)
        self.metric_y = {s: deque(maxlen=self.metric_window) for s in self.metric_series}
        self.chart = None
        self.lines = {}
        try:
            self.pl.subplot(0, 2)
        except Exception:
            pass
        self._init_chart()

        # Live pane threshold slider (global overlay)
        self._add_live_slider()

    def _add_speed_slider(self):
        def _cb(val):
            try:
                self.pb.speed = float(val)
            except Exception:
                pass
        try:
            self.speed_slider = self.pl.add_slider_widget(
                callback=_cb, rng=[0.1, 4.0], value=self.pb.speed,
                title="Playback speed (×)", pointa=(0.02, 0.05), pointb=(0.40, 0.05)
            )
        except Exception:
            self.speed_slider = self.pl.add_slider_widget(
                callback=_cb, rng=[0.1, 4.0], value=self.pb.speed,
                title="Playback speed (×)"
            )

    def _bind_keys(self):
        iren = getattr(self.pl, "iren", None)
        if iren is not None:
            if hasattr(iren, "AddObserver"):
                iren.AddObserver("KeyPressEvent", self._on_key)
        for k in ("space", "period", "comma", "home", "end", "1", "2", "3", "4", "5", "escape", "Esc"):
            try:
                self.pl.add_key_event(k, lambda k=k: self._on_key_simple(k))
            except Exception:
                pass

    def _on_key(self, obj, evt):
        try:
            sym = obj.GetKeySym()
        except Exception:
            return
        self._on_key_simple(sym)

    def _on_key_simple(self, sym: str):
        k = (sym or "").lower()
        if k in ("escape", "esc"):
            self.pb.playing = False
            try:
                self.pl.close()
            except Exception:
                pass
            return
        if k == "space":
            self.pb.playing = not self.pb.playing
        elif k == "period":
            if self.pb.p_seq is not None:
                self.pb.idx = min(self.pb.idx + 1, self.pb.T - 1)
        elif k == "comma":
            if self.pb.p_seq is not None:
                self.pb.idx = max(self.pb.idx - 1, 0)
        elif k == "home":
            self.pb.idx = 0
        elif k == "end":
            if self.pb.p_seq is not None:
                self.pb.idx = self.pb.T - 1
        elif k == "1":
            self.pb.speed = 0.5
        elif k == "2":
            self.pb.speed = 1.0
        elif k == "3":
            self.pb.speed = 1.5
        elif k == "4":
            self.pb.speed = 2.0
        elif k == "5":
            self.pb.speed = 3.0

    def set_gt_points(self, pts: np.ndarray):
        pts = np.asarray(pts, np.float32)
        if pts.size == 0:
            return
        # Downsample to a fixed budget if needed
        try:
            if pts.shape[0] > MAX_GT_PTS:
                sel = np.random.choice(pts.shape[0], MAX_GT_PTS, replace=False)
                pts = pts[sel]
        except Exception:
            pass
        # Remove previous actor if any
        try:
            if self.gt_actor is not None:
                self.pl.remove_actor(self.gt_actor)
        except Exception:
            pass
        # Build polydata and render as lightweight points (not shaded spheres)
        self.gt_poly = make_point_cloud_polydata(pts)
        try:
            self.gt_actor = self.pl.add_mesh(
                self.gt_poly,
                style="points",
                render_points_as_spheres=False,
                point_size=2.0,
                color="lightgray",
                lighting=False,
                pickable=False,
            )
        except TypeError:
            # Fallback for older pyvista that may not support some kwargs
            self.gt_actor = self.pl.add_mesh(
                self.gt_poly,
                style="points",
                point_size=2.0,
                color="lightgray",
            )
        # Disable MSAA/FXAA to reduce fragment work
        try:
            rw = getattr(self.pl, "ren_win", None) or getattr(self.pl, "render_window", None)
            if rw is not None:
                try:
                    rw.SetMultiSamples(0)
                except Exception:
                    pass
                try:
                    rw.SetUseFXAA(0)
                except Exception:
                    pass
        except Exception:
            pass

    def set_goal_points(self, pts: np.ndarray):
        pts = np.asarray(pts, np.float32)
        try:
            if self.goal_actor is not None:
                self.pl.remove_actor(self.goal_actor)
        except Exception:
            pass
        if pts.size == 0:
            self.goal_poly = None
            self.goal_actor = None
            return
        self.goal_poly = make_point_cloud_polydata(pts)
        # Distinct color & size for goals
        self.goal_actor = self.pl.add_mesh(
            self.goal_poly, color="gold", render_points_as_spheres=True, point_size=12.0
        )

    def ensure_drone_poly(self, N: int):
        need_new_poly = not (self.drone_poly is not None and int(self.drone_poly.n_points) == int(N))
        if need_new_poly:
            pts0 = np.zeros((N, 3), np.float32)
            self.drone_poly = make_point_cloud_polydata(pts0)
        # Remove any existing actors (both panes)
        for actor_attr in ("drone_actor_left", "drone_actor_mid", "drone_actor"):
            try:
                actor = getattr(self, actor_attr, None)
                if actor is not None:
                    self.pl.remove_actor(actor)
            except Exception:
                pass
            setattr(self, actor_attr, None)
        # Add actors in both left (GT) and middle (live) panes, sharing the same polydata
        try:
            self.pl.subplot(0, 0)
        except Exception:
            pass
        try:
            self.drone_actor_left = self.pl.add_mesh(
                self.drone_poly, render_points_as_spheres=True, point_size=14.0, color="tomato"
            )
        except Exception:
            self.drone_actor_left = self.pl.add_mesh(self.drone_poly, point_size=14.0, color="tomato")
        try:
            self.pl.subplot(0, 1)
        except Exception:
            pass
        try:
            self.drone_actor_mid = self.pl.add_mesh(
                self.drone_poly, render_points_as_spheres=True, point_size=14.0, color="tomato"
            )
        except Exception:
            self.drone_actor_mid = self.pl.add_mesh(self.drone_poly, point_size=14.0, color="tomato")

    def update_drones(self, p_now: np.ndarray):
        if self.drone_poly is None:
            self.ensure_drone_poly(p_now.shape[0])
        self.drone_poly.points = np.asarray(p_now, np.float32)

    # ---------- Live map pane ----------
    def _set_live_clim(self, lo: float, hi: float):
        try:
            if self.live_actor is not None and hasattr(self.live_actor, "mapper"):
                try:
                    self.live_actor.mapper.SetScalarRange(float(lo), float(hi))
                except Exception:
                    try:
                        self.live_actor.mapper.scalar_range = (float(lo), float(hi))
                    except Exception:
                        pass
        except Exception:
            pass

    def set_live_points(self, pts: np.ndarray):
        """Create right-pane point cloud for live reconstruction."""
        pts = np.asarray(pts, np.float32)
        try:
            self.pl.subplot(0, 1)
        except Exception:
            pass
        try:
            if self.live_actor is not None:
                self.pl.remove_actor(self.live_actor)
        except Exception:
            pass
        self.live_poly = make_point_cloud_polydata(pts)
        # initialize scalars
        self.live_poly.point_data["live"] = np.zeros((self.live_poly.n_points,), np.float32)
        # Lightweight rendering: plain points (no shaded spheres)
        try:
            self.live_actor = self.pl.add_mesh(
                self.live_poly,
                scalars="live",
                style="points",
                render_points_as_spheres=False,
                point_size=2.0,
                cmap="viridis",
                clim=[0.0, 1.0],
                lighting=False,
                nan_color=None,
                nan_opacity=0.0,
                pickable=False,
            )
        except TypeError:
            self.live_actor = self.pl.add_mesh(
                self.live_poly,
                scalars="live",
                style="points",
                point_size=2.0,
                cmap="viridis",
                clim=[0.0, 1.0],
                lighting=False,
            )

    def _current_thresh_value(self) -> float:
        try:
            return float(self.live_thresh_alpha)
        except Exception:
            return 1.0

    def _set_actor_clim(self, lo: float, hi: float):
        self._set_live_clim(lo, hi)

    def _apply_min_mask_to_field(self, field: np.ndarray) -> np.ndarray:
        """Return a copy where ONLY values ≤ threshold remain; values > threshold are hidden (NaN)."""
        thr = self._current_thresh_value()
        out = np.asarray(field, np.float32).copy()
        out[~np.isfinite(out)] = np.nan
        try:
            out[out > thr] = np.nan
        except Exception:
            out = out.reshape(-1)
            out[out > thr] = np.nan
        return out

    def _update_live_scalars(self, field: np.ndarray, clim_hint: Optional[list]):
        self.live_field_last = np.asarray(field, np.float32).copy()
        if self.live_poly is None:
            return
        # Update slider range from incoming data (finite min/max)
        vals = np.asarray(field, np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            fmin = float(np.nanmin(finite))
            fmax = float(np.nanmax(finite))
            if not np.isfinite(fmin) or not np.isfinite(fmax) or (fmax - fmin) < 1e-12:
                fmin, fmax = 0.0, 1.0
        else:
            fmin, fmax = 0.0, 1.0
        self._update_live_slider_range(fmin, fmax)
        # 1) Apply the ≤ threshold mask (NaNs are invisible)
        masked = self._apply_min_mask_to_field(field)

        pd = self.live_poly.point_data
        n = int(self.live_poly.n_points)
        if masked.size != n:
            masked = np.resize(masked.astype(np.float32), n)
        if "live" in pd and int(pd["live"].size) == n:
            pd["live"][:] = masked
        else:
            pd["live"] = masked

        # 2) Auto color limits from the visible subset only
        if self.live_auto_clim:
            vis_vals = masked[np.isfinite(masked)]
            if vis_vals.size:
                lo, hi = np.percentile(vis_vals, [2.0, 98.0]).astype(np.float64)
                if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                    lo, hi = float(np.nanmin(vis_vals)), float(np.nanmax(vis_vals))
                    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                        lo, hi = 0.0, 1.0
            else:
                # nothing visible: keep a harmless range
                lo, hi = 0.0, 1.0
            pad = 0.02 * (hi - lo + 1e-12)
            self._set_actor_clim(lo - pad, hi + pad)
        else:
            # If disabled, prefer passed hint or [0,1]
            if clim_hint and len(clim_hint) == 2:
                lo, hi = float(clim_hint[0]), float(clim_hint[1])
            else:
                lo, hi = 0.0, 1.0
            self._set_actor_clim(lo, hi)

    def update_live_values(self, field: np.ndarray, clim_hint: Optional[list] = None):
        """Update per-point live field values subject to threshold; auto-scale from visible subset."""
        self._update_live_scalars(field, clim_hint)

    def _update_live_slider_range(self, lo: float, hi: float):
        # Create slider on-demand in the middle pane
        if getattr(self, 'live_slider', None) is None:
            self._add_live_slider()
        try:
            rep = self.live_slider.GetRepresentation()
            rep.SetMinimumValue(float(lo))
            rep.SetMaximumValue(float(hi))
            # If user hasn't set a value yet, snap to hi (show everything)
            if not getattr(self, '_live_thresh_user_set', False):
                rep.SetValue(float(hi))
                self.live_thresh_alpha = float(hi)
            else:
                val = float(self.live_thresh_alpha)
                if not np.isfinite(val):
                    val = float(hi)
                val = float(min(max(val, lo), hi))
                rep.SetValue(val)
                self.live_thresh_alpha = val
        except Exception:
            # Fallback: ignore range update if widget API not available
            pass

    def _add_live_slider(self):
        def _slider_cb(val):
            try:
                self.live_thresh_alpha = float(val)
                self._live_thresh_user_set = True
                if self.live_field_last is not None:
                    self._update_live_scalars(self.live_field_last, None)
                    try:
                        self.pl.render()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            try:
                # Attach slider to middle pane
                self.pl.subplot(0, 1)
            except Exception:
                pass
            self.live_slider = self.pl.add_slider_widget(
                callback=_slider_cb, rng=[0.0, 1.0], value=self.live_thresh_alpha,
                title="Live threshold (show ≤)", pointa=(0.35, 0.02), pointb=(0.98, 0.02)
            )
        except Exception:
            try:
                try:
                    self.pl.subplot(0, 1)
                except Exception:
                    pass
                self.live_slider = self.pl.add_slider_widget(
                    callback=_slider_cb, rng=[0.0, 1.0], value=self.live_thresh_alpha,
                    title="Live threshold (show ≤)"
                )
            except Exception:
                self.live_slider = None

    # ---------- Metrics (right pane) ----------
    def _init_chart(self):
        try:
            if self.chart is not None:
                self.pl.remove_chart(self.chart)
        except Exception:
            pass
        try:
            self.pl.subplot(0, 2)
        except Exception:
            pass
        try:
            self.chart = pv.Chart2D()
        except Exception as e:
            # If Chart2D is unavailable, leave pane blank
            self.chart = None
            return
        self.lines.clear()
        for s in self.metric_series:
            self.lines[s] = self.chart.line([], [], label=s)
        try:
            self.chart.legend.visible = True
        except Exception:
            pass
        self.chart.x_label = "iteration"
        self.chart.y_label = "value"
        self.pl.add_chart(self.chart)

    def _redraw_chart(self):
        if self.chart is None or len(self.metric_x) == 0:
            return
        xs = np.fromiter(self.metric_x, dtype=np.int64)
        for s in self.metric_series:
            ys = np.fromiter(self.metric_y.get(s, []), dtype=np.float32)
            if ys.size and s in self.lines:
                try:
                    upd = getattr(self.lines[s], 'update', None)
                    if callable(upd):
                        upd(xs[-ys.size:], ys)
                    else:
                        self.lines[s].x = xs[-ys.size:]
                        self.lines[s].y = ys
                except Exception:
                    # Fallback: re-init chart
                    self._init_chart()
                    break

    def on_metrics(self, it: int, scalars: dict):
        # Expect a dict with at least 'loss'
        v = float(scalars.get('loss', np.nan))
        if not np.isfinite(v):
            return
        self.metric_x.append(int(it))
        self.metric_y['loss'].append(v)
        self._redraw_chart()

    def _remove_arrow_actor(self):
        """Safe remove for the arrows overlay."""
        try:
            if self.arrow_actor is not None:
                self.pl.remove_actor(self.arrow_actor)
        except Exception:
            pass
        self.arrow_actor = None

    def _ensure_arrow_lines(self, N: int):
        """Create or resize the persistent polyline actor for N drones."""
        if self.arrow_lines_poly is not None:
            try:
                if int(self.arrow_lines_poly.n_points) == 2 * N:
                    return
            except Exception:
                pass
        try:
            if self.arrow_lines_actor is not None:
                self.pl.remove_actor(self.arrow_lines_actor)
        except Exception:
            pass
        pts = np.zeros((2 * N, 3), np.float32)  # start/end per drone
        lines = np.empty((N, 3), np.int64)
        lines[:, 0] = 2
        base = np.arange(N, dtype=np.int64)
        lines[:, 1] = 2 * base
        lines[:, 2] = 2 * base + 1
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = lines
        self.arrow_lines_poly = poly
        try:
            self.arrow_lines_actor = self.pl.add_mesh(
                poly, color="cyan", line_width=2.0, lighting=False, pickable=False
            )
        except TypeError:
            self.arrow_lines_actor = self.pl.add_mesh(poly, color="cyan", line_width=2.0)

    def update_arrows(self):
        """Update heading arrows using a persistent line actor (no actor churn)."""
        if self.pb.p_seq is None or self.pb.q_seq is None or self.pb.T == 0:
            # Hide if no data
            if self.arrow_lines_actor is not None:
                try:
                    self.arrow_lines_actor.SetVisibility(False)
                except Exception:
                    pass
            return

        idx = int(self.pb.idx)
        p_now = np.asarray(self.pb.p_seq[idx], np.float32)  # (N,3)
        q_now = np.asarray(self.pb.q_seq[idx], np.float32)  # (N,4)
        N = p_now.shape[0]
        self._ensure_arrow_lines(N)
        if self.arrow_lines_actor is not None:
            try:
                self.arrow_lines_actor.SetVisibility(True)
            except Exception:
                pass

        fwd = np.zeros_like(p_now, np.float32)
        for i in range(N):
            try:
                fwd[i] = quat_to_forward_numpy(q_now[i])
            except Exception:
                fwd[i] = np.array([1.0, 0.0, 0.0], np.float32)
        fwd /= (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
        fwd *= 0.8  # arrow length ~0.8m

        # In-place point update: [start0, end0, start1, end1, ...]
        try:
            pts = self.arrow_lines_poly.points
            pts[0::2] = p_now
            pts[1::2] = p_now + fwd
            self.arrow_lines_poly.points = pts
        except Exception:
            # Fallback: rebuild lines on next call
            self.arrow_lines_poly = None

    def update_hud(self, throttle: float = 0.15):
        """Update HUD text in place, throttled to avoid render contention."""
        now = time.perf_counter()
        if (now - getattr(self, "_last_hud_update", 0.0)) < throttle:
            return
        self._last_hud_update = now

        t = int(self.pb.idx)
        T = max(1, int(self.pb.T))
        status = (
            f"frame {t+1}/{T}   "
            f"{'PLAY' if self.pb.playing else 'PAUSE'}   "
            f"speed×{self.pb.speed:.2f}   "
            f"N={self.pb.N}"
        )

        if self.hud_actor is None:
            self.hud_actor = self.pl.add_text("", font_size=10, color="white")
        try:
            self.hud_actor.SetInput(status)
        except Exception:
            # Fallback: recreate once if SetInput isn't available
            try:
                self.pl.remove_actor(self.hud_actor)
            except Exception:
                pass
            self.hud_actor = self.pl.add_text(status, font_size=10, color="white")

    def _pump_events(self):
        """Explicitly process UI events to keep interaction responsive."""
        try:
            iren = getattr(self.pl, "iren", None)
            if iren is not None:
                iren.ProcessEvents()
        except Exception:
            pass

    def run(self, msg_q):
        try:
            self.pl.show(auto_close=False, interactive_update=True)
        except TypeError:
            try:
                self.pl.show(auto_close=False)
            except TypeError:
                self.pl.show()
        # Print GL backend details once
        _report_gl_backend(self.pl)

        idle_ui_interval = 0.016  # ~60 Hz event processing when idle
        last_ui = 0.0
        alive = True
        while alive:
            try:
                # Wait briefly for new messages; avoids busy render loop
                m = msg_q.get(timeout=idle_ui_interval)
            except Empty:
                # Idle tick: advance playback if due, and throttle UI updates
                did_update = False
                if self.pb.step_if_due() and self.pb.p_seq is not None:
                    self.update_drones(self.pb.p_seq[self.pb.idx])
                    self.update_arrows()
                    self.update_hud()
                    try:
                        self.pl.render()
                    finally:
                        self._pump_events()
                    did_update = True

                # Keep window responsive at most ~10 Hz
                now = time.perf_counter()
                if (now - last_ui) >= idle_ui_interval and not did_update:
                    try:
                        self.pl.update()
                    finally:
                        self._pump_events()
                    last_ui = now
                continue

            # Drain a small burst of queued messages so we render once after applying them
            batch = [m]
            for _ in range(31):
                try:
                    batch.append(msg_q.get_nowait())
                except Empty:
                    break

            need_render = False
            for m in batch:
                tp = m.get("type")
                if tp == "init":
                    self.lb = np.asarray(m.get("lb", self.lb), np.float32)
                    self.ub = np.asarray(m.get("ub", self.ub), np.float32)
                    pts_gt = np.asarray(m.get("gt_points", np.zeros((0, 3), np.float32)), np.float32)
                    try:
                        self.pl.subplot(0, 0)
                    except Exception:
                        pass
                    self.set_gt_points(pts_gt)
                    goals = np.asarray(m.get("goal_points", np.zeros((0, 3), np.float32)), np.float32)
                    self.set_goal_points(goals)
                    live_pts = m.get("live_pts", None)
                    if live_pts is not None:
                        self.set_live_points(np.asarray(live_pts, np.float32))
                    need_render = True
                elif tp == "traj":
                    p = np.asarray(m["p"], np.float32)
                    qv = m.get("q", None)
                    q_seq = None if qv is None else np.asarray(qv, np.float32)
                    dt = float(m.get("dt", 0.05))
                    self.pb.set_traj(p, q_seq, dt)
                    # Reset playback state for new trajectories
                    self.pb.playing = True
                    self.pb.idx = 0
                    self.ensure_drone_poly(self.pb.N)
                    # Render initial frame immediately
                    if self.pb.p_seq is not None:
                        self.update_drones(self.pb.p_seq[self.pb.idx])
                        self.update_arrows()
                        need_render = True
                elif tp == "live":
                    field = np.asarray(m.get("field", np.zeros((0,), np.float32)), np.float32)
                    clim_hint = m.get("clim_hint", None)
                    self.update_live_values(field, clim_hint)
                    need_render = True
                elif tp == "metrics":
                    self.on_metrics(int(m.get("iter", 0)), dict(m.get("scalars", {})))
                    need_render = True
                elif tp == "bye":
                    alive = False

            self.update_hud()
            if need_render:
                try:
                    self.pl.render()
                finally:
                    self._pump_events()

        try:
            self.pl.close()
        except Exception:
            pass


def run_viz_basic(msg_q):
    app = BasicTrainViewer()
    app.run(msg_q)
