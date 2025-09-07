from __future__ import annotations
import time
from dataclasses import dataclass
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
            self.pl = pv.Plotter(window_size=(1200, 700), enable_keybindings=False)
        except TypeError:
            self.pl = pv.Plotter(window_size=(1200, 700))
            for attr in ("enable_keybindings", "enable_key_bindings"):
                if hasattr(self.pl, attr):
                    try:
                        setattr(self.pl, attr, False)
                    except Exception:
                        pass
        self.pl.add_text("live_mvp :: training viewer (GT + drones only)", font_size=10)
        self.pl.show_axes()

        self.gt_actor = None
        self.gt_poly: Optional[pv.PolyData] = None

        self.goal_poly: Optional[pv.PolyData] = None
        self.goal_actor = None

        self.drone_poly: Optional[pv.PolyData] = None
        self.drone_actor = None

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
        if self.drone_poly is not None and self.drone_poly.n_points == N:
            return
        try:
            if self.drone_actor is not None:
                self.pl.remove_actor(self.drone_actor)
        except Exception:
            pass
        pts0 = np.zeros((N, 3), np.float32)
        self.drone_poly = make_point_cloud_polydata(pts0)
        self.drone_actor = self.pl.add_mesh(
            self.drone_poly, render_points_as_spheres=True, point_size=14.0, color="tomato"
        )

    def update_drones(self, p_now: np.ndarray):
        if self.drone_poly is None:
            self.ensure_drone_poly(p_now.shape[0])
        self.drone_poly.points = np.asarray(p_now, np.float32)

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
            for _ in range(7):
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
                    pts = np.asarray(m.get("gt_points", np.zeros((0, 3), np.float32)), np.float32)
                    self.set_gt_points(pts)
                    goals = np.asarray(m.get("goal_points", np.zeros((0, 3), np.float32)), np.float32)
                    self.set_goal_points(goals)
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
