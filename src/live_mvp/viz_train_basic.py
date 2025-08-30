from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
from queue import Empty

import numpy as np
import pyvista as pv


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
        try:
            if self.gt_actor is not None:
                self.pl.remove_actor(self.gt_actor)
        except Exception:
            pass
        self.gt_poly = make_point_cloud_polydata(pts)
        self.gt_actor = self.pl.add_mesh(
            self.gt_poly, color="lightgray", render_points_as_spheres=True, point_size=4.5
        )

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

    def update_hud(self):
        try:
            if self.hud_actor is not None:
                self.pl.remove_actor(self.hud_actor)
        except Exception:
            pass
        t = self.pb.idx
        T = max(1, self.pb.T)
        status = (
            f"frame {t+1}/{T}   "
            f"{'PLAY' if self.pb.playing else 'PAUSE'}   "
            f"speed×{self.pb.speed:.2f}   "
            f"N={self.pb.N}"
        )
        self.hud_actor = self.pl.add_text(status, font_size=10, color="white")

    def run(self, msg_q):
        try:
            self.pl.show(auto_close=False, interactive_update=True)
        except TypeError:
            try:
                self.pl.show(auto_close=False)
            except TypeError:
                self.pl.show()

        alive = True
        while alive:
            drained = False
            for _ in range(8):
                try:
                    m = msg_q.get_nowait()
                except Empty:
                    break
                drained = True
                tp = m.get("type")
                if tp == "init":
                    self.lb = np.asarray(m.get("lb", self.lb), np.float32)
                    self.ub = np.asarray(m.get("ub", self.ub), np.float32)
                    pts = np.asarray(m.get("gt_points", np.zeros((0, 3), np.float32)), np.float32)
                    self.set_gt_points(pts)
                    goals = np.asarray(m.get("goal_points", np.zeros((0, 3), np.float32)), np.float32)
                    self.set_goal_points(goals)
                elif tp == "traj":
                    p = np.asarray(m["p"], np.float32)
                    qv = m.get("q", None)
                    q_seq = None if qv is None else np.asarray(qv, np.float32)
                    dt = float(m.get("dt", 0.05))
                    self.pb.set_traj(p, q_seq, dt)
                    self.ensure_drone_poly(self.pb.N)
                elif tp == "bye":
                    alive = False

            if self.pb.step_if_due() and self.pb.p_seq is not None:
                self.update_drones(self.pb.p_seq[self.pb.idx])

            if (not self.pb.playing) and self.pb.p_seq is not None:
                self.update_drones(self.pb.p_seq[self.pb.idx])

            self.update_hud()

            try:
                self.pl.update()
            except Exception:
                alive = False

            time.sleep(0.01)

        try:
            self.pl.close()
        except Exception:
            pass


def run_viz_basic(msg_q):
    app = BasicTrainViewer()
    app.run(msg_q)
