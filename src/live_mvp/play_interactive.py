"""
Interactive viewer for live_mvp:
- Keyboard control of one drone (body-frame accelerations + yaw rate).
- 3 synchronized views:
  (1) Ground-truth φ_gt cross-section with φ=0 contour and drone pose.
  (2) Drone "camera" depth image (fast discretized ray marcher).
  (3) Live-map reconstruction: Qη(x) heatmap + learned φ̂≈0 contour.

Keys:
  W/S  : forward/back (body-x)
  A/D  : left/right   (body-y)
  R/F  : up/down      (body-z)
  Q/E  : yaw left/right (body-z angular rate)
  Shift: boost (2x accel & yaw rate)
  Space: brake (zero velocity)
  C    : recenter slice z to drone z
  P    : pause/resume
  Esc  : quit
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from .dyn import State, DynCfg, step, R_from_q
from .env_gt import phi_gt, raycast_depth_gt
from .live_map import (
    init_live_map, update_geom, update_expo, MapState, v_Q, v_G, HASH_CFG
)

# ---------------------------
# Small helpers
# ---------------------------

def camera_dirs_body(n_az=40, n_el=30, fov_az_deg=80., fov_el_deg=45.):
    """Grid of unit directions in the body frame covering a rectangular FOV."""
    az = jnp.linspace(-jnp.deg2rad(fov_az_deg)/2, jnp.deg2rad(fov_az_deg)/2, n_az)
    el = jnp.linspace(-jnp.deg2rad(fov_el_deg)/2, jnp.deg2rad(fov_el_deg)/2, n_el)
    A, E = jnp.meshgrid(az, el, indexing='xy')
    x = jnp.cos(E) * jnp.cos(A)
    y = jnp.cos(E) * jnp.sin(A)
    z = jnp.sin(E)
    D = jnp.stack([x, y, z], axis=-1)
    D = D / (jnp.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)
    return D.reshape(-1, 3), (int(n_el), int(n_az))


def fast_raycast_depth_grid(o, D, t0=0.2, t1=12.0, S=96, eps=0.02):
    """
    Fast, discretized raycaster for visualization.
    o: (3,), D: (M,3). Returns t: (M,) with NaN for misses.
    """
    ts = jnp.linspace(t0, t1, S)  # (S,)
    xs = o[None, None, :] + ts[:, None, None] * D[None, :, :]  # (S,M,3)
    xs_flat = xs.reshape(-1, 3)
    phi = jax.vmap(phi_gt)(xs_flat).reshape(S, -1)              # (S,M)
    # Consider 'hit' when φ <= eps
    hit_mask = phi <= eps
    any_hit = hit_mask.any(axis=0)                               # (M,)
    first_idx = jnp.argmax(hit_mask, axis=0)                     # (M,), 0 if none
    t = ts[first_idx]
    t = jnp.where(any_hit, t, jnp.nan)
    return t


def grid_xy(lb, ub, res, z_plane):
    xs = jnp.linspace(lb[0], ub[0], res)
    ys = jnp.linspace(lb[1], ub[1], res)
    X, Y = jnp.meshgrid(xs, ys, indexing='xy')
    XY = jnp.stack([X, Y], axis=-1)  # (res,res,2)
    pts = jnp.concatenate([XY, jnp.full((res, res, 1), z_plane)], axis=-1).reshape(-1, 3)
    return xs, ys, X, Y, pts


@dataclass
class Controls:
    w: bool = False; s: bool = False
    a: bool = False; d: bool = False
    r: bool = False; f: bool = False
    q: bool = False; e: bool = False
    boost: bool = False
    brake: bool = False
    paused: bool = False
    recenter_slice: bool = False
    quit: bool = False


# ---------------------------
# Interactive App
# ---------------------------

class InteractiveApp:
    def __init__(self):
        # World bounds from HASH_CFG for visuals
        self.lb = np.array(HASH_CFG.lb)
        self.ub = np.array(HASH_CFG.ub)

        # Sim & map state
        key = jax.random.PRNGKey(0)
        self.mapstate: MapState = init_live_map(key)

        # One drone state
        p0 = jnp.array([0., 0., 1.6])
        v0 = jnp.zeros(3)
        q0 = jnp.array([1., 0., 0., 0.])   # identity
        w0 = jnp.zeros(3)
        self.state = State(p0, v0, q0, w0)
        self.cfg = DynCfg(dt=0.05)

        # Camera dirs & shapes
        self.cam_dirs_body, self.cam_shape = camera_dirs_body(n_az=40, n_el=30)
        self.cam_S = 96  # samples for fast camera

        # Mapping update subsample (coarse rays)
        self.map_dirs_body, _ = camera_dirs_body(n_az=24, n_el=3, fov_az_deg=100., fov_el_deg=40.)

        # Visualization grids
        self.slice_z = float(self.state.p[2])   # start at drone z
        self.res = 128
        self.xs, self.ys, self.Xm, self.Ym, self.slice_pts = grid_xy(HASH_CFG.lb, HASH_CFG.ub, self.res, self.slice_z)

        # Matplotlib figure
        self.fig, (self.ax_gt, self.ax_cam, self.ax_live) = plt.subplots(1, 3, figsize=(13.5, 4.5))
        self.fig.canvas.manager.set_window_title("live_mvp :: interactive")

        # Initialize images/contours
        self.im_cam = self.ax_cam.imshow(np.zeros(self.cam_shape), origin='lower', aspect='auto',
                                         extent=[-40, 40, -22.5, 22.5])
        self.ax_cam.set_title("Drone view (depth, deg az x deg el)")
        self.ax_cam.set_xlabel("azimuth (deg)"); self.ax_cam.set_ylabel("elevation (deg)")

        self.im_live = self.ax_live.imshow(np.zeros((self.res, self.res)),
                                           extent=[self.xs[0], self.xs[-1], self.ys[0], self.ys[-1]],
                                           origin='lower', interpolation='nearest', vmin=0.0, vmax=1.0)
        self.ax_live.set_title("Reconstruction Qη (coverage) @ z-slice")
        self.ax_live.set_xlabel("x"); self.ax_live.set_ylabel("y")
        self.cont_live = None  # learned φ̂≈0 contour

        self.im_gt = self.ax_gt.imshow(np.zeros((self.res, self.res)),
                                       extent=[self.xs[0], self.xs[-1], self.ys[0], self.ys[-1]],
                                       origin='lower', interpolation='nearest')
        self.ax_gt.set_title("Ground truth φ_gt @ z-slice")
        self.ax_gt.set_xlabel("x"); self.ax_gt.set_ylabel("y")
        self.cont_gt = None

        # Drone pose artists in GT panel
        p0_np = np.array(p0)
        self.pose_quiver = self.ax_gt.quiver([p0_np[0]], [p0_np[1]], [0.0], [0.0],
                                             scale=10, width=0.004, color='k')

        # Status text
        self.txt = self.fig.text(0.02, 0.95, "", ha='left', va='top')

        # Controls
        self.ctrl = Controls()

        # Keyboard bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key)

        # First render
        self.update_all(bootstrap=True)

    # -------- utils --------

    def _clear_contour(self, cs):
        """Remove all artists for a contour set across Matplotlib versions."""
        if cs is None:
            return
        try:
            if hasattr(cs, "collections"):
                for coll in cs.collections:
                    try:
                        coll.remove()
                    except Exception:
                        pass
                return
            if hasattr(cs, "artists"):
                for art in cs.artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                return
            if hasattr(cs, "lines"):
                for ln in cs.lines:
                    try:
                        ln.remove()
                    except Exception:
                        pass
                return
            if hasattr(cs, "remove"):
                cs.remove()
        except Exception:
            pass

    # -------- keyboard --------

    def on_key(self, event):
        name = event.key.lower() if event.key is not None else ""
        down = (event.name == 'key_press_event')

        if name in ('shift',):
            self.ctrl.boost = down
        elif name == ' ':
            self.ctrl.brake = down
        elif name == 'w':
            self.ctrl.w = down
        elif name == 's':
            self.ctrl.s = down
        elif name == 'a':
            self.ctrl.a = down
        elif name == 'd':
            self.ctrl.d = down
        elif name == 'r':
            self.ctrl.r = down
        elif name == 'f':
            self.ctrl.f = down
        elif name == 'q':
            self.ctrl.q = down
        elif name == 'e':
            self.ctrl.e = down
        elif name == 'c' and down:
            self.ctrl.recenter_slice = True
        elif name == 'p' and down:
            self.ctrl.paused = not self.ctrl.paused
        elif name == 'escape' and down:
            self.ctrl.quit = True

    # -------- sim step --------

    def control_to_u(self, st: State):
        """Map key state to (a_world, w_body)."""
        accel_unit = 2.0
        yaw_unit = 1.0
        if self.ctrl.boost:
            accel_unit *= 2.0
            yaw_unit *= 2.0

        # body-frame acceleration command
        ax = (1.0 if self.ctrl.w else 0.0) + (-1.0 if self.ctrl.s else 0.0)
        ay = (1.0 if self.ctrl.d else 0.0) + (-1.0 if self.ctrl.a else 0.0)
        az = (1.0 if self.ctrl.r else 0.0) + (-1.0 if self.ctrl.f else 0.0)
        a_body = jnp.array([ax, ay, az]) * accel_unit

        # convert to world
        Rw = R_from_q(st.q)
        a_world = (Rw @ a_body).astype(jnp.float32)

        # body angular-velocity command (yaw only for now)
        wz = (1.0 if self.ctrl.q else 0.0) + (-1.0 if self.ctrl.e else 0.0)
        w_body = jnp.array([0.0, 0.0, wz * yaw_unit], dtype=jnp.float32)

        # quick brake: zero velocity
        if self.ctrl.brake:
            st = State(st.p, jnp.zeros_like(st.v), st.q, st.w)

        u = jnp.concatenate([a_world, w_body])
        return st, u

    def mapping_update(self, st: State):
        """
        Update the live map from a sparse set of rays.
        """
        p, q = st.p, st.q
        Rw = R_from_q(q)
        rays_w = (self.map_dirs_body @ Rw.T)  # (M,3)
        # Subsample for speed
        idx = jnp.arange(0, rays_w.shape[0], 3)
        sel = rays_w[idx]

        SFS = 24
        ts = jnp.linspace(0.2, 12.0, SFS)

        def per_ray(d):
            t = raycast_depth_gt(p, d)                     # GT sphere tracing (matches training)
            stop_t = jnp.where(jnp.isnan(t), 12.0, t)
            xs = p[None, :] + ts[:, None] * d[None, :]
            m_free = (ts < stop_t).astype(jnp.float32)
            r = ts
            w_seen = m_free / (1.0 + r * r)
            x_hit = p + stop_t * d
            m_hit = jnp.isfinite(t).astype(jnp.float32) * (t <= 12.0)
            return x_hit, m_hit, xs, m_free, w_seen

        hits, m_hits, frees, m_frees, w_seens = jax.vmap(per_ray)(sel)
        ms = update_geom(self.mapstate, hits, m_hits, frees, m_frees)
        ms = update_expo(ms, frees, w_seens)
        self.mapstate = ms

    # -------- renderers --------

    def update_gt_panel(self):
        z = self.slice_z
        xs, ys, Xm, Ym, pts = grid_xy(HASH_CFG.lb, HASH_CFG.ub, self.res, z)
        phi = jax.vmap(phi_gt)(pts).reshape(self.res, self.res)
        phi_np = np.asarray(jax.device_get(phi))
        self.im_gt.set_data(phi_np)
        self.im_gt.set_clim(vmin=phi_np.min(), vmax=phi_np.max())

        # φ=0 contour
        self._clear_contour(self.cont_gt)
        self.cont_gt = self.ax_gt.contour(np.asarray(Xm), np.asarray(Ym), phi_np, levels=[0.0], linewidths=1.5)

        # pose arrow
        p = np.asarray(self.state.p)
        Rw = np.asarray(R_from_q(self.state.q))
        fwd = Rw @ np.array([1.0, 0.0, 0.0])
        self.pose_quiver.set_offsets(np.array([[p[0], p[1]]]))
        self.pose_quiver.set_UVC(np.array([fwd[0]]), np.array([fwd[1]]))

        self.ax_gt.set_xlim(xs[0], xs[-1]); self.ax_gt.set_ylim(ys[0], ys[-1])
        self.ax_gt.set_title(f"GT φ_gt @ z={z:.2f}")

    def update_live_panel(self):
        z = self.slice_z
        xs, ys, Xm, Ym, pts = grid_xy(HASH_CFG.lb, HASH_CFG.ub, self.res, z)

        # Coverage heatmap Qη
        Q = v_Q(pts, self.mapstate.expo.eta).reshape(self.res, self.res)
        Q_np = np.asarray(jax.device_get(Q))
        self.im_live.set_data(Q_np)
        self.im_live.set_clim(0.0, 1.0)
        self.im_live.set_extent([xs[0], xs[-1], ys[0], ys[-1]])

        # Learned φ̂≈0 contour
        phi_hat = v_G(pts, self.mapstate.geom.theta).reshape(self.res, self.res)
        phi_hat_np = np.asarray(jax.device_get(phi_hat))

        self._clear_contour(self.cont_live)
        self.cont_live = self.ax_live.contour(np.asarray(Xm), np.asarray(Ym), phi_hat_np, levels=[0.0], linewidths=1.2)

        self.ax_live.set_xlim(xs[0], xs[-1]); self.ax_live.set_ylim(ys[0], ys[-1])

    def update_cam_panel(self):
        # World-frame rays from body FOV
        Rw = R_from_q(self.state.q)
        rays_w = (self.cam_dirs_body @ Rw.T)  # (M,3)

        # Fast discrete raycast for visualization
        t = fast_raycast_depth_grid(self.state.p, rays_w, S=self.cam_S)  # (M,)
        T = np.asarray(jax.device_get(t)).reshape(self.cam_shape)

        # Map depth to image (NaN -> max)
        tmax = 12.0
        Timg = np.where(np.isfinite(T), T, tmax)
        self.im_cam.set_data(Timg)
        self.im_cam.set_clim(vmin=0.0, vmax=tmax)

    def update_status(self, fps):
        p = np.asarray(self.state.p)
        self.txt.set_text(
            f"pos=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})  "
            f"fps≈{fps:.1f}  "
            f"{'PAUSED' if self.ctrl.paused else ''}"
        )

    # -------- main loop --------

    def update_all(self, bootstrap=False, frame_idx=0):
        t0 = time.time()

        if self.ctrl.recenter_slice:
            self.slice_z = float(self.state.p[2])
            self.ctrl.recenter_slice = False

        # Simulate
        if not self.ctrl.paused:
            st, u = self.control_to_u(self.state)
            self.state = step(st, u, self.cfg)
            # Clamp to avoid going below ground too far (softly)
            zmin = 0.05
            self.state = State(
                p=jnp.where(self.state.p[2] < zmin,
                            self.state.p.at[2].set(zmin),
                            self.state.p),
                v=self.state.v,
                q=self.state.q,
                w=self.state.w
            )
            self.mapping_update(self.state)

        # Render
        self.update_cam_panel()

        # Update GT every frame (cheap)
        self.update_gt_panel()

        # Update live map panel every few frames (heavier)
        if bootstrap or (frame_idx % 5 == 0):
            self.update_live_panel()

        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        self.update_status(fps)

        plt.pause(0.001)

    def run(self):
        i = 0
        while not self.ctrl.quit:
            self.update_all(frame_idx=i)
            i += 1


def main():
    app = InteractiveApp()
    app.run()


if __name__ == "__main__":
    main()
