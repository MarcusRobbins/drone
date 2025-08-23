"""
3D interactive viewer for live_mvp with process isolation.

- Main/UI process: PyVista window (no JAX imports). Geometry is created ONCE and only
  per-vertex scalars are updated afterwards.
- Child/Sim process: all JAX work (dynamics, GT raycast, live mapping).

Views:
  Left   : Ground-truth shell (|phi_gt| <= eps) + drone pose + HUD.
  Middle : Drone "camera" depth image (triangulated plane textured by t).
  Right  : Live reconstruction grid; field depends on mode (see keys).

Keys:
  I/K: forward/back   J/L: left/right   U/O: down/up
  Z/C: yaw left/right
  Shift: boost   Space: brake (hold while pressed)
  H: toggle HOLD (one-tap stop; ignores motion until toggled off)
  P: pause physics   M: toggle mapping   A: toggle live auto-contrast
  1: vis (Q * shell mask)   2: Q (exposure)   3: |phi|   4: shell mask
  Slider (bottom): minimum density threshold (auto-scaled)
  Esc: quit

Run:
  set PYTHONPATH=%CD%\\src
  python -m live_mvp.play_interactive_3d

Env toggles (optional):
  LIVEMVP_ENABLE_JIT=1/0   # enable/disable JIT (default: 1)
  LIVEMVP_GPU=1/0          # prefer GPU (default: 1); set 0 to force CPU
  CUDA_VISIBLE_DEVICES=0   # select which GPU if needed
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import multiprocessing as mp
from queue import Empty

# ---------------------------------------------------------------------
# Logging & PyVista noise suppression
# ---------------------------------------------------------------------

log = logging.getLogger("live_mvp.play3d")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# Quell noisy destructor AttributeErrors on some PyVista builds
try:
    from pyvista.plotting import plotter as _pv_plotter_mod  # type: ignore
    _pv_plotter_mod.BasePlotter.__del__ = lambda self: None  # noqa: E731
except Exception:
    pass

# ---------------------------------------------------------------------
# UI-side math (NumPy only; no JAX imports here)
# ---------------------------------------------------------------------

def quat_to_R_numpy(q: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix from quaternion [w,x,y,z], NumPy only."""
    w, x, y, z = [float(v) for v in q]
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)]
    ], dtype=np.float32)

def make_point_cloud_polydata(pts: np.ndarray) -> pv.PolyData:
    """Create a PolyData with vertex cells; no scalars yet."""
    poly = pv.PolyData()
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"make_point_cloud_polydata expected (N,3), got {pts.shape}")
    n = pts.shape[0]
    poly.points = pts
    verts = np.empty(n*2, dtype=np.int64)
    verts[0::2] = 1
    verts[1::2] = np.arange(n, dtype=np.int64)
    poly.verts = verts
    return poly

def plane_from_image(img2d: np.ndarray, scalars_name: str = "values") -> pv.PolyData:
    """Triangulated plane with point scalars from a 2D image."""
    n_rows, n_cols = img2d.shape
    plane = pv.Plane(center=(0.0, 0.0, 0.0),
                     direction=(0.0, 0.0, 1.0),
                     i_size=float(n_cols),
                     j_size=float(n_rows),
                     i_resolution=max(1, n_cols - 1),
                     j_resolution=max(1, n_rows - 1))
    vals = img2d.astype(np.float32).ravel(order="C")
    if plane.n_points == vals.size:
        plane.point_data[scalars_name] = vals
    else:
        plane.point_data[scalars_name] = np.resize(vals, plane.n_points)
    return plane

# ---------------------------------------------------------------------
# Messages & Controls
# ---------------------------------------------------------------------

@dataclass
class Controls:
    fwd: bool = False; back: bool = False
    left: bool = False; right: bool = False
    down: bool = False; up: bool = False
    yaw_l: bool = False; yaw_r: bool = False
    boost: bool = False
    brake: bool = False
    hold: bool = False            # NEW: latched stop/hold
    paused: bool = False
    mapping: bool = False
    quit: bool = False
    # Live map visualization controls
    live_mode: int = 0            # 0: vis, 1: Q, 2: |phi|, 3: mask
    live_auto_clim: bool = True

# UI → Sim:
#   {"type": "ctrl", "ctrl": asdict(Controls)}
#   {"type": "quit"}
#
# Sim → UI:
#   {"type": "init", "gt_points": (Mg,3), "live_pts": (Nl,3), "cam_shape": (h,w), "tmax": float, "lb":(3,), "ub":(3,)}
#   {"type": "state", "p": (3,), "q": (4,), "v": (3,), "w": (3,), "steps": int, "sps": float, "mapping":bool, "paused":bool}
#   {"type": "cam", "img": (h,w) float32}
#   {"type": "live", "field": (Nl,) float32, "name": str, "clim_hint": [lo,hi], "stats": {...}}
#   {"type": "log", "msg": str}
#   {"type": "bye"}

# ---------------------------------------------------------------------
# Sim worker process (JAX lives only here)
# ---------------------------------------------------------------------

def _sim_worker(ui2sim: mp.Queue, sim2ui: mp.Queue,
                grid_gt_res=(80,80,40),
                grid_live_res=(48,48,24),
                shell_eps_gt=0.08,
                shell_eps_live=0.12,
                dt=0.05,
                cam_n_az=32, cam_n_el=24, cam_samples=72,
                cam_fov_az=80., cam_fov_el=45.,
                cam_hz=12.0, state_hz=30.0, live_hz=2.0,
                map_update_every_steps=6):
    """Child process target: owns all JAX imports & computation."""

    # --- Environment hygiene BEFORE importing jax ---
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    prefer_gpu = os.environ.get("LIVEMVP_GPU", "1") not in ("0", "false", "False")
    if not prefer_gpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    enable_jit = os.environ.get("LIVEMVP_ENABLE_JIT", "1") in ("1", "true", "True")

    import jax, jax.numpy as jnp
    from jax import tree_util as jtu

    from .env_gt import phi_gt, raycast_depth_gt
    from .dyn import State, DynCfg, step, R_from_q
    from .live_map import init_live_map, update_geom, update_expo, MapState, v_Q, v_G, HASH_CFG

    def _simlog(msg: str):
        try:
            sim2ui.put_nowait({"type": "log", "msg": msg})
        except Exception:
            pass

    def camera_dirs_body(n_az=32, n_el=24, fov_az_deg=80., fov_el_deg=45.):
        az = jnp.linspace(-jnp.deg2rad(fov_az_deg)/2, jnp.deg2rad(fov_az_deg)/2, n_az)
        el = jnp.linspace(-jnp.deg2rad(fov_el_deg)/2, jnp.deg2rad(fov_el_deg)/2, n_el)
        A, E = jnp.meshgrid(az, el, indexing='xy')
        x = jnp.cos(E) * jnp.cos(A)
        y = jnp.cos(E) * jnp.sin(A)
        z = jnp.sin(E)
        D = jnp.stack([x, y, z], axis=-1)
        D = D / (jnp.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)
        return D.reshape(-1, 3), (int(n_el), int(n_az))

    # Eager camera marcher; a jitted constant-S closure is built when JIT is on.
    def fast_raycast_depth_grid(o, D, t0=0.2, t1=12.0, S=72, eps=0.02):
        ts = jnp.linspace(t0, t1, S)
        xs = o[None, None, :] + ts[:, None, None] * D[None, :, :]  # (S,M,3)
        xs_flat = xs.reshape(-1, 3)
        phi = jax.vmap(phi_gt)(xs_flat).reshape(S, -1)
        hit_mask = phi <= eps
        any_hit = hit_mask.any(axis=0)
        first_idx = jnp.argmax(hit_mask, axis=0)
        t = ts[first_idx]
        return jnp.where(any_hit, t, jnp.nan)

    def make_grid(lb: jnp.ndarray, ub: jnp.ndarray, res_xyz: Tuple[int,int,int]):
        nx, ny, nz = res_xyz
        xs = jnp.linspace(lb[0], ub[0], nx)
        ys = jnp.linspace(lb[1], ub[1], ny)
        zs = jnp.linspace(lb[2], ub[2], nz)
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing='xy')
        pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        return xs, ys, zs, pts

    def thin_shell_points(pts_jax: jnp.ndarray, phi_fn, shell_eps: float) -> np.ndarray:
        phi = jax.vmap(phi_fn)(pts_jax)
        mask = jnp.abs(phi) <= shell_eps
        pts_np = np.asarray(jax.device_get(pts_jax))
        mask_np = np.asarray(jax.device_get(mask))
        return pts_np[mask_np].astype(np.float32)

    # --- Initialize world, map, state ---
    lb = jnp.asarray(HASH_CFG.lb); ub = jnp.asarray(HASH_CFG.ub)
    _, _, _, pts_gt = make_grid(lb, ub, grid_gt_res)
    _, _, _, pts_live = make_grid(lb, ub, grid_live_res)  # FIXED live grid

    key = jax.random.PRNGKey(0)
    mapstate: MapState = init_live_map(key)
    state = State(p=jnp.array([0., 0., 1.6]),
                  v=jnp.zeros(3),
                  q=jnp.array([1., 0., 0., 0.]),
                  w=jnp.zeros(3))
    cfg = DynCfg(dt=float(dt), drag_v=0.02, drag_w=0.01, a_max=5.0, w_max=2.0)

    cam_dirs, cam_shape = camera_dirs_body(cam_n_az, cam_n_el, cam_fov_az, cam_fov_el)
    tmax = 12.0

    # Send init
    t0_init = time.time()
    gt_points = thin_shell_points(pts_gt, phi_gt, shell_eps_gt)
    sim2ui.put({"type": "init",
                "gt_points": np.asarray(jax.device_get(gt_points), dtype=np.float32),
                "live_pts":  np.asarray(jax.device_get(pts_live), dtype=np.float32),
                "cam_shape": cam_shape,
                "tmax": float(tmax),
                "lb": np.asarray(jax.device_get(lb), dtype=np.float32),
                "ub": np.asarray(jax.device_get(ub), dtype=np.float32)})
    _simlog(f"[sim] init sent: gt_points={int(gt_points.shape[0])}, live_pts={int(pts_live.shape[0])}, "
            f"prep_time={time.time()-t0_init:.2f}s")

    # ---------------- JIT hot paths & warmup (with safe fallbacks) ----------------
    step_fn = step
    cam_march_fn = lambda o, D: fast_raycast_depth_grid(o, D, S=int(cam_samples))

    # NaN guard for map parameters
    def _sanitize_mapstate(ms: MapState, max_abs=1e3) -> MapState:
        def clean(x):
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.clip(x, -max_abs, max_abs)
        theta = jtu.tree_map(clean, ms.geom.theta)
        eta   = jtu.tree_map(clean, ms.expo.eta)
        return MapState(ms.geom._replace(theta=theta), ms.expo._replace(eta=eta))

    if enable_jit:
        _simlog("[sim] JIT enabled")
        try:
            step_fn = jax.jit(step)
        except Exception as e:
            _simlog(f"[sim] step jit disabled: {e!r}")
            step_fn = step

        # Jitted camera marcher with constant S
        def _make_fast_raycast_jit(S: int, t0c: float = 0.2, t1c: float = 12.0, ec: float = 0.02):
            S_const = int(S)
            @jax.jit
            def _fn(o, D):
                ts = jnp.linspace(t0c, t1c, S_const)
                xs = o[None, None, :] + ts[:, None, None] * D[None, :, :]
                xs_flat = xs.reshape(-1, 3)
                phi = jax.vmap(phi_gt)(xs_flat).reshape(S_const, -1)
                hit_mask = phi <= ec
                any_hit = hit_mask.any(axis=0)
                first_idx = jnp.argmax(hit_mask, axis=0)
                t = ts[first_idx]
                return jnp.where(any_hit, t, jnp.nan)
            return _fn
        try:
            cam_march_fn = _make_fast_raycast_jit(int(cam_samples))
        except Exception as e:
            _simlog(f"[sim] camera marcher jit disabled: {e!r}")
            cam_march_fn = lambda o, D: fast_raycast_depth_grid(o, D, S=int(cam_samples))

        # Mapping update (jitted): includes occluded negatives for exposure and diagnostics
        def _map_update_once(ms: MapState, st: State):
            p, q = st.p, st.q
            Rw = R_from_q(q)
            # Slightly denser than before: more elevation rows, no stride
            dirs_body, _ = camera_dirs_body(n_az=24, n_el=4, fov_az_deg=100., fov_el_deg=40.)
            rays_w = (dirs_body @ Rw.T)

            SFS = 24
            ts = jnp.linspace(0.2, 12.0, SFS)

            def per_ray(d):
                t = raycast_depth_gt(p, d)                        # robust GT raycast
                stop_t = jnp.where(jnp.isnan(t), 12.0, t)
                xs = p[None,:] + ts[:,None]*d[None,:]             # (SFS,3)
                # free-space strictly before the hit
                m_free = (ts < stop_t).astype(jnp.float32)
                w_free = m_free / (1.0 + ts*ts)
                # occluded region strictly behind the hit (bound it to [t_max])
                m_occ = (ts > stop_t).astype(jnp.float32)
                w_occ = m_occ / (1.0 + ts*ts)
                # hit point and mask
                x_hit = p + stop_t * d
                m_hit = jnp.isfinite(t).astype(jnp.float32) * (t <= 12.0)
                return x_hit, m_hit, xs, m_free, w_free, xs, m_occ, w_occ

            # Note: returns (hits, m_hits, frees, m_frees, w_frees, occ, m_occ, w_occ)
            hits, m_hits, frees, m_frees, w_frees, occs, m_occs, w_occs = jax.vmap(per_ray)(rays_w)

            # Diagnostics (counts)
            n_hit = jnp.sum(m_hits).astype(jnp.int32)
            n_free = jnp.sum(m_frees).astype(jnp.int32)
            n_occ = jnp.sum(m_occs).astype(jnp.int32)

            # Update geometry and exposure (with negatives)
            new_map = update_geom(ms, hits, m_hits, frees, m_frees)
            new_map = update_expo(new_map, frees, w_frees, occs, w_occs, neg_weight=0.7)

            # Emit a small tuple we can log outside jit
            return _sanitize_mapstate(new_map), (n_hit, n_free, n_occ)
        try:
            map_update_fn = jax.jit(_map_update_once)
        except Exception as e:
            _simlog(f"[sim] mapping jit disabled: {e!r}")
            def map_update_fn(ms, st):
                nm, counts = _map_update_once(ms, st)
                return nm, counts
    else:
        _simlog("[sim] JIT disabled")
        def map_update_fn(ms: MapState, st: State):
            p, q = st.p, st.q
            Rw = R_from_q(q)
            dirs_body, _ = camera_dirs_body(n_az=16, n_el=2, fov_az_deg=100., fov_el_deg=40.)
            rays_w = (dirs_body @ Rw.T)
            idx = jnp.arange(0, rays_w.shape[0], 2)
            sel = rays_w[idx]
            SFS = 16
            ts = jnp.linspace(0.2, 12.0, SFS)
            def per_ray(d):
                t = raycast_depth_gt(p, d)
                stop_t = jnp.where(jnp.isnan(t), 12.0, t)
                xs = p[None,:] + ts[:,None]*d[None,:]
                m_free = (ts < stop_t).astype(jnp.float32)
                w_seen = m_free / (1.0 + ts*ts)
                x_hit = p + stop_t * d
                m_hit = jnp.isfinite(t).astype(jnp.float32) * (t <= 12.0)
                # also construct occluded weights for diagnostics in non-JIT path
                m_occ = (ts > stop_t).astype(jnp.float32)
                w_occ = m_occ / (1.0 + ts*ts)
                return x_hit, m_hit, xs, m_free, w_seen, xs, m_occ, w_occ
            hits, m_hits, frees, m_frees, w_seens, occs, m_occs, w_occs = jax.vmap(per_ray)(sel)
            new_map = update_geom(ms, hits, m_hits, frees, m_frees)
            new_map = update_expo(new_map, frees, w_seens, occs, w_occs, neg_weight=0.7)
            # counts
            n_hit = int(jnp.sum(m_hits))
            n_free = int(jnp.sum(m_frees))
            n_occ = int(jnp.sum(m_occs))
            return _sanitize_mapstate(new_map), (n_hit, n_free, n_occ)

    # Jitted pack of live fields to avoid duplicating v_G/v_Q work
    if enable_jit:
        @jax.jit
        def _compute_live_pack(theta, eta, pts_live_const, eps_shell):
            G = v_G(pts_live_const, theta)                   # (N,)
            Q = v_Q(pts_live_const, eta)                     # (N,)
            absG = jnp.abs(G)
            mask = 1.0 - jnp.clip(absG / eps_shell, 0.0, 1.0)
            vis = jnp.clip(jnp.nan_to_num(Q * mask, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            return vis, Q, absG, mask
        compute_pack = lambda th, et: _compute_live_pack(th, et, pts_live, shell_eps_live)
    else:
        def compute_pack(theta, eta):
            G = v_G(pts_live, theta)
            Q = v_Q(pts_live, eta)
            absG = jnp.abs(G)
            mask = 1.0 - jnp.clip(absG / shell_eps_live, 0.0, 1.0)
            vis = jnp.clip(jnp.nan_to_num(Q * mask, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            return vis, Q, absG, mask

    # Warmup compilation once mapping is toggled ON
    did_warmup = False
    def _warmup_if_needed():
        nonlocal did_warmup, mapstate
        if did_warmup or not enable_jit:
            return
        try:
            t0 = time.time()
            _ = step_fn(state, jnp.zeros(6, dtype=jnp.float32), cfg)
            _ = cam_march_fn(state.p, cam_dirs)
            mapstate, _ = map_update_fn(mapstate, state)
            _ = compute_pack(mapstate.geom.theta, mapstate.expo.eta)
            _simlog(f"[sim] warmup compiled in {time.time()-t0:.2f}s (step+cam+map+pack)")
        except Exception as e:
            _simlog(f"[sim] warmup failed: {e!r}")
        did_warmup = True

    # Timers
    next_sim_t = time.perf_counter()
    last_cam_t = 0.0
    last_state_t = 0.0
    last_live_t = 0.0
    last_heartbeat = time.time()
    cam_dt = 1.0 / max(1e-6, cam_hz)
    state_dt = 1.0 / max(1e-6, state_hz)
    live_dt = 1.0 / max(1e-6, live_hz)

    steps = 0
    rate_win_t0 = time.time()
    rate_win_steps0 = 0
    steps_per_sec = 0.0

    ctrl = Controls()
    quit_flag = False

    try:
        while not quit_flag:
            # Drain UI messages
            while True:
                try:
                    m = ui2sim.get_nowait()
                except Empty:
                    break
                tp = m.get("type")
                if tp == "quit":
                    quit_flag = True
                    break
                if tp == "ctrl":
                    prev_map = ctrl.mapping
                    ctrl = Controls(**{**asdict(ctrl), **m.get("ctrl", {})})
                    if ctrl.mapping and not prev_map:
                        _simlog(f"[sim] mapping ENABLED at step {steps}")
                        _warmup_if_needed()
                    if (not ctrl.mapping) and prev_map:
                        _simlog(f"[sim] mapping DISABLED at step {steps}")

            # Fixed-step sim
            now = time.perf_counter()
            if now >= next_sim_t:
                if not ctrl.paused:
                    if ctrl.hold:
                        # Hold: zero velocities and ignore commands
                        state = State(state.p, jnp.zeros_like(state.v), state.q, jnp.zeros_like(state.w))
                    else:
                        # control → u
                        accel_unit = 3.0 * (2.0 if ctrl.boost else 1.0)
                        yaw_unit = 1.2 * (2.0 if ctrl.boost else 1.0)
                        ax = (1.0 if ctrl.fwd else 0.0) + (-1.0 if ctrl.back else 0.0)
                        ay = (1.0 if ctrl.right else 0.0) + (-1.0 if ctrl.left else 0.0)
                        az = (1.0 if ctrl.up else 0.0) + (-1.0 if ctrl.down else 0.0)
                        a_body = jnp.array([ax, ay, az], dtype=jnp.float32) * accel_unit
                        Rw = R_from_q(state.q)
                        a_world = Rw @ a_body
                        wz = (-1.0 if ctrl.yaw_l else 0.0) + (1.0 if ctrl.yaw_r else 0.0)
                        w_body = jnp.array([0.0, 0.0, wz * yaw_unit], dtype=jnp.float32)
                        st_in = state if not ctrl.brake else State(state.p, jnp.zeros_like(state.v), state.q, state.w)
                        u = jnp.concatenate([a_world, w_body])
                        try:
                            state = step_fn(st_in, u, cfg)
                        except Exception as e:
                            _simlog(f"[sim] step error: {e!r}")
                            state = st_in

                    if float(state.p[2]) < 0.05:
                        state = State(state.p.at[2].set(0.05), state.v, state.q, state.w)
                    steps += 1

                    # mapping update
                    if ctrl.mapping and (steps % int(max(1, map_update_every_steps)) == 0):
                        t0m = time.time()
                        try:
                            mapstate, counts = map_update_fn(mapstate, state)
                        except Exception as e:
                            _simlog(f"[sim] mapping_update error: {e!r}")
                        dtm = time.time() - t0m
                        try:
                            ch, cf, co = [int(v) for v in counts]
                            _simlog(f"[sim] map_update: hits={ch} free={cf} occ={co}")
                        except Exception:
                            pass
                        if dtm > 0.25:
                            _simlog(f"[sim] mapping_update took {dtm:.3f}s at step {steps}")

                next_sim_t += dt

                # steps/s heartbeat
                now_s = time.time()
                if now_s - rate_win_t0 >= 1.0:
                    steps_per_sec = (steps - rate_win_steps0) / (now_s - rate_win_t0)
                    rate_win_steps0 = steps
                    rate_win_t0 = now_s

            # State telemetry
            if time.time() - last_state_t >= state_dt:
                sim2ui.put({"type": "state",
                            "p": np.asarray(state.p, np.float32),
                            "q": np.asarray(state.q, np.float32),
                            "v": np.asarray(state.v, np.float32),
                            "w": np.asarray(state.w, np.float32),
                            "steps": int(steps),
                            "sps": float(steps_per_sec),
                            "mapping": bool(ctrl.mapping),
                            "paused": bool(ctrl.paused)})
                last_state_t = time.time()

            # Camera telemetry
            if time.time() - last_cam_t >= cam_dt:
                Rw = R_from_q(state.q)
                rays_w = (cam_dirs @ Rw.T)
                t0c = time.time()
                try:
                    t = cam_march_fn(state.p, rays_w)
                    img = np.asarray(jax.device_get(t)).reshape(cam_shape)
                    img = np.where(np.isfinite(img), img, tmax).astype(np.float32)  # far = tmax
                    sim2ui.put({"type": "cam", "img": img})
                except Exception as e:
                    _simlog(f"[sim] camera error: {e!r}")
                dtc = time.time() - t0c
                if dtc > 0.2:
                    _simlog(f"[sim] camera marcher took {dtc:.3f}s")
                last_cam_t = time.time()

            # Live field telemetry
            if ctrl.mapping and (time.time() - last_live_t >= live_dt):
                t0v = time.time()
                try:
                    vis_dev, Q_dev, absG_dev, mask_dev = compute_pack(mapstate.geom.theta, mapstate.expo.eta)
                    # choose field by mode
                    if ctrl.live_mode == 0:
                        field_dev, name = vis_dev, "vis"
                    elif ctrl.live_mode == 1:
                        field_dev, name = Q_dev, "Q"
                    elif ctrl.live_mode == 2:
                        field_dev, name = absG_dev, "|phi|"
                    else:
                        field_dev, name = mask_dev, "mask"

                    field = np.asarray(jax.device_get(field_dev))
                    # robust clim hint (2–98 percentiles)
                    if field.size:
                        lo, hi = np.nanpercentile(field, [2.0, 98.0])
                        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                            lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
                            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                                lo, hi = 0.0, 1.0
                    else:
                        lo, hi = 0.0, 1.0

                    # stats for logs
                    def _stats(x):
                        if x.size == 0: return (0.0, 0.0, 0.0)
                        return (float(np.nanmin(x)), float(np.nanmean(x)), float(np.nanmax(x)))
                    vmin, vmean, vmax = _stats(field)
                    qmin, qmean, qmax = _stats(np.asarray(jax.device_get(Q_dev)))
                    gmin, gmean, gmax = _stats(np.asarray(jax.device_get(absG_dev)))
                    mmin, mmean, mmax = _stats(np.asarray(jax.device_get(mask_dev)))

                    sim2ui.put({"type": "live",
                                "field": field.astype(np.float32),
                                "name": name,
                                "clim_hint": [float(lo), float(hi)],
                                "stats": {
                                    "field": [vmin, vmean, vmax],
                                    "Q": [qmin, qmean, qmax],
                                    "|phi|": [gmin, gmean, gmax],
                                    "mask": [mmin, mmean, mmax],
                                }})
                    _simlog(f"[sim] {name} stats: min={vmin:.3f} mean={vmean:.3f} max={vmax:.3f} | "
                            f"Q[{qmin:.3f},{qmean:.3f},{qmax:.3f}] | "
                            f"|phi|[{gmin:.3e},{gmean:.3e},{gmax:.3e}] | "
                            f"mask[{mmin:.3f},{mmean:.3f},{mmax:.3f}]")
                except Exception as e:
                    _simlog(f"[sim] compute_live_field error: {e!r}")
                dtv = time.time() - t0v
                if dtv > 0.2:
                    _simlog(f"[sim] compute_live_field took {dtv:.3f}s")
                last_live_t = time.time()

            # Periodic heartbeat
            if time.time() - last_heartbeat > 2.5:
                p = np.asarray(state.p)
                _simlog(f"[sim] heartbeat: steps={steps} sps~{steps_per_sec:.1f} "
                        f"p=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                        f"mapping={'ON' if ctrl.mapping else 'OFF'} "
                        f"hold={'ON' if ctrl.hold else 'OFF'} mode={ctrl.live_mode}")
                last_heartbeat = time.time()

            time.sleep(0.001)

    except Exception as e:
        _simlog(f"[sim-exception] {e!r}")
    finally:
        try:
            sim2ui.put({"type": "bye"})
        except Exception:
            pass


# ---------------------------------------------------------------------
# UI application (main process)
# ---------------------------------------------------------------------

class Interactive3DApp:
    def __init__(self):
        # IPC queues
        self.ui2sim: mp.Queue = mp.Queue(maxsize=8)
        self.sim2ui: mp.Queue = mp.Queue(maxsize=8)

        # Start sim process
        self.proc = mp.Process(target=_sim_worker, args=(self.ui2sim, self.sim2ui), daemon=True)
        self.proc.start()
        log.info("Sim process started.")

        # Placeholders filled by 'init'
        self.lb = np.array([-6.0, -6.0, 0.0], np.float32)
        self.ub = np.array([+6.0, +6.0, 4.0], np.float32)
        self.cam_shape: Tuple[int,int] = (24, 32)
        self.tmax = 12.0
        self.gt_points = np.zeros((0,3), np.float32)
        self.live_pts = np.zeros((0,3), np.float32)

        # Dynamic data
        self.p = np.array([0.,0.,1.6], np.float32)
        self.q = np.array([1.,0.,0.,0.], np.float32)
        self.v = np.zeros(3, np.float32)
        self.w = np.zeros(3, np.float32)
        self.sps = 0.0; self.steps = 0
        self.mapping = False; self.paused = False

        # Controls
        self.ctrl = Controls()

        # Live field / filtering state (UI side)
        self.live_field_name = "vis"
        self.live_auto_clim = True
        self.live_field_last: Optional[np.ndarray] = None
        self.live_lo_hint = 0.0
        self.live_hi_hint = 1.0
        self.live_thresh_alpha = 0.0  # slider in [0,1], maps into [lo_hint, hi_hint]

        # PyVista window
        pv.set_plot_theme("document")
        try:
            self.pl = pv.Plotter(shape=(1,3), window_size=(1500, 560), enable_keybindings=False)
        except TypeError:
            self.pl = pv.Plotter(shape=(1,3), window_size=(1500, 560))
            for attr in ("enable_keybindings", "enable_key_bindings"):
                if hasattr(self.pl, attr):
                    try: setattr(self.pl, attr, False)
                    except Exception: pass
        self.pl.add_text("live_mvp  3D interactive (decoupled UI)", font_size=10)

        # Left: GT + pose + HUD
        self.pl.subplot(0,0)
        self.gt_poly: Optional[pv.PolyData] = None
        self.gt_actor = None
        self.drone_actor = None
        self.hud_actor = None
        self.pl.show_axes()

        # Middle: Camera plane (defer until init)
        self.pl.subplot(0,1)
        self.cam_mesh: Optional[pv.PolyData] = None
        self.cam_actor = None
        self.pl.show_axes()

        # Right: Live cloud (defer until init)
        self.pl.subplot(0,2)
        self.live_poly: Optional[pv.PolyData] = None
        self.live_actor = None
        self.pl.show_axes()

        # Slider (global; controls min threshold over live field)
        self.live_slider = None  # created after actors
        self._bind_keys()

        # Wait briefly for 'init'
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self._drain_once():
                if (self.gt_points.size and self.live_pts.size and self.cam_actor is None):
                    self._build_static_actors()
                    break
            time.sleep(0.01)

    # ------------------ Messaging ------------------

    def _send_ctrl(self):
        try:
            self.ui2sim.put_nowait({"type": "ctrl", "ctrl": asdict(self.ctrl)})
        except Exception:
            pass

    def _drain_once(self) -> bool:
        got = False
        for _ in range(16):
            try:
                m = self.sim2ui.get_nowait()
            except Empty:
                break
            got = True
            tp = m.get("type")
            if tp == "init":
                self.gt_points = np.asarray(m["gt_points"], np.float32)
                self.live_pts = np.asarray(m["live_pts"], np.float32)
                self.cam_shape = tuple(m["cam_shape"])
                self.tmax = float(m.get("tmax", 12.0))
                self.lb = np.asarray(m.get("lb", self.lb), np.float32)
                self.ub = np.asarray(m.get("ub", self.ub), np.float32)
                if self.cam_actor is None or self.live_actor is None or self.gt_actor is None:
                    self._build_static_actors()
            elif tp == "state":
                self.p = np.asarray(m["p"], np.float32)
                self.q = np.asarray(m["q"], np.float32)
                self.v = np.asarray(m["v"], np.float32)
                self.w = np.asarray(m["w"], np.float32)
                self.steps = int(m.get("steps", self.steps))
                self.sps = float(m.get("sps", self.sps))
                self.mapping = bool(m.get("mapping", self.mapping))
                self.paused = bool(m.get("paused", self.paused))
            elif tp == "cam":
                img = np.asarray(m["img"], np.float32)
                self._update_cam_scalars(img)
            elif tp == "live":
                field = np.asarray(m["field"], np.float32)
                self.live_field_name = str(m.get("name", "vis"))
                clim_hint = m.get("clim_hint", None)
                if clim_hint and len(clim_hint) == 2:
                    self.live_lo_hint, self.live_hi_hint = float(clim_hint[0]), float(clim_hint[1])
                self._update_live_scalars(field, clim_hint)
                stats = m.get("stats", None)
                if stats:
                    log.info(f"[ui] field={self.live_field_name} "
                             f"min={stats['field'][0]:.3f} mean={stats['field'][1]:.3f} max={stats['field'][2]:.3f}")
            elif tp == "log":
                log.info(m.get("msg",""))
            elif tp == "bye":
                pass
        return got

    # ------------------ Actor construction & updates ------------------

    def _build_static_actors(self):
        # Left: GT shell
        self.pl.subplot(0,0)
        try:
            if self.gt_actor is not None:
                self.pl.remove_actor(self.gt_actor)
        except Exception:
            pass
        self.gt_poly = make_point_cloud_polydata(self.gt_points)
        self.gt_actor = self.pl.add_mesh(self.gt_poly, color="lightgray",
                                         render_points_as_spheres=True, point_size=5.0)

        # Middle: Camera plane
        self.pl.subplot(0,1)
        try:
            if self.cam_actor is not None:
                self.pl.remove_actor(self.cam_actor)
        except Exception:
            pass
        img0 = np.zeros(self.cam_shape, np.float32)
        self.cam_mesh = plane_from_image(img0, "values")
        self.cam_actor = self.pl.add_mesh(self.cam_mesh, scalars="values",
                                          cmap="viridis", clim=[0.0, float(self.tmax)],
                                          show_scalar_bar=False, lighting=False)

        # Right: Live point grid
        self.pl.subplot(0,2)
        try:
            if self.live_actor is not None:
                self.pl.remove_actor(self.live_actor)
        except Exception:
            pass
        self.live_poly = make_point_cloud_polydata(self.live_pts)
        self.live_poly.point_data["live"] = np.zeros((self.live_pts.shape[0],), dtype=np.float32)
        self.live_actor = self.pl.add_mesh(self.live_poly, scalars="live",
                                           render_points_as_spheres=False, point_size=3.0,
                                           clim=[0.0, 1.0], cmap="viridis",
                                           nan_color=None, nan_opacity=0.0)  # NaNs invisible

        # Slider widget for min threshold (normalized 0..1 in [lo_hint,hi_hint])
        try:
            if self.live_slider is not None:
                self.pl.remove_slider_widget(self.live_slider)
        except Exception:
            pass
        def _slider_cb(val):
            self._on_live_slider(val)
        try:
            # place along bottom: pointa/b set in normalized display coords if available
            self.live_slider = self.pl.add_slider_widget(
                callback=_slider_cb, rng=[0.0, 1.0], value=self.live_thresh_alpha,
                title="Live Min (auto-scale)", pointa=(0.35, 0.02), pointb=(0.98, 0.02)
            )
        except Exception:
            # fallback without positioning
            self.live_slider = self.pl.add_slider_widget(callback=_slider_cb, rng=[0.0, 1.0],
                                                         value=self.live_thresh_alpha, title="Live Min (auto-scale)")

    def _update_cam_scalars(self, img: np.ndarray):
        if self.cam_mesh is None or self.cam_actor is None:
            self._build_static_actors()
        vals = img.astype(np.float32).ravel(order="C")
        npts = self.cam_mesh.n_points if self.cam_mesh is not None else 0
        if self.cam_mesh is not None and vals.size == npts:
            pd = self.cam_mesh.point_data
            if "values" in pd and pd["values"].size == npts:
                pd["values"][:] = vals
            else:
                pd["values"] = vals
        else:
            self.cam_shape = img.shape
            self._build_static_actors()

    def _set_actor_clim(self, lo: float, hi: float):
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

    def _current_thresh_value(self) -> float:
        lo, hi = float(self.live_lo_hint), float(self.live_hi_hint)
        return float(lo + self.live_thresh_alpha * max(0.0, hi - lo))

    def _apply_min_mask_to_field(self, field: np.ndarray) -> np.ndarray:
        """Return a copy with values below threshold set to NaN (hidden)."""
        thr = self._current_thresh_value()
        masked = field.astype(np.float32).copy()
        masked[field < thr] = np.nan
        return masked

    def _update_live_scalars(self, field: np.ndarray, clim_hint: Optional[list]):
        self.live_field_last = field.copy()

        if self.live_poly is None:
            return

        # NaN-mask out points below threshold
        field_vis = self._apply_min_mask_to_field(field)

        pd = self.live_poly.point_data
        n = self.live_poly.n_points
        if field_vis.size != n:
            field_vis = np.resize(field_vis.astype(np.float32), n)
        if "live" in pd and pd["live"].size == n:
            pd["live"][:] = field_vis
        else:
            pd["live"] = field_vis

        # Adaptive color scaling (toggle with 'A')
        if self.live_auto_clim:
            if clim_hint is not None and len(clim_hint) == 2:
                lo, hi = float(clim_hint[0]), float(clim_hint[1])
            else:
                # fallback on UI side if no hint provided
                finite_vals = field[np.isfinite(field)]
                if finite_vals.size:
                    lo, hi = np.nanpercentile(finite_vals, [2.0, 98.0])
                    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                        lo, hi = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
                        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                            lo, hi = 0.0, 1.0
                else:
                    lo, hi = 0.0, 1.0
            pad = 0.02 * (hi - lo + 1e-12)
            self._set_actor_clim(lo - pad, hi + pad)
        else:
            self._set_actor_clim(0.0, 1.0)

    def _on_live_slider(self, alpha: float):
        """Slider callback: alpha in [0,1] mapped into [lo_hint, hi_hint]."""
        try:
            self.live_thresh_alpha = float(alpha)
        except Exception:
            return
        # apply to last field immediately for responsiveness
        if self.live_field_last is not None:
            self._update_live_scalars(self.live_field_last, [self.live_lo_hint, self.live_hi_hint])
        log.info(f"[ui] slider: min threshold alpha={self.live_thresh_alpha:.3f} "
                 f"→ value={self._current_thresh_value():.4f} "
                 f"({self.live_field_name})")

    def _update_pose_and_hud(self):
        # Recreate small pose meshes (cheap)
        self.pl.subplot(0,0)
        p = self.p.astype(np.float32)
        Rw = quat_to_R_numpy(self.q)
        fwd = (Rw @ np.array([1.0, 0.0, 0.0], np.float32))
        sphere = pv.Sphere(radius=0.08, center=p)
        arrow = pv.Arrow(start=p, direction=fwd, tip_length=0.2, tip_radius=0.05,
                         shaft_radius=0.02, scale=0.4)
        try:
            if self.drone_actor is not None:
                self.pl.remove_actor(self.drone_actor)
        except Exception:
            pass
        self.drone_actor = self.pl.add_mesh(sphere, color="orange")
        _ = self.pl.add_mesh(arrow, color="red")

        # HUD
        try:
            if self.hud_actor is not None:
                self.pl.remove_actor(self.hud_actor)
        except Exception:
            pass
        spd = float(np.linalg.norm(self.v))
        yaw_rate = float(self.w[2])
        mode_name = {0:"vis", 1:"Q", 2:"|phi|", 3:"mask"}.get(self.ctrl.live_mode, "vis")
        # visible count (best-effort; requires last field)
        vis_cnt = 0
        if self.live_field_last is not None:
            thr = self._current_thresh_value()
            vis_cnt = int(np.sum(np.isfinite(self.live_field_last) & (self.live_field_last >= thr)))
        def on(b): return "●" if b else "○"
        keys_line = (
            f"I{on(self.ctrl.fwd)} K{on(self.ctrl.back)}  "
            f"J{on(self.ctrl.left)} L{on(self.ctrl.right)}  "
            f"U{on(self.ctrl.down)} O{on(self.ctrl.up)}  "
            f"Z{on(self.ctrl.yaw_l)} C{on(self.ctrl.yaw_r)}  "
            f"Shift{on(self.ctrl.boost)} Space{on(self.ctrl.brake)}  "
            f"H(HOLD {on(self.ctrl.hold)})  "
            f"P{on(self.ctrl.paused)} M{on(self.ctrl.mapping)}  "
            f"A(auto-clim {on(self.live_auto_clim)})  "
            f"1/2/3/4(mode={mode_name})"
        )
        hud = (
            f"[steps={self.steps} ~{self.sps:.1f} sps]  "
            f"p=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})  "
            f"|v|={spd:.2f}  yaw={yaw_rate:.2f}  "
            f"min_th={self._current_thresh_value():.4f}  "
            f"visible≈{vis_cnt}"
            f"\nKeys: {keys_line}"
        )
        self.hud_actor = self.pl.add_text(hud, font_size=10, color="white")

    # ------------------ Input handling ------------------

    def _bind_keys(self):
        iren = getattr(self.pl, "iren", None)
        if iren is not None:
            if hasattr(iren, "add_observer"):
                iren.add_observer("KeyPressEvent", self._on_vtk_key_press)
                iren.add_observer("KeyReleaseEvent", self._on_vtk_key_release)
            elif hasattr(iren, "AddObserver"):
                iren.AddObserver("KeyPressEvent", self._on_vtk_key_press)
                iren.AddObserver("KeyReleaseEvent", self._on_vtk_key_release)
        if hasattr(self.pl, "add_key_event"):
            for k in ("i","k","j","l","u","o","z","c","space","p","h","m","a","1","2","3","4","escape","Esc","Shift_L","Shift_R"):
                self.pl.add_key_event(k, lambda k=k: self._apply_press(self._norm(k)))

    def _norm(self, token: str) -> str:
        t = token.lower()
        if t in ("escape","esc"): return "escape"
        if t in ("shift_l","shift_r","shift"): return "shift"
        if t == "space": return "space"
        return t

    def _on_vtk_key_press(self, obj, evt):
        try: token = obj.GetKeySym()
        except Exception: token = ""
        self._apply_press(self._norm(str(token)))

    def _on_vtk_key_release(self, obj, evt):
        try: token = obj.GetKeySym()
        except Exception: token = ""
        self._apply_release(self._norm(str(token)))

    def _apply_press(self, key: str):
        if key == "escape":
            self.ctrl.quit = True
            try: self.ui2sim.put_nowait({"type": "quit"})
            except Exception: pass
            return
        if key == "p":
            self.ctrl.paused = not self.ctrl.paused; self._send_ctrl(); return
        if key == "m":
            # toggle mapping on release
            return
        if key == "a":
            self.live_auto_clim = not self.live_auto_clim
            self.ctrl.live_auto_clim = self.live_auto_clim
            self._send_ctrl()
            return
        if key == "1":
            self.ctrl.live_mode = 0; self._send_ctrl(); return
        if key == "2":
            self.ctrl.live_mode = 1; self._send_ctrl(); return
        if key == "3":
            self.ctrl.live_mode = 2; self._send_ctrl(); return
        if key == "4":
            self.ctrl.live_mode = 3; self._send_ctrl(); return
        if key == "h":
            self.ctrl.hold = not self.ctrl.hold; self._send_ctrl(); return
        if key == "shift": self.ctrl.boost = True;  self._send_ctrl(); return
        if key == "space": self.ctrl.brake = True;  self._send_ctrl(); return
        if   key == "i": self.ctrl.fwd   = True
        elif key == "k": self.ctrl.back  = True
        elif key == "j": self.ctrl.left  = True
        elif key == "l": self.ctrl.right = True
        elif key == "u": self.ctrl.down  = True
        elif key == "o": self.ctrl.up    = True
        elif key == "z": self.ctrl.yaw_l = True
        elif key == "c": self.ctrl.yaw_r = True
        self._send_ctrl()

    def _apply_release(self, key: str):
        if key == "shift": self.ctrl.boost = False; self._send_ctrl(); return
        if key == "space": self.ctrl.brake = False; self._send_ctrl(); return
        if key == "m":
            self.ctrl.mapping = not self.ctrl.mapping; self._send_ctrl(); return
        if   key == "i": self.ctrl.fwd   = False
        elif key == "k": self.ctrl.back  = False
        elif key == "j": self.ctrl.left  = False
        elif key == "l": self.ctrl.right = False
        elif key == "u": self.ctrl.down  = False
        elif key == "o": self.ctrl.up    = False
        elif key == "z": self.ctrl.yaw_l = False
        elif key == "c": self.ctrl.yaw_r = False
        self._send_ctrl()

    # ------------------ Main UI loop ------------------

    def run(self):
        try:
            self.pl.show(auto_close=False, interactive_update=True)
        except TypeError:
            try: self.pl.show(auto_close=False)
            except TypeError: self.pl.show()

        try:
            while not self.ctrl.quit:
                self._drain_once()
                self._update_pose_and_hud()
                try: self.pl.update()
                except Exception: pass
                self._send_ctrl()
                time.sleep(1.0/30.0)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.ctrl.quit = True
                self.ui2sim.put_nowait({"type":"quit"})
            except Exception:
                pass
            try:
                if self.proc.is_alive():
                    self.proc.join(timeout=1.0)
            except Exception:
                pass
            try: self.pl.close()
            except Exception: pass

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    app = Interactive3DApp()
    app.run()

if __name__ == "__main__":
    main()
