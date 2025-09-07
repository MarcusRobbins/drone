import os
from dataclasses import dataclass
import jax, jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional

def smin(a, b, k=8.0):
    return -jnp.log(jnp.exp(-k*a) + jnp.exp(-k*b) + 1e-12) / k

def sd_box(x, h):
    q = jnp.abs(x) - h
    # robust for both (3,) and (...,3)
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside  = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside

# -------- Ground-truth config (module-local, process-wide) --------
@dataclass(frozen=True)
class GTConfig:
    include_plane: bool = True  # default ON (preserves current tests/behavior)


def _env_truthy(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on", "y", "t")


# Read once at import time; can be overridden programmatically via set_gt_config.
_GT_CFG = GTConfig(include_plane=_env_truthy("LIVEMVP_GROUND_PLANE", True))


def get_gt_config() -> GTConfig:
    return _GT_CFG


def set_gt_config(include_plane: Optional[bool] = None) -> GTConfig:
    """
    Override GT config for this process. Must be called before building grids / JITs.
    """
    global _GT_CFG
    if include_plane is not None:
        _GT_CFG = GTConfig(include_plane=bool(include_plane))
    return _GT_CFG


def phi_gt(x):
    # Sphere
    c_sph = jnp.array([3.0, 0.0, 1.0]); r_sph = 1.0
    phi_sphere = jnp.linalg.norm(x - c_sph) - r_sph
    # Box
    c_box = jnp.array([1.6, -1.4, 0.7]); he = jnp.array([0.6, 0.6, 0.8])
    phi_box = sd_box(x - c_box, he)
    if _GT_CFG.include_plane:
        # Ground plane z=0 (positive above)
        phi_plane = x[2]
        return smin(smin(phi_plane, phi_sphere), phi_box)
    else:
        return smin(phi_sphere, phi_box)

def raycast_depth_gt(o, d, t_max=12.0, eps=1e-3, iters=64):
    """
    Sphere tracing on the GT SDF, returning the first t where |phi|<eps.
    The direction is normalized; step length uses abs(phi) to avoid backtracking.
    """
    o = jnp.asarray(o, dtype=jnp.float32)
    d = jnp.asarray(d, dtype=jnp.float32)
    d = d / (jnp.linalg.norm(d) + 1e-12)

    def cond(state):
        t, i, t_hit, done = state
        return (i < iters) & (~done) & (t <= t_max)

    def body(state):
        t, i, t_hit, done = state
        x   = o + t * d
        phi = phi_gt(x)
        step = jnp.clip(jnp.abs(phi), 1e-3, 0.5)
        new_t = t + step
        is_hit = jnp.abs(phi) < eps
        # latch first hit time
        new_t_hit = jnp.where(done, t_hit, jnp.where(is_hit, t, t_hit))
        new_done  = done | is_hit
        return (new_t, i+1, new_t_hit, new_done)

    t0 = jnp.array(0.0, dtype=jnp.float32)
    i0 = jnp.array(0, dtype=jnp.int32)
    th0 = jnp.array(0.0, dtype=jnp.float32)
    dn0 = jnp.array(False)

    t_fin, _, t_hit, done = jax.lax.while_loop(cond, body, (t0, i0, th0, dn0))
    return jnp.where(done & (t_hit <= t_max), t_hit, jnp.nan)


# --- fast GT via precomputed grid ---
class GTGrid(NamedTuple):
    lb: jnp.ndarray
    ub: jnp.ndarray
    res: jnp.ndarray         # (3,) int32
    dx:  jnp.ndarray         # (3,) float32
    phi: jnp.ndarray         # (Nx,Ny,Nz) float32


def build_gt_grid(lb: jnp.ndarray, ub: jnp.ndarray, res_xyz=(160, 160, 80), use_numpy: bool = True) -> GTGrid:
    """Precompute GT SDF on a 3D grid.

    When use_numpy=True, compute on CPU with NumPy to avoid compiling a massive XLA graph.
    The resulting arrays are then converted to JAX DeviceArrays.
    """
    if use_numpy:
        lb_np = np.asarray(lb, np.float32)
        ub_np = np.asarray(ub, np.float32)
        res_np = np.asarray(res_xyz, np.int32)
        xs = np.linspace(lb_np[0], ub_np[0], int(res_np[0]), dtype=np.float32)
        ys = np.linspace(lb_np[1], ub_np[1], int(res_np[1]), dtype=np.float32)
        zs = np.linspace(lb_np[2], ub_np[2], int(res_np[2]), dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
        # Analytic SDF in NumPy (honor config)
        phi_plane = Z
        c_sph = np.array([3.0, 0.0, 1.0], np.float32); r_sph = 1.0
        phi_sphere = np.linalg.norm(np.stack([X - c_sph[0], Y - c_sph[1], Z - c_sph[2]], axis=-1), axis=-1) - r_sph
        c_box = np.array([1.6, -1.4, 0.7], np.float32); he = np.array([0.6, 0.6, 0.8], np.float32)
        # sd_box in NumPy
        qx = np.abs(X - c_box[0]) - he[0]
        qy = np.abs(Y - c_box[1]) - he[1]
        qz = np.abs(Z - c_box[2]) - he[2]
        outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
        inside = np.minimum(np.maximum.reduce([qx, qy, qz]), 0.0)
        phi_box = outside + inside
        # smin with k=8.0
        def smin_np(a, b, k=8.0):
            return -np.log(np.exp(-k * a) + np.exp(-k * b) + 1e-12) / k
        if _GT_CFG.include_plane:
            phi_np = smin_np(smin_np(phi_plane, phi_sphere), phi_box).astype(np.float32)
        else:
            phi_np = smin_np(phi_sphere, phi_box).astype(np.float32)
        dx_np = (ub_np - lb_np) / (res_np.astype(np.float32) - 1.0)
        return GTGrid(
            lb=jnp.asarray(lb_np),
            ub=jnp.asarray(ub_np),
            res=jnp.asarray(res_np),
            dx=jnp.asarray(dx_np),
            phi=jnp.asarray(phi_np),
        )
    else:
        lb = jnp.asarray(lb, jnp.float32)
        ub = jnp.asarray(ub, jnp.float32)
        res = jnp.asarray(jnp.array(res_xyz), jnp.int32)
        xs = jnp.linspace(lb[0], ub[0], res[0])
        ys = jnp.linspace(lb[1], ub[1], res[1])
        zs = jnp.linspace(lb[2], ub[2], res[2])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing='xy')
        P = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        PHI = jax.vmap(phi_gt)(P).reshape(res[0], res[1], res[2]).astype(jnp.float32)
        dx = (ub - lb) / (res.astype(jnp.float32) - 1.0)
        return GTGrid(lb=lb, ub=ub, res=res, dx=dx, phi=PHI)


def _phi_grid_trilinear(x: jnp.ndarray, grid: GTGrid) -> jnp.ndarray:
    # x: (3,)
    u = (x - grid.lb) / (grid.ub - grid.lb + 1e-9)
    p = u * (grid.res.astype(jnp.float32) - 1.0)
    i0 = jnp.floor(p).astype(jnp.int32)
    t = p - i0.astype(jnp.float32)
    # ensure we have valid 8-corner cube
    i0 = jnp.clip(i0, 0, grid.res - 2)

    offsets = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        jnp.int32,
    )
    corners = i0[None, :] + offsets  # (8,3)

    def sample(idx3):
        return grid.phi[idx3[0], idx3[1], idx3[2]]

    vals = jax.vmap(sample)(corners)  # (8,)
    wx = jnp.where(offsets[:, 0] == 1, t[0], 1.0 - t[0])
    wy = jnp.where(offsets[:, 1] == 1, t[1], 1.0 - t[1])
    wz = jnp.where(offsets[:, 2] == 1, t[2], 1.0 - t[2])
    w = (wx * wy * wz).astype(jnp.float32)
    return jnp.sum(w * vals)


def raycast_depth_grid(o, d, grid: GTGrid, t_max=12.0, eps=1e-3, iters=48):
    o = jnp.asarray(o, jnp.float32)
    d = jnp.asarray(d, jnp.float32)
    d = d / (jnp.linalg.norm(d) + 1e-12)

    def cond(s):
        t, i, t_hit, done = s
        return (i < iters) & (~done) & (t <= t_max)

    def body(s):
        t, i, t_hit, done = s
        x = o + t * d
        phi = _phi_grid_trilinear(x, grid)
        step = jnp.clip(jnp.abs(phi), 1e-3, 0.5)
        new_t = t + step
        is_hit = jnp.abs(phi) < eps
        new_t_hit = jnp.where(done, t_hit, jnp.where(is_hit, t, t_hit))
        new_done = done | is_hit
        return (new_t, i + 1, new_t_hit, new_done)

    t0 = jnp.array(0.0, jnp.float32)
    i0 = jnp.array(0, jnp.int32)
    th0 = jnp.array(0.0, jnp.float32)
    dn0 = jnp.array(False)
    _, _, t_hit, done = jax.lax.while_loop(cond, body, (t0, i0, th0, dn0))
    return jnp.where(done & (t_hit <= t_max), t_hit, jnp.nan)
