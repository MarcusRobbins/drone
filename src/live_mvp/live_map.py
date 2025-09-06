
from typing import NamedTuple, Optional
import os
import jax, jax.numpy as jnp, optax
from jax import tree_util as jtu

# ---------------------------
# Multi-resolution hash-grid
# ---------------------------

class HashCfg(NamedTuple):
    L: int = 12          # levels
    F: int = 2           # features per level
    N_min: int = 16      # min grid res
    N_max: int = 512     # max grid res
    T: int = 1 << 15     # table size per level (power of two)
    lb: jnp.ndarray = jnp.array([-6., -6.,  0.])   # world AABB lower
    ub: jnp.ndarray = jnp.array([ 6.,  6.,  4.])   # world AABB upper

HASH_CFG = HashCfg()

def _level_res_all(cfg: HashCfg):
    ls = jnp.arange(cfg.L, dtype=jnp.float32)
    g = ls / jnp.maximum(cfg.L - 1, 1)
    Ns = jnp.floor(cfg.N_min * (cfg.N_max / cfg.N_min) ** g).astype(jnp.int32)
    return Ns

def _level_res(l: int, cfg: HashCfg):
    g = l / max(cfg.L - 1, 1)
    return jnp.floor(cfg.N_min * (cfg.N_max / cfg.N_min) ** g).astype(jnp.int32)

def _hash_ijk(ijk, T):
    # 3D mix hash (Instant-NGP-style primes)
    ijk = ijk.astype(jnp.uint32)
    x, y, z = ijk[...,0], ijk[...,1], ijk[...,2]
    h = (x * jnp.uint32(0x9E3779B1) ^
         y * jnp.uint32(0x85EBCA77) ^
         z * jnp.uint32(0xC2B2AE3D))
    return (h & (jnp.uint32(T) - jnp.uint32(1))).astype(jnp.int32)

_OFFS = jnp.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], jnp.int32)  # (8,3)

def _encode_point_ref(tables, x, cfg: HashCfg):
    # Original per-level, per-corner encoder (reference)
    u = (x - cfg.lb) / (cfg.ub - cfg.lb + 1e-9)
    u = jnp.clip(u, 0.0, 1.0)
    feats = []
    for l in range(cfg.L):
        N = _level_res(l, cfg)
        u_grid = u * (N - 1).astype(jnp.float32)
        i0 = jnp.floor(u_grid).astype(jnp.int32)
        t = (u_grid - i0.astype(jnp.float32))
        i0 = jnp.clip(i0, 0, N - 2)
        emb = jnp.zeros((cfg.F,))
        for dz in (0,1):
            for dy in (0,1):
                for dx in (0,1):
                    corner = i0 + jnp.array([dx, dy, dz], dtype=jnp.int32)
                    wxyz = ((t[0] if dx else 1.0 - t[0]) *
                            (t[1] if dy else 1.0 - t[1]) *
                            (t[2] if dz else 1.0 - t[2]))
                    hid = _hash_ijk(corner, cfg.T)
                    emb_l = tables[l][hid]  # (F,)
                    emb = emb + wxyz * emb_l
        feats.append(emb)
    return jnp.concatenate(feats, axis=0)  # (L*F,)

def _encode_point_fused_LTF(tables_LTF: jnp.ndarray, x: jnp.ndarray, cfg: HashCfg):
    # Fused across levels and 8 corners; expects tables stacked as (L,T,F)
    u = (x - cfg.lb) / (cfg.ub - cfg.lb + 1e-9)
    u = jnp.clip(u, 0.0, 1.0)
    Ns = _level_res_all(cfg).astype(jnp.float32)            # (L,)
    ugrid = u[None,:] * (Ns - 1.0)[:,None]                  # (L,3)
    i0 = jnp.floor(ugrid).astype(jnp.int32)                 # (L,3)
    t = (ugrid - i0.astype(jnp.float32))                    # (L,3)
    # Clip per-level, broadcasting the max bound over xyz axis.
    i0 = jnp.clip(i0, 0, (Ns.astype(jnp.int32) - 2)[:, None])

    corners = i0[:,None,:] + _OFFS[None,:,:]                # (L,8,3)
    h = _hash_ijk(corners, cfg.T)                           # (L,8)
    # Gather features per level along the T axis.
    # Expand indices to (L,8,F) so they match arr's non-axis dims (L,*,F).
    L, F = tables_LTF.shape[0], tables_LTF.shape[2]
    h_exp = jnp.broadcast_to(h[..., None], (L, h.shape[1], F))  # (L,8,F)
    emb = jnp.take_along_axis(tables_LTF, h_exp, axis=1)        # (L,8,F)

    wx = jnp.where(_OFFS[None,:,0]==1, t[:,None,0], 1.0 - t[:,None,0])
    wy = jnp.where(_OFFS[None,:,1]==1, t[:,None,1], 1.0 - t[:,None,1])
    wz = jnp.where(_OFFS[None,:,2]==1, t[:,None,2], 1.0 - t[:,None,2])
    w = (wx * wy * wz).astype(emb.dtype)                    # (L,8)
    per_level = jnp.sum(w[...,None] * emb, axis=1)          # (L,F)
    return per_level.reshape(-1)                            # (L*F,)

def _encode_point_fused(tables_tuple, x, cfg: HashCfg):
    # Accepts tuple-of-levels (T,F), stacks to (L,T,F)
    tables_LTF = jnp.stack(tables_tuple, axis=0)
    return _encode_point_fused_LTF(tables_LTF, x, cfg)

# Toggle between reference and fused encoders via env var.
_FUSED_ENCODER = os.environ.get("LIVEMVP_FUSED_ENCODER", "1") in ("1", "true", "True")

def _encode_point(tables, x, cfg: HashCfg):
    return (_encode_point_fused(tables, x, cfg)
            if _FUSED_ENCODER else _encode_point_ref(tables, x, cfg))

v_encode = jax.vmap(_encode_point, in_axes=(None,0,None))

def init_hash_tables(key, cfg: HashCfg):
    keys = jax.random.split(key, cfg.L)
    return tuple(jax.random.normal(k, (cfg.T, cfg.F)) * 1e-4 for k in keys)

# ---------------------------
# Tiny MLP
# ---------------------------

def init_mlp(key, in_dim, hidden, out_dim, bias_last: float = 0.0):
    keys = jax.random.split(key, len(hidden)+1)
    params=[]
    prev = in_dim
    for k, h in zip(keys[:-1], hidden):
        W = jax.random.normal(k, (prev, h)) * (1.0/jnp.sqrt(prev))
        b = jnp.zeros((h,)); params.append((W,b)); prev=h
    k = keys[-1]
    W = jax.random.normal(k, (prev, out_dim)) * (1.0/jnp.sqrt(prev))
    b = jnp.full((out_dim,), float(bias_last)); params.append((W,b))
    return tuple(params)

def mlp_apply(params, x):
    for W,b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W,b = params[-1]
    return x @ W + b

# ---------------------------
# States and fields
# ---------------------------

class GeomParams(NamedTuple):
    tables: tuple  # tuple of (T,F) arrays per level
    mlp:    tuple  # MLP weights

class ExpoParams(NamedTuple):
    tables: tuple
    mlp:    tuple

class GeomState(NamedTuple):
    theta: GeomParams
    opt:   optax.OptState

class ExpoState(NamedTuple):
    eta: ExpoParams
    opt: optax.OptState

class MapState(NamedTuple):
    geom: GeomState
    expo: ExpoState

# Gradient accumulation for mapping via Optax apply_every.
# Controlled by env var LIVEMVP_MAP_ACCUM (default 1 = no accumulation).
_APPLY_EVERY_K = int(os.environ.get("LIVEMVP_MAP_ACCUM", "1"))

if _APPLY_EVERY_K <= 1:
    _G_OPT_TX = optax.adam(1e-3)
    _E_OPT_TX = optax.adam(1e-3)
else:
    # Accumulate grads for K steps, then apply once.
    _G_OPT_TX = optax.chain(optax.adam(1e-3), optax.apply_every(_APPLY_EVERY_K))
    _E_OPT_TX = optax.chain(optax.adam(1e-3), optax.apply_every(_APPLY_EVERY_K))

# ---------- Debug / safety helpers ----------
_DBG = os.environ.get("LIVEMVP_DEBUG", "0") in ("1", "true", "True")
_NO_EIK = os.environ.get("LIVEMVP_NO_EIK", "0") in ("1", "true", "True")

def _dbg_print(do, fmt, *xs):
    if not _DBG:
        return
    do = jnp.asarray(do, dtype=jnp.bool_)
    def _yes(_):
        jax.debug.print(fmt, *xs)
        return 0
    def _no(_):
        return 0
    _ = jax.lax.cond(do, _yes, _no, 0)

def _clean_float(x, max_abs: float = 1e6):
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = jnp.clip(x, -max_abs, max_abs)
    return x

def _tree_stats(tree):
    leaves = [jnp.asarray(x) for x in jtu.tree_leaves(tree)
              if hasattr(x, "dtype") and jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)]
    if not leaves:
        z = jnp.array(0.0, jnp.float32)
        return z, z, z
    v = jnp.concatenate([x.reshape(-1) for x in leaves])
    frac_bad = 1.0 - jnp.mean(jnp.isfinite(v).astype(jnp.float32))
    rms = jnp.sqrt(jnp.mean(v * v))
    maxabs = jnp.max(jnp.abs(v))
    return frac_bad, rms, maxabs

def init_live_map(key, geom_bias: float = 0.0):
    k1,k2,k3,k4 = jax.random.split(key, 4)
    tables_g = init_hash_tables(k1, HASH_CFG)
    tables_e = init_hash_tables(k2, HASH_CFG)
    in_dim   = HASH_CFG.L * HASH_CFG.F
    mlp_g    = init_mlp(k3, in_dim, [64,64], 1, bias_last=float(geom_bias))
    mlp_e    = init_mlp(k4, in_dim, [64,64], 1)
    theta    = GeomParams(tables=tables_g, mlp=mlp_g)
    eta      = ExpoParams(tables=tables_e, mlp=mlp_e)
    return MapState(GeomState(theta, _G_OPT_TX.init(theta)),
                    ExpoState(eta,   _E_OPT_TX.init(eta)))

def G_phi(x, theta: GeomParams):
    z = _encode_point(theta.tables, x, HASH_CFG)
    return mlp_apply(theta.mlp, z)[0]

def Q_expo(x, eta: ExpoParams):
    z = _encode_point(eta.tables, x, HASH_CFG)
    return jax.nn.sigmoid(mlp_apply(eta.mlp, z)[0])

v_G = jax.vmap(G_phi, in_axes=(0, None))
v_Q = jax.vmap(Q_expo, in_axes=(0, None))

# ---------------------------
# Online updates (masked, JIT-safe)
# ---------------------------

def update_geom(mapstate: MapState,
                hits_xyz, hits_mask,           # (R,3), (R,)
                frees_xyz, frees_mask):        # (R,S,3), (R,S)
    theta, opt = mapstate.geom

    def loss_fn(params: GeomParams):
        mu = 0.2
        # hits
        l_hit, eik = 0.0, 0.0
        if hits_xyz.shape[0] > 0:
            g_hits = v_G(hits_xyz, params)                      # (R,)
            w_hits = hits_mask.astype(jnp.float32)
            l_hit  = (w_hits * (g_hits**2)).sum() / (w_hits.sum() + 1e-6)
            # eikonal near hits (masked)
            if not _NO_EIK:
                def grad_norm(x):
                    return jnp.linalg.norm(jax.grad(lambda xx: G_phi(xx, params))(x))
                gn = jax.vmap(grad_norm)(hits_xyz)
                eik = ((w_hits * (gn - 1.0)**2).sum() / (w_hits.sum() + 1e-6))

        # free space
        l_free = 0.0
        if frees_xyz.size > 0:
            xs = frees_xyz.reshape(-1,3)                        # (R*S,3)
            wm = frees_mask.reshape(-1).astype(jnp.float32)     # (R*S,)
            g_free = v_G(xs, params)
            l_free = (wm * (g_free - mu)**2).sum() / (wm.sum() + 1e-6)

        # weight eik term unless disabled
        return l_hit + 0.5*l_free + (0.0 if _NO_EIK else 0.1)*eik

    # grads -> sanitize
    g = jax.grad(loss_fn)(theta)
    g = jtu.tree_map(_clean_float, g)
    # opt update -> sanitize
    updates, opt2 = _G_OPT_TX.update(g, opt, theta)
    updates = jtu.tree_map(_clean_float, updates)
    opt2    = jtu.tree_map(_clean_float, opt2)
    theta2  = optax.apply_updates(theta, updates)
    theta2  = jtu.tree_map(_clean_float, theta2)

    # diagnostics
    gb, grms, gmax = _tree_stats(g)
    ub, urms, umax = _tree_stats(updates)
    tb, trms, tmax = _tree_stats(theta2)
    ob, orms, omax = _tree_stats(opt2)
    _dbg_print(True,
               "[lm] geom: g_rms={:.3e} upd_rms={:.3e} theta_rms={:.3e} opt_rms={:.3e} bad[g,u,θ,opt]=({:.3f},{:.3f},{:.3f},{:.3f}) max[θ]={:.3e}",
               grms, urms, trms, orms, gb, ub, tb, ob, tmax)
    return MapState(GeomState(theta2, opt2), mapstate.expo)

def update_expo(mapstate: MapState,
                seen_xyz, w_seen,               # (R,S,3), (R,S)  : free before hit  (target=1)
                occ_xyz: Optional[jnp.ndarray] = None,
                w_occ: Optional[jnp.ndarray] = None,
                neg_weight: float = 0.5):       # relative weight for negatives
    """
    Balanced BCE on exposure Q: encourage Q→1 along free segments and Q→0 in occluded segments.
    If occ_xyz/w_occ are None, falls back to the old positive-only update.
    """
    eta, opt = mapstate.expo

    def loss_fn(params: ExpoParams):
        loss = 0.0
        eps = 1e-6

        if seen_xyz.size > 0:
            xs = seen_xyz.reshape(-1,3)
            wp = w_seen.reshape(-1).astype(jnp.float32)
            p  = v_Q(xs, params)
            pos = -(wp * jnp.log(p + eps)).sum() / (wp.sum() + 1e-6)
            loss = loss + pos

        if (occ_xyz is not None) and (w_occ is not None) and (occ_xyz.size > 0):
            xn = occ_xyz.reshape(-1,3)
            wn = w_occ.reshape(-1).astype(jnp.float32)
            p  = v_Q(xn, params)
            neg = -(wn * jnp.log(1.0 - p + eps)).sum() / (wn.sum() + 1e-6)
            loss = loss + neg_weight * neg

        return loss

    g = jax.grad(loss_fn)(eta)
    g = jtu.tree_map(_clean_float, g)
    updates, opt2 = _E_OPT_TX.update(g, opt, eta)
    updates = jtu.tree_map(_clean_float, updates)
    opt2    = jtu.tree_map(_clean_float, opt2)
    eta2    = optax.apply_updates(eta, updates)
    eta2    = jtu.tree_map(_clean_float, eta2)

    # diagnostics
    gb, grms, gmax = _tree_stats(g)
    ub, urms, umax = _tree_stats(updates)
    eb, erms, emax = _tree_stats(eta2)
    ob, orms, omax = _tree_stats(opt2)
    _dbg_print(True,
               "[lm] expo: g_rms={:.3e} upd_rms={:.3e} eta_rms={:.3e} opt_rms={:.3e} bad[g,u,η,opt]=({:.3f},{:.3f},{:.3f},{:.3f}) max[η]={:.3e}",
               grms, urms, erms, orms, gb, ub, eb, ob, emax)
    return MapState(mapstate.geom, ExpoState(eta2, opt2))
