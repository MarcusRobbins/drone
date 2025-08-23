
from typing import NamedTuple, Optional
import jax, jax.numpy as jnp, optax

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

def _encode_point(tables, x, cfg: HashCfg):
    # Normalize to [0,1]^3 within AABB
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
                    hid = _hash_ijk(corner, HASH_CFG.T)
                    emb_l = tables[l][hid]  # (F,)
                    emb = emb + wxyz * emb_l
        feats.append(emb)
    return jnp.concatenate(feats, axis=0)  # (L*F,)

v_encode = jax.vmap(_encode_point, in_axes=(None,0,None))

def init_hash_tables(key, cfg: HashCfg):
    keys = jax.random.split(key, cfg.L)
    return tuple(jax.random.normal(k, (cfg.T, cfg.F)) * 1e-4 for k in keys)

# ---------------------------
# Tiny MLP
# ---------------------------

def init_mlp(key, in_dim, hidden, out_dim):
    keys = jax.random.split(key, len(hidden)+1)
    params=[]
    prev = in_dim
    for k, h in zip(keys[:-1], hidden):
        W = jax.random.normal(k, (prev, h)) * (1.0/jnp.sqrt(prev))
        b = jnp.zeros((h,)); params.append((W,b)); prev=h
    k = keys[-1]
    W = jax.random.normal(k, (prev, out_dim)) * (1.0/jnp.sqrt(prev))
    b = jnp.zeros((out_dim,)); params.append((W,b))
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

_G_OPT_TX = optax.adam(1e-3)
_E_OPT_TX = optax.adam(1e-3)

def init_live_map(key):
    k1,k2,k3,k4 = jax.random.split(key, 4)
    tables_g = init_hash_tables(k1, HASH_CFG)
    tables_e = init_hash_tables(k2, HASH_CFG)
    in_dim   = HASH_CFG.L * HASH_CFG.F
    mlp_g    = init_mlp(k3, in_dim, [64,64], 1)
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
            def grad_norm(x): return jnp.linalg.norm(jax.grad(lambda xx: G_phi(xx, params))(x))
            gn = jax.vmap(grad_norm)(hits_xyz)
            eik = ((w_hits * (gn - 1.0)**2).sum() / (w_hits.sum() + 1e-6))

        # free space
        l_free = 0.0
        if frees_xyz.size > 0:
            xs = frees_xyz.reshape(-1,3)                        # (R*S,3)
            wm = frees_mask.reshape(-1).astype(jnp.float32)     # (R*S,)
            g_free = v_G(xs, params)
            l_free = (wm * (g_free - mu)**2).sum() / (wm.sum() + 1e-6)

        return l_hit + 0.5*l_free + 0.1*eik

    g = jax.grad(loss_fn)(theta)
    updates, opt2 = _G_OPT_TX.update(g, opt, theta)
    theta2 = optax.apply_updates(theta, updates)
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
    updates, opt2 = _E_OPT_TX.update(g, opt, eta)
    eta2 = optax.apply_updates(eta, updates)
    return MapState(mapstate.geom, ExpoState(eta2, opt2))
