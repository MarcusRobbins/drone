
import jax, jax.numpy as jnp
from typing import NamedTuple
from .live_map import G_phi, Q_expo
from .dyn import R_from_q, body_rays_world
from .render import sdf_to_sigma, RenderCfg

# --------
# Utility MLP for the policy
# --------
def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes)-1)
    params=[]; prev=sizes[0]
    for k, n in zip(keys, sizes[1:]):
        W = jax.random.normal(k, (prev, n)) * (1.0/jnp.sqrt(prev))
        b = jnp.zeros((n,)); params.append((W,b)); prev=n
    return tuple(params)

def mlp_apply(params, x):
    for W,b in params[:-1]: 
        x = jax.nn.tanh(x @ W + b)
    W,b = params[-1]; 
    return x @ W + b

# --------
# (A) Non-learned 3D Anchor-Lattice features
# --------
def anchor_grid_body(K=12, R=3.0, H=5, z_min=-1.0, z_max=1.0, include_center=True):
    """Body-frame anchor offsets: K points on a ring at radius R,
    replicated at H evenly spaced z-levels in [z_min, z_max], plus optional center."""
    az = jnp.linspace(0.0, 2*jnp.pi, K, endpoint=False)
    ring = jnp.stack([R*jnp.cos(az), R*jnp.sin(az), jnp.zeros_like(az)], axis=-1)  # (K,3)
    zs = jnp.linspace(z_min, z_max, H)
    def add_z(z): return ring + jnp.array([0.0, 0.0, z])
    anchors = jax.vmap(add_z)(zs).reshape(-1, 3)  # (H*K,3)
    if include_center:
        anchors = jnp.concatenate([anchors, jnp.array([[0.0, 0.0, 0.0]])], axis=0)
    return anchors  # (N,3)

ANCHORS_BODY = anchor_grid_body(K=12, R=3.0, H=5, z_min=-1.0, z_max=1.0, include_center=True)
ANCHOR_FEAT_DIM = ANCHORS_BODY.shape[0] * 2  # channels: [phi_clip, exposure]

def anchor_features(p, q, theta, eta):
    """Non-learned 3D features: for each anchor point around the drone, return:
      - clipped & normalized SDF ([-1,1])
      - exposure Q in [0,1]
    Output shape: (ANCHOR_FEAT_DIM,)"""
    Rw = R_from_q(q)
    anchors_world = (ANCHORS_BODY @ Rw.T) + p  # (N,3)
    phi = jax.vmap(G_phi, in_axes=(0,None))(anchors_world, theta)   # (N,)
    E   = jax.vmap(Q_expo, in_axes=(0,None))(anchors_world, eta)    # (N,)
    phi_clip = jnp.clip(phi, -0.5, 0.5) / 0.5  # [-1,1] for stability
    return jnp.stack([phi_clip, E], axis=1).reshape(-1)             # (2N,)

# --------
# (B) Non-learned "Unseen Compass"
# --------
def compass_rays_body(n_az=24, n_el=3, fov_az_deg=360., el_min_deg=-25., el_max_deg=25.):
    az = jnp.linspace(-jnp.deg2rad(fov_az_deg)/2, jnp.deg2rad(fov_az_deg)/2, n_az, endpoint=False)
    el = jnp.deg2rad(jnp.linspace(el_min_deg, el_max_deg, n_el))
    A,E = jnp.meshgrid(az, el, indexing='xy')
    x = jnp.cos(E)*jnp.cos(A); y = jnp.cos(E)*jnp.sin(A); z = jnp.sin(E)
    D = jnp.stack([x,y,z], axis=-1)
    D = D/(jnp.linalg.norm(D, axis=-1, keepdims=True)+1e-9)
    return D.reshape(-1,3)  # (M,3)

COMPASS_BODY = compass_rays_body(n_az=24, n_el=3)
COMPASS_M = COMPASS_BODY.shape[0]

def unseen_potential_ray(o, d, theta, eta, rcfg: RenderCfg, S=32):
    """Non-learned info-gain-ish integral along ray using live map only."""
    ts = jnp.linspace(rcfg.t0, rcfg.t1, S)
    dt = (rcfg.t1 - rcfg.t0) / (S - 1 + 1e-9)
    xs = o[None,:] + ts[:,None]*d[None,:]                  # (S,3)

    phi   = jax.vmap(G_phi, in_axes=(0,None))(xs, theta)   # (S,)
    sigma = sdf_to_sigma(phi, rcfg.eps_shell)              # shell around surfaces
    alpha = 1.0 - jnp.exp(-sigma * dt)                     # absorption
    T     = jnp.cumprod(jnp.concatenate([jnp.ones((1,)), 1.0 - alpha[:-1]]))
    E     = jax.vmap(Q_expo, in_axes=(0,None))(xs, eta)    # exposure in [0,1]
    unseen = 1.0 - jax.lax.stop_gradient(E)                # don't backprop into map
    w_dist = 1.0 / (1.0 + ts*ts)                           # prefer nearer opportunities
    w = T * alpha * unseen * w_dist
    return w.sum()                                         # scalar potential per ray

def unseen_compass_features(p, q, theta, eta, rcfg: RenderCfg):
    """Returns per-direction potentials (M,) and a 3D direction summary (vec)."""
    rays_w = body_rays_world(q, COMPASS_BODY)              # (M,3)
    o = p
    pot = jax.vmap(lambda d: unseen_potential_ray(o, d, theta, eta, rcfg))(rays_w)  # (M,)
    vec = (pot[:,None] * rays_w).sum(0) / (pot.sum() + 1e-9)                         # (3,)
    return pot, vec  # (M,), (3,)
