
from typing import NamedTuple
import jax, jax.numpy as jnp
from .live_map import G_phi, Q_expo

class RenderCfg(NamedTuple):
    t0: float = 0.2
    t1: float = 12.0
    S:  int   = 64      # samples per ray
    eps_shell: float = 0.18
    r0: float = 2.5
    tau_r: float = 0.8
    theta_min_deg: float = 20.0
    theta_max_deg: float = 60.0
    tau_theta: float = 0.08
    unseen_gamma: float = 1.3

def sdf_to_sigma(phi, eps): 
    return jax.nn.softplus(-phi/eps)

def expected_hit_live(o, d, theta, rcfg: RenderCfg):
    ts = jnp.linspace(rcfg.t0, rcfg.t1, rcfg.S)
    dt = (rcfg.t1-rcfg.t0)/(rcfg.S-1+1e-9)
    xs = o[None,:] + ts[:,None]*d[None,:]
    phi = jax.vmap(G_phi, in_axes=(0,None))(xs, theta)
    sigma = sdf_to_sigma(phi, rcfg.eps_shell)
    alpha = 1.0 - jnp.exp(-sigma*dt)
    T = jnp.cumprod(jnp.concatenate([jnp.ones((1,)), 1.0-alpha[:-1]]))
    w = (T*alpha); w = w/(w.sum()+1e-9)
    xh = (w[:,None]*xs).sum(0)
    seen = (T*alpha).sum()
    return xh, seen, xs, T, alpha

def incidence_weight(cos_inc, rcfg):
    cmin = jnp.cos(jnp.deg2rad(rcfg.theta_max_deg))
    cmax = jnp.cos(jnp.deg2rad(rcfg.theta_min_deg))
    w_lo = jax.nn.sigmoid((cos_inc - cmin)/(rcfg.tau_theta+1e-9))
    w_hi = jax.nn.sigmoid((cmax - cos_inc)/(rcfg.tau_theta+1e-9))
    return w_lo * w_hi

def close_weight(r, rcfg): 
    return jax.nn.sigmoid((rcfg.r0 - r)/(rcfg.tau_r+1e-9))

def recon_reward_for_ray(o, d, theta, eta, rcfg: RenderCfg):
    d = d/(jnp.linalg.norm(d)+1e-9)
    xh, seen, xs, T, alpha = expected_hit_live(o, d, theta, rcfg)
    # normals from live SDF (stop grad through the normal to avoid brittle 2nd-order paths)
    n = jax.grad(G_phi)(xh, theta)
    n_hat = n/(jnp.linalg.norm(n)+1e-9)
    n_hat = jax.lax.stop_gradient(n_hat)
    v = -d
    cos_inc = jnp.abs(jnp.dot(n_hat, v))
    dist = jnp.linalg.norm(xh - o)
    # small floors keep signal alive early
    w_inc = 0.2 + 0.8 * incidence_weight(cos_inc, rcfg)
    w_close = 0.2 + 0.8 * close_weight(dist, rcfg)
    # unseen from live exposure
    E = Q_expo(xh, eta)
    w_unseen = (1.0 - jax.lax.stop_gradient(E)) ** rcfg.unseen_gamma
    return seen * w_inc * w_close * w_unseen
