
from typing import NamedTuple
import jax, jax.numpy as jnp

class State(NamedTuple): 
    p: jnp.ndarray; v: jnp.ndarray; q: jnp.ndarray; w: jnp.ndarray

class DynCfg(NamedTuple):
    dt: float = 0.05
    drag_v: float = 0.05
    drag_w: float = 0.02
    a_max: float = 3.5
    w_max: float = 1.2

def quat_mul(q, r):
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = r
    return jnp.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                      w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2,
                      w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_normalize(q): 
    return q/(jnp.linalg.norm(q)+1e-9)

def R_from_q(q):
    w,x,y,z = q
    xx,yy,zz = x*x,y*y,z*z; wx,wy,wz = w*x,w*y,w*z; xy,xz,yz = x*y,x*z,y*z
    return jnp.array([[1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
                      [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
                      [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]])

def clamp_u(u, cfg: DynCfg):
    a = jnp.clip(u[:3], -cfg.a_max, cfg.a_max)
    w = jnp.clip(u[3:], -cfg.w_max, cfg.w_max)
    return jnp.concatenate([a,w])

def step(st: State, u_raw: jnp.ndarray, cfg: DynCfg) -> State:
    u = clamp_u(u_raw, cfg)
    a, w_cmd = u[:3], u[3:]
    v = st.v + cfg.dt * (a - cfg.drag_v*st.v)
    p = st.p + cfg.dt * v
    dq = 0.5 * quat_mul(st.q, jnp.array([0., *w_cmd]))
    q = quat_normalize(st.q + cfg.dt * dq)
    w = (1.0 - cfg.drag_w*cfg.dt) * w_cmd
    return State(p,v,q,w)

def body_rays_world(q, rays_body):
    return (R_from_q(q) @ rays_body.T).T
