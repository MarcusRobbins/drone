import jax, jax.numpy as jnp

def smin(a, b, k=8.0):
    return -jnp.log(jnp.exp(-k*a) + jnp.exp(-k*b) + 1e-12) / k

def sd_box(x, h):
    q = jnp.abs(x) - h
    # robust for both (3,) and (...,3)
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside  = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside

def phi_gt(x):
    # Ground plane z=0 (positive above)
    phi_plane = x[2]
    # Sphere
    c_sph = jnp.array([3.0, 0.0, 1.0]); r_sph = 1.0
    phi_sphere = jnp.linalg.norm(x - c_sph) - r_sph
    # Box
    c_box = jnp.array([1.6, -1.4, 0.7]); he = jnp.array([0.6, 0.6, 0.8])
    phi_box = sd_box(x - c_box, he)
    return smin(smin(phi_plane, phi_sphere), phi_box)

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
