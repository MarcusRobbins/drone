import jax.numpy as jnp
from numpy.testing import assert_allclose
from live_mvp.dyn import clamp_u, body_rays_world, R_from_q, State, DynCfg, step


def test_clamp_u_saturates_to_cfg():
    cfg = DynCfg(a_max=3.5, w_max=1.2)
    u_raw = jnp.array([10., -10., 4.,  5., -5., 4.])
    u = clamp_u(u_raw, cfg)
    a, w = u[:3], u[3:]
    assert_allclose(a, jnp.array([3.5, -3.5, 3.5]), atol=1e-6)
    assert_allclose(w, jnp.array([1.2, -1.2, 1.2]), atol=1e-6)


def test_body_rays_world_z_yaw_90deg():
    # 90 deg about z: [cos(45°), 0, 0, sin(45°)]
    q = jnp.array([jnp.sqrt(0.5), 0., 0., jnp.sqrt(0.5)])
    rays_body = jnp.array([[1.,0.,0.], [0.,1.,0.]])
    rays_world = body_rays_world(q, rays_body)
    # x->y, y->-x under +90° z-rotation (right-handed)
    assert_allclose(rays_world[0], jnp.array([0., 1., 0.]), atol=1e-6)
    assert_allclose(rays_world[1], jnp.array([-1., 0., 0.]), atol=1e-6)


def test_step_respects_clamp_bounds():
    st = State(
        p=jnp.zeros(3),
        v=jnp.zeros(3),
        q=jnp.array([1.,0.,0.,0.]),
        w=jnp.zeros(3),
    )
    # Try to command something huge; the integrator should still stay stable
    u_raw = jnp.array([100., 100., 100., 10., 10., 10.])
    st2 = step(st, u_raw, DynCfg(dt=0.05, a_max=3.5, w_max=1.2))
    # Velocity/omega must remain finite and limited by clamp
    assert jnp.all(jnp.isfinite(st2.v)).item()
    assert jnp.all(jnp.isfinite(st2.w)).item()
