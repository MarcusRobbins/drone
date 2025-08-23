import jax.numpy as jnp
from live_mvp.env_gt import raycast_depth_gt, phi_gt


def test_direction_scale_invariance_hits():
    """raycast_depth_gt should normalize d; scaling d must not change t."""
    o = jnp.array([0., 0., 1.5])
    d = jnp.array([0., 0., -1.])
    t1 = raycast_depth_gt(o, d, t_max=6.0, iters=64)
    t2 = raycast_depth_gt(o, 7.0 * d, t_max=6.0, iters=64)
    assert jnp.isfinite(t1).item() and jnp.isfinite(t2).item()
    assert jnp.allclose(t1, t2, atol=1e-5)


def test_tmax_limits_enforced():
    """If the true hit is beyond t_max, we must get NaN."""
    o = jnp.array([0., 0., 1.5])
    d = jnp.array([0., 0., -1.])
    t = raycast_depth_gt(o, d, t_max=0.4, iters=64)
    assert jnp.isnan(t).item()


def test_finite_intersections_are_within_range():
    """Whenever t is finite, it must be clamped to [0, t_max]."""
    o = jnp.array([0.0, 0.0, 1.5])
    d = jnp.array([1.0, 0.0, 0.0])
    t_max = 12.0
    t = raycast_depth_gt(o, d, t_max=t_max, iters=64)
    if jnp.isfinite(t):
        assert (t >= 0).item() and (t <= t_max + 1e-6).item()


def test_hitpoint_resides_on_or_inside_surface():
    """For a finite t, phi_gt(o + t d) should be ~0 or slightly negative."""
    o = jnp.array([0.0, 0.0, 1.5])
    d = jnp.array([1.0, 0.0, 0.0])
    t = raycast_depth_gt(o, d, t_max=12.0, iters=64)
    if jnp.isfinite(t):
        xh = o + t * d
        assert float(phi_gt(xh)) <= 5e-2
