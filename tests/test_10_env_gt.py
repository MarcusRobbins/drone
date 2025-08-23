import jax.numpy as jnp
from live_mvp.env_gt import raycast_depth_gt, phi_gt


def test_downward_hits_scene_surface_or_inside():
    """
    Full-scene raycast should produce a finite depth for a downward ray.
    We assert the hitpoint is on/inside some surface (phi <= small tol),
    not an exact numeric plane distance.
    """
    o = jnp.array([0., 0., 1.5]); d = jnp.array([0., 0., -1.])
    t = raycast_depth_gt(o, d, t_max=5.0, iters=64)
    assert jnp.isfinite(t)
    x = o + t * d
    assert float(phi_gt(x)) <= 1e-2  # on/inside any surface


def test_parallel_along_x_hits_object():
    """
    A ray parallel to the ground along +x from z=1.5 should hit the sphere/box.
    Old test expected NaN; for the full scene this should be a finite hit.
    """
    o = jnp.array([0., 0., 1.5]); d = jnp.array([1., 0., 0.])
    t = raycast_depth_gt(o, d, t_max=12.0, iters=64)
    assert jnp.isfinite(t)
    x = o + t * d
    assert float(phi_gt(x)) <= 5e-2  # on/inside object surface


def test_upward_miss_returns_nan():
    o = jnp.array([0., 0., 1.5]); d = jnp.array([0., 0., 1.])
    t = raycast_depth_gt(o, d, t_max=12.0, iters=64)
    assert jnp.isnan(t)
