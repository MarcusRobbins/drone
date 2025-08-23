import jax.numpy as jnp
from numpy.testing import assert_allclose
from live_mvp.env_gt import smin, sd_box, phi_gt, raycast_depth_gt


def test_smin_bounds():
    """
    Smooth-min must lie between min(a,b) - ln(2)/k and min(a,b).
    """
    k = 8.0
    pairs = [
        (1.0, 2.0),
        (2.0, 1.0),
        (0.0, -1.0),
        (-2.0, -3.0),
        (10.0, -10.0),
    ]
    for a, b in pairs:
        sm = float(smin(jnp.array(a), jnp.array(b), k))
        mn = min(a, b)
        lower = mn - jnp.log(2.0) / k
        assert sm <= mn + 1e-6
        assert sm >= float(lower) - 1e-6


def test_sd_box_known_points():
    """
    Check exact distances for a few easy box cases.
    """
    h = jnp.array([1.0, 2.0, 3.0])
    # Inside at origin -> -min half-extent (distance to nearest face)
    d0 = float(sd_box(jnp.array([0.0, 0.0, 0.0]), h))
    assert_allclose(d0, -1.0, atol=1e-6)

    # Outside along +x at (2,0,0) with h=(1,1,1) should be +1
    h1 = jnp.array([1.0, 1.0, 1.0])
    d1 = float(sd_box(jnp.array([2.0, 0.0, 0.0]), h1))
    assert_allclose(d1, 1.0, atol=1e-6)

    # On the surface along +z at (0,0,3) with h=(1,2,3) should be 0
    d2 = float(sd_box(jnp.array([0.0, 0.0, 3.0]), h))
    assert_allclose(d2, 0.0, atol=1e-6)


def test_phi_gt_sphere_center_is_minus_radius():
    """
    Scene contains a sphere of radius 1 at (3,0,1).
    Distance at the center should be -1 (inside by radius).
    """
    x = jnp.array([3.0, 0.0, 1.0])
    d = float(phi_gt(x))
    assert_allclose(d, -1.0, atol=1e-6)


def test_raycast_hits_sphere_along_centerline():
    """
    Aim directly from origin-line toward the sphere's center: should hit the sphere,
    and the hit point should be on/inside the surface (phi <= small tol).
    """
    o = jnp.array([0.0, 0.0, 1.5])
    dir_to_center = jnp.array([3.0, 0.0, 1.0]) - o
    d = dir_to_center / (jnp.linalg.norm(dir_to_center) + 1e-9)
    t = raycast_depth_gt(o, d, t_max=12.0, iters=64)
    assert jnp.isfinite(t).item()
    xh = o + t * d
    assert float(phi_gt(xh)) <= 5e-2  # allow small sphere-tracing error


def test_raycast_upward_miss_nan():
    o = jnp.array([0.0, 0.0, 1.5]); d = jnp.array([0.0, 0.0, 1.0])
    t = raycast_depth_gt(o, d, t_max=12.0, iters=64)
    assert jnp.isnan(t).item()
