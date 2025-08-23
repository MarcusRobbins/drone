import jax, jax.numpy as jnp
from live_mvp.policy import (
    anchor_grid_body, ANCHORS_BODY, ANCHOR_FEAT_DIM,
    compass_rays_body, COMPASS_BODY, unseen_compass_features
)
from live_mvp.live_map import init_live_map
from live_mvp.dyn import R_from_q


def test_anchor_grid_shapes_and_radii():
    K, H, R = 8, 3, 2.5
    anchors = anchor_grid_body(K=K, R=R, H=H, z_min=-1.0, z_max=1.0, include_center=True)
    # count = H*K + 1
    assert anchors.shape == (H * K + 1, 3)
    # ring points have ~radius R at each z (excluding center)
    ring = anchors[:-1]
    radii = jnp.linalg.norm(ring[:, :2], axis=1)
    assert jnp.allclose(radii, R, atol=1e-6)


def test_compass_rays_normalized_and_counts():
    n_az, n_el = 16, 4
    D = compass_rays_body(n_az=n_az, n_el=n_el)
    assert D.shape == (n_az * n_el, 3)
    norms = jnp.linalg.norm(D, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_unseen_compass_shapes_and_nonneg():
    key = jax.random.PRNGKey(0)
    ms = init_live_map(key)
    theta = ms.geom.theta
    eta = ms.expo.eta
    p = jnp.array([0.0, 0.0, 1.5])
    q = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity
    # sanity: body->world conversion uses R_from_q; COMPASS_BODY is normalized
    R = R_from_q(q)
    assert jnp.allclose(R @ jnp.eye(3), jnp.eye(3), atol=1e-6)

    from live_mvp.render import RenderCfg
    pot, vec = unseen_compass_features(p, q, theta, eta, RenderCfg(S=24))
    # shapes
    assert pot.shape == (COMPASS_BODY.shape[0],)
    assert vec.shape == (3,)
    # non-negative potentials and finite
    assert (pot >= 0).all().item()
    assert jnp.isfinite(pot).all().item()
    assert jnp.isfinite(vec).all().item()
