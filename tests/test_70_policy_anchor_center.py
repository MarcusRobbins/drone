import jax, jax.numpy as jnp
from live_mvp.policy import anchor_features, ANCHORS_BODY, ANCHOR_FEAT_DIM
from live_mvp.live_map import init_live_map, G_phi, Q_expo


def test_anchor_features_shape_and_center_matches_model():
    key = jax.random.PRNGKey(0)
    ms = init_live_map(key)
    theta = ms.geom.theta
    eta = ms.expo.eta

    p = jnp.array([0.2, -0.1, 1.3])
    q = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity

    feats = anchor_features(p, q, theta, eta)
    assert feats.shape == (ANCHOR_FEAT_DIM,)

    # The last anchor is the center (include_center=True in ANCHORS_BODY)
    # anchor_features packs [phi_clip, E] per anchor, flattened.
    N = ANCHORS_BODY.shape[0]
    phi_clip_center = feats[2*(N-1) + 0]
    E_center        = feats[2*(N-1) + 1]

    # Ground truth from model:
    phi_raw = G_phi(p, theta)
    phi_clip_gt = jnp.clip(phi_raw, -0.5, 0.5) / 0.5
    E_gt = Q_expo(p, eta)

    assert jnp.allclose(phi_clip_center, phi_clip_gt, atol=1e-6).item()
    assert jnp.allclose(E_center,        E_gt,        atol=1e-6).item()
