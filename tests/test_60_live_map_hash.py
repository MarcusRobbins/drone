import jax, jax.numpy as jnp
from live_mvp.live_map import (
    HASH_CFG, init_hash_tables, _level_res, _hash_ijk, _encode_point,
    init_live_map, G_phi, Q_expo
)


def test_level_res_monotone():
    Ns = [int(_level_res(l, HASH_CFG)) for l in range(HASH_CFG.L)]
    assert all(Ns[i] <= Ns[i + 1] for i in range(len(Ns) - 1))


def test_hash_range_and_encode_shapes():
    key = jax.random.PRNGKey(0)
    tables = init_hash_tables(key, HASH_CFG)
    assert len(tables) == HASH_CFG.L
    for t in tables:
        assert t.shape == (HASH_CFG.T, HASH_CFG.F)

    # hash range
    idx = _hash_ijk(jnp.array([10, 20, 30], dtype=jnp.int32), HASH_CFG.T)
    assert int(idx) >= 0 and int(idx) < HASH_CFG.T

    # encode one point (in-AABB)
    x = jnp.array([0.0, 0.0, 0.0])
    z = _encode_point(tables, x, HASH_CFG)
    assert z.shape == (HASH_CFG.L * HASH_CFG.F,)

    # clamping at borders (below lb and above ub)
    x_lo = HASH_CFG.lb - 10.0
    x_hi = HASH_CFG.ub + 10.0
    z_lo = _encode_point(tables, x_lo, HASH_CFG)
    z_hi = _encode_point(tables, x_hi, HASH_CFG)
    assert jnp.isfinite(z_lo).all().item()
    assert jnp.isfinite(z_hi).all().item()


def test_G_phi_and_Q_range_and_grad():
    key = jax.random.PRNGKey(0)
    ms = init_live_map(key)
    theta = ms.geom.theta
    eta = ms.expo.eta
    x = jnp.array([0.5, -0.4, 0.8])
    g = G_phi(x, theta)
    q = Q_expo(x, eta)
    assert jnp.isfinite(g).item()
    assert (q >= 0).item() and (q <= 1).item()

    # gradient exists and finite
    grad_g = jax.grad(lambda xx: G_phi(xx, theta))(x)
    assert grad_g.shape == (3,)
    assert jnp.isfinite(grad_g).all().item()
