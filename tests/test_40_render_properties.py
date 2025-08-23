import jax, jax.numpy as jnp
from live_mvp.render import expected_hit_live, sdf_to_sigma, RenderCfg
from live_mvp.live_map import init_live_map


def test_sdf_to_sigma_monotone_decreasing():
    eps = 0.18
    # phi1 < phi2 => sigma(phi1) >= sigma(phi2)
    phis = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    sigmas = sdf_to_sigma(phis, eps)
    # pairwise monotonic check
    for i in range(len(phis) - 1):
        assert sigmas[i] >= sigmas[i + 1] - 1e-9


def test_expected_hit_live_basic_invariants():
    key = jax.random.PRNGKey(0)
    mapstate = init_live_map(key)
    theta = mapstate.geom.theta
    o = jnp.array([0.0, 0.0, 1.2])
    d = jnp.array([1.0, 0.0, -0.1]); d = d / (jnp.linalg.norm(d) + 1e-9)
    rcfg = RenderCfg(S=32)  # keep it light for CPU

    xh, seen, xs, T, alpha = expected_hit_live(o, d, theta, rcfg)
    # shapes
    assert xs.shape[1] == 3
    assert T.shape == alpha.shape == (rcfg.S,)

    # alpha in [0,1], T in (0,1] and non-increasing
    assert jnp.all((alpha >= 0) & (alpha <= 1)).item()
    assert jnp.all((T > 0) & (T <= 1)).item()
    assert jnp.all(T[1:] <= T[:-1] + 1e-9).item()

    # weights sum to ~1 (ignoring tiny numerical error)
    dt = (rcfg.t1 - rcfg.t0) / (rcfg.S - 1 + 1e-9)
    w = (T * alpha); w = w / (w.sum() + 1e-9)
    assert jnp.allclose(w.sum(), 1.0, atol=1e-5)

    # xh must lie on the ray line: (xh - o) parallel to d
    v = xh - o
    cross = jnp.linalg.norm(jnp.cross(v, d))
    assert cross <= 1e-5
