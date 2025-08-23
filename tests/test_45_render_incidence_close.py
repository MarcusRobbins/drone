import jax.numpy as jnp
from numpy.testing import assert_allclose
from live_mvp.render import incidence_weight, close_weight, RenderCfg, recon_reward_for_ray
from live_mvp.live_map import init_live_map


def _cos(deg):
    return jnp.cos(jnp.deg2rad(deg))


def test_incidence_weight_window_prefers_mid_angles():
    rcfg = RenderCfg(theta_min_deg=20.0, theta_max_deg=60.0, tau_theta=0.08)
    # Construct cos_inc values
    w0   = float(incidence_weight(_cos(0.0),  rcfg))   # head-on (outside window)
    w40  = float(incidence_weight(_cos(40.0), rcfg))   # inside window
    w80  = float(incidence_weight(_cos(80.0), rcfg))   # very grazing (outside window)
    assert w40 > w0 + 1e-4
    assert w40 > w80 + 1e-4


def test_close_weight_monotone():
    rcfg = RenderCfg(r0=2.5, tau_r=0.8)
    w_near = float(close_weight(0.5, rcfg))
    w_far  = float(close_weight(5.0, rcfg))
    assert w_near > w_far + 1e-4


def test_recon_reward_direction_scale_invariance():
    """recon_reward_for_ray normalizes d; scaling should not change the value."""
    ms = init_live_map(jnp.array([0, 1], dtype=jnp.uint32))  # cheap key stub
    theta = ms.geom.theta
    eta = ms.expo.eta
    from live_mvp.render import RenderCfg
    rcfg = RenderCfg(S=24)
    o = jnp.array([0., 0., 1.2])
    d = jnp.array([1., 0., -0.2])
    r1 = recon_reward_for_ray(o, d, theta, eta, rcfg)
    r2 = recon_reward_for_ray(o, 3.0 * d, theta, eta, rcfg)
    assert_allclose(r1, r2, atol=1e-5)
