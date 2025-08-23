import jax, jax.numpy as jnp
from live_mvp.train_live import train_step, SimCfg, POLICY_OPT
from live_mvp.live_map import init_live_map
from live_mvp.dyn import State
from live_mvp.policy import init_mlp, ANCHOR_FEAT_DIM, COMPASS_M

def _pick_first_device():
    for backend in ("cuda", "rocm"):
        try:
            devs = jax.devices(backend)
            if devs:
                return devs[0]
        except RuntimeError:
            # backend not present
            pass
    return jax.devices()[0]  # CPU fallback


def test_train_step_smoke():
    dev = _pick_first_device()
    with jax.default_device(dev):
        key = jax.random.PRNGKey(0)
        mapstate = init_live_map(key)
        states0 = State(
            p=jnp.array([[0.,0.,1.6],[0.,0.9,1.6]]),
            v=jnp.zeros((2,3)),
            q=jnp.tile(jnp.array([1.,0.,0.,0.]), (2,1)),
            w=jnp.zeros((2,3)),
        )
        obs_dim = 3 + 3 + 3 + ANCHOR_FEAT_DIM + COMPASS_M + 3
        policy_params = init_mlp(jax.random.PRNGKey(1), [obs_dim, 32, 32, 6])
        opt_state = POLICY_OPT.init(policy_params)
        sim = SimCfg(steps=5)
        policy_params, opt_state, mapstate, loss, metrics = train_step(
            policy_params, opt_state, mapstate, states0, key, sim
        )
        assert jnp.isfinite(loss)
        assert all(jnp.isfinite(jnp.array(list(metrics.values()))))
