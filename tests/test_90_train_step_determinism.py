import jax, jax.numpy as jnp
from live_mvp.train_live import train_step, SimCfg, POLICY_OPT
from live_mvp.live_map import init_live_map
from live_mvp.dyn import State
from live_mvp.policy import init_mlp, ANCHOR_FEAT_DIM, COMPASS_M


def _mk_inputs():
    key = jax.random.PRNGKey(0)
    mapstate = init_live_map(key)
    states0 = State(
        p=jnp.array([[0., 0., 1.6], [0., 0.9, 1.6]]),
        v=jnp.zeros((2, 3)),
        q=jnp.tile(jnp.array([1., 0., 0., 0.]), (2, 1)),
        w=jnp.zeros((2, 3)),
    )
    obs_dim = 3 + 3 + 3 + ANCHOR_FEAT_DIM + COMPASS_M + 3
    params = init_mlp(jax.random.PRNGKey(1), [obs_dim, 32, 32, 6])
    opt_state = POLICY_OPT.init(params)
    sim = SimCfg(steps=3)  # keep small to avoid JIT on CPU
    return key, mapstate, states0, params, opt_state, sim


def test_train_step_deterministic_given_same_inputs():
    key, map1, st1, pp1, opt1, sim = _mk_inputs()
    out1 = train_step(pp1, opt1, map1, st1, key, sim)

    key2, map2, st2, pp2, opt2, sim2 = _mk_inputs()  # fresh but identical
    out2 = train_step(pp2, opt2, map2, st2, key2, sim2)

    # Compare loss and metrics exactly/closely
    _, _, _, loss1, metrics1 = out1
    _, _, _, loss2, metrics2 = out2

    assert jnp.allclose(loss1, loss2, atol=1e-8).item()
    for k in metrics1.keys():
        assert jnp.allclose(metrics1[k], metrics2[k], atol=1e-8).item()
