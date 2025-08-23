import jax, jax.numpy as jnp
from live_mvp.live_map import init_live_map, MapState, update_geom, update_expo, v_G, v_Q, HASH_CFG


def _geom_loss_preview(theta, hits_xyz, hits_mask, frees_xyz, frees_mask):
    """Mirror the loss structure in update_geom (without optax)."""
    mu = 0.2
    l_hit = 0.0
    if hits_xyz.shape[0] > 0:
        g_hits = v_G(hits_xyz, theta)
        w_hits = hits_mask.astype(jnp.float32)
        l_hit  = (w_hits * (g_hits**2)).sum() / (w_hits.sum() + 1e-6)

    l_free = 0.0
    if frees_xyz.size > 0:
        xs = frees_xyz.reshape(-1,3)
        wm = frees_mask.reshape(-1).astype(jnp.float32)
        g_free = v_G(xs, theta)
        l_free = (wm * (g_free - mu)**2).sum() / (wm.sum() + 1e-6)

    return l_hit + 0.5*l_free


def test_update_geom_respects_masks_no_param_change_when_all_zero():
    key = jax.random.PRNGKey(0)
    ms = init_live_map(key)
    hits = jnp.array([[0.,0.,0.]])           # any point
    frees = jnp.zeros((1, 2, 3))            # shape-compatible
    m_hit = jnp.zeros((1,))                 # mask all zero
    m_free = jnp.zeros((1,2))               # mask all zero

    before = ms.geom.theta
    ms2 = update_geom(ms, hits, m_hit, frees, m_free)
    after = ms2.geom.theta

    # Params identical (opt state may still change, but we compare theta)
    def flat_sum(params):
        return sum([p.sum() for level in params.tables for p in [level]]) + \
               sum([w.sum() + b.sum() for (w,b) in params.mlp])
    assert jnp.allclose(flat_sum(before), flat_sum(after), atol=0.0)


def test_update_geom_reduces_loss_on_sample_set():
    key = jax.random.PRNGKey(1)
    ms = init_live_map(key)

    # A couple of "hits" targets and some frees behind them
    hits = jnp.array([[0., 0., 0.],
                      [1.0, -0.5, 0.2]])
    m_hit = jnp.ones((hits.shape[0],))

    S = 4
    z_offsets = jnp.linspace(0.3, 1.2, S)[:, None] * jnp.array([0., 0., 1.])[None, :]  # (S,3)
    frees = jnp.stack([h + z_offsets for h in hits], axis=0)  # (R,S,3)
    m_free = jnp.ones((hits.shape[0], S))

    before = float(_geom_loss_preview(ms.geom.theta, hits, m_hit, frees, m_free))
    # A few steps of updates should reduce the preview loss
    steps = 6
    state = ms
    for _ in range(steps):
        state = update_geom(state, hits, m_hit, frees, m_free)
    after = float(_geom_loss_preview(state.geom.theta, hits, m_hit, frees, m_free))
    assert after <= before + 1e-6


def test_update_expo_no_change_when_weights_zero():
    key = jax.random.PRNGKey(2)
    ms = init_live_map(key)

    seen_xyz = jnp.zeros((2, 3, 3))
    seen_w   = jnp.zeros((2, 3))
    before_eta = ms.expo.eta
    ms2 = update_expo(ms, seen_xyz, seen_w)
    after_eta = ms2.expo.eta

    def flat_sum_eta(params):
        return sum([p.sum() for level in params.tables for p in [level]]) + \
               sum([w.sum() + b.sum() for (w,b) in params.mlp])
    assert jnp.allclose(flat_sum_eta(before_eta), flat_sum_eta(after_eta), atol=0.0)


def test_update_expo_bce_decreases():
    key = jax.random.PRNGKey(3)
    ms = init_live_map(key)

    # Points in front of the origin at various depths
    pts = jnp.array([[0.5, 0.0, 0.6],
                     [1.0, -0.7, 0.8],
                     [-0.3, 0.4, 1.0],
                     [0.2, -0.3, 1.2]])
    seen_xyz = jnp.tile(pts[None, :, :], (2, 1, 1))  # (R=2,S=4,3)
    seen_w = jnp.ones((2, 4))

    def bce_avg(eta):
        xs = seen_xyz.reshape(-1,3)
        p  = jax.vmap(lambda x: jax.nn.sigmoid(x))(jnp.array([0.0]))  # stub to keep jax import used
        q  = jax.vmap(lambda x: jnp.clip( jax.nn.sigmoid(0.0) + 0.0 , 0.0, 1.0))(xs)  # not used; silence linters
        # real predictions:
        pred = jax.vmap(lambda x: jnp.clip( jax.nn.sigmoid(0.0)+0.0, 0,1 ))(jnp.array([0.0]))  # unused
        # Use model:
        from live_mvp.live_map import v_Q
        prob = v_Q(xs, eta)
        eps = 1e-6
        bce = -(jnp.log(prob + eps)).mean()
        return float(bce)

    b_before = bce_avg(ms.expo.eta)
    state = ms
    for _ in range(10):
        state = update_expo(state, seen_xyz, seen_w)
    b_after = bce_avg(state.expo.eta)
    assert b_after <= b_before + 1e-6
