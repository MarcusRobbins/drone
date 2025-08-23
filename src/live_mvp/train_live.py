from functools import partial
from typing import NamedTuple
import jax, jax.numpy as jnp, optax

from .env_gt import raycast_depth_gt
from .live_map import init_live_map, update_geom, update_expo, MapState
from .render import RenderCfg, recon_reward_for_ray
from .dyn import State, DynCfg, step, body_rays_world, R_from_q
from .policy import init_mlp, mlp_apply, anchor_features, ANCHOR_FEAT_DIM
from .policy import unseen_compass_features, COMPASS_M


def ray_dirs_body(n_az=12, n_el=3, fov_az=100., fov_el=40.):
    az = jnp.linspace(-jnp.deg2rad(fov_az)/2, jnp.deg2rad(fov_az)/2, n_az)
    el = jnp.linspace(-jnp.deg2rad(fov_el)/2, jnp.deg2rad(fov_el)/2, n_el)
    A, E = jnp.meshgrid(az, el, indexing='xy')
    x = jnp.cos(E) * jnp.cos(A); y = jnp.cos(E) * jnp.sin(A); z = jnp.sin(E)
    D = jnp.stack([x, y, z], axis=-1)
    D = D / (jnp.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)
    return D.reshape(-1, 3)


RAYS_BODY = ray_dirs_body()


class SimCfg(NamedTuple):
    steps: int = 80
    rcfg: RenderCfg = RenderCfg()
    dyn: DynCfg = DynCfg()
    w_coll: float = 4.0
    margin: float = 0.12
    w_ctrl: float = 0.01


def soft_collision_live(p, theta, margin):
    from .live_map import G_phi
    return jax.nn.softplus(margin - G_phi(p, theta))


def _rollout_loss_impl(policy_params, mapstate: MapState, states0, key, sim: SimCfg):
    """One environment, N drones, train policy; update map online each step."""
    N = states0.p.shape[0]

    def one_step(carry, _):
        key, states, mapstate = carry

        # === 1) Online mapping: simulate depth on GT, update live map ===
        def update_from_drone(i, ms):
            p = states.p[i]; q = states.q[i]
            rays_w = body_rays_world(q, RAYS_BODY)
            idx = jnp.arange(0, RAYS_BODY.shape[0], 4)           # subsample for speed
            sel = rays_w[idx]

            # Fixed sample positions along each ray; mask beyond measured depth
            SFS = 24
            ts = jnp.linspace(sim.rcfg.t0, sim.rcfg.t1, SFS)     # (SFS,)

            def per_ray(d):
                t = raycast_depth_gt(p, d)                        # full-scene GT depth
                stop_t = jnp.where(jnp.isnan(t), sim.rcfg.t1, t)
                xs = p[None, :] + ts[:, None] * d[None, :]       # (SFS,3)
                # free-space mask: everything strictly before stop_t
                m_free = (ts < stop_t).astype(jnp.float32)       # (SFS,)
                # exposure weights ~ 1/r^2 (softened)
                r = ts
                w_seen = m_free / (1.0 + r * r)
                # hit point & mask
                x_hit = p + stop_t * d
                m_hit = jnp.isfinite(t).astype(jnp.float32) * (t <= sim.rcfg.t1)
                return x_hit, m_hit, xs, m_free, w_seen

            hits, m_hits, frees, m_frees, w_seens = jax.vmap(per_ray)(sel)
            # Update geometry (masked)
            ms = update_geom(ms, hits, m_hits, frees, m_frees)
            # Update exposure (weighted)
            ms = update_expo(ms, frees, w_seens)
            return ms

        mapstate = jax.lax.fori_loop(0, N, update_from_drone, mapstate)

        # === 2) Policy obs: non-learned anchor-lattice + unseen compass + kinematics ===
        Rws = jax.vmap(R_from_q)(states.q)
        fwd = jnp.einsum('nij,j->ni', Rws, jnp.array([1., 0., 0.]))
        def embed_one(i):
            anch = anchor_features(states.p[i], states.q[i], mapstate.geom.theta, mapstate.expo.eta)
            pot, vec = unseen_compass_features(states.p[i], states.q[i],
                                               mapstate.geom.theta, mapstate.expo.eta, sim.rcfg)
            obs = jnp.concatenate([states.p[i], states.v[i], fwd[i], anch, pot, vec])
            return obs
        obs = jax.vmap(embed_one)(jnp.arange(N))
        u_raw = jax.vmap(lambda o: mlp_apply(policy_params, o))(obs)

        # === 3) Dynamics ===
        states_next = State(
            p=jax.vmap(lambda st, u: step(st, u, sim.dyn).p)(states, u_raw),
            v=jax.vmap(lambda st, u: step(st, u, sim.dyn).v)(states, u_raw),
            q=jax.vmap(lambda st, u: step(st, u, sim.dyn).q)(states, u_raw),
            w=jax.vmap(lambda st, u: step(st, u, sim.dyn).w)(states, u_raw),
        )

        # === 4) Reward from LIVE map only ===
        def drone_reward(i):
            o = states_next.p[i]; q = states_next.q[i]
            rays_w = body_rays_world(q, RAYS_BODY)
            idx = jnp.arange(0, RAYS_BODY.shape[0], 4)
            sel = rays_w[idx]
            rws = jax.vmap(lambda d: recon_reward_for_ray(o, d, mapstate.geom.theta, mapstate.expo.eta, sim.rcfg))(sel)
            return rws.mean()
        recon_rew = jax.vmap(drone_reward)(jnp.arange(N))

        # penalties
        coll = jax.vmap(lambda p: soft_collision_live(p, mapstate.geom.theta, sim.margin))(states_next.p)
        ctrl = (u_raw ** 2).sum(axis=1)

        # --- sanitize numerics (avoid NaNs/Infs propagating) ---
        recon_rew = jnp.nan_to_num(recon_rew, nan=0.0, posinf=0.0, neginf=0.0)
        coll = jnp.nan_to_num(coll, nan=0.0, posinf=0.0, neginf=0.0)
        ctrl = jnp.nan_to_num(ctrl, nan=0.0, posinf=0.0, neginf=0.0)

        loss = (sim.w_coll * coll + sim.w_ctrl * ctrl).mean() - recon_rew.mean()
        loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        metrics = dict(
            recon=jnp.nan_to_num(recon_rew.mean(), nan=0.0, posinf=0.0, neginf=0.0),
            coll=jnp.nan_to_num(coll.mean(),      nan=0.0, posinf=0.0, neginf=0.0),
            ctrl=jnp.nan_to_num(ctrl.mean(),      nan=0.0, posinf=0.0, neginf=0.0),
        )
        return (key, states_next, mapstate), (loss, metrics)

    (carry_f, outs) = jax.lax.scan(one_step, (key, states0, mapstate), None, length=sim.steps)
    losses, metrics_seq = outs
    loss = jnp.nan_to_num(losses.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    # metrics_seq is a pytree dict of arrays with leading time dimension
    metrics = {k: jnp.nan_to_num(v.mean(), nan=0.0, posinf=0.0, neginf=0.0) for k, v in metrics_seq.items()}
    return loss, metrics, carry_f[2]  # updated mapstate


# -------------
# JIT-fused training step (GPU-resident)
# -------------
POLICY_OPT = optax.adam(3e-3)


def _train_step_impl(policy_params, opt_state, mapstate, states0, key, sim: SimCfg):
    def _loss_with_aux(pp):
        loss, metrics, mapstate_out = _rollout_loss_impl(pp, mapstate, states0, key, sim)
        return loss, (metrics, mapstate_out)

    (loss, (metrics, mapstate_new)), grads = jax.value_and_grad(_loss_with_aux, has_aux=True)(policy_params)
    # sanitize grads (rarely needed, but safe)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

    updates, opt_state = POLICY_OPT.update(grads, opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)
    return policy_params, opt_state, mapstate_new, loss, metrics


# JIT-compiled variant (used when it helps)
@partial(jax.jit, static_argnames=("sim",), donate_argnums=(0, 1, 2))
def _train_step_jit(policy_params, opt_state, mapstate, states0, key, sim: SimCfg):
    return _train_step_impl(policy_params, opt_state, mapstate, states0, key, sim)


def _pick_first_device():
    # Prefer CUDA/ROCm if present; otherwise fall back to CPU.
    for backend in ("cuda", "rocm"):
        try:
            devs = jax.devices(backend)
            if devs:
                return devs[0]
        except RuntimeError:
            pass
    return jax.devices()[0]


def train_step(policy_params, opt_state, mapstate, states0, key, sim: SimCfg):
    """
    Wrapper that runs JIT on GPU or larger runs, and plain eager on tiny CPU runs.
    This makes the smoke test fast and robust on CPU-only machines.
    """
    backend = jax.default_backend()
    use_jit = (backend in ("cuda", "rocm")) or (sim.steps >= 20)
    if use_jit:
        return _train_step_jit(policy_params, opt_state, mapstate, states0, key, sim)
    else:
        return _train_step_impl(policy_params, opt_state, mapstate, states0, key, sim)


def main():
    # Prefer to place initial arrays directly on GPU if available
    dev = _pick_first_device()
    with jax.default_device(dev):
        key = jax.random.PRNGKey(0)
        mapstate = init_live_map(key)

        # two drones
        def init_state(i):
            p = jnp.array([0., 0.9 * i, 1.6]); v = jnp.zeros(3); q = jnp.array([1., 0., 0., 0.]); w = jnp.zeros(3)
            return State(p, v, q, w)
        states0 = jax.vmap(init_state)(jnp.arange(2))

        # policy net (no CNN/BEV embed; we use fixed features)
        obs_dim = 3 + 3 + 3 + ANCHOR_FEAT_DIM + COMPASS_M + 3  # p, v, fwd, anchors, compass bins, compass vec
        policy_params = init_mlp(jax.random.PRNGKey(2), [obs_dim, 64, 64, 6])  # -> [ax,ay,az,wx,wy,wz]
        opt_state = POLICY_OPT.init(policy_params)

    sim = SimCfg()

    # Warm-up compile or first run
    print("Compiling… (or running eager on CPU small run)")
    key, sub = jax.random.split(key)
    policy_params, opt_state, mapstate, loss, metrics = train_step(
        policy_params, opt_state, mapstate, states0, sub, sim
    )

    # Training loop (host logs tiny scalars)
    for it in range(1, 401):
        key, sub = jax.random.split(key)
        policy_params, opt_state, mapstate, loss, metrics = train_step(
            policy_params, opt_state, mapstate, states0, sub, sim
        )
        if it % 20 == 0 or it == 1:
            recon = float(metrics["recon"])
            coll = float(metrics["coll"])
            ctrl = float(metrics["ctrl"])
            print(f"[{it:03d}] loss={float(loss):.4f} | recon={recon:.3f} | coll={coll:.3f} | ctrl={ctrl:.3f}")


if __name__ == "__main__":
    main()
