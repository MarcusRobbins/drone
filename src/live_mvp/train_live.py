from functools import partial
from typing import NamedTuple, Optional
import argparse, time, os, sys
import jax, jax.numpy as jnp, optax
from jax import tree_util as jtu

from .env_gt import raycast_depth_gt
from .live_map import init_live_map, update_geom, update_expo, MapState, GeomState, ExpoState
from .render import RenderCfg, recon_reward_for_ray
from .dyn import State, DynCfg, step, body_rays_world, R_from_q
from .policy import init_mlp, mlp_apply, anchor_features, ANCHOR_FEAT_DIM
from .policy import unseen_compass_features, COMPASS_M
from .timing import StepTimes, time_blocked

# ---------- Debug toggles (env) ----------
# Enable lightweight JIT-time prints for the first step, or every K steps.
_DBG = os.environ.get("LIVEMVP_DEBUG", "0") in ("1", "true", "True")
try:
    _DBG_EVERY = int(os.environ.get("LIVEMVP_DEBUG_EVERY", "0"))
except Exception:
    _DBG_EVERY = 0

# ----- Debug print helpers that work across JAX versions -----
def _dbg_mask_for_step(t):
    """Return jnp.bool_ mask whether to print at step t."""
    if not _DBG:
        return jnp.array(False, dtype=jnp.bool_)
    if _DBG_EVERY <= 0:
        return (t == 0)
    else:
        return (t % _DBG_EVERY == 0)

def _dbg_print(do, fmt, *xs):
    """Conditionally print inside JIT without using unsupported kwargs."""
    if not _DBG:
        return
    do = jnp.asarray(do, dtype=jnp.bool_)
    def _yes(_):
        jax.debug.print(fmt, *xs)
        return jnp.int32(0)
    def _no(_):
        return jnp.int32(0)
    _ = jax.lax.cond(do, _yes, _no, 0)


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


def _rollout_loss_impl(pp, mapstate: MapState, states0, key, sim: SimCfg):
    """One environment, N drones, train policy; update map online each step."""
    N = states0.p.shape[0]

    def _sanitize_mapstate(ms: MapState, max_abs: float = 1e3) -> MapState:
        """Clip & de-NaN the map params (mirrors viewer's guard)."""
        def clean(x):
            x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.clip(x, -max_abs, max_abs)
        theta = jtu.tree_map(clean, ms.geom.theta)
        eta = jtu.tree_map(clean, ms.expo.eta)
        return MapState(GeomState(theta, ms.geom.opt), ExpoState(eta, ms.expo.opt))

    def one_step(carry, t):
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
                # --- CUT GRADIENTS through GT sensing & sampling geometry ---
                p_sg = jax.lax.stop_gradient(p)
                d_sg = jax.lax.stop_gradient(d)
                t = raycast_depth_gt(p_sg, d_sg)                  # full-scene GT depth (no-grad)
                stop_t = jnp.where(jnp.isnan(t), sim.rcfg.t1, t)
                xs = p_sg[None, :] + ts[:, None] * d_sg[None, :]  # (SFS,3) (no-grad)
                # free-space mask: everything strictly before stop_t
                m_free = (ts < stop_t).astype(jnp.float32)        # (SFS,)
                # exposure weights ~ 1/r^2 (softened)
                r = ts
                w_seen = m_free / (1.0 + r * r)
                # hit point & mask
                x_hit = p_sg + stop_t * d_sg
                m_hit = jnp.isfinite(t).astype(jnp.float32) * (t <= sim.rcfg.t1)
                return x_hit, m_hit, xs, m_free, w_seen

            hits, m_hits, frees, m_frees, w_seens = jax.vmap(per_ray)(sel)
            # simple counts for debug
            n_hit = jnp.sum(m_hits)
            n_free = jnp.sum(m_frees)
            # Update live map (masked/weighted). These updates optimize map params internally,
            # but for policy training we treat the resulting mapstate as a constant.
            ms = update_geom(ms, hits, m_hits, frees, m_frees)
            ms = update_expo(ms, frees, w_seens)
            ms = jtu.tree_map(jax.lax.stop_gradient, ms)  # CUT GRADIENTS into map
            # print for first drone only, at the chosen cadence
            _dbg_print(jnp.logical_and(_dbg_mask_for_step(t), i == 0),
                       "[dbg] t={:d} map_update: hits={} free={}", t, n_hit, n_free)
            return ms

        mapstate = jax.lax.fori_loop(0, N, update_from_drone, mapstate)
        # Keep map numerically healthy (same guard as viewer)
        mapstate = _sanitize_mapstate(mapstate)

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

        # minimal NaN diagnostics on observation features
        def frac_nonfinite(x):
            x = jnp.asarray(x)
            return 1.0 - jnp.mean(jnp.isfinite(x).astype(jnp.float32))
        obs_nan = frac_nonfinite(obs)
        # Pass params explicitly so JAX sees dependency and propagates grads
        u_raw = jax.vmap(mlp_apply, in_axes=(None, 0))(pp, obs)

        # Obs / action stats (cadenced)
        def _stats(x):
            x = jnp.asarray(x)
            return (jnp.nanmin(x), jnp.nanmean(x), jnp.nanmax(x))
        omin, omean, omax = _stats(obs)
        umin, umean, umax = _stats(u_raw)
        _dbg_print(_dbg_mask_for_step(t),
                   "[dbg] t={:d} obs[min,mean,max]=({:.3e},{:.3e},{:.3e}) "
                   "u[min,mean,max]=({:.3e},{:.3e},{:.3e}) u_rms={:.3e} obs_nan={:.3f}",
                   t, omin, omean, omax, umin, umean, umax,
                   jnp.sqrt(jnp.mean(u_raw**2)),
                   1.0 - jnp.mean(jnp.isfinite(obs).astype(jnp.float32)))

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
            _dbg_print(_dbg_mask_for_step(t),
                       "[dbg] t={:d} drone {:d} recon[min,mean,max]=({:.3e},{:.3e},{:.3e})",
                       t, i, jnp.nanmin(rws), jnp.nanmean(rws), jnp.nanmax(rws))
            return rws.mean()
        recon_rew = jax.vmap(drone_reward)(jnp.arange(N))

        # penalties
        coll = jax.vmap(lambda p: soft_collision_live(p, mapstate.geom.theta, sim.margin))(states_next.p)
        ctrl = (u_raw ** 2).sum(axis=1)
        # RMS magnitude of action per-iteration (diagnostic)
        u_rms = jnp.sqrt(jnp.maximum(ctrl.mean(), 0.0))

        # --- compute RAW loss (no sanitization) for gradient ---
        recon_raw = recon_rew.mean()
        coll_raw  = coll.mean()
        ctrl_raw  = ctrl.mean()
        # --- base loss
        loss_raw  = sim.w_coll * coll_raw + sim.w_ctrl * ctrl_raw - recon_raw
        # --- debug: isolate the ctrl term to confirm gradient flow
        if bool(int(os.environ.get("LIVEMVP_DEBUG_CTRL_ONLY", "0"))):
            loss_raw = sim.w_ctrl * ctrl_raw

        # --- sanitized copies for logging only ---
        recon_s = jnp.nan_to_num(recon_raw, nan=0.0, posinf=0.0, neginf=0.0)
        coll_s  = jnp.nan_to_num(coll_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        ctrl_s  = jnp.nan_to_num(ctrl_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        loss_s  = jnp.nan_to_num(loss_raw,  nan=0.0, posinf=0.0, neginf=0.0)

        # extra visibility on action scale & NaNs
        u_rms = jnp.sqrt(jnp.mean(u_raw ** 2))
        u_nan = jnp.any(~jnp.isfinite(u_raw)).astype(jnp.float32)

        any_nan = ~jnp.all(jnp.isfinite(jnp.array([recon_raw, coll_raw, ctrl_raw, loss_raw])))
        _dbg_print(_dbg_mask_for_step(t),
                   "[dbg] t={:d} loss={:.3e} recon={:.3e} coll={:.3e} ctrl={:.3e} any_nan={}",
                   t, loss_raw, recon_raw, coll_raw, ctrl_raw, any_nan)

        # --- tiny in-graph probe: gradient only through control penalty path ---
        def _tree_rms(t):
            sq = jax.tree.map(lambda x: jnp.sum(x * x), t)
            return jnp.sqrt(jtu.tree_reduce(lambda a, b: a + b, sq, 0.0))

        def _ctrl_probe(pp, obs_):
            u = jax.vmap(mlp_apply, in_axes=(None, 0))(pp, obs_)
            return jnp.mean(u * u)

        g_probe = jax.grad(_ctrl_probe)(pp, obs)
        grad_probe_rms = _tree_rms(g_probe)

        metrics = dict(
            recon=recon_s,
            coll=coll_s,
            ctrl=ctrl_s,
            u_rms=jnp.nan_to_num(u_rms, nan=0.0, posinf=0.0, neginf=0.0),
            loss_isfinite=jnp.isfinite(loss_raw).astype(jnp.float32),
            u_nan=u_nan,
            obs_nan=obs_nan,
            grad_probe=jnp.asarray(grad_probe_rms, jnp.float32),
        )
        return (key, states_next, mapstate), (loss_raw, metrics)

    (carry_f, outs) = jax.lax.scan(one_step, (key, states0, mapstate), jnp.arange(sim.steps))
    losses, metrics_seq = outs
    loss = jnp.nan_to_num(losses.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    # metrics_seq is a pytree dict of arrays with leading time dimension
    metrics = {k: jnp.nan_to_num(v.mean(), nan=0.0, posinf=0.0, neginf=0.0) for k, v in metrics_seq.items()}
    return loss, metrics, carry_f[2]  # updated mapstate


# -------------
# JIT-fused training step (GPU-resident)
# -------------
POLICY_OPT = optax.adam(3e-3)


def _tree_rms(tree):
    leaves, _ = jtu.tree_flatten(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    num = sum(jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves)
    den = sum(jnp.size(jnp.asarray(x)) for x in leaves)
    return jnp.sqrt(num / (den + 1e-9))


def _train_step_impl(policy_params, opt_state, mapstate, states0, key, sim: SimCfg):
    def _loss_with_aux(pp):
        loss, metrics, mapstate_out = _rollout_loss_impl(pp, mapstate, states0, key, sim)
        return loss, (metrics, mapstate_out)

    (loss, (metrics, mapstate_new)), grads = jax.value_and_grad(_loss_with_aux, has_aux=True)(policy_params)

    # ---- pre-sanitization gradient stats (detect pruned/NaN/flat) ----
    def _grad_tree_stats(g):
        leaves, _ = jtu.tree_flatten(g)
        n_param = len(jtu.tree_leaves(policy_params))
        n_grad = len(leaves)
        leaf_ratio = 0.0 if n_param == 0 else (n_grad / float(n_param))
        if n_grad == 0:
            return (jnp.array(leaf_ratio, jnp.float32),
                    jnp.array(0.0, jnp.float32),
                    jnp.array(0.0, jnp.float32),
                    jnp.array(0.0, jnp.float32))
        total = 0
        finite_total = 0
        sqsum = 0.0
        maxabs = 0.0
        for x in leaves:
            x = jnp.asarray(x)
            total += x.size
            finite = jnp.isfinite(x)
            finite_total += finite.sum()
            xx = jnp.where(finite, x, 0.0)
            sqsum = sqsum + jnp.sum(xx * xx)
            maxabs = jnp.maximum(maxabs, jnp.max(jnp.abs(xx)))
        total_f = float(total)
        finite_frac = (finite_total / total_f)
        pre_rms = jnp.sqrt(sqsum / total_f)
        return (jnp.array(leaf_ratio, jnp.float32),
                jnp.array(finite_frac, jnp.float32),
                jnp.array(maxabs, jnp.float32),
                jnp.array(pre_rms, jnp.float32))
    gleaf_ratio, gfinite_frac, gmaxabs, gpre_rms = _grad_tree_stats(grads)

    # sanitize grads (rarely needed, but safe)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

    updates, opt_state = POLICY_OPT.update(grads, opt_state, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)
    # add training diagnostics (RMS norms) to metrics
    grad_rms = _tree_rms(grads)
    param_rms = _tree_rms(policy_params)
    update_rms = _tree_rms(updates)
    metrics = dict(
        metrics,
        grad_rms=jnp.nan_to_num(grad_rms, nan=0.0, posinf=0.0, neginf=0.0),
        param_rms=jnp.nan_to_num(param_rms, nan=0.0, posinf=0.0, neginf=0.0),
        update_rms=jnp.nan_to_num(update_rms, nan=0.0, posinf=0.0, neginf=0.0),
        grad_leaf_ratio=gleaf_ratio,          # 0 → grads pruned to ad.Zero
        grad_finite_frac=gfinite_frac,        # <1 → NaNs/±Inf pre-sanitize
        grad_pre_maxabs=gmaxabs,              # large → potential explosion
        grad_pre_rms=gpre_rms,                # RMS before sanitization
    )
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


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Train live_mvp policy with timing.")
    parser.add_argument("--drones", type=int, default=2, help="Number of drones N.")
    parser.add_argument("--steps", type=int, default=80, help="Sim steps per train_step.")
    parser.add_argument("--iters", type=int, default=400, help="Training iterations.")
    parser.add_argument("--timing", action="store_true", help="Print timing each iteration and summary.")
    parser.add_argument("--profile", type=str, default="", help="Start TensorBoard trace to this directory.")
    parser.add_argument("--csv", type=str, default="", help="Write per-iter metrics to CSV.")
    args = parser.parse_args(argv)

    # Prefer to place initial arrays directly on GPU if available
    dev = _pick_first_device()
    with jax.default_device(dev):
        key = jax.random.PRNGKey(0)
        mapstate = init_live_map(key)

        # N drones (shared map)
        N = int(max(1, args.drones))
        def init_state(i):
            # spread along y for a non-degenerate start
            p = jnp.array([0., 0.9 * (i - (N - 1) / 2.0), 1.6])
            v = jnp.zeros(3); q = jnp.array([1., 0., 0., 0.]); w = jnp.zeros(3)
            return State(p, v, q, w)
        states0 = jax.vmap(init_state)(jnp.arange(N))

        # policy net (no CNN/BEV embed; we use fixed features)
        obs_dim = 3 + 3 + 3 + ANCHOR_FEAT_DIM + COMPASS_M + 3  # p, v, fwd, anchors, compass bins, compass vec
        policy_params = init_mlp(jax.random.PRNGKey(2), [obs_dim, 64, 64, 6])  # -> [ax,ay,az,wx,wy,wz]
        opt_state = POLICY_OPT.init(policy_params)

    sim = SimCfg(steps=int(max(1, args.steps)))

    # Optional TensorBoard trace
    tracedir = args.profile.strip()
    if tracedir:
        try:
            from jax import profiler as _jprof
            _jprof.start_trace(tracedir)
            print(f"[trace] TensorBoard trace started: {tracedir}")
        except Exception as e:
            print(f"[trace] Failed to start trace: {e!r}")
            tracedir = ""

    # Warm-up compile or first run (captures compile + first-exec time)
    print("Compiling (first train_step)…")
    key, sub = jax.random.split(key)
    (policy_params, opt_state, mapstate, loss, metrics), t_compile = time_blocked(
        train_step, policy_params, opt_state, mapstate, states0, sub, sim
    )
    print(f"[timing] compile+first: {t_compile:.3f}s  "
          f"(backend={jax.default_backend()}, device={dev})")

    # Optional CSV (unbuffered-ish)
    csv_path = args.csv.strip()
    csv_fh = None
    if csv_path:
        csv_fh = open(csv_path, "w", buffering=1)
        csv_fh.write("iter,dt_ms,wall_ms,gap_ms,env_steps_s,drone_steps_s,loss,recon,coll,ctrl,u_rms,grad_rms,param_rms,update_rms\n")

    # Training loop (host logs tiny scalars)
    perstep = StepTimes("train_step")
    total_iters = int(max(1, args.iters))
    for it in range(1, total_iters + 1):
        key, sub = jax.random.split(key)
        iter_t0 = time.perf_counter()
        (policy_params, opt_state, mapstate, loss, metrics), dt = time_blocked(
            train_step, policy_params, opt_state, mapstate, states0, sub, sim
        )
        wall = time.perf_counter() - iter_t0
        gap = wall - dt
        perstep.add(dt)

        # scalars
        loss_f = float(loss)  # raw loss (may be NaN)
        recon_f = float(metrics.get("recon", 0.0))
        coll_f = float(metrics.get("coll", 0.0))
        ctrl_f = float(metrics.get("ctrl", 0.0))
        urms_f = float(metrics.get("u_rms", 0.0))
        grads_f = float(metrics.get("grad_rms", 0.0))
        prms_f = float(metrics.get("param_rms", 0.0))
        up_f = float(metrics.get("update_rms", 0.0))
        loss_fin = float(metrics.get("loss_isfinite", 0.0))
        u_nan_f = float(metrics.get("u_nan", 0.0))
        obs_nan_f = float(metrics.get("obs_nan", 0.0))
        gprobe_f = float(metrics.get("grad_probe", 0.0))
        gleaf_f  = float(metrics.get("grad_leaf_ratio", 0.0))
        gfin_f   = float(metrics.get("grad_finite_frac", 1.0))
        gpremax  = float(metrics.get("grad_pre_maxabs", 0.0))
        gprerms  = float(metrics.get("grad_pre_rms", 0.0))

        env_steps = sim.steps / max(1e-9, dt)
        drone_steps = (sim.steps * states0.p.shape[0]) / max(1e-9, dt)

        # When --timing is set, print EVERY iteration so cadence is visible
        if args.timing:
            print(
                f"[{it:03d}] dt={dt*1e3:7.1f} ms | wall={wall*1e3:7.1f} ms | gap={gap*1e3:6.1f} ms | "
                f"env-steps/s={env_steps:8.1f} | drone-steps/s={drone_steps:8.1f} | "
                f"loss={loss_f:.3e} (finite={loss_fin:.0f}) | recon={recon_f:.3e} | coll={coll_f:.3e} | ctrl={ctrl_f:.3e} | "
                f"u_rms={urms_f:.3e} (u_nan={u_nan_f:.0f}) | obs_nan={obs_nan_f:.3f} | "
                f"gprobe={gprobe_f:.3e} | grad={grads_f:.3e} | upd={up_f:.3e} | prm={prms_f:.3e} | "
                f"gpre={gprerms:.3e} gfin={gfin_f:.2f} gleaf={gleaf_f:.2f} gmax={gpremax:.2e}",
                flush=True,
            )
        elif (it % 20 == 0) or (it == 1):
            print(
                f"[{it:03d}] dt={dt*1e3:7.1f} ms | wall={wall*1e3:7.1f} ms | gap={gap*1e3:6.1f} ms | "
                f"env-steps/s={env_steps:8.1f} | drone-steps/s={drone_steps:8.1f} | "
                f"loss={loss_f:.3e} (finite={loss_fin:.0f}) | recon={recon_f:.3e} | coll={coll_f:.3e} | ctrl={ctrl_f:.3e} | "
                f"u_rms={urms_f:.3e} (u_nan={u_nan_f:.0f}) | obs_nan={obs_nan_f:.3f} | "
                f"gprobe={gprobe_f:.3e} | grad={grads_f:.3e} | upd={up_f:.3e} | prm={prms_f:.3e} | "
                f"gpre={gprerms:.3e} gfin={gfin_f:.2f} gleaf={gleaf_f:.2f} gmax={gpremax:.2e}",
                flush=True,
            )

        if csv_fh:
            csv_fh.write(
                f"{it},{dt*1e3:.3f},{wall*1e3:.3f},{gap*1e3:.3f},"
                f"{env_steps:.3f},{drone_steps:.3f},"
                f"{loss_f:.6e},{recon_f:.6e},{coll_f:.6e},{ctrl_f:.6e},{urms_f:.6e},{grads_f:.6e},{prms_f:.6e},{up_f:.6e}\n"
            )

    # Timing summary (exclude the compile+first time from steady-state stats)
    stats = perstep.summary()
    if stats:
        mean_dt = stats["mean"]; med_dt = stats["median"]; p95_dt = stats["p95"]
        env_sps_mean = sim.steps / max(1e-9, mean_dt)
        env_sps_med  = sim.steps / max(1e-9, med_dt)
        env_sps_p95  = sim.steps / max(1e-9, p95_dt)
        drones = int(states0.p.shape[0])
        print("\n=== Timing summary ===")
        print(f"compile+first: {t_compile:.3f} s")
        print(f"steady-state per train_step: mean={mean_dt:.4f}s  median={med_dt:.4f}s  p95={p95_dt:.4f}s")
        print(f"env-steps/s   : mean={env_sps_mean:,.1f}  median={env_sps_med:,.1f}  p95={env_sps_p95:,.1f}")
        print(f"drone-steps/s : mean={(env_sps_mean*drones):,.1f}  "
              f"median={(env_sps_med*drones):,.1f}  p95={(env_sps_p95*drones):,.1f}  (N={drones}, steps={sim.steps})")

    if csv_fh:
        csv_fh.close()

    if tracedir:
        try:
            from jax import profiler as _jprof
            _jprof.stop_trace()
            print(f"[trace] TensorBoard trace written to: {tracedir}")
        except Exception as e:
            print(f"[trace] Failed to stop trace: {e!r}")


if __name__ == "__main__":
    main()
