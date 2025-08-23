import jax.numpy as jnp
from numpy.testing import assert_allclose
from live_mvp.dyn import quat_mul, quat_normalize, R_from_q, step, State, DynCfg


def random_unit_quat():
    q = jnp.array([0.7, 0.2, -0.3, 0.6])
    return quat_normalize(q)


def test_quat_mul_identity():
    q = random_unit_quat()
    I = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert_allclose(quat_mul(q, I), q, atol=1e-7)
    assert_allclose(quat_mul(I, q), q, atol=1e-7)


def test_rotation_matrix_is_orthonormal_and_det_one():
    q = random_unit_quat()
    R = R_from_q(q)
    I = jnp.eye(3)
    assert_allclose(R @ R.T, I, atol=1e-6)
    det = jnp.linalg.det(R)
    assert_allclose(det, 1.0, atol=1e-5)


def test_step_zero_control_no_change():
    st = State(
        p=jnp.array([1.0, 2.0, 3.0]),
        v=jnp.zeros(3),
        q=jnp.array([1.0, 0.0, 0.0, 0.0]),
        w=jnp.zeros(3),
    )
    u = jnp.zeros(6)
    cfg = DynCfg(dt=0.05)
    st2 = step(st, u, cfg)
    assert_allclose(st2.p, st.p, atol=1e-7)
    assert_allclose(st2.v, st.v, atol=1e-7)
    assert_allclose(st2.q, st.q, atol=1e-7)
    assert_allclose(st2.w, st.w, atol=1e-7)
