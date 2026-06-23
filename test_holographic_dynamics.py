"""Propagator binding (B4): dynamics as an algebra of binds."""
import numpy as np
from holographic_dynamics import Propagator
from holographic_ai import bind, cosine, random_vector


def _bind_trajectory(D=256, T=400, seed=0):
    rng = np.random.default_rng(seed)
    U = random_vector(D, rng)
    s = random_vector(D, rng)
    traj = [s]
    for _ in range(T):
        s = bind(U, s) + 0.01 * rng.standard_normal(D)
        s /= np.linalg.norm(s)
        traj.append(s)
    return np.array(traj)


def test_step_is_literally_a_bind():
    # "dynamics as an algebra of binds" is exact, not a metaphor: step == bind(U, state).
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    x = traj[310]
    assert np.allclose(prop.step(x), bind(prop.U, x))


def test_propagator_predicts_bind_shaped_dynamics():
    # when the dynamics ARE a bind, the propagator recovers the operator and predicts full states,
    # far better than persistence (binding scrambles, so the last state poorly predicts the next).
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    pred = np.mean([cosine(prop.step(traj[300 + i]), traj[301 + i]) for i in range(80)])
    persist = np.mean([cosine(traj[300 + i], traj[301 + i]) for i in range(80)])
    assert pred > 0.9 and pred > persist + 0.2


def test_trajectory_is_content_addressable():
    # the durable win: forward k then back k returns the start -- past states are recoverable.
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    x = traj[350]
    fwd = prop.rollout(x, 4)[-1]
    back = prop.recall_at(fwd, 4)
    assert cosine(x, back) > 0.99


def test_rollout_shape():
    traj = _bind_trajectory()
    prop = Propagator.learn(traj[:300])
    assert prop.rollout(traj[10], 5).shape == (5, traj.shape[1])
