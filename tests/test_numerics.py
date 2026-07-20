import numpy as np
import torch

from util.rk4_step import closed_loop_rk4_step, rk4_step
from util.simulation import rollout_running_cost


def test_fixed_action_and_continuous_feedback_rk4_differ():
    def dynamics(state, action):
        return action

    state = np.array([[1.0]])
    dt = 0.2
    held = rk4_step(dynamics, state, state, dt)
    feedback = closed_loop_rk4_step(dynamics, lambda current: current, state, dt)

    expected_rk4 = 1 + dt + dt**2 / 2 + dt**3 / 6 + dt**4 / 24
    np.testing.assert_allclose(held, [[1 + dt]])
    np.testing.assert_allclose(feedback, [[expected_rk4]])


def test_rollout_marks_convergence_and_horizon():
    initial = torch.tensor([[1.0], [0.01]])

    convergent = rollout_running_cost(
        initial_state=initial,
        dynamics=lambda state, action: -state,
        policy=torch.zeros_like,
        dt=0.1,
        max_steps=100,
        convergence_threshold=0.05,
        running_cost_limit=100.0,
    )
    assert convergent.converged.tolist() == [True, True]
    assert convergent.diverged.tolist() == [False, False]
    assert torch.all(convergent.zubov_target(0.2) < 1)

    horizon = rollout_running_cost(
        initial_state=initial[:1],
        dynamics=lambda state, action: torch.zeros_like(state),
        policy=torch.zeros_like,
        dt=0.1,
        max_steps=2,
        convergence_threshold=0.05,
        running_cost_limit=100.0,
    )
    assert horizon.converged.tolist() == [False]
    assert horizon.diverged.tolist() == [True]
    torch.testing.assert_close(horizon.zubov_target(0.2), torch.ones(1))
