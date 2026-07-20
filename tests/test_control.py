import numpy as np
import torch

from agents.agent_factory import AgentFactory
from config import get_config
from util.riccati_solver import RiccatiSolver


def test_continuous_and_discrete_riccati_residuals():
    solver = RiccatiSolver()
    a_continuous = np.array([[0.0, 1.0], [1.0, 0.0]])
    b_continuous = np.array([[0.0], [1.0]])
    q = np.eye(2)
    r = np.eye(1)
    p_continuous = solver.solve_continuous_are(a_continuous, b_continuous, q, r)
    assert np.all(np.linalg.eigvalsh(p_continuous) > 0)

    a_discrete = np.array([[1.0, 0.1], [0.0, 1.0]])
    b_discrete = np.array([[0.0], [0.1]])
    p_discrete = solver.solve_discrete_are(a_discrete, b_discrete, q, r)
    assert np.all(np.linalg.eigvalsh(p_discrete) > 0)


def test_blending_matches_beta_at_local_boundary_and_preserves_jacobian():
    config = get_config("las-td3")
    config["device"] = "cpu"
    config["beta"] = 0.7
    agent = AgentFactory.create_agent(config)

    direction = torch.tensor([[0.4, -0.2]])
    value = agent.blending_function.get_lyapunov_value(direction)
    boundary_state = direction * torch.sqrt(torch.tensor(agent.c_star) / value)
    _, h1, h2 = agent.blending_function.get_all_blending_terms(boundary_state)
    torch.testing.assert_close(h1, torch.tensor([agent.beta]), atol=1e-6, rtol=0)
    torch.testing.assert_close(h2, torch.tensor([agent.beta]), atol=1e-6, rtol=0)

    origin = torch.zeros(2, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda state: agent.action_tensor(state, noise=False), origin
    )
    torch.testing.assert_close(
        jacobian,
        -agent.lqr_agent.K,
        atol=1e-5,
        rtol=1e-5,
    )


def test_las_td3_local_q_uses_reward_sign_and_update_runs():
    config = get_config("las-td3")
    config.update(device="cpu", batch_size=2)
    agent = AgentFactory.create_agent(config)

    state = torch.tensor([[0.2, -0.1]])
    action = torch.tensor([[0.05]])
    local_q = agent._compute_Q_loc(state, action)
    assert local_q.item() < 0
    torch.testing.assert_close(
        agent._compute_Q_loc(torch.zeros_like(state), torch.zeros_like(action)),
        torch.zeros((1, 1)),
    )

    for offset in (0.0, 0.1):
        agent.add_transition(
            (
                np.array([0.2 + offset, -0.1]),
                np.array([0.05]),
                np.array([-0.1]),
                np.array([0.19 + offset, -0.08]),
                False,
            )
        )
    actor_loss, critic_loss = agent.update()
    assert actor_loss is None
    assert np.isfinite(critic_loss)
