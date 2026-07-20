"""Inspect the dual-policy blending values at selected LQR levels."""

import torch

from agents.agent_factory import AgentFactory
from config import get_config


def get_state_for_V_level(agent, target_V_val, start_state=None):
    if start_state is None:
        start_state = torch.randn(1, agent.state_dim, device=agent.device)

    if torch.all(start_state == 0):
        start_state[0, 0] = 1.0

    blending_func = agent.blending_function

    V_start = blending_func.get_lyapunov_value(start_state)

    if V_start.item() < 1e-9:
        return get_state_for_V_level(agent, target_V_val, start_state + 0.1)

    scaling_factor = torch.sqrt(target_V_val / V_start)

    target_state = scaling_factor * start_state
    return target_state


def analyze_blending_values():
    config = get_config("las-td3")
    config["beta"] = 0.8

    agent = AgentFactory.create_agent(config)
    c_star = agent.c_star
    blending_function = agent.blending_function
    print("-" * 50)

    points_of_interest = {
        "Deep Inside DoA": 0.01 * c_star,
        "A Bit Inside DoA": 0.5 * c_star,
        "On DoA Boundary": 1.0 * c_star,
        "Far Outside DoA": 3.0 * c_star,
    }

    print(f"{'Location':<20} | {'V(x) / c*':<10} | {'h1(x)':<10} | {'h2(x)':<10}")
    print("-" * 65)

    for name, target_v in points_of_interest.items():
        state = get_state_for_V_level(agent, target_v)
        v_norm, h1, h2 = blending_function.get_all_blending_terms(state)
        print(
            f"{name:<20} | {v_norm.item():<10.3f} | {h1.item():<10.4f} | {h2.item():<10.4f}"
        )


if __name__ == "__main__":
    analyze_blending_values()
