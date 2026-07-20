import numpy as np
import torch

from util.dreal import require_dreal


def pendulum_dynamics_torch(
    state: torch.Tensor,
    action: torch.Tensor,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5,
    b: float = 0.1,
) -> torch.Tensor:
    theta = state[..., 0]
    theta_dot = state[..., 1]

    action = action.squeeze(-1)

    theta_ddot = (
        (g / l) * torch.sin(theta) - (b / (m * l**2)) * theta_dot + action / (m * l**2)
    )

    dtheta = theta_dot
    dtheta_ddot = theta_ddot

    dxdt = torch.stack([dtheta, dtheta_ddot], dim=-1)
    return dxdt


def pendulum_dynamics_np(
    state: np.ndarray,
    action: np.ndarray,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5,
    b: float = 0.1,
) -> np.ndarray:
    state = np.asarray(state)
    action = np.asarray(action)
    if state.ndim == 1:  # Single sample
        state = state.reshape(1, -1)
    if action.ndim == 0:  # Single scalar action
        action = np.array([action]).reshape(1, -1)
    elif (
        action.ndim == 1 and action.shape[0] == state.shape[0]
    ):  # Batch of scalar actions
        action = action.reshape(-1, 1)

    theta = state[:, 0]
    theta_dot = state[:, 1]
    u = action[:, 0]

    theta_ddot = (g / l) * np.sin(theta) - (b / (m * l**2)) * theta_dot + u / (m * l**2)

    dtheta = theta_dot
    dtheta_dot = theta_ddot

    dxdt = np.stack([dtheta, dtheta_dot], axis=1)
    return dxdt


def pendulum_dynamics_dreal(
    state,
    action,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5,
    b: float = 0.1,
):
    d = require_dreal()
    theta, omega = state
    action = action[0]

    theta_dot = omega
    theta_ddot = (g / l) * d.sin(theta) - (b / (m * l**2)) * omega + action / (m * l**2)

    dtheta = theta_dot
    dtheta_dot = theta_ddot

    dxdt = np.array([dtheta, dtheta_dot])
    return dxdt


def compute_pendulum_reward(state: np.ndarray, action: float) -> float:
    state_array = np.asarray(state)
    action_array = np.asarray(action)
    if action_array.ndim > 0 and action_array.shape[-1] == 1:
        action_array = action_array[..., 0]
    theta = state_array[..., 0]
    theta_dot = state_array[..., 1]
    cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * action_array**2
    reward = -cost
    return float(reward) if np.ndim(reward) == 0 else reward
