from copy import deepcopy

import numpy as np

from util.dynamics import (
    compute_pendulum_reward,
    pendulum_dynamics_dreal,
    pendulum_dynamics_np,
    pendulum_dynamics_torch,
)

DT = 0.003
PENDULUM_G = 9.81
PENDULUM_M = 0.15
PENDULUM_L = 0.5
PENDULUM_B = 0.1
MAX_ACTION = 1.0


config_ldp_pendulum = {
    "agent_str": "LDP",
    "model_name": "LDP",
    "environment": "InvertedPendulum",
    "max_action": MAX_ACTION,
    "beta": 0.9,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "dynamics_fn": pendulum_dynamics_torch,
    "LQR": {
        "agent_str": "LQR",
        "environment": "InvertedPendulum",
        "discrete_discounted": False,
        "g": PENDULUM_G,
        "m": PENDULUM_M,
        "l": PENDULUM_L,
        "b": PENDULUM_B,
        "max_action": MAX_ACTION,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
    },
    "alpha": 0.2,
    "lr": 2e-3,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "dt": DT,
    "rollout_steps": 3000,
    "rollout_refresh_interval": 10,
    "plot_interval": 0,
    "plot_grid_size": 201,
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (10, 10),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "normalize_gradients": False,
    "c_star": 1.1523,
}

config_lac_pendulum = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "environment": "InvertedPendulum",
    "alpha": 0.2,
    "lr": 2e-3,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "dt": DT,
    "rollout_steps": 3000,
    "rollout_refresh_interval": 10,
    "plot_interval": 0,
    "plot_grid_size": 201,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (10, 10),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": 1.0,
    "normalize_gradients": False,
}

config_las_td3_pendulum = {
    "agent_str": "LAS_TD3",
    "model_name": "LAS_TD3",
    "environment": "InvertedPendulum",
    "max_action": MAX_ACTION,
    "beta": 0.6,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "LQR": {
        "agent_str": "LQR",
        "environment": "InvertedPendulum",
        "discrete_discounted": True,
        "gamma": 0.9,
        "dt": DT,
        "g": PENDULUM_G,
        "m": PENDULUM_M,
        "l": PENDULUM_L,
        "b": PENDULUM_B,
        "max_action": MAX_ACTION,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
    },
    "gamma": 0.9,
    "tau": 0.005,
    "policy_freq": 2,
    "batch_size": 256,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "expl_noise": 0.1,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "actor_hidden_sizes": (256, 256),
    "critic_hidden_sizes": (256, 256),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "c_star": 1.1982,
}


config_td3_pendulum = {
    "agent_str": "TD3",
    "model_name": "TD3",
    "environment": "InvertedPendulum",
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "policy_freq": 2,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "start_episodes": 100,
    "expl_noise": 0.1,
    "actor_hidden_sizes": (256, 256),
    "critic_hidden_sizes": (256, 256),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": MAX_ACTION,
    "dynamics_fn": pendulum_dynamics_np,
    "rewards_fn": compute_pendulum_reward,
}

config_lqr_pendulum = {
    "agent_str": "LQR",
    "model_name": "LQR",
    "environment": "InvertedPendulum",
    "discrete_discounted": False,
    "gamma": 0.99,
    "dt": DT,
    "g": PENDULUM_G,
    "m": PENDULUM_M,
    "l": PENDULUM_L,
    "b": PENDULUM_B,
    "max_action": MAX_ACTION,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}


CONFIGS = {
    "ldp": config_ldp_pendulum,
    "lac": config_lac_pendulum,
    "las-td3": config_las_td3_pendulum,
    "td3": config_td3_pendulum,
    "lqr": config_lqr_pendulum,
}


def get_config(name: str) -> dict:
    """Return an isolated experiment configuration."""
    try:
        return deepcopy(CONFIGS[name.lower()])
    except KeyError as exc:
        choices = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config {name!r}; choose from {choices}") from exc
