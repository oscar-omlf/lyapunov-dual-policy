import numpy as np
import torch

from agents.agent_factory import AgentFactory
from agents.lqr_agent import LQRAgent
from config import get_config
from util.csv_utils import write_2d_csv, write_counts_csv
from util.dynamics import compute_pendulum_reward, pendulum_dynamics_np
from util.logger_utils import save_json, setup_run_directory_and_logging
from util.reproducibility import seed_everything
from util.rk4_step import closed_loop_rk4_step, rk4_step


@torch.no_grad()
def _batch_policy(agent, states):
    if isinstance(agent, LQRAgent):
        return agent.policy_np(states)
    if hasattr(agent, "action_tensor"):
        actions = agent.action_tensor(states)
        return actions.detach().cpu().numpy()
    return np.stack([agent.policy(state) for state in states])


def _simulate_run(
    agent,
    initial_states,
    *,
    dt,
    steps,
    stabilization_threshold,
    consecutive_stable,
    integration,
    project_state,
):
    state = initial_states.copy()
    episodes = state.shape[0]
    cumulative_cost = np.zeros(episodes, dtype=np.float64)
    stable_count = np.zeros(episodes, dtype=np.int32)
    stabilized = np.zeros(episodes, dtype=bool)
    steps_to_stabilize = np.full(episodes, np.nan, dtype=np.float64)

    def policy(current_state):
        return _batch_policy(agent, current_state)

    for step in range(steps):
        action = policy(state)
        reward = compute_pendulum_reward(state, action)
        cumulative_cost -= np.asarray(reward, dtype=np.float64)
        if integration == "continuous":
            next_state = closed_loop_rk4_step(pendulum_dynamics_np, policy, state, dt)
        elif integration == "sample-hold":
            next_state = rk4_step(pendulum_dynamics_np, state, action, dt)
        else:
            raise ValueError(f"Unknown integration mode: {integration}")

        if project_state:
            next_state[:, 0] = (next_state[:, 0] + np.pi) % (2 * np.pi) - np.pi
            next_state[:, 1] = np.clip(next_state[:, 1], -8.0, 8.0)
        state = next_state

        inside = np.linalg.norm(state, ord=2, axis=1) < stabilization_threshold
        stable_count = np.where(inside, stable_count + 1, 0)
        newly_stable = (~stabilized) & (stable_count >= consecutive_stable)
        steps_to_stabilize[newly_stable] = step + 1
        stabilized |= newly_stable

    return cumulative_cost, steps_to_stabilize, stabilized


def evaluate_agents(
    agent_names,
    *,
    checkpoints=None,
    episodes_by_agent=None,
    beta_by_agent=None,
    seed=0,
    output_dir="runs",
    device="auto",
    runs=50,
    episodes=500,
    steps=3000,
    dt=0.003,
    stabilization_threshold=5e-4,
    consecutive_stable=10,
    integration="continuous",
    project_state=False,
):
    if runs <= 0 or episodes <= 0 or steps <= 0:
        raise ValueError("runs, episodes, and steps must be positive")
    if dt <= 0 or stabilization_threshold <= 0:
        raise ValueError("dt and stabilization_threshold must be positive")
    if consecutive_stable <= 0:
        raise ValueError("consecutive_stable must be positive")
    checkpoints = checkpoints or {}
    episodes_by_agent = episodes_by_agent or {}
    beta_by_agent = beta_by_agent or {}
    seed_everything(seed)
    evaluation_config = {
        "model_name": "EVALUATION",
        "environment": "InvertedPendulum",
        "seed": seed,
        "agents": list(agent_names),
        "runs": runs,
        "episodes": episodes,
        "steps": steps,
        "dt": dt,
        "integration": integration,
        "project_state": project_state,
        "stabilization_threshold": stabilization_threshold,
        "consecutive_stable": consecutive_stable,
    }
    run_dir, logger = setup_run_directory_and_logging(
        evaluation_config, base_log_dir=output_dir, evaluation=True
    )

    rng = np.random.default_rng(seed)
    initial_states = np.empty((runs, episodes, 2), dtype=np.float32)
    initial_states[..., 0] = rng.uniform(-np.pi, np.pi, size=(runs, episodes))
    initial_states[..., 1] = rng.uniform(-8.0, 8.0, size=(runs, episodes))
    np.save(run_dir / "initial_states.npy", initial_states)

    summaries = {}
    for agent_name in agent_names:
        config = get_config(agent_name)
        config["device"] = device
        if agent_name in beta_by_agent:
            config["beta"] = beta_by_agent[agent_name]
        agent = AgentFactory.create_agent(config)
        if agent_name != "lqr":
            checkpoint = checkpoints.get(agent_name)
            if checkpoint is None:
                raise ValueError(f"Checkpoint required for agent {agent_name!r}")
            agent.load(
                checkpoint,
                episode=episodes_by_agent.get(agent_name, "latest"),
            )

        costs = np.empty((runs, episodes), dtype=np.float64)
        stabilize_steps = np.full_like(costs, np.nan)
        counts = np.zeros(runs, dtype=np.int64)
        for run in range(runs):
            run_cost, run_steps, stabilized = _simulate_run(
                agent,
                initial_states[run],
                dt=dt,
                steps=steps,
                stabilization_threshold=stabilization_threshold,
                consecutive_stable=consecutive_stable,
                integration=integration,
                project_state=project_state,
            )
            costs[run] = run_cost
            stabilize_steps[run] = run_steps
            counts[run] = stabilized.sum()
            logger.info(
                "%s run=%d/%d cost=%.6f convergence=%.2f%%",
                agent_name,
                run + 1,
                runs,
                run_cost.mean(),
                100.0 * stabilized.mean(),
            )

        agent_dir = run_dir / agent_name
        write_2d_csv(costs, agent_dir / "costs.csv")
        write_2d_csv(stabilize_steps, agent_dir / "steps_to_stabilize.csv")
        write_counts_csv(counts, agent_dir / "episodes_stabilized.csv")
        finite_steps = stabilize_steps[np.isfinite(stabilize_steps)]
        run_mean_steps = np.array(
            [
                row[np.isfinite(row)].mean() if np.isfinite(row).any() else np.nan
                for row in stabilize_steps
            ]
        )
        finite_run_means = run_mean_steps[np.isfinite(run_mean_steps)]
        summary = {
            "mean_cost": float(costs.mean()),
            "sd_run_mean_cost": float(costs.mean(axis=1).std(ddof=1))
            if runs > 1
            else 0.0,
            "convergence_rate": float(counts.sum() / costs.size),
            "mean_steps_to_stabilize": (
                float(finite_steps.mean()) if finite_steps.size else None
            ),
            "sd_run_mean_steps": (
                float(finite_run_means.std(ddof=1))
                if finite_run_means.size > 1
                else None
            ),
        }
        summaries[agent_name] = summary
        save_json(summary, agent_dir / "summary.json")

    save_json(summaries, run_dir / "summary.json")
    return run_dir
