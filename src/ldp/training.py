from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

from agents.agent_factory import AgentFactory
from config import get_config
from util.certification import bisect_verified_level
from util.doa_utils import estimate_doa
from util.dreal import extract_ce_from_model
from util.logger_utils import save_json, setup_run_directory_and_logging
from util.reproducibility import seed_everything
from util.rk4_step import rk4_step


def _save_loss_history(run_dir, actor_losses, critic_losses):
    steps = np.arange(1, len(actor_losses) + 1)
    history = np.column_stack((steps, actor_losses, critic_losses))
    np.savetxt(
        Path(run_dir) / "losses.csv",
        history,
        delimiter=",",
        header="step,actor_loss,critic_loss",
        comments="",
    )


def train_lyapunov_cegar(
    agent_name,
    *,
    seed=0,
    output_dir="runs",
    device="auto",
    beta=None,
    outer_loops=80,
    steps_per_loop=1000,
    log_interval=50,
    certify=True,
    level_lower=0.01,
    level_upper=0.95,
    level_tolerance=0.01,
    bisection_iterations=12,
    eps=0.25,
    bridge_level=None,
    local_margin=1e-4,
    delta=1e-4,
    verification_scale=2.0,
    doa_samples=50_000,
    minimum_stop_level=0.4,
    minimum_relative_gain=0.01,
    patience=3,
    counterexample_capacity=128,
    rollout_steps=None,
    rollout_refresh_interval=None,
    plot_interval=None,
):
    if agent_name not in {"ldp", "lac"}:
        raise ValueError("agent_name must be 'ldp' or 'lac'")
    if outer_loops <= 0 or steps_per_loop <= 0:
        raise ValueError("outer_loops and steps_per_loop must be positive")
    seed_everything(seed)
    config = get_config(agent_name)
    config["seed"] = int(seed)
    config["device"] = device
    if beta is not None:
        if agent_name != "ldp":
            raise ValueError("beta is only defined for LDP")
        if not 0 < beta < 1:
            raise ValueError("beta must lie strictly between zero and one")
        config["beta"] = float(beta)
        config["model_name"] = f"LDP_beta_{beta:.2f}"
    if rollout_steps is not None:
        config["rollout_steps"] = int(rollout_steps)
    if rollout_refresh_interval is not None:
        config["rollout_refresh_interval"] = int(rollout_refresh_interval)
    if plot_interval is not None:
        config["plot_interval"] = int(plot_interval)

    run_dir, logger = setup_run_directory_and_logging(config, base_log_dir=output_dir)
    config["run_dir"] = str(run_dir)
    save_json(config, run_dir / "config.json")
    agent = AgentFactory.create_agent(config)
    counterexamples = deque(maxlen=int(counterexample_capacity))
    actor_losses = []
    critic_losses = []
    certificate_history = []
    doa_history = []
    stagnant = 0
    best_level = -np.inf

    logger.info(
        "Starting %s training: %d outer loops x %d updates",
        agent_name.upper(),
        outer_loops,
        steps_per_loop,
    )
    for outer in range(1, outer_loops + 1):
        for inner in range(1, steps_per_loop + 1):
            actor_loss, critic_loss = agent.update(
                counter_examples=list(counterexamples),
                normalize_gradients=config["normalize_gradients"],
            )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            if log_interval and inner % log_interval == 0:
                losses = agent.trainer.last_losses
                logger.info(
                    "outer=%d update=%d/%d total=%.6f pde=%.6f rollout=%.6f",
                    outer,
                    inner,
                    steps_per_loop,
                    losses["total"],
                    losses["pde"],
                    losses["rollout"],
                )

        if not certify:
            continue

        verifier_options = {
            "scale": verification_scale,
            "eps": eps,
            "delta": delta,
        }
        if agent_name == "ldp":
            verifier_options.update(
                bridge_level=bridge_level,
                local_margin=local_margin,
            )
        verifier = partial(
            agent.trainer.check_lyapunov_with_ce,
            **verifier_options,
        )
        verified_metadata = {}

        def check_fn(*, level):
            outcome = verifier(level=level)
            if outcome[0]:
                verified_metadata[float(level)] = dict(agent.trainer.last_verification)
            return outcome

        result = bisect_verified_level(
            check_fn,
            lower=level_lower,
            upper=level_upper,
            tolerance=level_tolerance,
            max_iterations=bisection_iterations,
        )
        level = result.verified_level
        verification = dict(
            verified_metadata.get(
                float(level) if level is not None else None,
                agent.trainer.last_verification,
            )
        )
        verification.update(
            {
                "outer_loop": outer,
                "verified_level": level,
                "lower_verified": result.lower_verified,
                "upper_verified": result.upper_verified,
                "checks": result.checks,
            }
        )

        if level is not None:
            lb, ub = config["r1_bounds"]
            doa_fraction = estimate_doa(
                level=level,
                lb=2 * lb,
                ub=2 * ub,
                value_fn=(
                    agent.get_composite_W_value
                    if agent_name == "ldp"
                    else agent.get_W_value
                ),
                device=agent.device,
                n_samples=doa_samples,
            )
            verification["doa_fraction_r2"] = doa_fraction
            doa_history.append(doa_fraction)
            logger.info(
                "outer=%d verified_level=%.6f doa_fraction=%.6f",
                outer,
                level,
                doa_fraction,
            )
            if level > best_level:
                best_level = level
                agent.save(run_dir, episode="best")

            if len(doa_history) >= 2 and level >= minimum_stop_level:
                relative_gain = (doa_history[-1] - doa_history[-2]) / max(
                    doa_history[-2], 1e-12
                )
                verification["relative_doa_gain"] = relative_gain
                stagnant = stagnant + 1 if relative_gain < minimum_relative_gain else 0
                if stagnant >= patience:
                    certificate_history.append(verification)
                    save_json(
                        certificate_history,
                        run_dir / "certificates.json",
                    )
                    logger.info(
                        "Early stopping after %d low-gain certificates",
                        patience,
                    )
                    break
        else:
            logger.warning("No verified level at lower bracket %.6f", level_lower)

        counterexample = extract_ce_from_model(result.counterexample, agent.state_dim)
        if counterexample is not None:
            counterexamples.append(counterexample)
            verification["counterexample"] = counterexample.tolist()
            logger.info("Added counterexample %s", counterexample)

        certificate_history.append(verification)
        save_json(certificate_history, run_dir / "certificates.json")
        if counterexamples:
            np.savetxt(
                run_dir / "counterexamples.csv",
                np.asarray(counterexamples),
                delimiter=",",
            )

    total_updates = len(actor_losses)
    agent.save(run_dir, episode="latest")
    _save_loss_history(run_dir, actor_losses, critic_losses)
    save_json(
        {
            "agent": agent_name,
            "seed": seed,
            "updates": total_updates,
            "best_verified_level": (None if best_level == -np.inf else best_level),
            "counterexamples": len(counterexamples),
            "certification_enabled": certify,
        },
        run_dir / "summary.json",
    )
    logger.info("Finished %d optimizer updates", total_updates)
    return run_dir


def train_td3(
    agent_name,
    *,
    seed=0,
    output_dir="runs",
    device="auto",
    beta=None,
    episodes=1000,
    steps_per_episode=3000,
    warmup_steps=1000,
    log_interval=10,
    project_state=True,
):
    if agent_name not in {"td3", "las-td3"}:
        raise ValueError("agent_name must be 'td3' or 'las-td3'")
    if episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("episodes and steps_per_episode must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    seed_everything(seed)
    config = get_config(agent_name)
    config["seed"] = int(seed)
    config["device"] = device
    if beta is not None:
        if agent_name != "las-td3":
            raise ValueError("beta is only defined for LAS-TD3")
        if not 0 < beta < 1:
            raise ValueError("beta must lie strictly between zero and one")
        config["beta"] = float(beta)
        config["model_name"] = f"LAS_TD3_beta_{beta:.2f}"

    run_dir, logger = setup_run_directory_and_logging(config, base_log_dir=output_dir)
    config["run_dir"] = str(run_dir)
    save_json(config, run_dir / "config.json")
    agent = AgentFactory.create_agent(config)
    dynamics = get_config("td3")["dynamics_fn"]
    reward_fn = get_config("td3")["rewards_fn"]
    dt = config.get("dt", get_config("lqr")["dt"])
    returns = []
    actor_losses = []
    critic_losses = []
    total_steps = 0

    for episode in range(1, episodes + 1):
        state = np.array(
            [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-8.0, 8.0),
            ],
            dtype=np.float32,
        )
        episode_return = 0.0
        episode_actor_losses = []
        episode_critic_losses = []
        for step in range(steps_per_episode):
            if total_steps < warmup_steps:
                action = np.random.uniform(
                    -config["max_action"],
                    config["max_action"],
                    size=agent.action_dim,
                )
            elif agent_name == "td3":
                action = agent.policy(state, noise=True)
            else:
                action = agent.policy(state, noise=True)

            next_state = rk4_step(dynamics, state, action, dt).reshape(-1)
            if project_state:
                next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
                next_state[1] = np.clip(next_state[1], -8.0, 8.0)
            reward = float(reward_fn(state, float(action[0])))
            done = step == steps_per_episode - 1
            agent.add_transition((state, action, np.array([reward]), next_state, done))
            total_steps += 1
            episode_return += reward
            if total_steps >= warmup_steps:
                actor_loss, critic_loss = agent.update()
                if actor_loss is not None:
                    episode_actor_losses.append(actor_loss)
                if critic_loss is not None:
                    episode_critic_losses.append(critic_loss)
            state = next_state

        returns.append(episode_return)
        actor_losses.append(
            np.nan if not episode_actor_losses else float(np.mean(episode_actor_losses))
        )
        critic_losses.append(
            np.nan
            if not episode_critic_losses
            else float(np.mean(episode_critic_losses))
        )
        if log_interval and episode % log_interval == 0:
            logger.info(
                "episode=%d/%d return=%.6f replay=%d",
                episode,
                episodes,
                episode_return,
                len(agent._replay_buffer),
            )

    agent.save(run_dir, episode="latest")
    metrics = np.column_stack(
        (np.arange(1, episodes + 1), returns, actor_losses, critic_losses)
    )
    np.savetxt(
        run_dir / "training_metrics.csv",
        metrics,
        delimiter=",",
        header="episode,return,actor_loss,critic_loss",
        comments="",
    )
    save_json(
        {
            "agent": agent_name,
            "seed": seed,
            "episodes": episodes,
            "steps": total_steps,
            "mean_return": float(np.mean(returns)),
            "project_state": bool(project_state),
        },
        run_dir / "summary.json",
    )
    return run_dir
