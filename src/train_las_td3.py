import os
import torch
import numpy as np

from util.rk4_step import rk4_step
from agents.las_td3_agent import LAS_TD3Agent
from util.metrics_tracker import MetricsTracker
from util.logger_utils import setup_run_directory_and_logging
from util.dynamics import (
    pendulum_dynamics_np,
    pendulum_dynamics_dreal, 
    compute_pendulum_reward
)
from config import config_las_td3_pendulum


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    DT = 0.003
    MAX_ACTION_VAL = 1.0
    
    CFG = config_las_td3_pendulum

    run_dir, logger = setup_run_directory_and_logging(CFG)
    CFG["run_dir"] = run_dir

    agent = LAS_TD3Agent(CFG)
    exit()

    tracker = MetricsTracker()

    NUM_EPISODES = 1000
    NUM_STEPS_PER_EPISODE = 150
    PRINT_EVERY_EPISODES = 10

    print(f"Initial c* used by blending function: {agent.blending_function.c_star:.4f}")

    initial_exploration_steps = 1000
    total_steps_taken = 0
    total_returns = []
    total_actor_losses = []
    total_critic_losses = []

    for episode in range(NUM_EPISODES):
        ep_actor_losses = []
        ep_critic_losses = []

        current_state = np.array([
            np.random.uniform(-np.pi, np.pi),   
            np.random.uniform(-8.0, 8.0)
        ])
        
        episode_reward = 0
        episode_steps = 0
        actor_loss_ep, critic_loss_ep = None, None

        for step in range(NUM_STEPS_PER_EPISODE):

            if total_steps_taken < initial_exploration_steps:
                action = np.random.uniform(-MAX_ACTION_VAL, MAX_ACTION_VAL, size=(agent.action_dim,)) 
            else:
                action = agent.policy(current_state) 

            next_state = rk4_step(pendulum_dynamics_np, current_state, action, DT).squeeze()

            next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
            next_state[1] = np.clip(next_state[1], -8.0, 8.0) 

            reward_float = compute_pendulum_reward(
                current_state,
                action.item()
            )

            reward_np = np.array([reward_float])

            episode_reward += reward_float
            episode_steps += 1
            total_steps_taken += 1

            done_bool = (step == NUM_STEPS_PER_EPISODE - 1)
            done_np = np.array([float(done_bool)])

            agent.add_transition((
                current_state,
                action, 
                reward_np, 
                next_state, 
                done_np
            ))

            actor_loss, critic_loss = None, None
            if total_steps_taken > initial_exploration_steps:
                actor_loss, critic_loss = agent.update()


            if actor_loss is not None: ep_actor_losses.append(actor_loss)
            if critic_loss is not None:
                ep_critic_losses.append(critic_loss)

            if critic_loss is None and total_steps_taken > initial_exploration_steps:
                print('Potential Error: update() returned no loss when it should have.', step, total_steps_taken)
                    
            current_state = next_state

            if step % 50 == 0 and (episode + 1) % PRINT_EVERY_EPISODES == 0:
                with torch.no_grad():
                    current_state_t_for_log = torch.as_tensor(current_state, dtype=torch.float32, device=DEVICE)
                    v_x = agent.blending_function.get_normalized_lyapunov_value(current_state_t_for_log.unsqueeze(0))

                    log_msg = (
                        f"Ep {episode+1}, Step {step+1}: v(x)={v_x.item():.4f} | "
                        f"State=[{current_state[0]:.2f}, {current_state[1]:.2f}] | "
                        f"Action={action.item():.2f} | Reward={reward_float:.2f}")
                    
                    if actor_loss is not None and critic_loss is not None:
                        log_msg += f" | Losses A={actor_loss:.4f}, C={critic_loss:.4f}"
                    
                    logger.info(log_msg)
            
            if done_bool:
                break

        total_returns.append(episode_reward)
        if ep_actor_losses:
            total_actor_losses.append(np.mean(ep_actor_losses))
        if ep_critic_losses:
            total_critic_losses.append(np.mean(ep_critic_losses))

        if (episode + 1) % PRINT_EVERY_EPISODES == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | Steps: {episode_steps} | Reward: {episode_reward:.2f}")

    logger.info("Training Finished")

    logger.info(f"Saving final model to {run_dir}")
    agent.save(run_dir)

    model_name = CFG["model_name"]
    tracker.add_run_returns(model_name, total_returns)
    tracker.add_run_losses(model_name, total_actor_losses, total_critic_losses)
    tracker.save_top10_plots(folder=run_dir)
    logger.info(f"Metrics plots saved to {run_dir}")


if __name__ == "__main__":
    main()
