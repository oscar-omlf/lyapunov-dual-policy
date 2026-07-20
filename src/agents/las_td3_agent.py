import copy
from pathlib import Path

import numpy as np
import torch

from agents.dual_policy_agent import DualPolicyAgent
from models.td3_actor import TD3Actor
from models.td3_critic import TD3Critic
from trainers.las_td3_trainer import LAS_TD3Trainer
from util.checkpoints import load_checkpoint, validate_checkpoint


class LAS_TD3Agent(DualPolicyAgent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.gamma = config.get("gamma")
        self.tau = config.get("tau")
        self.policy_freq = config.get("policy_freq")
        self.batch_size = config.get("batch_size")
        self.policy_noise = config.get("policy_noise") * self.max_action
        self.noise_clip = config.get("noise_clip") * self.max_action
        self.expl_noise = config.get("expl_noise")

        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")

        # Get architecture hyperparameters from config
        actor_hidden_sizes = config.get("actor_hidden_sizes", (64, 64))
        critic_hidden_sizes = config.get("critic_hidden_sizes", (64, 64))

        # Initialize the actor and critic models with the tunable architectures
        self.actor_model = TD3Actor(
            input_size=self.state_dim,
            hidden_sizes=actor_hidden_sizes,
            action_dim=self.action_dim,
            max_action=self.max_action,
        ).to(device=self.device)

        self.critic_model = TD3Critic(
            state_dim=self.state_dim,
            hidden_sizes=critic_hidden_sizes,
            action_dim=self.action_dim,
        ).to(device=self.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor_model)
        self.critic_target = copy.deepcopy(self.critic_model)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.critic_lr
        )

        H_matrix, H_matrix_np = self._initialize_H_matrix()
        self.H_matrix = H_matrix
        self.H_matrix_np = H_matrix_np

        self.trainer = LAS_TD3Trainer(
            agent=self,
            buffer=self._replay_buffer,
            gamma=self.gamma,
            tau=self.tau,
            policy_freq=self.policy_freq,
            batch_size=self.batch_size,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            device=self.device,
        )

    def _initialize_H_matrix(self):
        # Compute and store the H matrix for Q_loc
        if not self.lqr_agent.discrete_discounted:
            raise ValueError("TD3 requires a discrete-time LQR agent.")

        A = self.lqr_agent.A_np
        B = self.lqr_agent.B_np

        P = self.lqr_agent.P_np

        lqr_gamma = self.lqr_agent.gamma

        Q_gamma = self.lqr_agent.Q_gamma
        R_gamma = self.lqr_agent.R_gamma

        H_matrix_np = self.riccati_solver.compute_H_matrix(
            A, B, Q_gamma, R_gamma, P, lqr_gamma
        )

        H_matrix = torch.from_numpy(H_matrix_np).to(
            dtype=torch.float32, device=self.device
        )

        return H_matrix, H_matrix_np

    def _compute_Q_loc(
        self, state_torch: torch.Tensor, action_torch: torch.Tensor
    ) -> torch.Tensor:
        current_x_star = self.x_star.unsqueeze(0)
        current_u_star = self.u_star.unsqueeze(0)

        delta_x = state_torch - current_x_star
        delta_u = action_torch - current_u_star

        if delta_u.shape[0] != delta_x.shape[0] and delta_u.shape[0] == 1:
            delta_u = delta_u.expand(delta_x.shape[0], -1)

        z_error = torch.cat((delta_x, delta_u), dim=-1)

        term1 = torch.matmul(z_error, self.H_matrix)
        # TD3 maximizes discounted rewards while H defines a positive cost.
        Q_loc_val = -torch.sum(term1 * z_error, dim=1, keepdim=True)
        return Q_loc_val

    def get_composite_Q_values(
        self,
        state_torch: torch.Tensor,
        action_torch: torch.Tensor,
        use_target_critic: bool = False,
    ):
        critic_net_to_use = (
            self.critic_target if use_target_critic else self.critic_model
        )
        omega_q1, omega_q2 = critic_net_to_use(state_torch, action_torch)

        # return omega_q1, omega_q2

        q_loc = self._compute_Q_loc(state_torch, action_torch)

        with torch.no_grad():
            h2_val = self.blending_function.get_h2(state_torch)
            if h2_val.ndim == 1:
                h2_val = h2_val.unsqueeze(-1)

        Q_comp1 = q_loc + h2_val * (omega_q1 - q_loc)
        Q_comp2 = q_loc + h2_val * (omega_q2 - q_loc)
        return Q_comp1, Q_comp2

    def get_composite_Q1_value(
        self,
        state_torch: torch.Tensor,
        action_torch: torch.Tensor,
    ):
        omega_q1 = self.critic_model.Q1_value(state_torch, action_torch)
        # return omega_q1

        q_loc = self._compute_Q_loc(state_torch, action_torch)

        with torch.no_grad():
            h2_val = self.blending_function.get_h2(state_torch)
            if h2_val.ndim == 1:
                h2_val = h2_val.unsqueeze(-1)

        Q_comp1 = q_loc + h2_val * (omega_q1 - q_loc)
        return Q_comp1

    def _get_global_action(
        self, state_torch: torch.Tensor, noise: bool = True
    ) -> torch.Tensor:
        mu_theta_action = self.actor_model(state_torch)

        if self.expl_noise > 0 and noise is True:
            noise = torch.normal(
                0,
                self.expl_noise * self.max_action,
                size=mu_theta_action.shape,
                device=self.device,
            )
            mu_theta_action += noise

        mu_theta_action = torch.clamp(
            mu_theta_action, -self.max_action, self.max_action
        )
        return mu_theta_action

    def add_transition(self, transition: tuple) -> None:
        """
        Add a transition to the replay buffer.
        :param transition: (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition
        state_t = torch.as_tensor(state, dtype=torch.float32).cpu()
        action_t = torch.as_tensor(action, dtype=torch.float32).cpu()
        reward_t = torch.as_tensor(reward, dtype=torch.float32).cpu()
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32).cpu()
        done_value = float(np.asarray(done).reshape(-1)[0])
        done_t = torch.tensor([done_value], dtype=torch.float32)
        self._replay_buffer.push((state_t, action_t, reward_t, next_state_t, done_t))

    def update(self) -> None:
        """
        Perform a gradient descent step on both actor and critic.
        """
        return self.trainer.train()

    def save(self, file_path, episode=None) -> None:
        """
        Save the actor and critic networks.
        """
        directory = Path(file_path)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "latest" if episode is None else str(episode)
        torch.save(
            {
                "agent": "las-td3",
                "episode": episode,
                "actor": self.actor_model.state_dict(),
                "critic": self.critic_model.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "beta": self.beta,
                "c_star": self.c_star,
                "update_counter": self.trainer._update_counter,
            },
            directory / f"checkpoint_{suffix}.pt",
        )

    def load(self, file_path, episode=None) -> None:
        """
        Load the actor and critic networks, and synchronize the trainer's target networks.
        """
        directory = Path(file_path)
        suffix = "latest" if episode is None else str(episode)
        checkpoint_path = directory / f"checkpoint_{suffix}.pt"
        if checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            validate_checkpoint(
                checkpoint,
                expected_agent="las-td3",
                beta=self.beta,
                c_star=self.c_star,
            )
            self.actor_model.load_state_dict(checkpoint["actor"])
            self.critic_model.load_state_dict(checkpoint["critic"])
            self.actor_target.load_state_dict(
                checkpoint.get("actor_target", checkpoint["actor"])
            )
            self.critic_target.load_state_dict(
                checkpoint.get("critic_target", checkpoint["critic"])
            )
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            self.trainer._update_counter = int(
                checkpoint.get("update_counter", self.trainer._update_counter)
            )
            return

        self.actor_model.load_state_dict(
            load_checkpoint(directory / f"td3_actor_{episode}.pth", self.device)
        )
        self.critic_model.load_state_dict(
            load_checkpoint(directory / f"td3_critic_{episode}.pth", self.device)
        )
        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.critic_target.load_state_dict(self.critic_model.state_dict())
