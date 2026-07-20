import copy

import torch
import torch.nn.functional as F

from agents.abstract_agent import ReplayBuffer
from trainers.abstract_trainer import Trainer


class TD3Trainer(Trainer):
    def __init__(
        self,
        buffer: ReplayBuffer,
        actor,
        critic,
        gamma: float,
        tau: float,
        policy_freq: int,
        batch_size: int,
        policy_noise: float,
        noise_clip: float,
        actor_lr: float,
        critic_lr: float,
        device: str,
    ):
        super().__init__()
        self.buffer = buffer
        self.actor_model = actor
        self.critic_model = critic
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = device

        self._update_counter = 1

        # Create target networks as deep copies
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=critic_lr
        )

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        transitions = self.buffer.sample(self.batch_size)
        states = torch.stack([t[0] for t in transitions]).to(self.device)
        actions = torch.stack([t[1] for t in transitions]).to(self.device)
        rewards = torch.stack([t[2] for t in transitions]).to(self.device).squeeze(-1)
        next_states = torch.stack([t[3] for t in transitions]).to(self.device)
        dones = torch.stack([t[4] for t in transitions]).to(self.device).squeeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.actor_model.max_action, self.actor_model.max_action
            )
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2).squeeze(-1)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic_model(states, actions)
        current_Q1 = current_Q1.squeeze(-1)
        current_Q2 = current_Q2.squeeze(-1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self._update_counter % self.policy_freq == 0:
            actor_loss = -self.critic_model.Q1_value(
                states, self.actor_model(states)
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft-update target networks
            for param, target_param in zip(
                self.critic_model.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.actor_model.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self._update_counter += 1

        if actor_loss is None:
            return None, critic_loss.item()
        return actor_loss.item(), critic_loss.item()
