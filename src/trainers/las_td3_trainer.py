import torch
import torch.nn.functional as F

from agents.abstract_agent import AbstractAgent, ReplayBuffer


class LAS_TD3Trainer:
    def __init__(
        self,
        agent: AbstractAgent,
        buffer: ReplayBuffer,
        gamma: float,
        tau: float,
        policy_freq: int,
        batch_size: int,
        policy_noise: float,
        noise_clip: float,
        device: str,
    ):
        self.agent = agent
        self.buffer = buffer
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = device

        self._update_counter = 1

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        transitions = self.buffer.sample(self.batch_size)

        states = torch.stack([t[0] for t in transitions]).to(self.device)
        actions = torch.stack([t[1] for t in transitions]).to(self.device)
        rewards = torch.stack([t[2] for t in transitions]).to(self.device)
        next_states = torch.stack([t[3] for t in transitions]).to(self.device)
        dones = torch.stack([t[4] for t in transitions]).to(self.device)

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.ndim == 1:
            dones = dones.unsqueeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.agent.actor_target(next_states) + noise).clamp(
                -self.agent.max_action, self.agent.max_action
            )
            next_actions_target_blended = self.agent._get_blended_action(
                state_torch=next_states,
                global_action_torch=next_actions,
            )

            target_Q1, target_Q2 = self.agent.get_composite_Q_values(
                state_torch=next_states,
                action_torch=next_actions_target_blended,
                use_target_critic=True,
            )
            target_Q_comp = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q_comp

        current_Q1, current_Q2 = self.agent.get_composite_Q_values(
            state_torch=states, action_torch=actions, use_target_critic=False
        )

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        actor_loss_val = None
        if self._update_counter % self.policy_freq == 0:
            actor_actions_global = self.agent.actor_model(states)

            actor_actions_blended = self.agent._get_blended_action(
                state_torch=states,
                global_action_torch=actor_actions_global,
            )

            q1_for_actor_loss = self.agent.get_composite_Q1_value(
                state_torch=states, action_torch=actor_actions_blended
            )
            actor_loss = -q1_for_actor_loss.mean()

            self.agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            for param, target_param in zip(
                self.agent.critic_model.parameters(),
                self.agent.critic_target.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.agent.actor_model.parameters(),
                self.agent.actor_target.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self._update_counter += 1

        return actor_loss_val, critic_loss.item()
