from pathlib import Path

import numpy as np
import torch

from agents.abstract_agent import AbstractAgent
from models.td3_actor import TD3Actor
from models.td3_critic import TD3Critic
from trainers.td3_trainer import TD3Trainer
from util.checkpoints import load_checkpoint, validate_checkpoint


class TD3Agent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.max_action = config.get("max_action")
        self.gamma = config.get("gamma")
        self.tau = config.get("tau")
        self.policy_freq = config.get("policy_freq")
        self.batch_size = config.get("batch_size")

        # Noise used in target updates (for training)
        self.policy_noise = config.get("policy_noise") * self.max_action
        self.noise_clip = config.get("noise_clip") * self.max_action

        # Exploration noise added to actions when interacting with the environment
        self.expl_noise = config.get("expl_noise", 0.1)

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

        self.trainer = TD3Trainer(
            buffer=self._replay_buffer,
            actor=self.actor_model,
            critic=self.critic_model,
            gamma=self.gamma,
            tau=self.tau,
            policy_freq=self.policy_freq,
            batch_size=self.batch_size,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            device=self.device,
        )

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

    def policy(self, state, noise=True) -> np.array:
        """
        Get the action to take based on the current state, adding Gaussian exploration noise.
        This mimics the original code:
          action = actor(state) + N(0, expl_noise * max_action)
        and clips the result between -max_action and max_action.
        """
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor_model(state_t.unsqueeze(0))
        action = action.cpu().numpy().flatten()
        if noise:
            noise = np.random.normal(
                0, self.expl_noise * self.max_action, size=action.shape
            )
            action += noise

        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def action_tensor(self, state) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        single = state_tensor.ndim == 1
        if single:
            state_tensor = state_tensor.unsqueeze(0)
        action = self.actor_model(state_tensor)
        return action.squeeze(0) if single else action

    def save(self, file_path="./saved_models/", episode=None) -> None:
        """
        Save the actor and critic networks.
        """
        directory = Path(file_path)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "latest" if episode is None else str(episode)
        torch.save(
            {
                "agent": "td3",
                "episode": episode,
                "actor": self.actor_model.state_dict(),
                "critic": self.critic_model.state_dict(),
                "actor_target": self.trainer.actor_target.state_dict(),
                "critic_target": self.trainer.critic_target.state_dict(),
                "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
                "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
                "update_counter": self.trainer._update_counter,
            },
            directory / f"checkpoint_{suffix}.pt",
        )

    def load(self, file_path="./saved_models/", episode=None) -> None:
        """
        Load the actor and critic networks, and synchronize the trainer's target networks.
        """
        directory = Path(file_path)
        suffix = "latest" if episode is None else str(episode)
        checkpoint_path = directory / f"checkpoint_{suffix}.pt"
        if checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            validate_checkpoint(checkpoint, expected_agent="td3")
            self.actor_model.load_state_dict(checkpoint["actor"])
            self.critic_model.load_state_dict(checkpoint["critic"])
            self.trainer.actor_target.load_state_dict(
                checkpoint.get("actor_target", checkpoint["actor"])
            )
            self.trainer.critic_target.load_state_dict(
                checkpoint.get("critic_target", checkpoint["critic"])
            )
            if "actor_optimizer" in checkpoint:
                self.trainer.actor_optimizer.load_state_dict(
                    checkpoint["actor_optimizer"]
                )
            if "critic_optimizer" in checkpoint:
                self.trainer.critic_optimizer.load_state_dict(
                    checkpoint["critic_optimizer"]
                )
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
        self.trainer.actor_target.load_state_dict(self.actor_model.state_dict())
        self.trainer.critic_target.load_state_dict(self.critic_model.state_dict())
