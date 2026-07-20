from pathlib import Path

import torch

from agents.abstract_agent import AbstractAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from trainers.lyapunov_trainer import LyapunovTrainer
from util.checkpoints import load_checkpoint, validate_checkpoint


class LyapunovAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.alpha = config.get("alpha")
        self.lr = config.get("lr")
        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.dt = config.get("dt")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.r1_bounds = config.get("r1_bounds")
        self.rollout_steps = int(config.get("rollout_steps", 3000))
        self.rollout_refresh_interval = int(config.get("rollout_refresh_interval", 10))
        self.plot_interval = int(config.get("plot_interval", 0))
        self.plot_grid_size = int(config.get("plot_grid_size", 201))

        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")

        self.max_action = config.get("max_action")

        self.actor_model = LyapunovActor(
            self.state_dim,
            actor_hidden_sizes,
            self.action_dim,
            max_action=self.max_action,
        ).to(device=self.device)
        self.critic_model = LyapunovCritic(self.state_dim, critic_hidden_sizes).to(
            device=self.device
        )

        self.run_dir = config.get("run_dir")

        self.trainer = LyapunovTrainer(
            actor=self.actor_model,
            critic=self.critic_model,
            lr=self.lr,
            alpha=self.alpha,
            batch_size=self.batch_size,
            num_paths_sampled=self.num_paths_sampled,
            norm_threshold=self.norm_threshold,
            integ_threshold=self.integ_threshold,
            dt=self.dt,
            dynamics_fn=self.dynamics_fn,
            dynamics_fn_dreal=self.dynamics_fn_dreal,
            state_dim=self.state_dim,
            r1_bounds=self.r1_bounds,
            run_dir=self.run_dir,
            device=self.device,
            rollout_steps=self.rollout_steps,
            rollout_refresh_interval=self.rollout_refresh_interval,
            plot_interval=self.plot_interval,
            plot_grid_size=self.plot_grid_size,
        )

    def add_transition(self, transition: tuple) -> None:
        pass

    def update(self, counter_examples: list = None, normalize_gradients: bool = False):
        loss = self.trainer.train(
            counter_examples=counter_examples, normalize_gradients=normalize_gradients
        )
        return loss

    @torch.no_grad()
    def policy(self, state, noise=False):
        return self.action_tensor(state).detach().cpu().numpy().reshape(-1)

    def action_tensor(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_tensor.ndim == 1
        if single:
            state_tensor = state_tensor.unsqueeze(0)
        action = self.actor_model(state_tensor)
        return action.squeeze(0) if single else action

    def get_W_value(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.critic_model(state_tensor)

    def save(self, file_path, episode=None):
        directory = Path(file_path)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "latest" if episode is None else str(episode)
        torch.save(
            {
                "agent": "lac",
                "episode": episode,
                "actor": self.actor_model.state_dict(),
                "critic": self.critic_model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "trainer_step": self.trainer.timesteps,
            },
            directory / f"checkpoint_{suffix}.pt",
        )

    def load(self, file_path, episode=None):
        directory = Path(file_path)
        suffix = "latest" if episode is None else str(episode)
        checkpoint_path = directory / f"checkpoint_{suffix}.pt"
        if checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            validate_checkpoint(checkpoint, expected_agent="lac")
            self.actor_model.load_state_dict(checkpoint["actor"])
            self.critic_model.load_state_dict(checkpoint["critic"])
            if "optimizer" in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
            self.trainer.timesteps = int(
                checkpoint.get("trainer_step", self.trainer.timesteps)
            )
            return
        legacy_suffix = str(episode)
        self.actor_model.load_state_dict(
            load_checkpoint(directory / f"actor_model_{legacy_suffix}.pth", self.device)
        )
        self.critic_model.load_state_dict(
            load_checkpoint(
                directory / f"critic_model_{legacy_suffix}.pth", self.device
            )
        )
