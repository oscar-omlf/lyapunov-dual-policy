from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR

from agents.dual_policy_agent import DualPolicyAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from trainers.las_lyapunov_trainer import LAS_LyapunovAC_Trainer
from util.checkpoints import load_checkpoint, validate_checkpoint


class LAS_LyapunovAgent(DualPolicyAgent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.alpha_zubov = config.get("alpha")
        self.lr = config.get("lr")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.dt = config.get("dt")
        self.r1_bounds = config.get("r1_bounds")
        self.rollout_steps = int(config.get("rollout_steps", 3000))
        self.rollout_refresh_interval = int(config.get("rollout_refresh_interval", 10))
        self.plot_interval = int(config.get("plot_interval", 0))
        self.plot_grid_size = int(config.get("plot_grid_size", 201))

        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")

        self.actor_model = LyapunovActor(
            self.state_dim,
            actor_hidden_sizes,
            self.action_dim,
            max_action=self.max_action,
        ).to(device=self.device)
        self.critic_model = LyapunovCritic(self.state_dim, critic_hidden_sizes).to(
            device=self.device
        )

        self.optimizer = torch.optim.Adam(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()),
            lr=self.lr,
        )
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)

        self.run_dir = config.get("run_dir")

        self.trainer = LAS_LyapunovAC_Trainer(
            agent=self,
            alpha_zubov=self.alpha_zubov,
            batch_size=self.batch_size,
            num_paths_sampled=self.num_paths_sampled,
            norm_threshold=self.norm_threshold,
            integ_threshold=self.integ_threshold,
            dt=self.dt,
            dynamics_fn=self.dynamics_fn,
            dynamics_fn_dreal=self.dynamics_fn_dreal,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            r1_bounds=self.r1_bounds,
            run_dir=self.run_dir,
            device=self.device,
            rollout_steps=self.rollout_steps,
            rollout_refresh_interval=self.rollout_refresh_interval,
            plot_interval=self.plot_interval,
            plot_grid_size=self.plot_grid_size,
        )

    def _get_global_action(
        self, state_torch: torch.Tensor, noise: bool = False
    ) -> torch.Tensor:
        return self.actor_model(state_torch)

    def get_composite_W_value(self, state_torch: torch.Tensor) -> torch.Tensor:
        state_torch = torch.as_tensor(
            state_torch, dtype=torch.float32, device=self.device
        )
        single = state_torch.ndim == 1
        if single:
            state_torch = state_torch.unsqueeze(0)

        W_glo = self.critic_model(state_torch)
        V_loc = self.blending_function.get_lyapunov_value(state_torch)
        W_loc = torch.tanh(self.alpha_zubov * V_loc)
        h2_val = self.blending_function.get_h2(state_torch)
        W_comp_val = W_loc + h2_val * (W_glo - W_loc)
        return W_comp_val.squeeze(0) if single else W_comp_val

    def symbolic_W_value(self, x):
        from util.dreal import require_dreal

        d = require_dreal()
        W_global = self.critic_model.forward_dreal(x)[0]
        V_local = self.symbolic_local_lyapunov(x)
        W_local = d.tanh(self.alpha_zubov * V_local)
        normalized_value = V_local / self.c_star
        h2 = d.tanh(
            float(self.blending_function.s_factor.item())
            * normalized_value
            * d.sqrt(normalized_value)
        )
        return W_local + h2 * (W_global - W_local)

    def add_transition(self, transition: tuple) -> None:
        return None

    def update(self, counter_examples: list = None, normalize_gradients: bool = False):
        loss = self.trainer.train(
            counter_examples=counter_examples, normalize_gradients=normalize_gradients
        )
        return loss

    def save(self, file_path, episode=None):
        directory = Path(file_path)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "latest" if episode is None else str(episode)
        torch.save(
            {
                "agent": "ldp",
                "episode": episode,
                "actor": self.actor_model.state_dict(),
                "critic": self.critic_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "beta": self.beta,
                "c_star": self.c_star,
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
            validate_checkpoint(
                checkpoint,
                expected_agent="ldp",
                beta=self.beta,
                c_star=self.c_star,
            )
            self.actor_model.load_state_dict(checkpoint["actor"])
            self.critic_model.load_state_dict(checkpoint["critic"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.trainer.timesteps = int(
                checkpoint.get("trainer_step", self.trainer.timesteps)
            )
            return

        # Backward compatibility with thesis-era two-file checkpoints.
        legacy_suffix = str(episode)
        self.actor_model.load_state_dict(
            load_checkpoint(directory / f"actor_model_{legacy_suffix}.pth", self.device)
        )
        self.critic_model.load_state_dict(
            load_checkpoint(
                directory / f"critic_model_{legacy_suffix}.pth", self.device
            )
        )
