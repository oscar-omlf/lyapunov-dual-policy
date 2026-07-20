from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR

from trainers.abstract_trainer import Trainer
from util.dreal import (
    dreal_var,
    in_box,
    is_unsat,
    on_boundary,
    require_dreal,
    squared_norm,
)
from util.sampling import (
    sample_in_region_torch,
    sample_out_of_region_torch,
)
from util.simulation import RolloutResult, rollout_running_cost


class LyapunovTrainer(Trainer):
    """Standalone physics-informed Lyapunov actor-critic baseline."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        lr: float,
        alpha: float,
        batch_size: int,
        num_paths_sampled: int,
        norm_threshold: float,
        integ_threshold: float,
        dt: float,
        dynamics_fn: Callable,
        dynamics_fn_dreal: Callable,
        state_dim: int,
        r1_bounds: list,
        run_dir: str,
        device: str,
        rollout_steps: int = 3000,
        rollout_refresh_interval: int = 10,
        plot_interval: int = 0,
        plot_grid_size: int = 201,
    ):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic
        self.alpha_zubov = float(alpha)
        self.batch_size = int(batch_size)
        self.num_paths_sampled = int(num_paths_sampled)
        self.norm_threshold = float(norm_threshold)
        self.integ_threshold = float(integ_threshold)
        self.dt = float(dt)
        self.dynamics_fn = dynamics_fn
        self.dynamics_fn_dreal = dynamics_fn_dreal
        self.state_dim = int(state_dim)
        self.lb, self.ub = (
            np.asarray(r1_bounds[0]),
            np.asarray(r1_bounds[1]),
        )
        self.run_dir = Path(run_dir) if run_dir else None
        self.device = device
        self.rollout_steps = int(rollout_steps)
        self.rollout_refresh_interval = int(rollout_refresh_interval)
        self.plot_interval = int(plot_interval)
        self.plot_grid_size = int(plot_grid_size)
        if self.rollout_refresh_interval <= 0:
            raise ValueError("rollout_refresh_interval must be positive")

        self.lb_tensor = torch.as_tensor(
            self.lb, dtype=torch.float32, device=self.device
        )
        self.ub_tensor = torch.as_tensor(
            self.ub, dtype=torch.float32, device=self.device
        )
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=lr
        )
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)
        self.timesteps = 0
        self._rollout_states: Optional[torch.Tensor] = None
        self._rollout_targets: Optional[torch.Tensor] = None
        self.last_rollout: Optional[RolloutResult] = None
        self.last_losses = {}
        self.last_verification = {}

    def _action(self, state):
        return self.actor_model(state)

    def _refresh_rollout_targets(self):
        states = sample_in_region_torch(
            self.num_paths_sampled,
            self.lb_tensor,
            self.ub_tensor,
            self.device,
        )
        rollout = rollout_running_cost(
            initial_state=states,
            dynamics=self.dynamics_fn,
            policy=self._action,
            dt=self.dt,
            max_steps=self.rollout_steps,
            convergence_threshold=self.norm_threshold,
            running_cost_limit=self.integ_threshold,
        )
        self._rollout_states = states
        self._rollout_targets = rollout.zubov_target(self.alpha_zubov)
        self.last_rollout = rollout

    def _physics_samples(self, counter_examples):
        if not counter_examples:
            return sample_in_region_torch(
                self.batch_size,
                self.lb_tensor,
                self.ub_tensor,
                self.device,
            )
        counterexample_tensor = torch.as_tensor(
            counter_examples, dtype=torch.float32, device=self.device
        ).reshape(-1, self.state_dim)
        counterexample_tensor = counterexample_tensor[-self.batch_size :]
        random_count = self.batch_size - counterexample_tensor.shape[0]
        random_samples = sample_in_region_torch(
            random_count,
            self.lb_tensor,
            self.ub_tensor,
            self.device,
        )
        return torch.cat((counterexample_tensor, random_samples), dim=0)

    def train(self, counter_examples=None, normalize_gradients=False):
        if (
            self._rollout_states is None
            or self.timesteps % self.rollout_refresh_interval == 0
        ):
            self._refresh_rollout_targets()

        zero = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        loss_zero = 5.0 * self.critic_model(zero).square().mean()
        loss_rollout = F.mse_loss(
            self.critic_model(self._rollout_states),
            self._rollout_targets,
        )

        physics_states = self._physics_samples(counter_examples).detach()
        physics_states.requires_grad_(True)
        values = self.critic_model(physics_states)
        (value_gradient,) = torch.autograd.grad(
            values.sum(), physics_states, create_graph=True
        )
        actions = self.actor_model(physics_states)
        vector_field = self.dynamics_fn(physics_states, actions)
        phi = torch.linalg.vector_norm(physics_states, ord=2, dim=1)
        residual = (
            torch.sum(value_gradient * vector_field.detach(), dim=1)
            + self.alpha_zubov * (1.0 - values.square()) * phi
        )
        loss_pde = residual.square().mean()

        actor_gradient = value_gradient.detach()
        if normalize_gradients:
            actor_gradient = actor_gradient / torch.linalg.vector_norm(
                actor_gradient, ord=2, dim=1, keepdim=True
            ).clamp_min(1e-8)
        loss_control = torch.sum(actor_gradient * vector_field, dim=1).mean()

        boundary_states = sample_out_of_region_torch(
            self.batch_size,
            self.lb_tensor,
            self.ub_tensor,
            scale=2.0,
            device=self.device,
        )
        boundary_values = self.critic_model(boundary_states)
        loss_boundary = 5.0 * F.l1_loss(
            boundary_values, torch.ones_like(boundary_values)
        )

        actor_loss = loss_control
        critic_loss = loss_zero + loss_rollout + loss_pde + loss_boundary
        total_loss = 0.5 * actor_loss + critic_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.timesteps += 1
        self.last_losses = {
            "zero": loss_zero.item(),
            "rollout": loss_rollout.item(),
            "pde": loss_pde.item(),
            "control": loss_control.item(),
            "boundary": loss_boundary.item(),
            "total": total_loss.item(),
        }
        if self.plot_interval and self.timesteps % self.plot_interval == 0:
            self.plot_level_set_and_trajectories()
        return actor_loss.item(), critic_loss.item()

    @torch.no_grad()
    def simulate_trajectories(self, x, max_steps=None):
        rollout = rollout_running_cost(
            initial_state=x,
            dynamics=self.dynamics_fn,
            policy=self._action,
            dt=self.dt,
            max_steps=max_steps or self.rollout_steps,
            convergence_threshold=self.norm_threshold,
            running_cost_limit=self.integ_threshold,
            store_trajectory=True,
        )
        return rollout.trajectory, rollout.running_cost, rollout.converged

    @torch.no_grad()
    def plot_level_set_and_trajectories(self):
        if self.state_dim != 2 or self.run_dir is None:
            return
        import matplotlib.pyplot as plt

        x_axis = np.linspace(self.lb[0] * 2, self.ub[0] * 2, self.plot_grid_size)
        y_axis = np.linspace(self.lb[1] * 2, self.ub[1] * 2, self.plot_grid_size)
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        points = torch.as_tensor(
            np.stack((grid_x.ravel(), grid_y.ravel()), axis=1),
            dtype=torch.float32,
            device=self.device,
        )
        values = self.critic_model(points).cpu().numpy().reshape(grid_x.shape)
        initial_states = sample_in_region_torch(
            5, self.lb_tensor, self.ub_tensor, self.device
        )
        trajectories, _, _ = self.simulate_trajectories(
            initial_states, max_steps=min(self.rollout_steps, 1000)
        )

        figure, axis = plt.subplots(figsize=(6, 5))
        contour = axis.contourf(grid_x, grid_y, values, levels=50, cmap="viridis")
        figure.colorbar(contour, ax=axis)
        for trajectory in trajectories.cpu().numpy():
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=0.7,
                color="red",
            )
        axis.set(
            title="Standalone Zubov candidate",
            xlabel="x[0]",
            ylabel="x[1]",
        )
        axis.set_aspect("equal")
        output_dir = self.run_dir / "level_sets"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_dir / f"level_set_{self.timesteps}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _sanity_network(self):
        d = require_dreal()
        x_numpy = np.random.randn(self.state_dim)
        x_symbolic = dreal_var(self.state_dim)
        y_torch = self.actor_model(
            torch.as_tensor(x_numpy, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
        )[0, 0].item()
        y_symbolic = self.actor_model.forward_dreal(x_symbolic)[0]
        formula = d.And(
            *[x_symbolic[i] == x_numpy[i] for i in range(self.state_dim)],
            y_symbolic >= y_torch - 1e-3,
            y_symbolic <= y_torch + 1e-3,
        )
        assert d.CheckSatisfiability(formula, 1e-4) is not None

    def _sanity_dynamics(self):
        d = require_dreal()
        x_numpy = np.random.randn(self.state_dim)
        u_numpy = np.random.randn(1)
        x_symbolic = dreal_var(self.state_dim)
        u_symbolic = dreal_var(1, "u")
        f_torch = self.dynamics_fn(
            torch.as_tensor(x_numpy, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(u_numpy, dtype=torch.float32).unsqueeze(0),
        )[0].numpy()
        f_symbolic = self.dynamics_fn_dreal(x_symbolic, u_symbolic)
        constraints = [x_symbolic[i] == x_numpy[i] for i in range(self.state_dim)]
        constraints.append(u_symbolic[0] == u_numpy[0])
        constraints.extend(
            f_symbolic[i] >= f_torch[i] - 1e-3 for i in range(self.state_dim)
        )
        constraints.extend(
            f_symbolic[i] <= f_torch[i] + 1e-3 for i in range(self.state_dim)
        )
        assert d.CheckSatisfiability(d.And(*constraints), 1e-4) is not None

    def check_lyapunov(self, level=0.9, scale=2.0, eps=0.5, delta=1e-4):
        passed, model = self.check_lyapunov_with_ce(
            level=level, scale=scale, eps=eps, delta=delta
        )
        return (None, None) if passed else (model, model)

    def check_lyapunov_with_ce(self, level=0.9, scale=2.0, eps=0.5, delta=1e-4):
        """Verify the learned candidate on an annulus outside ``eps``."""
        d = require_dreal()
        with torch.no_grad():
            value_at_origin = self.critic_model(
                torch.zeros(
                    (1, self.state_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            ).item()

        x = dreal_var(self.state_dim)
        action = self.actor_model.forward_dreal(x)
        vector_field = self.dynamics_fn_dreal(x, action)
        candidate = self.critic_model.forward_dreal(x)[0]
        derivative = sum(
            vector_field[i] * candidate.Differentiate(x[i])
            for i in range(self.state_dim)
        )
        norm_sq = squared_norm(x)
        violation = d.And(
            norm_sq >= eps**2,
            in_box(x, self.lb, self.ub, scale),
            candidate <= level,
            d.Or(derivative >= 0, candidate <= value_at_origin),
        )
        violation_result = d.CheckSatisfiability(violation, delta)
        if not is_unsat(violation_result):
            self.last_verification = {
                "passed": False,
                "scope": "annular",
                "reason": "candidate",
                "eps": eps,
                "level": level,
                "delta": delta,
            }
            return False, violation_result

        boundary_contact = d.And(
            on_boundary(x, self.lb, self.ub, scale),
            candidate <= level,
        )
        boundary_result = d.CheckSatisfiability(boundary_contact, delta)
        passed = is_unsat(boundary_result)
        self.last_verification = {
            "passed": passed,
            "scope": "annular",
            "reason": None if passed else "outer_boundary",
            "eps": eps,
            "level": level,
            "delta": delta,
        }
        return (True, None) if passed else (False, boundary_result)
