from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from agents.abstract_agent import AbstractAgent
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


class LAS_LyapunovAC_Trainer(Trainer):
    """Physics-informed trainer and verifier for the composite LDP system."""

    def __init__(
        self,
        agent: AbstractAgent,
        alpha_zubov: float,
        batch_size: int,
        num_paths_sampled: int,
        norm_threshold: float,
        integ_threshold: float,
        dt: float,
        dynamics_fn: callable,
        dynamics_fn_dreal: callable,
        state_dim: int,
        action_dim: int,
        r1_bounds: list,
        run_dir: str,
        device: str,
        rollout_steps: int = 3000,
        rollout_refresh_interval: int = 10,
        plot_interval: int = 0,
        plot_grid_size: int = 201,
    ):
        super().__init__()
        self.agent = agent
        self.alpha_zubov = float(alpha_zubov)
        self.batch_size = int(batch_size)
        self.num_paths_sampled = int(num_paths_sampled)
        self.norm_threshold = float(norm_threshold)
        self.integ_threshold = float(integ_threshold)
        self.dt = float(dt)
        self.dynamics_fn = dynamics_fn
        self.dynamics_fn_dreal = dynamics_fn_dreal
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
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
        self.timesteps = 0
        self._rollout_states: Optional[torch.Tensor] = None
        self._rollout_targets: Optional[torch.Tensor] = None
        self.last_rollout: Optional[RolloutResult] = None
        self.last_losses = {}
        self.last_verification = {}
        self._local_bridge_cache = {}

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
            policy=lambda x: self.agent.action_tensor(x, noise=False),
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

        rollout_states = self._rollout_states
        rollout_targets = self._rollout_targets

        zero = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        loss_zero = 5.0 * self.agent.critic_model(zero).square().mean()

        rollout_values = self.agent.get_composite_W_value(rollout_states)
        loss_rollout = F.mse_loss(rollout_values, rollout_targets)

        physics_states = self._physics_samples(counter_examples).detach()
        physics_states.requires_grad_(True)
        values = self.agent.get_composite_W_value(physics_states)
        (value_gradient,) = torch.autograd.grad(
            values.sum(), physics_states, create_graph=True
        )

        actions = self.agent.action_tensor(physics_states, noise=False)
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
        boundary_values = self.agent.get_composite_W_value(boundary_states)
        loss_boundary = 5.0 * F.l1_loss(
            boundary_values, torch.ones_like(boundary_values)
        )

        actor_loss = loss_control
        critic_loss = loss_zero + loss_rollout + loss_pde + loss_boundary
        total_loss = 0.5 * actor_loss + critic_loss
        self.agent.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.agent.optimizer.step()
        self.agent.scheduler.step()
        self._local_bridge_cache.clear()

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
    def simulate_trajectories(self, x: torch.Tensor, max_steps: Optional[int] = None):
        """Compatibility wrapper returning trajectory, cost, and convergence."""
        rollout = rollout_running_cost(
            initial_state=x,
            dynamics=self.dynamics_fn,
            policy=lambda state: self.agent.action_tensor(state, noise=False),
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
        points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
        point_tensor = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        values = (
            self.agent.get_composite_W_value(point_tensor)
            .cpu()
            .numpy()
            .reshape(grid_x.shape)
        )

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
            title="Composite Zubov candidate",
            xlabel="x[0]",
            ylabel="x[1]",
            xlim=(x_axis[0], x_axis[-1]),
            ylim=(y_axis[0], y_axis[-1]),
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

    def check_lyapunov_with_ce(
        self,
        level=0.9,
        scale=2.0,
        eps=0.25,
        delta=1e-4,
        bridge_level=None,
        local_margin=1e-4,
    ):
        """Verify a piecewise local/composite Lyapunov certificate.

        ``V_local`` proves convergence inside the bridge. The learned
        composite candidate proves decrease outside it. This closes the
        historical unverified epsilon-ball gap.
        """
        d = require_dreal()
        if bridge_level is None:
            lambda_min = np.linalg.eigvalsh(self.agent.lqr_agent.P_np).min()
            bridge_level = float(lambda_min * eps**2)
        if bridge_level <= 0:
            raise ValueError("bridge_level must be positive")

        x = dreal_var(self.state_dim)
        action = self.agent.symbolic_policy(x)
        candidate = self.agent.symbolic_W_value(x)
        local_value = self.agent.symbolic_local_lyapunov(x)
        vector_field = self.dynamics_fn_dreal(x, action)
        candidate_derivative = sum(
            vector_field[i] * candidate.Differentiate(x[i])
            for i in range(self.state_dim)
        )
        local_derivative = sum(
            vector_field[i] * local_value.Differentiate(x[i])
            for i in range(self.state_dim)
        )
        norm_sq = squared_norm(x)
        domain = in_box(x, self.lb, self.ub, scale)

        cache_key = (
            float(scale),
            float(bridge_level),
            float(local_margin),
            float(delta),
        )
        cached_local = self._local_bridge_cache.get(cache_key)
        if cached_local is None:
            local_violation = d.And(
                domain,
                local_value <= bridge_level,
                local_derivative + local_margin * norm_sq > 0,
            )
            local_result = d.CheckSatisfiability(local_violation, delta)
            bridge_boundary_contact = d.And(
                on_boundary(x, self.lb, self.ub, scale),
                local_value <= bridge_level,
            )
            bridge_boundary_result = d.CheckSatisfiability(
                bridge_boundary_contact, delta
            )
            cached_local = (local_result, bridge_boundary_result)
            self._local_bridge_cache[cache_key] = cached_local
        else:
            local_result, bridge_boundary_result = cached_local

        if not is_unsat(local_result):
            self.last_verification = {
                "passed": False,
                "reason": "local_bridge_decrease",
                "level": float(level),
                "bridge_level": bridge_level,
                "eps": eps,
                "delta": delta,
            }
            return False, local_result

        if not is_unsat(bridge_boundary_result):
            self.last_verification = {
                "passed": False,
                "reason": "local_bridge_boundary",
                "level": float(level),
                "bridge_level": bridge_level,
                "eps": eps,
                "delta": delta,
            }
            return False, bridge_boundary_result

        outer_violation = d.And(
            domain,
            local_value >= bridge_level,
            candidate <= level,
            d.Or(candidate_derivative >= 0, candidate <= 0),
        )
        outer_result = d.CheckSatisfiability(outer_violation, delta)
        if not is_unsat(outer_result):
            self.last_verification = {
                "passed": False,
                "reason": "composite_candidate",
                "level": float(level),
                "bridge_level": bridge_level,
                "eps": eps,
                "delta": delta,
            }
            return False, outer_result

        boundary_contact = d.And(
            on_boundary(x, self.lb, self.ub, scale),
            candidate <= level,
        )
        boundary_result = d.CheckSatisfiability(boundary_contact, delta)
        if not is_unsat(boundary_result):
            self.last_verification = {
                "passed": False,
                "reason": "outer_boundary",
                "level": float(level),
                "bridge_level": bridge_level,
                "eps": eps,
                "delta": delta,
            }
            return False, boundary_result

        self.last_verification = {
            "passed": True,
            "reason": None,
            "level": float(level),
            "bridge_level": bridge_level,
            "eps": eps,
            "local_margin": local_margin,
            "delta": delta,
        }
        return True, None
