from dataclasses import dataclass
from typing import Callable, Optional

import torch

from util.rk4_step import closed_loop_rk4_step

TensorPolicy = Callable[[torch.Tensor], torch.Tensor]
TensorDynamics = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RolloutResult:
    """Outputs from a batch of continuous-feedback rollouts."""

    running_cost: torch.Tensor
    converged: torch.Tensor
    diverged: torch.Tensor
    final_state: torch.Tensor
    steps: int
    trajectory: Optional[torch.Tensor] = None

    def zubov_target(self, alpha: float) -> torch.Tensor:
        """Return finite targets for converged paths and one otherwise."""
        finite_target = torch.tanh(alpha * self.running_cost)
        return torch.where(
            self.converged,
            finite_target,
            torch.ones_like(finite_target),
        )


@torch.no_grad()
def rollout_running_cost(
    initial_state: torch.Tensor,
    dynamics: TensorDynamics,
    policy: TensorPolicy,
    dt: float,
    max_steps: int,
    convergence_threshold: float,
    running_cost_limit: float,
    state_norm_limit: float = float("inf"),
    store_trajectory: bool = False,
) -> RolloutResult:
    """Integrate ``||x(t)||`` for batched continuous-feedback trajectories.

    A path is finite only when it reaches the convergence ball. Reaching the
    horizon, exceeding the running-cost limit, or exceeding the state-norm
    limit marks the path as non-convergent, for which the Zubov target is one.
    """
    if initial_state.ndim != 2:
        raise ValueError("initial_state must have shape [batch, state_dim]")
    if dt <= 0 or max_steps <= 0:
        raise ValueError("dt and max_steps must be positive")

    x = initial_state.clone()
    batch_size = x.shape[0]
    running_cost = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
    active = torch.ones(batch_size, dtype=torch.bool, device=x.device)
    converged = torch.zeros_like(active)
    diverged = torch.zeros_like(active)
    history = [x.clone()] if store_trajectory else None
    completed_steps = 0

    for step in range(max_steps):
        norms = torch.linalg.vector_norm(x, ord=2, dim=1)
        newly_converged = active & (norms <= convergence_threshold)
        converged |= newly_converged
        active &= ~newly_converged

        newly_diverged = active & (
            (running_cost >= running_cost_limit)
            | (norms >= state_norm_limit)
            | ~torch.isfinite(norms)
        )
        diverged |= newly_diverged
        active &= ~newly_diverged

        if not torch.any(active):
            completed_steps = step
            break

        running_cost[active] += norms[active] * dt
        active_indices = torch.where(active)[0]
        next_active = closed_loop_rk4_step(dynamics, policy, x[active_indices], dt)
        finite_active = torch.isfinite(next_active).all(dim=1)
        failed_indices = active_indices[~finite_active]
        diverged[failed_indices] = True
        active[failed_indices] = False
        finite_indices = active_indices[finite_active]
        x[finite_indices] = next_active[finite_active]

        if history is not None:
            history.append(x.clone())
        completed_steps = step + 1
    else:
        final_norms = torch.linalg.vector_norm(x, ord=2, dim=1)
        final_converged = active & (final_norms <= convergence_threshold)
        converged |= final_converged
        active &= ~final_converged
        # Any remaining active trajectories reached the finite horizon.
        diverged |= active

    trajectory = torch.stack(history, dim=1) if history is not None else None
    return RolloutResult(
        running_cost=running_cost,
        converged=converged,
        diverged=diverged,
        final_state=x,
        steps=completed_steps,
        trajectory=trajectory,
    )
