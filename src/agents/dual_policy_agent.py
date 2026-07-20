import numpy as np
import torch

from agents.abstract_agent import AbstractAgent
from agents.lqr_agent import LQRAgent
from util.blending_function import BlendingFunction
from util.dreal import (
    dreal_clip_vector,
    dreal_var,
    in_box,
    is_unsat,
    on_boundary,
    quadratic_form,
    require_dreal,
    squared_norm,
)
from util.riccati_solver import RiccatiSolver


class DualPolicyAgent(AbstractAgent):
    """Base for local/global policies with Taylor-order-preserving blending."""

    def __init__(self, config):
        super().__init__(config)
        lqr_config = config.get("LQR")
        if lqr_config is None:
            raise ValueError("LQR configuration is missing")

        lqr_config = dict(lqr_config)
        lqr_config.setdefault("device", self.device)
        self.lqr_agent = LQRAgent(lqr_config)

        self.beta = float(config.get("beta", 0.5))
        if not 0 < self.beta < 1:
            raise ValueError("beta must lie strictly between zero and one")

        self.x_star = torch.zeros(
            self.state_dim, dtype=torch.float32, device=self.device
        )
        self.u_star = torch.zeros(
            self.action_dim, dtype=torch.float32, device=self.device
        )
        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")
        if self.dynamics_fn_dreal is None:
            raise ValueError("dynamics_fn_dreal is required")

        bounds = config.get("r1_bounds")
        if bounds is None:
            raise ValueError("r1_bounds is required")
        self.lb = np.asarray(bounds[0], dtype=np.float64)
        self.ub = np.asarray(bounds[1], dtype=np.float64)
        self.max_action = float(config.get("max_action"))

        c_star = config.get("c_star")
        if c_star is None:
            c_star = self._estimate_domain_of_attraction(self.lqr_check)
            if c_star <= 0:
                raise RuntimeError("Could not certify a positive local LQR level")
        self.c_star = float(c_star)

        self.blending_function = BlendingFunction(
            self.lqr_agent,
            beta_h=self.beta,
            c_star=self.c_star,
            device=self.device,
        )
        self.riccati_solver = RiccatiSolver()
        self.actor_model = None
        self.critic_model = None
        self.timesteps = 0

    def add_transition(self, transition):
        return None

    def update(self):
        return self._train()

    @torch.no_grad()
    def policy(self, state, noise: bool = False) -> np.ndarray:
        action = self.action_tensor(state, noise=noise)
        return action.detach().cpu().numpy().reshape(-1)

    def action_tensor(self, state, noise: bool = False) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        single = state_tensor.ndim == 1
        if single:
            state_tensor = state_tensor.unsqueeze(0)
        global_action = self._get_global_action(state_tensor, noise=noise)
        action = self._get_blended_action(state_tensor, global_action)
        return action.squeeze(0) if single else action

    def _get_blended_action(
        self,
        state_torch: torch.Tensor,
        global_action_torch: torch.Tensor,
    ) -> torch.Tensor:
        local_action = self._get_local_action(state_torch)
        h1 = self.blending_function.get_h1(state_torch).unsqueeze(-1)
        action = local_action + h1 * (global_action_torch - local_action)
        return torch.clamp(action, -self.max_action, self.max_action)

    def _get_local_action(self, state_torch: torch.Tensor) -> torch.Tensor:
        state_torch = torch.as_tensor(
            state_torch, dtype=torch.float32, device=self.device
        )
        single = state_torch.ndim == 1
        if single:
            state_torch = state_torch.unsqueeze(0)
        action = self.u_star.unsqueeze(0) - state_torch @ self.lqr_agent.K.T
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.squeeze(0) if single else action

    def _get_global_action(
        self, state_torch: torch.Tensor, noise: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def symbolic_local_lyapunov(self, x):
        return quadratic_form(x, self.lqr_agent.P_np)

    def symbolic_local_policy(self, x):
        raw = [
            -sum(self.lqr_agent.K_np[i, j] * x[j] for j in range(self.state_dim))
            for i in range(self.action_dim)
        ]
        return dreal_clip_vector(raw, -self.max_action, self.max_action)

    def symbolic_policy(self, x):
        d = require_dreal()
        local_action = self.symbolic_local_policy(x)
        global_action = self.actor_model.forward_dreal(x)
        local_value = self.symbolic_local_lyapunov(x)
        normalized_value = local_value / self.c_star
        h1 = d.tanh(float(self.blending_function.s_factor.item()) * normalized_value)
        blended = [
            local_action[i] + h1 * (global_action[i] - local_action[i])
            for i in range(self.action_dim)
        ]
        return dreal_clip_vector(blended, -self.max_action, self.max_action)

    def load(self, file_path, episode=None):
        raise NotImplementedError

    def save(self, file_path, episode=None):
        raise NotImplementedError

    def _lqr_check(self, level, scale, eps, delta):
        d = require_dreal()
        x = dreal_var(self.state_dim)
        value = self.symbolic_local_lyapunov(x)
        action = self.symbolic_local_policy(x)
        dynamics = self.dynamics_fn_dreal(x, action)
        derivative = sum(
            dynamics[i] * value.Differentiate(x[i]) for i in range(self.state_dim)
        )
        norm_sq = squared_norm(x)

        violation = d.And(
            value <= level,
            norm_sq >= eps**2,
            in_box(x, self.lb, self.ub, scale),
            derivative >= 0,
        )
        boundary_contact = d.And(
            on_boundary(x, self.lb, self.ub, scale),
            value <= level,
        )
        return (
            d.CheckSatisfiability(violation, delta),
            d.CheckSatisfiability(boundary_contact, delta),
        )

    def lqr_check_discrete(self, level, scale=2.0, eps=0.5, delta=1e-3):
        return self._lqr_check(level, scale, eps, delta)

    def lqr_check_continuous(self, level, scale=2.0, eps=1e-5, delta=1e-3):
        return self._lqr_check(level, scale, eps, delta)

    def lqr_check(self, level, scale=2.0, eps=None, delta=1e-4):
        if self.lqr_agent.discrete_discounted:
            eps = 0.5 if eps is None else eps
            return self.lqr_check_discrete(level, scale, eps, delta)
        eps = 1e-5 if eps is None else eps
        return self.lqr_check_continuous(level, scale, eps, delta)

    def _estimate_domain_of_attraction(
        self,
        check_fn,
        c_max=0.95,
        tol=1e-3,
        it_max=40,
    ):
        """Find a verified lower level with an explicitly checked bracket."""
        lower = 0.0
        lower_result = check_fn(lower)
        if not all(is_unsat(result) for result in lower_result):
            return 0.0

        upper = float(c_max)
        upper_result = check_fn(upper)
        if all(is_unsat(result) for result in upper_result):
            return upper

        for _ in range(it_max):
            if upper - lower <= tol:
                break
            midpoint = 0.5 * (lower + upper)
            result = check_fn(midpoint)
            if all(is_unsat(item) for item in result):
                lower = midpoint
            else:
                upper = midpoint
        return lower
