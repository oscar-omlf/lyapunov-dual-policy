from typing import Tuple

import numpy as np
import torch

from agents.lqr_agent import LQRAgent


class BlendingFunction:
    def __init__(
        self,
        lqr_agent: LQRAgent,
        beta_h: float = 0.9,
        c_star: float = 1.0,
        device: str = "cpu",
    ):
        if c_star <= 0:
            raise ValueError("c_star must be strictly positive")
        self.P_lqr = lqr_agent.P
        self.setpoint_x = torch.zeros(1, self.P_lqr.shape[0], device=device)
        self.c_star = c_star
        self.c_star_torch = torch.tensor(c_star, dtype=torch.float32, device=device)
        self.device = device

        if not (0 < beta_h < 1):
            raise ValueError("beta_h for blending must be between 0 and 1 (exclusive)")

        s_val = float(np.arctanh(beta_h))
        self.s_factor = torch.tensor(s_val, dtype=torch.float32, device=device)

    def _calculate_V_local(self, state_torch: torch.Tensor) -> torch.Tensor:
        """Calculates V_local(x) = (x-x*)^T P (x-x*)"""
        if state_torch.ndim == 1:
            state_torch_batch = state_torch.unsqueeze(0)
        else:
            state_torch_batch = state_torch

        delta_x = state_torch_batch - self.setpoint_x
        term1 = torch.matmul(delta_x, self.P_lqr)
        v_val = torch.sum(term1 * delta_x, dim=1)
        return v_val

    def get_lyapunov_value(self, state_torch: torch.Tensor) -> torch.Tensor:
        """Calculates and returns V_local(x)"""
        V_local_val = self._calculate_V_local(state_torch)
        return V_local_val

    def get_normalized_lyapunov_value(self, state_torch: torch.Tensor) -> torch.Tensor:
        """Calculates and returns v(x) = V_local(x) / c_star"""
        V_local_val = self._calculate_V_local(state_torch)

        v_x_normalized = V_local_val / self.c_star_torch
        v_x_normalized = torch.clamp(v_x_normalized, min=0.0)

        return v_x_normalized

    def get_h1(self, state_torch: torch.Tensor) -> torch.Tensor:
        """Calculates and returns h1(x) = tanh(s * v(x))"""
        v_x = self.get_normalized_lyapunov_value(state_torch)
        h1_val = torch.tanh(self.s_factor * v_x)
        return h1_val

    def get_h2(self, state_torch: torch.Tensor) -> torch.Tensor:
        """Calculates and returns h2(x) = tanh(s * v(x)^(3/2))"""
        v_x = self.get_normalized_lyapunov_value(state_torch)
        # Ensure v_x is non-negative for the fractional power
        h2_val = torch.tanh(self.s_factor * v_x ** (3.0 / 2.0))
        return h2_val

    def get_all_blending_terms(
        self, state_torch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates and returns v(x), h1(x), h2(x) efficiently"""
        v_x = self.get_normalized_lyapunov_value(state_torch)
        h1_val = torch.tanh(self.s_factor * v_x)
        h2_val = torch.tanh(self.s_factor * v_x ** (3.0 / 2.0))
        return v_x, h1_val, h2_val
