import numpy as np
import torch

from util.sampling import sample_in_region_torch


def calculate_lqr_roa_parameters(P: np.ndarray, c_star: float, scale: float = 2.0):
    eigenvalues = np.linalg.eigvalsh(P)
    if np.any(eigenvalues <= 0):
        raise ValueError("Matrix P must be positive definite.")

    r1_max_radius = np.sqrt(c_star / np.min(eigenvalues))

    r2_radius = scale * r1_max_radius

    L = torch.linalg.cholesky(torch.as_tensor(P, dtype=torch.float32))
    L_inv = torch.linalg.inv(L)

    print(f"LQR RoA defined by P matrix and c*={c_star:.4f}")
    print(f"Max radius of R1 (ellipsoid): {r1_max_radius:.4f}")
    print(f"Radius of R2 (circle): {r2_radius:.4f}")

    return L_inv, r2_radius


@torch.no_grad()
def estimate_doa(
    level,
    lb,
    ub,
    critic_model=None,
    device="cpu",
    n_samples=50000,
    value_fn=None,
):
    """Estimate the fraction of a box contained in a candidate sublevel set."""
    if value_fn is None:
        if critic_model is None:
            raise ValueError("critic_model or value_fn is required")
        value_fn = critic_model
    pts = sample_in_region_torch(
        n_samples,
        torch.as_tensor(lb, device=device, dtype=torch.float32),
        torch.as_tensor(ub, device=device, dtype=torch.float32),
        device,
    )
    W = value_fn(pts).reshape(-1)
    return (W <= level).float().mean().item()
