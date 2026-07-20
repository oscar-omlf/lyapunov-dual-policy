import numpy as np
import torch


def sample_in_region(num_samples: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.random.uniform(lb, ub, size=(num_samples, lb.shape[0]))


def sample_out_of_region(
    num_samples: int, lb: np.ndarray, ub: np.ndarray, scale: float = 2.0
) -> np.ndarray:
    """Sample uniformly on faces of the scaled box boundary.

    The historical name is retained for API compatibility. The Zubov boundary
    condition is defined on ``boundary(scale * R1)``, not at arbitrary points
    outside that box.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    lb_scaled = np.asarray(lb) * scale
    ub_scaled = np.asarray(ub) * scale
    x = np.random.uniform(lb_scaled, ub_scaled, size=(num_samples, len(lb)))
    if num_samples == 0:
        return x
    dimensions = np.random.randint(0, len(lb), size=num_samples)
    upper_face = np.random.randint(0, 2, size=num_samples).astype(bool)
    rows = np.arange(num_samples)
    x[rows, dimensions] = np.where(
        upper_face, ub_scaled[dimensions], lb_scaled[dimensions]
    )
    return x


def sample_in_region_torch(
    num_samples: int, lb: torch.Tensor, ub: torch.Tensor, device: str
) -> torch.Tensor:
    return torch.rand(num_samples, lb.shape[0], device=device) * (ub - lb) + lb


def sample_out_of_region_torch(
    num_samples: int, lb: torch.Tensor, ub: torch.Tensor, scale: float, device: str
) -> torch.Tensor:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    lb_scaled = lb * scale
    ub_scaled = ub * scale
    x = (
        torch.rand(num_samples, lb.shape[0], device=device) * (ub_scaled - lb_scaled)
        + lb_scaled
    )
    if num_samples == 0:
        return x
    dimensions = torch.randint(0, lb.shape[0], (num_samples,), device=device)
    upper_face = torch.randint(0, 2, (num_samples,), device=device).bool()
    rows = torch.arange(num_samples, device=device)
    face_values = torch.where(upper_face, ub_scaled[dimensions], lb_scaled[dimensions])
    x[rows, dimensions] = face_values
    return x


def sample_in_lqr_ellipsoid_torch(
    num_samples: int, c_star: float, L_inv: torch.Tensor, device: str
) -> torch.Tensor:
    state_dim = L_inv.shape[0]
    L_inv = L_inv.to(device)

    z = torch.randn(num_samples, state_dim, device=device)

    z_norm = z / torch.linalg.vector_norm(z, ord=2, dim=1, keepdim=True).clamp_min(
        torch.finfo(z.dtype).eps
    )

    u = torch.rand(num_samples, 1, device=device)
    r = u.pow(1.0 / state_dim)

    points_in_unit_sphere = z_norm * r
    points_in_unit_ellipsoid = points_in_unit_sphere @ L_inv

    return float(np.sqrt(c_star)) * points_in_unit_ellipsoid


def sample_on_circle_boundary_torch(
    num_samples: int, state_dim: int, r2_radius: float, device: str
) -> torch.Tensor:
    z = torch.randn(num_samples, state_dim, device=device)
    z_norm = z / torch.linalg.vector_norm(z, ord=2, dim=1, keepdim=True)

    return z_norm * r2_radius
