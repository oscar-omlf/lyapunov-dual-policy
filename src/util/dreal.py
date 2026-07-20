"""Small, shared dReal helpers.

dReal is intentionally optional at import time. Numerical training and
evaluation can run without it; certification calls fail with an actionable
message when the solver bindings are unavailable.
"""

from typing import Any

import numpy as np

try:
    import dreal as _dreal
except ImportError:  # pragma: no cover - exercised on platforms without dReal
    _dreal = None


def require_dreal():
    if _dreal is None:
        raise RuntimeError(
            "dReal Python bindings are required for certification. Install "
            "the platform-specific dReal package described in README.md."
        )
    return _dreal


def dreal_var(n: int, prefix: str = "x"):
    d = require_dreal()
    return np.array([d.Variable(f"{prefix}{i}") for i in range(n)])


def dreal_elementwise(x, func):
    return np.array([func(xi) for xi in x])


def dreal_sigmoid(z):
    d = require_dreal()
    return 1 / (1 + d.exp(-z))


def dreal_clip(value, lower: float, upper: float):
    d = require_dreal()
    return d.if_then_else(
        value > upper,
        upper,
        d.if_then_else(value < lower, lower, value),
    )


def dreal_clip_vector(values, lower: float, upper: float):
    return np.array([dreal_clip(value, lower, upper) for value in values])


def quadratic_form(x, matrix):
    return sum(
        x[i] * sum(matrix[i, j] * x[j] for j in range(len(x))) for i in range(len(x))
    )


def squared_norm(x):
    return sum(xi * xi for xi in x)


def in_box(x, lb, ub, scale: float = 1.0):
    d = require_dreal()
    return d.And(
        *[d.And(x[i] >= lb[i] * scale, x[i] <= ub[i] * scale) for i in range(len(x))]
    )


def on_boundary(x, lb, ub, scale: float = 2.0, pad: float = 0.01):
    """Return a thin inner shell containing the complete box boundary."""
    if not 0 < pad < 1:
        raise ValueError("pad must lie in (0, 1)")
    d = require_dreal()
    outer = in_box(x, lb, ub, scale)
    inner = in_box(x, lb, ub, scale * (1 - pad))
    return d.And(outer, d.Not(inner))


def is_unsat(result: Any) -> bool:
    """Return true only when dReal proved the query UNSAT."""
    return result is None or (isinstance(result, str) and result.lower() == "unsat")


def dreal_in_circle(x, radius: float):
    return squared_norm(x) <= radius**2


def dreal_on_circle_boundary(x, radius: float, tolerance: float = 1e-4):
    d = require_dreal()
    norm_sq = squared_norm(x)
    return d.And(
        norm_sq <= (radius + tolerance) ** 2,
        norm_sq >= (radius - tolerance) ** 2,
    )


def extract_ce_from_model(model, state_dim: int):
    d = require_dreal()
    if model is None:
        return None

    counterexample = np.zeros(state_dim, dtype=np.float64)
    found = np.zeros(state_dim, dtype=bool)
    for variable, value in model.items():
        name = str(variable)
        if not name.startswith("x"):
            continue
        try:
            index = int(name[1:])
        except ValueError:
            continue
        if not 0 <= index < state_dim:
            continue
        counterexample[index] = (
            value.mid() if isinstance(value, d.Interval) else float(value)
        )
        found[index] = True
    return counterexample if found.all() else None
