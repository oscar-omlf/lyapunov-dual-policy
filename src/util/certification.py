from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

CheckFunction = Callable[[float], Tuple[bool, Any]]


@dataclass(frozen=True)
class BisectionResult:
    verified_level: Optional[float]
    counterexample: Any
    lower_verified: bool
    upper_verified: bool
    checks: int


def bisect_verified_level(
    check_fn: CheckFunction,
    lower: float = 0.01,
    upper: float = 0.95,
    tolerance: float = 0.01,
    max_iterations: int = 12,
) -> BisectionResult:
    """Find the largest verified level in a checked monotone bracket."""
    if not 0 <= lower < upper:
        raise ValueError("Expected 0 <= lower < upper")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")

    checks = 1
    lower_ok, lower_ce = check_fn(level=lower)
    if not lower_ok:
        return BisectionResult(
            verified_level=None,
            counterexample=lower_ce,
            lower_verified=False,
            upper_verified=False,
            checks=checks,
        )

    checks += 1
    upper_ok, upper_ce = check_fn(level=upper)
    if upper_ok:
        return BisectionResult(
            verified_level=upper,
            counterexample=None,
            lower_verified=True,
            upper_verified=True,
            checks=checks,
        )

    low = lower
    high = upper
    counterexample = upper_ce
    for _ in range(max_iterations):
        if high - low <= tolerance:
            break
        midpoint = 0.5 * (low + high)
        checks += 1
        ok, model = check_fn(level=midpoint)
        if ok:
            low = midpoint
        else:
            high = midpoint
            counterexample = model

    return BisectionResult(
        verified_level=low,
        counterexample=counterexample,
        lower_verified=True,
        upper_verified=False,
        checks=checks,
    )
