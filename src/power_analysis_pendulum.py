"""Pilot-based power calculations for paired evaluation runs."""

import argparse
import itertools
from pathlib import Path

import numpy as np
from statsmodels.stats.power import TTestPower


def _run_means(path, *, nan_aware=False):
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = np.atleast_2d(values)
    reducer = np.nanmean if nan_aware else np.mean
    return reducer(values, axis=1)


def _paired_effect(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    differences = first[valid] - second[valid]
    if differences.size < 2:
        return None
    standard_deviation = differences.std(ddof=1)
    if standard_deviation == 0:
        return np.inf if differences.mean() != 0 else 0.0
    return abs(differences.mean()) / standard_deviation


def required_pairs(effect_size, *, alpha, power):
    if effect_size is None:
        return None
    if np.isinf(effect_size):
        return 2
    if effect_size <= 0:
        return None
    estimate = TTestPower().solve_power(
        effect_size=effect_size,
        nobs=None,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return max(2, int(np.ceil(estimate)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate paired run counts from pilot evaluation data"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.95)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    costs = {
        agent: _run_means(args.run_dir / agent / "costs.csv") for agent in args.agents
    }
    pairs = list(itertools.combinations(args.agents, 2))
    adjusted_alpha = args.alpha / max(len(pairs), 1)

    print("metric,comparison,paired_effect,alpha,power,required_runs")
    for first, second in pairs:
        effect = _paired_effect(costs[first], costs[second])
        required = required_pairs(effect, alpha=adjusted_alpha, power=args.power)
        effect_text = "" if effect is None else f"{effect:.6g}"
        required_text = "" if required is None else str(required)
        print(
            f"cost,{first} vs {second},{effect_text},"
            f"{adjusted_alpha:.6g},{args.power:.6g},{required_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
