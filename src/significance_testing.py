"""Paired statistical analysis for an ``ldp evaluate`` output directory."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _load_run_means(path, *, nan_aware=False):
    values = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    if not nan_aware:
        return values.mean(axis=1)
    return np.array(
        [
            row[np.isfinite(row)].mean() if np.isfinite(row).any() else np.nan
            for row in values
        ]
    )


def _holm_adjust(raw_p_values):
    ordered = sorted(raw_p_values, key=raw_p_values.get)
    adjusted = {}
    running_max = 0.0
    total = len(ordered)
    for rank, pair in enumerate(ordered):
        candidate = min((total - rank) * raw_p_values[pair], 1.0)
        running_max = max(running_max, candidate)
        adjusted[pair] = running_max
    return adjusted


def _paired_tests(data):
    raw = {}
    details = {}
    for first, second in itertools.combinations(sorted(data), 2):
        first_values = data[first]
        second_values = data[second]
        if first_values.shape != second_values.shape:
            raise ValueError(f"Paired runs differ in shape: {first!r} and {second!r}")
        valid = np.isfinite(first_values) & np.isfinite(second_values)
        if valid.sum() < 2:
            continue
        differences = first_values[valid] - second_values[valid]
        difference_sd = differences.std(ddof=1)
        if difference_sd == 0:
            if differences.mean() == 0:
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = None, 0.0
        else:
            test = stats.ttest_rel(first_values[valid], second_values[valid])
            statistic, p_value = float(test.statistic), float(test.pvalue)
        pair = (first, second)
        raw[pair] = p_value
        details[pair] = {
            "n_pairs": int(valid.sum()),
            "mean_difference": float(differences.mean()),
            "paired_t": statistic,
            "p_raw": p_value,
        }
    adjusted = _holm_adjust(raw)
    for pair, value in adjusted.items():
        details[pair]["p_holm"] = float(value)
    return {
        f"{first} vs {second}": result for (first, second), result in details.items()
    }


def analyze(run_dir, agents):
    costs = {agent: _load_run_means(run_dir / agent / "costs.csv") for agent in agents}
    step_means = {
        agent: _load_run_means(
            run_dir / agent / "steps_to_stabilize.csv", nan_aware=True
        )
        for agent in agents
    }
    convergence = {
        agent: np.atleast_1d(
            np.loadtxt(
                run_dir / agent / "episodes_stabilized.csv",
                delimiter=",",
                skiprows=1,
                usecols=1,
            )
        )
        for agent in agents
    }

    run_counts = {len(values) for values in costs.values()}
    if len(run_counts) != 1:
        raise ValueError("All agents must have the same paired run count")

    result = {
        "descriptive": {
            agent: {
                "mean_cost": float(costs[agent].mean()),
                "sd_run_mean_cost": (
                    float(costs[agent].std(ddof=1)) if costs[agent].size > 1 else 0.0
                ),
                "mean_steps_successful_runs": (
                    float(np.nanmean(step_means[agent]))
                    if np.isfinite(step_means[agent]).any()
                    else None
                ),
                "mean_episodes_stabilized": float(convergence[agent].mean()),
            }
            for agent in agents
        },
        "paired_cost_tests": _paired_tests(costs),
        "paired_conditional_step_tests": _paired_tests(step_means),
        "paired_convergence_count_tests": _paired_tests(convergence),
    }
    if len(agents) >= 3:
        friedman = stats.friedmanchisquare(*(costs[agent] for agent in agents))
        result["friedman_cost"] = {
            key: float(value) if np.isfinite(value) else None
            for key, value in zip(
                ("statistic", "p_value"),
                friedman,
            )
        }
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run paired tests on ldp evaluation outputs"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--agents", nargs="+")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    agents = args.agents or sorted(
        path.name
        for path in args.run_dir.iterdir()
        if path.is_dir() and (path / "costs.csv").exists()
    )
    if len(agents) < 2:
        raise ValueError("At least two evaluated agents are required")
    result = analyze(args.run_dir, agents)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
