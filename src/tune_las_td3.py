"""Run a reproducible beta sweep through the production training pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np

from ldp.training import train_td3
from util.logger_utils import save_json


def build_parser():
    parser = argparse.ArgumentParser(description="Tune LAS-TD3 beta")
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.1 * index for index in range(1, 10)],
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-state-projection", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    results = []
    for beta in args.betas:
        scores = []
        run_directories = []
        for run_index in range(args.runs):
            run_dir = train_td3(
                "las-td3",
                beta=beta,
                seed=args.seed + run_index,
                output_dir=args.output_dir,
                device=args.device,
                episodes=args.episodes,
                steps_per_episode=args.steps,
                warmup_steps=args.warmup_steps,
                project_state=not args.no_state_projection,
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            scores.append(float(summary["mean_return"]))
            run_directories.append(str(run_dir))
        results.append(
            {
                "beta": float(beta),
                "mean_return": float(np.mean(scores)),
                "sd_return": (
                    float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                ),
                "runs": run_directories,
            }
        )

    best = max(results, key=lambda item: item["mean_return"])
    output = {
        "agent": "las-td3",
        "base_seed": args.seed,
        "runs_per_beta": args.runs,
        "best_beta": best["beta"],
        "results": results,
    }
    output_path = args.output_dir / "las_td3_beta_search.json"
    save_json(output, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
