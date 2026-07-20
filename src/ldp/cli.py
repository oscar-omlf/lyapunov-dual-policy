import argparse
import sys

from util.logger_utils import save_json


def _key_value(items, value_type=str):
    parsed = {}
    for item in items or []:
        try:
            key, value = item.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected NAME=VALUE, got {item!r}"
            ) from exc
        parsed[key] = value_type(value)
    return parsed


def _add_common(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default="auto")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ldp",
        description="Lyapunov Dual-Policy experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a controller")
    _add_common(train)
    train.add_argument(
        "--agent",
        choices=("ldp", "lac", "td3", "las-td3"),
        default="ldp",
    )
    train.add_argument("--beta", type=float)
    train.add_argument("--outer-loops", type=int, default=80)
    train.add_argument("--steps-per-loop", type=int, default=1000)
    train.add_argument("--no-certify", action="store_true")
    train.add_argument("--level-lower", type=float, default=0.01)
    train.add_argument("--level-upper", type=float, default=0.95)
    train.add_argument("--level-tolerance", type=float, default=0.01)
    train.add_argument("--bisection-iterations", type=int, default=12)
    train.add_argument("--eps", type=float, default=0.25)
    train.add_argument("--bridge-level", type=float)
    train.add_argument("--local-margin", type=float, default=1e-4)
    train.add_argument("--delta", type=float, default=1e-4)
    train.add_argument("--verification-scale", type=float, default=2.0)
    train.add_argument("--doa-samples", type=int, default=50_000)
    train.add_argument("--minimum-stop-level", type=float, default=0.4)
    train.add_argument("--minimum-relative-gain", type=float, default=0.01)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--counterexample-capacity", type=int, default=128)
    train.add_argument("--rollout-steps", type=int)
    train.add_argument("--rollout-refresh-interval", type=int)
    train.add_argument("--plot-interval", type=int)
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--episode-steps", type=int, default=3000)
    train.add_argument("--warmup-steps", type=int, default=1000)
    train.add_argument("--log-interval", type=int, default=50)
    train.add_argument("--no-state-projection", action="store_true")

    certify = subparsers.add_parser("certify", help="Certify a trained controller")
    certify.add_argument("--agent", choices=("ldp", "lac", "lqr"), default="ldp")
    certify.add_argument("--checkpoint")
    certify.add_argument("--episode", default="latest")
    certify.add_argument("--device", default="auto")
    certify.add_argument("--beta", type=float)
    certify.add_argument("--level-lower", type=float, default=0.01)
    certify.add_argument("--level-upper", type=float, default=0.95)
    certify.add_argument("--tolerance", type=float, default=0.001)
    certify.add_argument("--max-iterations", type=int, default=40)
    certify.add_argument("--eps", type=float, default=0.25)
    certify.add_argument("--bridge-level", type=float)
    certify.add_argument("--local-margin", type=float, default=1e-4)
    certify.add_argument("--scale", type=float, default=2.0)
    certify.add_argument("--delta", type=float, default=1e-4)
    certify.add_argument("--output")

    evaluate = subparsers.add_parser(
        "evaluate", help="Run paired controller evaluation"
    )
    _add_common(evaluate)
    evaluate.add_argument(
        "--agents",
        nargs="+",
        choices=("ldp", "lac", "td3", "las-td3", "lqr"),
        required=True,
    )
    evaluate.add_argument(
        "--checkpoint",
        action="append",
        metavar="AGENT=DIR",
        help="Repeat for each learned agent",
    )
    evaluate.add_argument(
        "--checkpoint-episode",
        action="append",
        metavar="AGENT=EPISODE",
    )
    evaluate.add_argument("--beta", action="append", metavar="AGENT=VALUE")
    evaluate.add_argument("--runs", type=int, default=50)
    evaluate.add_argument("--episodes", type=int, default=500)
    evaluate.add_argument("--steps", type=int, default=3000)
    evaluate.add_argument("--dt", type=float, default=0.003)
    evaluate.add_argument(
        "--integration",
        choices=("continuous", "sample-hold"),
        default="continuous",
    )
    evaluate.add_argument("--project-state", action="store_true")
    evaluate.add_argument("--stabilization-threshold", type=float, default=5e-4)
    evaluate.add_argument("--consecutive-stable", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from ldp.training import train_lyapunov_cegar, train_td3

        if args.agent in {"ldp", "lac"}:
            run_dir = train_lyapunov_cegar(
                args.agent,
                seed=args.seed,
                output_dir=args.output_dir,
                device=args.device,
                beta=args.beta,
                outer_loops=args.outer_loops,
                steps_per_loop=args.steps_per_loop,
                log_interval=args.log_interval,
                certify=not args.no_certify,
                level_lower=args.level_lower,
                level_upper=args.level_upper,
                level_tolerance=args.level_tolerance,
                bisection_iterations=args.bisection_iterations,
                eps=args.eps,
                bridge_level=args.bridge_level,
                local_margin=args.local_margin,
                delta=args.delta,
                verification_scale=args.verification_scale,
                doa_samples=args.doa_samples,
                minimum_stop_level=args.minimum_stop_level,
                minimum_relative_gain=args.minimum_relative_gain,
                patience=args.patience,
                counterexample_capacity=args.counterexample_capacity,
                rollout_steps=args.rollout_steps,
                rollout_refresh_interval=args.rollout_refresh_interval,
                plot_interval=args.plot_interval,
            )
        else:
            run_dir = train_td3(
                args.agent,
                seed=args.seed,
                output_dir=args.output_dir,
                device=args.device,
                beta=args.beta,
                episodes=args.episodes,
                steps_per_episode=args.episode_steps,
                warmup_steps=args.warmup_steps,
                log_interval=args.log_interval,
                project_state=not args.no_state_projection,
            )
        print(run_dir)
        return 0

    if args.command == "certify":
        from ldp.certifying import certify_agent

        result = certify_agent(
            args.agent,
            checkpoint=args.checkpoint,
            episode=args.episode,
            device=args.device,
            beta=args.beta,
            level_lower=args.level_lower,
            level_upper=args.level_upper,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            eps=args.eps,
            bridge_level=args.bridge_level,
            local_margin=args.local_margin,
            scale=args.scale,
            delta=args.delta,
        )
        if args.output:
            save_json(result, args.output)
        print(result)
        return 0 if result["verified_level"] is not None else 2

    if args.command == "evaluate":
        from ldp.evaluation import evaluate_agents

        run_dir = evaluate_agents(
            args.agents,
            checkpoints=_key_value(args.checkpoint),
            episodes_by_agent=_key_value(args.checkpoint_episode),
            beta_by_agent=_key_value(args.beta, float),
            seed=args.seed,
            output_dir=args.output_dir,
            device=args.device,
            runs=args.runs,
            episodes=args.episodes,
            steps=args.steps,
            dt=args.dt,
            stabilization_threshold=args.stabilization_threshold,
            consecutive_stable=args.consecutive_stable,
            integration=args.integration,
            project_state=args.project_state,
        )
        print(run_dir)
        return 0
    raise AssertionError("Unhandled command")


def entrypoint(argv=None):
    try:
        return main(argv)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
