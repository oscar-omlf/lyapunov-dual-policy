import json

import numpy as np

from ldp.cli import _key_value, build_parser
from ldp.training import train_lyapunov_cegar


def test_cli_parses_key_value_options():
    assert _key_value(["ldp=0.9"], float) == {"ldp": 0.9}
    args = build_parser().parse_args(
        ["evaluate", "--agents", "ldp", "lqr", "--runs", "2"]
    )
    assert args.command == "evaluate"
    assert args.agents == ["ldp", "lqr"]


def test_one_step_ldp_training_without_dreal(tmp_path):
    run_dir = train_lyapunov_cegar(
        "ldp",
        seed=3,
        output_dir=tmp_path,
        device="cpu",
        outer_loops=1,
        steps_per_loop=1,
        certify=False,
        rollout_steps=2,
        rollout_refresh_interval=1,
        doa_samples=8,
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["updates"] == 1
    assert (run_dir / "checkpoint_latest.pt").exists()
    losses = np.loadtxt(run_dir / "losses.csv", delimiter=",", skiprows=1)
    assert np.isfinite(losses).all()
