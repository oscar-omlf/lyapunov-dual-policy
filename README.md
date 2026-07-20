# Lyapunov Dual-Policy Control

Research code for the Lyapunov Dual-Policy (LDP) controller described in
[`thesis-oscar.pdf`](thesis-oscar.pdf). LDP combines a trusted local LQR with a
physics-informed neural controller and a composite Zubov candidate. The
repository also includes standalone LQR, Lyapunov actor-critic (LAC), TD3, and
locally asymptotically stable TD3 (LAS-TD3) baselines.

The implementation has one command-line interface for training, certification,
and paired evaluation. Numerical training and evaluation do not require dReal;
formal certification does.

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and analysis scripts:

```bash
python -m pip install -e ".[dev]"
```

dReal is intentionally not a Python package dependency because its supported
installation route is platform-specific. Install the official dReal Python
bindings in the same environment before running `ldp certify` or certified
CEGAR training. On unsupported systems, use a supported Linux environment and
follow the [dReal installation guide](https://dreal.github.io/).

## Commands

Train LDP with certification:

```bash
ldp train --agent ldp --beta 0.9
```

Train without dReal:

```bash
ldp train --agent ldp --beta 0.9 --no-certify
ldp train --agent lac --no-certify
ldp train --agent td3
ldp train --agent las-td3 --beta 0.9
```

Certify a saved LDP checkpoint:

```bash
ldp certify \
  --agent ldp \
  --checkpoint runs/InvertedPendulum/LDP_beta_0.90/run_000 \
  --episode best
```

Run paired evaluation from identical initial states:

```bash
ldp evaluate \
  --agents ldp lac lqr \
  --checkpoint ldp=PATH_TO_LDP_RUN \
  --checkpoint lac=PATH_TO_LAC_RUN \
  --checkpoint-episode ldp=best
```

Use `ldp COMMAND --help` for all options. Historical scripts such as
`src/train_las_lac_cegar.py` are thin compatibility launchers for this CLI.

## Controller Semantics

For `v(x) = x^T P x / c_star` and `s = arctanh(beta)`, LDP uses

```text
h1(x) = tanh(s v(x))
h2(x) = tanh(s v(x)^(3/2))
```

The executed policy is exactly

```text
u_local = clip(-Kx)
u = clip(u_local + h1(x) (u_global - u_local))
```

and the dReal translation applies clipping in the same order. Since
`h1(x) = O(||x||^2)`, the blended controller has the local LQR Jacobian at the
origin. Since `h2(x) = O(||x||^3)`, the composite Zubov candidate preserves the
local quadratic behavior through second order.

LDP certification has two explicit parts:

1. The local quadratic function proves decrease for the actual blended,
   saturated controller inside a verified bridge.
2. The learned composite Zubov function proves positivity and decrease outside
   that bridge and excludes contact with the verification-box boundary.

This closes the unproved epsilon-ball gap in the thesis-era implementation.
Standalone LAC has no trusted local bridge, so its dReal result is explicitly
an **annular certificate**, not a proof that includes the equilibrium.

See [Methodology](docs/METHODOLOGY.md) for the equations and implementation
contract, and [Literature](docs/LITERATURE.md) for the research context.

## Reproducibility

Each run receives a unique directory containing its resolved configuration,
logs, metrics, summary, and checkpoint. Python, NumPy, and PyTorch are seeded;
paired evaluation stores the exact initial-state tensor.

The certification model assumes **unprojected continuous state feedback**:
the policy is recomputed at every RK4 substage. TD3 training uses
**sample-and-hold** actions. Passing `--integration sample-hold` or
`--project-state` during evaluation changes the closed-loop system and the
formal certificate does not transfer automatically.

Important artifact limitation: this repository currently tracks no trained
checkpoints, raw experiment outputs, or dReal solver transcripts. Therefore,
the numerical tables in the thesis cannot be independently reproduced from
versioned artifacts alone. New runs are reproducible from their generated
configuration, but exact training resume also requires a replay buffer, which
checkpoints do not store.

## Development

```bash
ruff check .
pytest
python -m compileall -q src
```

The focused tests cover integration semantics, rollout labels, samplers,
Riccati residuals, replay capacity, blending behavior, LAS-TD3 critic sign and
updates, checkpoint compatibility, certification bisection, and a one-step LDP
training run.

## Layout

```text
src/ldp/       CLI and train/certify/evaluate orchestration
src/agents/    Controller definitions and checkpoint contracts
src/trainers/  LAC, LDP, TD3, and LAS-TD3 optimization
src/models/    Neural policy and critic modules
src/util/      Dynamics, simulation, certification, sampling, and I/O
tests/         Regression and smoke tests
docs/          Methodology, literature map, and implementation audit
```

The project is released under the [MIT License](LICENSE).
