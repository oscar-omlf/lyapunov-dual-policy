# Implementation Audit

## Correctness Fixes

- Replaced missing configuration imports and hard-coded executable scripts with
  one CLI.
- Fixed the LDP trainer's global-policy call contract.
- Made runtime and dReal policies use identical local clipping, blending, and
  final clipping.
- Replaced the unproved LDP epsilon-ball exclusion with a verified local
  quadratic bridge under the actual blended policy.
- Labeled standalone LAC verification as annular.
- Corrected LAS-TD3's local positive-cost quadratic to TD3's negative-reward
  convention.
- Fixed the LAS-TD3 trainer's stale blending keyword, which prevented updates.
- Replaced custom Riccati iterations with validated SciPy CARE/DARE solvers.
- Made bisection verify both bracket endpoints and retain metadata from the
  accepted level.
- Added strict `agent`, `beta`, and `c_star` checkpoint checks.
- Corrected paired statistical tests and Holm adjustment.

## Efficiency Fixes

- Added batched continuous-feedback rollouts.
- Avoided full trajectory allocation during training.
- Cached finite-rollout targets between configurable refreshes.
- Cached the level-independent local dReal bridge during bisection.
- Replaced unbounded replay lists and full-buffer sampling copies with a
  bounded ring buffer and indexed sampling.
- Stored replay transitions on CPU and transferred only sampled batches.
- Batched controller evaluation over episodes.
- Replaced per-episode Welford object graphs with vectorized run statistics.

## Reproducibility and Release

- Added deterministic seeding, isolated config copies, unique run directories,
  resolved config JSON, summaries, and paired initial-state storage.
- Added unified checkpoint dictionaries with optimizer, scheduler, target
  network, and update-counter state plus thesis-era loading compatibility.
- Added `pyproject.toml`, a minimal cross-platform dependency list, CLI entry
  point, focused tests, and methodology/literature documentation.

## Remaining Limitations

- No trained checkpoints, raw thesis results, or solver transcripts are
  tracked. Thesis tables are not reproducible from versioned artifacts alone.
- dReal is an optional platform-specific dependency and certificates must be
  rerun in a supported environment.
- Replay buffers are intentionally not checkpointed, so optimizer state alone
  does not provide exact mid-run TD3 resumption.
- Hard-coded historical `c_star` blending levels have no versioned solver
  transcript establishing their provenance.
- The model has no uncertainty set, disturbance model, sensor error, or delay.
- The LAC certificate is annular.
- A continuous-time certificate does not automatically cover state projection
  or sample-and-hold deployment.
- The thesis's LAS-TD3 local identity-weighted quadratic prior differs from its
  reported DRL performance cost. The implementation preserves this reported
  design and documents the local critic as a stability prior.
