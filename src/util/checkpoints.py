import inspect
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path, device):
    """Load a trusted local checkpoint across supported PyTorch versions."""
    checkpoint_path = Path(path)
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(checkpoint_path, **kwargs)


def validate_checkpoint(
    checkpoint,
    *,
    expected_agent,
    beta=None,
    c_star=None,
):
    """Reject checkpoints that describe a different closed-loop controller."""
    checkpoint_agent = checkpoint.get("agent")
    if checkpoint_agent is not None and checkpoint_agent != expected_agent:
        raise ValueError(
            f"Checkpoint is for {checkpoint_agent!r}, expected {expected_agent!r}"
        )

    expected_values = {"beta": beta, "c_star": c_star}
    for name, expected in expected_values.items():
        stored = checkpoint.get(name)
        if stored is None or expected is None:
            continue
        if not np.isclose(float(stored), float(expected), rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"Checkpoint {name}={stored!r} does not match configured "
                f"{name}={expected!r}"
            )
