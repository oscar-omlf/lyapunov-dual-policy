import torch


def fetch_device(preferred: str = "auto") -> str:
    """Resolve an explicit device or choose the best available accelerator."""
    if preferred != "auto":
        return str(torch.device(preferred))
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
