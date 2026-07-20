import csv
from pathlib import Path

import numpy as np


def write_2d_csv(arr: np.ndarray, path: str, header_prefix: str = "ep") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"{header_prefix}{i}" for i in range(arr.shape[1])]
        writer.writerow(header)
        writer.writerows(arr)


def write_counts_csv(counts: np.ndarray, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "episodes_stabilized"])
        for run_idx, c in enumerate(counts):
            writer.writerow([run_idx, int(c)])
