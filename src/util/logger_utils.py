import json
import logging
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", repr(value))
        return f"{module}.{name}".lstrip(".")
    return value


def save_json(data, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(data), stream, indent=2, sort_keys=True)
        stream.write("\n")


def setup_run_directory_and_logging(
    config,
    base_log_dir="runs",
    evaluation=False,
):
    try:
        environment = config["environment"]
        model_name = config["model_name"]
    except KeyError as exc:
        raise KeyError("Configuration requires environment and model_name") from exc

    model_dir = Path(base_log_dir) / environment / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    run_number = 0
    while True:
        run_dir = model_dir / f"run_{run_number:03d}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            run_number += 1

    logger = logging.getLogger(f"{model_name}.run_{run_number:03d}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    log_name = "evaluation.log" if evaluation else "training.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / log_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(stream_handler)

    save_json(config, run_dir / "config.json")
    logger.info("Run directory: %s", run_dir)
    return run_dir, logger
