"""Load and validate dataset paths configured for this project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "input" / "data_paths.json"


def load_data_paths() -> dict[str, Any]:
    """Return the complete path configuration."""
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def get_data_path(*keys: str, must_exist: bool = True) -> Path:
    """Return one configured path selected by its nested JSON keys."""
    value: Any = load_data_paths()
    try:
        for key in keys:
            value = value[key]
    except (KeyError, TypeError) as exc:
        joined = " -> ".join(keys)
        raise KeyError(f"Unknown data-path key: {joined}") from exc

    if not isinstance(value, str):
        joined = " -> ".join(keys)
        raise TypeError(f"Configured value is not a path: {joined}")

    path = Path(value)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Configured path does not exist: {path}")
    return path


if __name__ == "__main__":
    config = load_data_paths()
    print(f"Loaded configuration: {CONFIG_FILE}")
    print(f"Data root: {config['data_root']}")
