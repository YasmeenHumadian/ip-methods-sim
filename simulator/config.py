"""Load and check the simulation settings from the YAML file."""

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "mini_methods.yaml"


def check_config(cfg):
    """Raise ValueError if any setting is out of range. Returns cfg if all is well."""
    w = cfg["world"]

    for key in ("malicious_share", "shared_share", "legit_error_rate", "low_and_slow_share"):
        if not 0 <= w[key] <= 1:
            raise ValueError(f"world.{key} must be between 0 and 1, got {w[key]}")

    low, high = w["users_per_shared_ip"]
    if low < 2 or high < low:
        raise ValueError("world.users_per_shared_ip must be [low, high] with 2 <= low <= high")

    if w["n_ips"] < 100:
        raise ValueError("world.n_ips must be at least 100")

    for name, rate in cfg["databases"]["detection_rates"].items():
        if not 0 <= rate <= 1:
            raise ValueError(f"databases.detection_rates.{name} must be between 0 and 1")

    agent = cfg["human_agent"]
    for key in ("recall", "precision"):
        if not 0 < agent[key] <= 1:
            raise ValueError(f"human_agent.{key} must be above 0 and at most 1")

    return cfg


def load_config(path=DEFAULT_PATH):
    """Read the YAML settings file, check it, and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return check_config(cfg)