"""BlackEye-style methods (Jeon & Tak 2022), built step by step."""
import numpy as np


def human_agent(df, cfg, seed=None):
    """The human security agent: catches ~25% of attackers, 90% of their blocks are correct.

    Returns True = the agent blacklisted this IP. Used as the (noisy) training labels.
    """
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng([seed, 2])  # its own random stream
    a = cfg["human_agent"]

    bad = df["is_malicious"].to_numpy()
    shared = (df["kind"] == "shared").to_numpy()

    blocked = np.zeros(len(df), dtype=bool)
    blocked[bad] = rng.random(bad.sum()) < a["recall"]  # misses most attackers

    # wrong blocks: enough to make precision exactly what the paper reports
    n_right = int(blocked.sum())
    n_wrong = round(n_right * (1 - a["precision"]) / a["precision"])
    n_wrong = min(n_wrong, int((~bad).sum()))

    weights = np.where(shared, cfg["databases"]["shared_fp_multiplier"], 1.0) * (~bad)
    chosen = rng.choice(len(df), size=n_wrong, replace=False, p=weights / weights.sum())
    blocked[chosen] = True
    return blocked