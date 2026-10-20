"""Simulated public blacklist databases (Lewis et al. 2020).

Each database flags some malicious IPs (its detection rate, taken from the paper)
and, by mistake, some innocent IPs. Shared IPs are flagged by mistake more often
(an assumption, see config).
"""

import numpy as np


def database_names(cfg):
    """The names of the four databases, in config order."""
    return list(cfg["databases"]["detection_rates"])


def database_flags(df, cfg, seed=None):
    """Return a copy of df with one True/False column per database, plus db_hits.

    db_<name> is True if that database lists the IP. db_hits counts how many do.
    Uses its own random stream, so changing database settings never changes the world.
    """
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng([seed, 1])

    db = cfg["databases"]
    base_fp = db["false_positive_rate"]
    shared_fp = min(1.0, base_fp * db["shared_fp_multiplier"])

    is_mal = df["is_malicious"].to_numpy()
    is_shared = (df["kind"] == "shared").to_numpy()
    fp_rate = np.where(is_shared, shared_fp, base_fp)

    out = df.copy()
    hits = np.zeros(len(df), dtype=int)
    for name, recall in db["detection_rates"].items():
        p_flag = np.where(is_mal, recall, fp_rate)
        flagged = rng.random(len(df)) < p_flag
        out[f"db_{name}"] = flagged
        hits += flagged
    out["db_hits"] = hits
    return out