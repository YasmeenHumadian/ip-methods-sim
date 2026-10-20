"""Lewis-style blocking methods (Lewis et al. 2020).

Every method takes the world table and returns a True/False array: True = block this IP.
Methods may only use what a defender can see (database flags and log features),
never the hidden columns.
"""

import numpy as np


def single_database(df, name):
    """Block an IP if this one database lists it."""
    return df[f"db_{name}"].to_numpy()


def crosscheck_weights(cfg):
    """Weight of each database in the cross-check score. The weights add up to 1."""
    rates = cfg["databases"]["detection_rates"]
    mode = cfg["crosscheck"]["weights"]
    if mode == "by_detection_rate":
        total = sum(rates.values())
        return {name: rate / total for name, rate in rates.items()}
    if mode == "equal":
        return {name: 1 / len(rates) for name in rates}
    raise ValueError(f"unknown crosscheck weights: {mode}")


def crosscheck_score(df, cfg):
    """Weighted share of databases that list the IP: a number from 0 to 1."""
    score = np.zeros(len(df))
    for name, weight in crosscheck_weights(cfg).items():
        score += weight * df[f"db_{name}"].to_numpy()
    return score


def crosscheck(df, cfg):
    """Block an IP if its cross-check score reaches the threshold."""
    return crosscheck_score(df, cfg) >= cfg["crosscheck"]["threshold"]