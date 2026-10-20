"""Build the simulated world: which IPs exist, who uses them, and what they do."""

import numpy as np
import pandas as pd

# What a defender can see in its logs. Methods may use ONLY these (plus region).
FEATURE_COLUMNS = [
    "total_events",        # how many login attempts in the window
    "active_days",         # on how many days the IP was active
    "failure_rate",        # share of attempts that failed
    "distinct_accounts",   # how many different accounts it tried
    "accounts_per_event",  # accounts tried divided by attempts
    "night_share",         # share of attempts at night
    "peak_day_share",      # share of all attempts on the busiest day
]

# The hidden truth. Used only to measure results, never as an input to a method.
HIDDEN_COLUMNS = ["kind", "is_malicious", "n_users"]


def make_rng(cfg, seed=None):
    """A random number generator. The same seed always gives the same world."""
    return np.random.default_rng(cfg["seed"] if seed is None else seed)


def build_population(cfg, rng):
    """One row per IP: kind, number of users, region, and the hidden label.

    kinds: single (1 user), shared (many users), attacker, low_and_slow
    """
    w = cfg["world"]
    n_ips = w["n_ips"]

    n_mal = round(n_ips * w["malicious_share"])
    n_innocent = n_ips - n_mal
    n_shared = round(n_innocent * w["shared_share"])
    n_single = n_innocent - n_shared
    n_slow = round(n_mal * w["low_and_slow_share"])
    n_loud = n_mal - n_slow

    kinds = (
        ["single"] * n_single
        + ["shared"] * n_shared
        + ["attacker"] * n_loud
        + ["low_and_slow"] * n_slow
    )

    low, high = w["users_per_shared_ip"]
    users = np.concatenate([
        np.ones(n_single, dtype=int),
        rng.integers(low, high + 1, size=n_shared),
        np.ones(n_mal, dtype=int),
    ])

    # Innocent IPs: any region equally likely. Attackers: high-risk regions more likely.
    regions = list(w["regions"])
    weights = np.ones(len(regions))
    for r in w["high_risk_regions"]:
        weights[regions.index(r)] = w["high_risk_weight"]
    region_innocent = rng.choice(regions, size=n_innocent)
    region_mal = rng.choice(regions, size=n_mal, p=weights / weights.sum())
    region = np.concatenate([region_innocent, region_mal])

        # Address block (like a /16 network). Attackers cluster in "bad neighbourhoods".
    n_prefix = w["n_prefixes"]
    n_bad = max(1, round(n_prefix * w["bad_prefix_share"]))
    prefix_innocent = rng.integers(0, n_prefix, size=n_innocent)
    in_bad = rng.random(n_mal) < w["attackers_in_bad_prefix"]
    prefix_mal = np.where(
        in_bad,
        rng.integers(0, n_bad, size=n_mal),
        rng.integers(0, n_prefix, size=n_mal),
    )
    prefix = np.concatenate([prefix_innocent, prefix_mal])
    prefix = rng.permutation(n_prefix)[prefix]  # shuffle the numbering so ids reveal nothing

    df = pd.DataFrame({"kind": kinds, "n_users": users, "region": region, "prefix": prefix})
    df["is_malicious"] = df["kind"].isin(["attacker", "low_and_slow"])

    # Shuffle so the row order does not reveal who is malicious, then number the IPs.
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "ip_id", np.arange(len(df)))
    return df


def add_activity(df, cfg, rng):
    """Add what each IP did over the window (the features a defender would see)."""
    w = cfg["world"]
    a = cfg["activity"]
    n = len(df)
    days = w["days"]

    kind = df["kind"].to_numpy()
    users = df["n_users"].to_numpy()
    is_att = kind == "attacker"
    is_slow = kind == "low_and_slow"
    is_shared = kind == "shared"

    # 1. Attempts per day: average per IP, with some natural variation, then daily draws.
    daily_mean = np.where(
        is_att,
        a["attacker_attempts_per_day"],
        np.where(is_slow, a["slow_attempts_per_day"], users * w["logins_per_user_per_day"]),
    ).astype(float)
    daily_mean = daily_mean * rng.lognormal(0.0, 0.3, size=n)
    daily = rng.poisson(daily_mean[:, None], size=(n, days))

    total = daily.sum(axis=1)
    active_days = (daily > 0).sum(axis=1)
    peak_share = daily.max(axis=1) / np.maximum(total, 1)

    # 2. Failures. Honest users make typos. Some shared IPs have a bad week.
    fail_p = np.full(n, w["legit_error_rate"], dtype=float)
    burst = is_shared & (rng.random(n) < a["shared_burst_chance"])
    fail_p[burst] = a["shared_burst_failure_rate"]
    fail_p[is_att] = a["attacker_failure_rate"]
    fail_p[is_slow] = a["slow_failure_rate"]
    failures = rng.binomial(total, fail_p)
    failure_rate = failures / np.maximum(total, 1)

    # 3. Accounts tried. Half of the loud attackers spray many accounts, half hit a few.
    spray = rng.random(n) < 0.5
    acc_loud = np.where(spray, rng.integers(50, 301, size=n), rng.integers(1, 6, size=n))
    acc_slow = rng.integers(5, 31, size=n)
    accounts = np.where(is_att, acc_loud, np.where(is_slow, acc_slow, users))
    accounts = np.minimum(accounts, total)

    # 4. When it happens: attackers spread attempts across the whole day.
    night_mean = np.where(is_att, 0.33, np.where(is_slow, 0.25, 0.12))
    night_share = np.clip(rng.normal(night_mean, 0.06, size=n), 0.0, 1.0)

    out = df.copy()
    out["total_events"] = total
    out["active_days"] = active_days
    out["failure_rate"] = failure_rate
    out["distinct_accounts"] = accounts
    out["accounts_per_event"] = accounts / np.maximum(total, 1)
    out["night_share"] = night_share
    out["peak_day_share"] = peak_share
    return out


def build_world(cfg, seed=None):
    """The full world: population plus activity. Same seed, same world."""
    rng = make_rng(cfg, seed)
    return add_activity(build_population(cfg, rng), cfg, rng)