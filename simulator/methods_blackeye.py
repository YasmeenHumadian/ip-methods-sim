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

# ---- Data cleansing (step 13) ----
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def crossover_point(score, labels):
    """The score where the 'safe' and 'malicious' bell curves cross."""
    s0, s1 = score[~labels], score[labels]
    m0, m1 = s0.mean(), s1.mean()
    sd0, sd1 = max(s0.std(), 1e-9), max(s1.std(), 1e-9)
    grid = np.linspace(m0, m1, 2001)
    gap = stats.norm.pdf(grid, m0, sd0) - stats.norm.pdf(grid, m1, sd1)
    return grid[np.argmin(np.abs(gap))]


def cleanse(X, labels, cfg):
    """Returns True for the training points to KEEP; False = looks mislabelled."""
    labels = np.asarray(labels, dtype=bool)
    keep = np.ones(len(labels), dtype=bool)
    if not cfg["cleansing"]["enabled"] or labels.sum() < 5 or (~labels).sum() < 5:
        return keep

    Xs = StandardScaler().fit_transform(X)
    score = Ridge(alpha=1.0).fit(Xs, labels.astype(float)).predict(Xs)
    if score[labels].mean() <= score[~labels].mean():
        return keep  # the first model found nothing useful, so do not trust it to drop anything

    cross = crossover_point(score, labels)
    return np.where(labels, score >= cross, score < cross)