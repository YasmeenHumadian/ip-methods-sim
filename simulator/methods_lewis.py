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

# ---- Lewis-style machine learning (step 11) ----
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

ML_MODELS = ["naive_bayes", "random_forest", "logistic"]


def ml_features(df, use_region):
    """What the ML models may see: address block, and optionally region (geolocation)."""
    cols = ["prefix"] + (["region"] if use_region else [])
    return pandas.get_dummies(df[cols].astype(str), dtype=float)


def lewis_ml(df, cfg, model, use_region=None, seed=None, return_probability=False):
    """Train on 70% of IPs, predict the other 30%. Returns True = blocked, indexed like df."""
    m = cfg["ml"]
    if use_region is None:
        use_region = m["use_geolocation"]
    seed = cfg["seed"] if seed is None else seed

    X = ml_features(df, use_region)
    y = df["is_malicious"].astype(int)  # the label only, never an input
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=m["train_share"], stratify=y, random_state=seed
    )

    if model == "naive_bayes":
        clf = BernoulliNB(class_prior=[0.5, 0.5])  # balanced, because attackers are rare
    elif model == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=m["rf_trees"], class_weight="balanced", random_state=seed, n_jobs=-1
        )
    elif model == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    else:
        raise ValueError(f"unknown model: {model}")

    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    if return_probability:
        return pandas.Series(prob, index=X_te.index, name=model)
    threshold = m["logistic_threshold"] if model == "logistic" else 0.5
    return pandas.Series(prob >= threshold, index=X_te.index, name=model)