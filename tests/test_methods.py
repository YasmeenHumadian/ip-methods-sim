import numpy as np

from simulator.config import load_config
from simulator.databases import database_flags
from simulator.methods_lewis import (
    crosscheck,
    crosscheck_score,
    crosscheck_weights,
    single_database,
)
from simulator.world import HIDDEN_COLUMNS, build_world


def flagged_world():
    """A bigger world so the measured rates are close to the settings."""
    cfg = load_config()
    cfg["world"]["n_ips"] = 20000
    df = database_flags(build_world(cfg, seed=1), cfg, seed=1)
    return cfg, df


def recall(blocked, df):
    return blocked[df["is_malicious"].to_numpy()].mean()


def false_positive_rate(blocked, df):
    return blocked[~df["is_malicious"].to_numpy()].mean()


def test_single_database_matches_its_flags():
    cfg, df = flagged_world()
    assert (single_database(df, "apility") == df["db_apility"].to_numpy()).all()


def test_weights_add_up_to_one_and_favour_better_databases():
    cfg, _ = flagged_world()
    weights = crosscheck_weights(cfg)
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert weights["apility"] > weights["virustotal"]


def test_crosscheck_catches_more_than_the_best_single_database():
    cfg, df = flagged_world()
    best_single = max(
        recall(single_database(df, name), df)
        for name in cfg["databases"]["detection_rates"]
    )
    assert recall(crosscheck(df, cfg), df) > best_single


def test_crosscheck_detection_is_near_the_paper():
    cfg, df = flagged_world()
    assert abs(recall(crosscheck(df, cfg), df) - 0.82) < 0.06


def test_crosscheck_false_positive_rate_is_small():
    cfg, df = flagged_world()
    assert false_positive_rate(crosscheck(df, cfg), df) < 0.10


def test_no_flags_means_no_block_and_score_in_range():
    cfg, df = flagged_world()
    for name in cfg["databases"]["detection_rates"]:
        df[f"db_{name}"] = False
    assert not crosscheck(df, cfg).any()
    score = crosscheck_score(df, cfg)
    assert (score >= 0).all() and (score <= 1).all()


def test_methods_do_not_use_hidden_columns():
    cfg, df = flagged_world()
    visible = df.drop(columns=HIDDEN_COLUMNS)
    assert np.array_equal(crosscheck(df, cfg), crosscheck(visible, cfg))
    assert np.array_equal(single_database(df, "myip"), single_database(visible, "myip"))


# ---- Lewis ML tests (step 11) ----
from sklearn.metrics import balanced_accuracy_score
from simulator.config import load_config
from simulator.world import HIDDEN_COLUMNS, build_world
from simulator.methods_lewis import ML_MODELS, lewis_ml, ml_features


def _ml_world(n):
    cfg = load_config()
    cfg["world"]["n_ips"] = n
    return cfg, build_world(cfg)


def test_ml_features_do_not_contain_hidden_columns():
    cfg, df = _ml_world(2000)
    for use_region in (True, False):
        X = ml_features(df, use_region)
        for hidden in HIDDEN_COLUMNS:
            assert not any(c.startswith(hidden) for c in X.columns)


def test_ml_models_predict_the_test_share():
    cfg, df = _ml_world(2000)
    for model in ML_MODELS:
        pred = lewis_ml(df, cfg, model)
        assert abs(len(pred) - 0.3 * len(df)) <= 1
        assert pred.dtype == bool
        assert pred.index.isin(df.index).all()


from sklearn.metrics import roc_auc_score


def test_ml_beats_chance_and_region_helps():
    cfg, df = _ml_world(20000)

    def auc(model, region):
        prob = lewis_ml(df, cfg, model, use_region=region, return_probability=True)
        return roc_auc_score(df.loc[prob.index, "is_malicious"], prob)

    for model in ("logistic", "naive_bayes"):
        assert auc(model, False) > 0.6
        assert auc(model, True) > auc(model, False)