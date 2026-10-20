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