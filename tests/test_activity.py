import pandas as pd

from simulator.config import load_config
from simulator.world import FEATURE_COLUMNS, HIDDEN_COLUMNS, build_world


def world(seed=1):
    cfg = load_config()
    return cfg, build_world(cfg, seed)


def test_feature_columns_exist_and_have_no_gaps():
    _, df = world()
    assert df[FEATURE_COLUMNS].notna().all().all()
    assert df["failure_rate"].between(0, 1).all()
    assert (df["total_events"] >= 0).all()


def test_features_exclude_hidden_columns():
    assert not set(FEATURE_COLUMNS) & set(HIDDEN_COLUMNS)


def test_same_seed_gives_same_world():
    _, a = world(seed=3)
    _, b = world(seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_failure_rates_by_kind():
    _, df = world()
    means = df.groupby("kind")["failure_rate"].mean()
    assert means["attacker"] > 0.9
    assert 0.02 < means["single"] < 0.10


def test_shared_ips_are_busier():
    _, df = world()
    means = df.groupby("kind")["total_events"].mean()
    assert means["shared"] > means["single"]


def test_low_and_slow_looks_less_extreme():
    _, df = world()
    means = df.groupby("kind")["failure_rate"].mean()
    assert means["low_and_slow"] < means["attacker"]


def test_accounts_never_exceed_events():
    _, df = world()
    assert (df["distinct_accounts"] <= df["total_events"]).all()


def test_some_shared_ips_look_suspicious():
    _, df = world()
    shared = df[df["kind"] == "shared"]
    assert (shared["failure_rate"] > 0.2).sum() > 0