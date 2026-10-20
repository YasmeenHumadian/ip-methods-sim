import pandas as pd

from simulator.config import load_config
from simulator.world import build_population, make_rng


def make(seed=1):
    cfg = load_config()
    return cfg, build_population(cfg, make_rng(cfg, seed))


def test_counts_match_config():
    cfg, df = make()
    assert len(df) == 2000
    assert df["is_malicious"].sum() == 200
    assert (df["kind"] == "shared").sum() == 720
    assert (df["kind"] == "low_and_slow").sum() == 60


def test_same_seed_gives_same_world():
    _, a = make(seed=5)
    _, b = make(seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_world():
    _, a = make(seed=5)
    _, b = make(seed=6)
    assert not a.equals(b)


def test_user_counts():
    cfg, df = make()
    low, high = cfg["world"]["users_per_shared_ip"]
    assert df[df["kind"] == "shared"]["n_users"].between(low, high).all()
    assert (df[df["kind"] != "shared"]["n_users"] == 1).all()


def test_ip_id_does_not_reveal_label():
    _, df = make()
    gap = abs(
        df.loc[df["is_malicious"], "ip_id"].mean()
        - df.loc[~df["is_malicious"], "ip_id"].mean()
    )
    assert gap < 0.1 * len(df)


def test_high_risk_regions_hold_more_attackers():
    cfg, df = make()
    high = df["region"].isin(cfg["world"]["high_risk_regions"])
    assert high[df["is_malicious"]].mean() > high[~df["is_malicious"]].mean()

def test_attackers_cluster_in_some_address_blocks():
    cfg, df = make()
    counts = df[df["is_malicious"]]["prefix"].value_counts()
    n_bad = round(cfg["world"]["n_prefixes"] * cfg["world"]["bad_prefix_share"])
    assert counts.head(n_bad).sum() > 0.5 * df["is_malicious"].sum()
    assert df["prefix"].between(0, cfg["world"]["n_prefixes"] - 1).all()