import pandas as pd

from simulator.config import load_config
from simulator.databases import database_flags, database_names
from simulator.world import build_world


def big_world():
    """A bigger world so the measured rates are close to the settings."""
    cfg = load_config()
    cfg["world"]["n_ips"] = 20000
    return cfg, build_world(cfg, seed=1)


def test_detection_rates_match_the_paper():
    cfg, df = big_world()
    flagged = database_flags(df, cfg, seed=1)
    mal = flagged[flagged["is_malicious"]]
    for name, rate in cfg["databases"]["detection_rates"].items():
        assert abs(mal[f"db_{name}"].mean() - rate) < 0.03


def test_false_positive_rates():
    cfg, df = big_world()
    flagged = database_flags(df, cfg, seed=1)
    fp = cfg["databases"]["false_positive_rate"]
    mult = cfg["databases"]["shared_fp_multiplier"]
    single = flagged[flagged["kind"] == "single"]
    shared = flagged[flagged["kind"] == "shared"]
    assert abs(single["db_virustotal"].mean() - fp) < 0.01
    assert abs(shared["db_virustotal"].mean() - fp * mult) < 0.02


def test_same_seed_gives_same_flags():
    cfg, df = big_world()
    a = database_flags(df, cfg, seed=2)
    b = database_flags(df, cfg, seed=2)
    pd.testing.assert_frame_equal(a, b)


def test_db_hits_is_the_sum_of_the_flags():
    cfg, df = big_world()
    flagged = database_flags(df, cfg, seed=1)
    cols = [f"db_{n}" for n in database_names(cfg)]
    assert (flagged[cols].sum(axis=1) == flagged["db_hits"]).all()


def test_no_false_alarms_when_rate_is_zero():
    cfg, df = big_world()
    cfg["databases"]["false_positive_rate"] = 0.0
    flagged = database_flags(df, cfg, seed=1)
    assert flagged.loc[~flagged["is_malicious"], "db_hits"].sum() == 0