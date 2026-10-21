import pandas

from simulator.config import load_config
from simulator.experiments import run_m1, run_m2, summarise


def small_cfg():
    cfg = load_config()
    cfg["world"]["n_ips"] = 1000
    cfg["n_seeds"] = 2
    return cfg


def test_m1_has_every_method_for_every_seed():
    results = run_m1(small_cfg())
    assert results["method"].nunique() == 11
    assert results["seed"].nunique() == 2
    assert len(results) == 22


def test_m1_is_reproducible():
    cfg = small_cfg()
    pandas.testing.assert_frame_equal(run_m1(cfg), run_m1(cfg))


def test_m2_runs_with_cleansing_on_and_off():
    results = run_m2(small_cfg())
    assert set(results["cleansing"]) == {"on", "off"}
    assert len(results) == 8


def test_summary_has_one_row_per_method():
    results = run_m1(small_cfg())
    assert len(summarise(results)) == 11



from simulator.experiments import cleansing_demo, run_once


def test_run_once_scores_every_method():
    df, results = run_once(small_cfg(), 1)
    assert len(results) == 11 and len(df) == 1000


def test_cleansing_demo_numbers_are_consistent():
    counts, table = cleansing_demo(small_cfg(), 1)
    assert counts["wrong_dropped"] <= counts["dropped"] <= counts["labels"]
    assert counts["wrong_dropped"] <= counts["wrong_labels"]
    assert list(table["cleansing"]) == ["off", "on"]