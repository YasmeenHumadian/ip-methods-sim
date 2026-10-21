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