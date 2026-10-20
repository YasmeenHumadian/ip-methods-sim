from simulator.config import load_config
from simulator.methods_blackeye import human_agent
from simulator.world import build_world


def agent_world():
    cfg = load_config()
    cfg["world"]["n_ips"] = 20000
    return cfg, build_world(cfg)


def test_agent_recall_and_precision_match_the_paper():
    cfg, df = agent_world()
    blocked = human_agent(df, cfg)
    bad = df["is_malicious"].to_numpy()
    assert abs(blocked[bad].mean() - 0.25) < 0.03
    assert abs(bad[blocked].mean() - 0.90) < 0.02


def test_agent_wrongly_blocks_shared_ips_more_often():
    cfg, df = agent_world()
    blocked = human_agent(df, cfg)
    shared = (df["kind"] == "shared").to_numpy()
    single = (df["kind"] == "single").to_numpy()
    assert blocked[shared].mean() > blocked[single].mean()

# ---- cleansing tests (step 13) ----
import numpy as np
from simulator.methods_blackeye import cleanse
from simulator.world import FEATURE_COLUMNS


def test_cleansing_keeps_most_points_but_not_all():
    cfg, df = agent_world()
    keep = cleanse(df[FEATURE_COLUMNS], human_agent(df, cfg), cfg)
    assert 0.80 < keep.mean() < 1.0


def test_cleansing_drops_wrong_labels_more_often_than_right_ones():
    cfg, df = agent_world()
    labels = human_agent(df, cfg)
    keep = cleanse(df[FEATURE_COLUMNS], labels, cfg)
    wrong = labels != df["is_malicious"].to_numpy()
    assert (~keep[wrong]).mean() > (~keep[~wrong]).mean()


def test_cleansing_can_be_switched_off():
    cfg, df = agent_world()
    cfg["cleansing"]["enabled"] = False
    assert cleanse(df[FEATURE_COLUMNS], human_agent(df, cfg), cfg).all()