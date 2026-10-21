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



# ---- BlackEye model tests (step 14) ----
from sklearn.metrics import roc_auc_score
from simulator.databases import database_flags
from simulator.methods_blackeye import BLACKEYE_MODELS, blackeye, blackeye_features
from simulator.methods_lewis import lewis_ml
from simulator.world import HIDDEN_COLUMNS


def be_world(n=20000):
    cfg = load_config()
    cfg["world"]["n_ips"] = n
    return cfg, database_flags(build_world(cfg), cfg)


def test_blackeye_features_do_not_contain_hidden_columns():
    cfg, df = be_world(2000)
    X = blackeye_features(df, cfg)
    assert not set(X.columns) & set(HIDDEN_COLUMNS)


def test_blackeye_and_lewis_are_tested_on_the_same_ips():
    cfg, df = be_world(2000)
    for model in BLACKEYE_MODELS:
        assert blackeye(df, cfg, model).index.equals(lewis_ml(df, cfg, "logistic").index)


def test_blackeye_beats_the_human_agent_and_lewis_ml():
    cfg, df = be_world()
    pred = blackeye(df, cfg, "logistic")
    truth = df.loc[pred.index, "is_malicious"]
    assert pred[truth].mean() > cfg["human_agent"]["recall"]  # finds more than the agent it learnt from

    be = blackeye(df, cfg, "logistic", return_probability=True)
    lw = lewis_ml(df, cfg, "logistic", return_probability=True)
    assert roc_auc_score(truth, be) > roc_auc_score(truth, lw)