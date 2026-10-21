import numpy as np

from simulator.config import load_config
from simulator.evaluate import agent_wrong_blocks_corrected, evaluate
from simulator.methods_blackeye import human_agent
from simulator.methods_lewis import lewis_ml
from simulator.world import build_world


def eval_world():
    cfg = load_config()
    cfg["world"]["n_ips"] = 20000
    return cfg, build_world(cfg)


def test_a_perfect_method_scores_perfectly():
    cfg, df = eval_world()
    r = evaluate(df, df["is_malicious"].to_numpy())
    assert r["caught"] == 1 and r["precision"] == 1 and r["wrongly_blocked"] == 0


def test_blocking_everything_catches_all_but_blocks_every_innocent():
    cfg, df = eval_world()
    r = evaluate(df, np.ones(len(df), dtype=bool))
    assert r["caught"] == 1
    assert r["wrongly_blocked_share"] == 1
    assert abs(r["precision"] - df["is_malicious"].mean()) < 1e-9


def test_evaluate_handles_ml_methods_that_return_test_rows_only():
    cfg, df = eval_world()
    r = evaluate(df, lewis_ml(df, cfg, "logistic"))
    assert 0 < r["caught"] < 1
    assert r["n_blocked"] < len(df)


def test_correcting_the_agent():
    cfg, df = eval_world()
    agent = human_agent(df, cfg)
    assert agent_wrong_blocks_corrected(df, agent, agent) == 0.0
    assert agent_wrong_blocks_corrected(df, agent, np.zeros(len(df), dtype=bool)) == 1.0