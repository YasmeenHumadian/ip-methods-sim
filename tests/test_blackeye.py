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