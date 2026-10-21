"""The experiments. M1 compares every method; M2 tests the cleansing step."""
import pandas

from simulator.config import load_config
from simulator.databases import database_flags
from simulator.evaluate import agent_wrong_blocks_corrected, evaluate
from simulator.methods_blackeye import BLACKEYE_MODELS, blackeye, human_agent
from simulator.methods_lewis import ML_MODELS, crosscheck, lewis_ml, single_database
from simulator.world import build_world

METRICS = ["caught", "precision", "f1", "wrongly_blocked_share", "shared_wrongly_blocked_share"]


def seeds_for(cfg):
    return [cfg["seed"] + i for i in range(cfg["n_seeds"])]


def make_world(cfg, seed):
    return database_flags(build_world(cfg, seed=seed), cfg, seed=seed)


def all_methods(df, cfg, seed):
    """Every method's answer on this world: True = blocked."""
    methods = {"human agent": human_agent(df, cfg, seed)}
    for name in cfg["databases"]["detection_rates"]:
        methods[f"database: {name}"] = single_database(df, name)
    methods["cross-check (Lewis)"] = crosscheck(df, cfg)
    for m in ML_MODELS:
        methods[f"{m} (Lewis ML)"] = lewis_ml(df, cfg, m, seed=seed)
    for m in BLACKEYE_MODELS:
        methods[f"{m} (BlackEye)"] = blackeye(df, cfg, m, seed=seed)
    return methods


def run_m1(cfg=None):
    cfg = cfg or load_config()
    rows = []
    for seed in seeds_for(cfg):
        df = make_world(cfg, seed)
        for name, blocked in all_methods(df, cfg, seed).items():
            row = evaluate(df, blocked)
            row.update(seed=seed, method=name)
            rows.append(row)
    return pandas.DataFrame(rows)


def run_m2(cfg=None):
    cfg = cfg or load_config()
    rows = []
    for seed in seeds_for(cfg):
        df = make_world(cfg, seed)
        agent = human_agent(df, cfg, seed)
        for model in BLACKEYE_MODELS:
            for cleansing in (False, True):
                blocked = blackeye(df, cfg, model, use_cleansing=cleansing, seed=seed)
                row = evaluate(df, blocked)
                row["agent_wrong_corrected"] = agent_wrong_blocks_corrected(df, agent, blocked)
                row.update(seed=seed, model=model, cleansing="on" if cleansing else "off")
                rows.append(row)
    return pandas.DataFrame(rows)


def summarise(results, by="method"):
    """Mean ± standard deviation over the seeds."""
    cols = [c for c in METRICS + ["agent_wrong_corrected"] if c in results]
    grouped = results.groupby(by, sort=False)[cols]
    return grouped.mean().round(3).astype(str) + " ± " + grouped.std().round(3).astype(str)