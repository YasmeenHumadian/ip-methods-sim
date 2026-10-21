"""Run both experiments and save the raw results to runs/."""
from pathlib import Path

from simulator.experiments import run_m1, run_m2, summarise

Path("runs").mkdir(exist_ok=True)

m1 = run_m1()
m1.to_csv("runs/m1_comparison.csv", index=False)
print("M1: every method (mean ± sd over seeds)")
print(summarise(m1, "method").to_string())

m2 = run_m2()
m2.to_csv("runs/m2_cleansing.csv", index=False)
print()
print("M2: BlackEye with cleansing on and off")
print(summarise(m2, ["model", "cleansing"]).to_string())