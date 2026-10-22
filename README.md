# Wrongly blocked IP addresses: a mini simulation of two papers

COMP516 Research Methods, University of Liverpool, Group 4. Companion to our literature review
*Wrongly Blocked IP Addresses and Reputation Recovery: A Scoping Review of Detection, Mitigation and Recovery Methods*.

Live demo: https://yasmeenhumadian-ip-methods-sim-app-y2kcjc.streamlit.app (may take a moment to wake up)

## What this is

A small, fully reproducible simulation that puts two published ways of deciding which IP addresses to block side by side, in one invented world where the truth is known.

- **Lewis et al. (2020):** check public blacklist databases, combine them in a cross-check score, and train ML models on an IP's address block and region.
- **BlackEye (Jeon & Tak, 2022):** learn from what each IP did in the login logs, trained on a human agent's noisy labels that are cleaned first.

We ask what both papers care about: how many attackers get caught, and how many innocent IPs, especially innocent shared IPs (a campus or an office behind one address), get blocked by mistake.

It is a simulation. Every number describes how the methods behave under our assumptions, not how they behave on real traffic.

## The simulated world

2,000 IP addresses (settings in `config/mini_methods.yaml`): normal single-user IPs, shared IPs with 5 to 50 users, obvious attackers, and "low-and-slow" attackers that look like normal users. Each IP has login activity over 7 days. The truth (`kind`, `is_malicious`, `n_users`) is hidden from every method and used only to score them.

## Methods compared

| Method | From | What it may see |
|---|---|---|
| Human agent | BlackEye baseline | Catches about 25% of attackers, 90% of its blocks are right (paper, Table 4) |
| One blacklist database (4) | Lewis | Its own list (detection rates 13%, 16%, 46%, 73%) |
| Cross-check | Lewis | Weighted vote of the four databases |
| Naive Bayes, Random Forest, Logistic | Lewis | Address block, optionally region |
| Logistic, Random Forest | BlackEye | Log features and the cross-check reputation score, trained on the agent's labels, with data cleansing |

Every method is scored the same way (`simulator/evaluate.py`): share of attackers caught, precision, share of innocent IPs blocked, and share of innocent shared IPs blocked.

## Main results

Mean over 10 random worlds (full tables with standard deviations in `run_log.md`):

| Method | Attackers caught | Innocent shared IPs wrongly blocked |
|---|---|---|
| Human agent | 26% | 0.5% |
| Lewis cross-check | 86% | 11.8% |
| Lewis Logistic (address and region) | 56% | 21.8% |
| BlackEye Logistic (with cleansing) | 85% | 0.5% |

BlackEye's data cleansing matters: without it the Logistic model wrongly blocks 5.1% of innocent shared IPs, and the Random Forest catches only 41% of attackers because it copies the agent's blind spots.

## Run it

Windows PowerShell:

```
git clone https://github.com/YasmeenHumadian/ip-methods-sim
cd ip-methods-sim
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest                      # 52 tests
streamlit run app.py        # the interactive demo
python run_mini_methods.py  # both experiments, raw results go to runs/
```

## Project layout

```
app.py                      Streamlit GUI (display only, no logic)
run_mini_methods.py         runs experiments M1 and M2
config/mini_methods.yaml    every setting, marked (paper) or (assumption)
simulator/world.py          the invented world and its login activity
simulator/databases.py      simulated blacklist databases
simulator/methods_lewis.py  cross-check and Lewis-style ML
simulator/methods_blackeye.py  human agent, cleansing, BlackEye models
simulator/evaluate.py       scoring shared by every method
simulator/experiments.py    experiments M1 (all methods) and M2 (cleansing on/off)
tests/                      52 tests
assumptions.md              every choice that is ours, not the papers'
run_log.md                  results of each run
docs/method_table.md        the methods as described in each paper
```

## Limits

- Synthetic data. Features, attacker behaviour and the address-block feature are our inventions.
- The Lewis-style ML sees only address block and region and is trained on true labels. It shows what that information can do in our world, not what Lewis' models do on real data.
- BlackEye's advantage comes partly from the world: attackers leave behaviour in the logs by design.
- Time-to-blacklist and recovery of wrongly blocked IPs are not simulated. Neither main paper studies recovery.

See `assumptions.md` for the full list.

## Next: the big simulation (planned)

Adds time, time-to-blacklist, and recovery policies (fixed expiry, decaying score, probation) for wrongly blocked IPs, over 30 seeds and about four experiments. The mini version is a true subset of it.

## References

- Lewis, Tambaliuc, Narman, Yoo (2020). IP Reputation Analysis of Public Databases and Machine Learning Techniques. ICNC 2020. doi:10.1109/ICNC47757.2020.9049760
- Jeon, Tak (2022). BlackEye: automatic IP blacklisting using machine learning from security logs. Wireless Networks 28:937-948. doi:10.1007/s11276-019-02201-5
