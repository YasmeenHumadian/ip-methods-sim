# Assumptions (mini simulation)

Everything below is a choice we made, not a finding from Lewis et al. (2020) or Jeon & Tak (2022). Numbers marked "paper" come from the papers.

## From the papers
- Lewis: four blacklist databases catch 13%, 16%, 46% and 73% of attackers (paper). Combining them caught about 82% (paper), which we use as a sanity target, not a setting.
- BlackEye: the human agent catches about 25% of attackers and 90% of the IPs they blacklist are really bad (paper, Table 4).
- BlackEye: noisy agent labels are cleaned by scoring with a first model, fitting two distributions, and removing points beyond the cross-over point (paper). Exact details are to be checked against the paper.
- BlackEye: the IP reputation score is one of its input features (paper); we use the Lewis cross-check score for it.

## Our assumptions
- World: 2,000 IPs, 10% attackers, 40% shared IPs (5 to 50 users each), 7 days, 3 logins per user per day.
- Four kinds of IP: single-user, shared, attacker, low-and-slow attacker (30% of attackers). Low-and-slow attackers look like normal users, so the task is not trivial.
- Log activity settings (failure rates, bursts, attempts per day) are invented. Shared IPs sometimes have a bad week of failed logins, so innocent shared IPs can look suspicious.
- Databases wrongly flag 2% of innocent IPs, and 3 times more often for shared IPs.
- Cross-check: databases weighted by detection rate, block at score 0.2. The threshold was chosen to land near Lewis' 82%.
- Regions are abstract (A to F). Attackers are 3 times more likely in E and F.
- Address blocks: 200 blocks, 20% are "bad neighbourhoods", 60% of attackers live in them. We have no real IP addresses, so this stands in for Lewis' address feature.
- Lewis-style ML sees only address block and (optionally) region. It is trained on true labels, with balanced class weights because attackers are rare. 70/30 split and 100 trees are settings to verify against the paper.
- The human agent wrongly blocks shared IPs 3 times more often than single-user IPs.
- Cleansing: ridge regression (alpha 1.0), normal curves for both classes, and we drop wrongly-sided points on both sides of the cross-over point.
- BlackEye models are trained on the agent's labels only and tested against the truth.

## Not in the mini simulation
- Time-to-blacklist (BlackEye's 66.2 vs 39.4 days) and any recovery of wrongly blocked IPs. Neither main paper studies recovery. These are in the big simulation plan.
- Real traffic. All results describe how the methods behave under these assumptions.