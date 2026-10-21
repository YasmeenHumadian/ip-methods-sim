# Run log

## Run 0: first look (single seed, not for slides)
World: 2,000 IPs, seed 1. Agent and cross-check scored on all IPs. ML and BlackEye scored on their 600 test IPs.

| method | caught | precision | wrongly blocked (share of innocent) | shared IPs wrongly blocked |
|---|---|---|---|---|
| human agent | 0.26 | 0.90 | 0.003 | 0.006 |
| cross-check (Lewis) | 0.87 | 0.58 | 0.068 | 0.111 |
| Naive Bayes (Lewis) | 0.55 | 0.26 | 0.176 | 0.211 |
| Random Forest (Lewis) | 0.37 | 0.26 | 0.115 | 0.128 |
| Logistic (Lewis) | 0.67 | 0.22 | 0.269 | 0.300 |
| BlackEye Logistic | 0.93 | 0.97 | 0.004 | 0.009 |
| BlackEye Random Forest | 0.83 | 1.00 | 0.000 | 0.000 |

Notes: with only about 60 attackers in the test set, ML numbers move a lot between seeds. The official 10-seed results (step 16) go in the next entry.