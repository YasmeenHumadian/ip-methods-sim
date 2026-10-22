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

## Run 1: official mini results (M1, M2)
Settings: config/mini_methods.yaml, 2,000 IPs per world, 10 seeds (1 to 10), mean ± sd. Command: python run_mini_methods.py. Raw per-seed results are in runs/*.csv (not committed). Re-run and confirmed identical on a second machine.

### M1: all methods
| method | caught | precision | wrongly blocked (innocent IPs) | wrongly blocked (innocent shared IPs) |
|---|---|---|---|---|
| human agent | 0.259 ± 0.029 | 0.899 ± 0.005 | 0.003 | 0.005 ± 0.002 |
| database: virustotal | 0.127 ± 0.025 | 0.272 ± 0.038 | 0.038 | 0.065 ± 0.007 |
| database: myip | 0.166 ± 0.022 | 0.335 ± 0.048 | 0.037 | 0.064 ± 0.009 |
| database: abuseipdb | 0.447 ± 0.033 | 0.577 ± 0.035 | 0.036 | 0.063 ± 0.008 |
| database: apility | 0.742 ± 0.042 | 0.703 ± 0.020 | 0.035 | 0.059 ± 0.008 |
| cross-check (Lewis) | 0.860 ± 0.018 | 0.578 ± 0.021 | 0.070 | 0.118 ± 0.012 |
| Naive Bayes (Lewis ML) | 0.520 ± 0.075 | 0.229 ± 0.034 | 0.195 | 0.200 ± 0.028 |
| Random Forest (Lewis ML) | 0.288 ± 0.056 | 0.232 ± 0.051 | 0.109 | 0.109 ± 0.023 |
| Logistic (Lewis ML) | 0.555 ± 0.070 | 0.223 ± 0.037 | 0.218 | 0.218 ± 0.047 |
| Logistic (BlackEye) | 0.853 ± 0.079 | 0.974 ± 0.041 | 0.003 | 0.005 ± 0.007 |
| Random Forest (BlackEye) | 0.808 ± 0.071 | 0.996 ± 0.008 | 0.000 | 0.001 ± 0.002 |

### M2: BlackEye with and without cleansing
| model | cleansing | caught | precision | wrongly blocked (shared IPs) | agent's wrong blocks corrected |
|---|---|---|---|---|---|
| Logistic | off | 0.923 ± 0.044 | 0.828 ± 0.071 | 0.051 ± 0.028 | 0.906 ± 0.186 |
| Logistic | on | 0.853 ± 0.079 | 0.974 ± 0.041 | 0.005 ± 0.007 | 1.000 ± 0.000 |
| Random Forest | off | 0.408 ± 0.120 | 0.987 ± 0.032 | 0.000 | 1.000 ± 0.000 |
| Random Forest | on | 0.808 ± 0.071 | 0.996 ± 0.008 | 0.001 ± 0.002 | 1.000 ± 0.000 |

### Checks
- Databases catch what Lewis reports (13, 16, 46, 73%): matched within noise.
- Human agent matches BlackEye Table 4 (about 25% caught, 90% precision): matched.
- Cross-check catches 86% (Lewis reports 82%): close; threshold is an assumption.
- "Agent's wrong blocks corrected" rests on about 2 wrong blocks per test set, so it is weak evidence.
- Lewis ML precision is low partly because of our choices (address block and region only, balanced class weights).
