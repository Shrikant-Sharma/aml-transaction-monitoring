# Step 6: scored the way the published papers score

F1 for the laundering class, in percent. Mean and spread over five seeds. Higher is better.

| Model | Source | HI-Small | LI-Small |
|---|---|---|---|
| LightGBM + graph features | Published | 62.86 ± 0.25 | 20.83 ± 1.50 |
| XGBoost + graph features | Published | 63.23 ± 0.17 | 27.30 ± 0.33 |
| GIN (graph neural network) | Published | 28.70 ± 1.13 | 7.90 ± 2.78 |
| PNA (graph neural network) | Published | 56.77 ± 2.41 | 14.85 ± 1.46 |
| Multi-GIN+EU (graph neural network) | Published | 64.79 ± 1.22 | 26.88 ± 6.63 |
| Multi-PNA+EU (graph neural network, best published) | Published | 68.16 ± 2.65 | 33.07 ± 2.63 |
| Boosted trees. Same fields: account behavior (22 features) | This run | 62.62 ± 0.98 | 26.53 ± 2.34 |
| Boosted trees. Same fields: + network (38 features) | This run | 64.39 ± 1.71 | 29.18 ± 2.28 |
| Boosted trees. Same fields: + 7-day pair history (23 features) | This run | 65.86 ± 0.44 | 35.27 ± 1.03 |
| Boosted trees. Version 2 (33 features) | This run | 72.97 ± 0.64 | 37.69 ± 5.38 |

Published numbers: Egressy and others, Provably Powerful Graph Neural Networks for Directed Multigraphs, arXiv 2306.11586v3, Table 2.

## What is the same and what is not

| | Published | This run |
|---|---|---|
| Split | Whole days, closest to 60/20/20 | The same rule: train days 0 to 5, validation days 6 and 7, test day 8 onward |
| Measure | F1 for the laundering class on the test days | The same |
| Days after day 9 | In the test part | In the test part here. Left out everywhere else in this project |
| Chosen on the validation days | Settings, loss weights, the training round to keep | The number of trees and the alert line |
| Seeds | Five, averaged | Five, averaged |
| Fields the model sees | Sender, receiver, time, amount received, currency received, payment format | "Same fields": those six. Version 2: also the customer file, the bank and the currency paid |
| What a test transfer may use | The whole graph, later transfers included | Only transfers up to the end of its own day |
| Training rows | Days 0 to 5 | Days 2 to 5. Days 0 and 1 are history only |
| Features | Learned by the network, or the publisher's graph features | The features of features.py |

## The test part, split in two

The published test part mixes two very different periods. Days 8 and 9 are regular days. After day 9 the files hold only the last legs of laundering schemes.

| Portfolio | Period | Transfers | Laundering | Same fields: account behavior | Same fields: + network | Same fields: + 7-day pair history | Version 2 |
|---|---|---|---|---|---|---|---|
| HI-Small | Day 8 onward (published test) | 863,900 | 1,611 | 62.62 ± 0.98 | 64.39 ± 1.71 | 65.86 ± 0.44 | 72.97 ± 0.64 |
| HI-Small | Days 8 and 9 only | 862,792 | 956 | 49.39 ± 1.46 | 49.65 ± 2.04 | 54.93 ± 1.08 | 61.78 ± 0.95 |
| HI-Small | After day 9 only | 1,108 | 655 | 83.10 ± 0.76 | 87.29 ± 0.36 | 81.06 ± 0.98 | 87.36 ± 0.92 |
| LI-Small | Day 8 onward (published test) | 1,174,673 | 802 | 26.53 ± 2.34 | 29.18 ± 2.28 | 35.27 ± 1.03 | 37.69 ± 5.38 |
| LI-Small | Days 8 and 9 only | 1,174,450 | 668 | 18.90 ± 1.88 | 20.44 ± 1.65 | 27.13 ± 1.04 | 27.85 ± 4.79 |
| LI-Small | After day 9 only | 223 | 134 | 79.00 ± 3.40 | 81.18 ± 1.20 | 74.05 ± 1.95 | 81.34 ± 1.67 |

## With the line at 0.5

The published code calls a transfer laundering when the model's score for laundering is the larger one, which is a line at 0.5 on scores shaped by a loss weight tuned on the validation days. A model with no such weight needs its line set on the validation days. With the line simply left at 0.5:

| Portfolio | Same fields: account behavior | Same fields: + network | Same fields: + 7-day pair history | Version 2 |
|---|---|---|---|---|
| HI-Small | 59.86 ± 0.50 | 62.93 ± 0.73 | 61.16 ± 0.87 | 71.33 ± 0.57 |
| LI-Small | 29.15 ± 0.93 | 30.76 ± 0.79 | 29.13 ± 1.90 | 36.33 ± 2.03 |

## Each seed

| Portfolio | Features | Seed | Trees | Line | Alerts | Found | Laundering | Precision | Share found | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| HI-Small | Same fields: account behavior (22) | 20261122 | 171 | 0.254 | 1,402 | 927 | 1,611 | 66.1% | 57.5% | 61.53 |
| HI-Small | Same fields: account behavior (22) | 20261123 | 176 | 0.302 | 1,180 | 893 | 1,611 | 75.7% | 55.4% | 63.99 |
| HI-Small | Same fields: account behavior (22) | 20261124 | 141 | 0.262 | 1,255 | 905 | 1,611 | 72.1% | 56.2% | 63.15 |
| HI-Small | Same fields: account behavior (22) | 20261125 | 137 | 0.245 | 1,347 | 916 | 1,611 | 68.0% | 56.9% | 61.93 |
| HI-Small | Same fields: account behavior (22) | 20261126 | 156 | 0.245 | 1,387 | 937 | 1,611 | 67.6% | 58.2% | 62.51 |
| HI-Small | Same fields: + network (38) | 20261122 | 156 | 0.229 | 1,511 | 985 | 1,611 | 65.2% | 61.1% | 63.10 |
| HI-Small | Same fields: + network (38) | 20261123 | 183 | 0.249 | 1,289 | 963 | 1,611 | 74.7% | 59.8% | 66.41 |
| HI-Small | Same fields: + network (38) | 20261124 | 155 | 0.217 | 1,471 | 985 | 1,611 | 67.0% | 61.1% | 63.92 |
| HI-Small | Same fields: + network (38) | 20261125 | 173 | 0.248 | 1,318 | 966 | 1,611 | 73.3% | 60.0% | 65.96 |
| HI-Small | Same fields: + network (38) | 20261126 | 214 | 0.212 | 1,595 | 1,003 | 1,611 | 62.9% | 62.3% | 62.57 |
| HI-Small | Same fields: + 7-day pair history (23) | 20261122 | 113 | 0.292 | 1,051 | 885 | 1,611 | 84.2% | 54.9% | 66.49 |
| HI-Small | Same fields: + 7-day pair history (23) | 20261123 | 184 | 0.263 | 1,116 | 899 | 1,611 | 80.6% | 55.8% | 65.93 |
| HI-Small | Same fields: + 7-day pair history (23) | 20261124 | 134 | 0.291 | 1,094 | 884 | 1,611 | 80.8% | 54.9% | 65.36 |
| HI-Small | Same fields: + 7-day pair history (23) | 20261125 | 166 | 0.308 | 1,034 | 873 | 1,611 | 84.4% | 54.2% | 66.01 |
| HI-Small | Same fields: + 7-day pair history (23) | 20261126 | 128 | 0.240 | 1,188 | 917 | 1,611 | 77.2% | 56.9% | 65.52 |
| HI-Small | Version 2 (33) | 20261122 | 186 | 0.300 | 1,130 | 1,014 | 1,611 | 89.7% | 62.9% | 73.99 |
| HI-Small | Version 2 (33) | 20261123 | 173 | 0.297 | 1,177 | 1,010 | 1,611 | 85.8% | 62.7% | 72.45 |
| HI-Small | Version 2 (33) | 20261124 | 101 | 0.356 | 1,018 | 958 | 1,611 | 94.1% | 59.5% | 72.88 |
| HI-Small | Version 2 (33) | 20261125 | 130 | 0.319 | 1,088 | 987 | 1,611 | 90.7% | 61.3% | 73.14 |
| HI-Small | Version 2 (33) | 20261126 | 150 | 0.293 | 1,132 | 993 | 1,611 | 87.7% | 61.6% | 72.40 |
| LI-Small | Same fields: account behavior (22) | 20261122 | 173 | 0.120 | 1,140 | 248 | 802 | 21.8% | 30.9% | 25.54 |
| LI-Small | Same fields: account behavior (22) | 20261123 | 137 | 0.126 | 1,350 | 247 | 802 | 18.3% | 30.8% | 22.96 |
| LI-Small | Same fields: account behavior (22) | 20261124 | 132 | 0.151 | 831 | 235 | 802 | 28.3% | 29.3% | 28.78 |
| LI-Small | Same fields: account behavior (22) | 20261125 | 115 | 0.187 | 746 | 211 | 802 | 28.3% | 26.3% | 27.26 |
| LI-Small | Same fields: account behavior (22) | 20261126 | 113 | 0.127 | 835 | 230 | 802 | 27.5% | 28.7% | 28.10 |
| LI-Small | Same fields: + network (38) | 20261122 | 144 | 0.144 | 960 | 233 | 802 | 24.3% | 29.1% | 26.45 |
| LI-Small | Same fields: + network (38) | 20261123 | 141 | 0.123 | 942 | 241 | 802 | 25.6% | 30.0% | 27.64 |
| LI-Small | Same fields: + network (38) | 20261124 | 121 | 0.172 | 709 | 222 | 802 | 31.3% | 27.7% | 29.38 |
| LI-Small | Same fields: + network (38) | 20261125 | 112 | 0.150 | 614 | 229 | 802 | 37.3% | 28.6% | 32.34 |
| LI-Small | Same fields: + network (38) | 20261126 | 130 | 0.151 | 747 | 233 | 802 | 31.2% | 29.1% | 30.08 |
| LI-Small | Same fields: + 7-day pair history (23) | 20261122 | 139 | 0.193 | 424 | 216 | 802 | 50.9% | 26.9% | 35.24 |
| LI-Small | Same fields: + 7-day pair history (23) | 20261123 | 158 | 0.183 | 389 | 215 | 802 | 55.3% | 26.8% | 36.10 |
| LI-Small | Same fields: + 7-day pair history (23) | 20261124 | 80 | 0.135 | 617 | 239 | 802 | 38.7% | 29.8% | 33.69 |
| LI-Small | Same fields: + 7-day pair history (23) | 20261125 | 84 | 0.171 | 367 | 212 | 802 | 57.8% | 26.4% | 36.27 |
| LI-Small | Same fields: + 7-day pair history (23) | 20261126 | 105 | 0.158 | 424 | 215 | 802 | 50.7% | 26.8% | 35.07 |
| LI-Small | Version 2 (33) | 20261122 | 143 | 0.296 | 333 | 227 | 802 | 68.2% | 28.3% | 40.00 |
| LI-Small | Version 2 (33) | 20261123 | 76 | 0.183 | 380 | 252 | 802 | 66.3% | 31.4% | 42.64 |
| LI-Small | Version 2 (33) | 20261124 | 138 | 0.254 | 309 | 228 | 802 | 73.8% | 28.4% | 41.04 |
| LI-Small | Version 2 (33) | 20261125 | 133 | 0.252 | 474 | 226 | 802 | 47.7% | 28.2% | 35.42 |
| LI-Small | Version 2 (33) | 20261126 | 106 | 0.213 | 813 | 237 | 802 | 29.2% | 29.6% | 29.35 |
