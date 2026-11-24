# Step 3: the rule, the forest, the regression and the boosted models

A replay on a public, synthetic benchmark. No real customer or bank is in it.

## The days

| Part | Days | Transfers | Laundering | One in |
|---|---|---|---|---|
| Warm-up (history only) | 0 to 1 | 1,869,370 | 730 | 2,561 |
| Training | 2 to 5 | 1,379,551 | 1,800 | 766 |
| Tuning | 6 to 7 | 965,524 | 1,036 | 932 |
| Test | 8 to 9 | 862,792 | 956 | 903 |
| After day 9 (left out) | 10 to 17 | 1,108 | 655 | 2 |

Day 0 is September 1, 2022. Most features look back 3 days, so days 0 and 1 have less history behind them than the rest. They feed the history of later days and no model learns from their rows. Regular activity stops after day 9. The later days hold only the last legs of laundering schemes, so they are scored and left out of every result.

## Test days at 1,000 alerts a day

| Method | Alerts | Laundering found | Precision | 95% interval | Share of laundering found | 95% interval |
|---|---|---|---|---|---|---|
| Peer-group rule (z-scores) | 2,000 | 215 | 10.8% | 9.5% to 12.2% | 22.5% | 17.7% to 27.4% |
| Isolation forest (no labels) | 2,000 | 38 | 1.9% | 1.4% to 2.6% | 4.0% | 2.6% to 5.5% |
| Logistic regression | 2,000 | 514 | 25.7% | 23.8% to 27.7% | 53.8% | 48.9% to 58.3% |
| Boosted trees: transfer fields | 2,000 | 237 | 11.8% | 10.5% to 13.3% | 24.8% | 21.0% to 28.4% |
| Boosted trees: + customer fields | 2,000 | 334 | 16.7% | 15.1% to 18.4% | 34.9% | 30.8% to 38.8% |
| Boosted trees: + account behavior (adopted) | 2,000 | 582 | 29.1% | 27.2% to 31.1% | 60.9% | 55.8% to 65.5% |
| Boosted trees: + network | 2,000 | 592 | 29.6% | 27.6% to 31.6% | 61.9% | 57.0% to 66.3% |

The precision interval is a Wilson interval. The interval for the share found resamples whole schemes, because transfers of one scheme are found or missed together.

## Test days at other numbers of alerts a day

Laundering found, as a share of all laundering on those days (956 laundering transfers in 862,792).

| Method | 250 a day | 500 a day | 1,000 a day | 2,000 a day | 5,000 a day | Average precision | ROC AUC |
|---|---|---|---|---|---|---|---|
| Peer-group rule (z-scores) | 5.1% | 6.8% | 22.5% | 31.4% | 34.5% | 0.054 | 0.722 |
| Isolation forest (no labels) | 1.3% | 3.1% | 4.0% | 6.3% | 9.3% | 0.004 | 0.778 |
| Logistic regression | 35.5% | 44.9% | 53.8% | 62.6% | 73.0% | 0.436 | 0.967 |
| Boosted trees: transfer fields | 10.0% | 15.9% | 24.8% | 35.9% | 50.7% | 0.069 | 0.921 |
| Boosted trees: + customer fields | 19.2% | 26.0% | 34.9% | 45.6% | 62.7% | 0.203 | 0.970 |
| Boosted trees: + account behavior | 47.0% | 54.3% | 60.9% | 69.7% | 79.4% | 0.567 | 0.981 |
| Boosted trees: + network | 47.2% | 55.1% | 61.9% | 70.1% | 80.8% | 0.578 | 0.982 |

## Tuning days

Laundering found, as a share of all laundering on those days (1,036 laundering transfers in 965,524).

| Method | 250 a day | 500 a day | 1,000 a day | 2,000 a day | 5,000 a day | Average precision | ROC AUC |
|---|---|---|---|---|---|---|---|
| Peer-group rule (z-scores) | 5.2% | 7.2% | 25.4% | 38.6% | 44.2% | 0.071 | 0.754 |
| Isolation forest (no labels) | 3.5% | 4.2% | 6.4% | 7.7% | 11.4% | 0.010 | 0.786 |
| Logistic regression | 39.4% | 50.0% | 57.8% | 65.5% | 79.6% | 0.518 | 0.973 |
| Boosted trees: transfer fields | 9.4% | 15.1% | 23.6% | 34.8% | 50.2% | 0.076 | 0.920 |
| Boosted trees: + customer fields | 19.4% | 25.8% | 34.6% | 43.9% | 59.9% | 0.222 | 0.970 |
| Boosted trees: + account behavior | 47.2% | 58.5% | 67.2% | 77.2% | 84.0% | 0.649 | 0.989 |
| Boosted trees: + network | 46.5% | 59.9% | 67.0% | 77.4% | 84.4% | 0.652 | 0.989 |

## Does each richer feature set pay for itself?

Judged on the tuning days at 1,000 alerts a day. A set is adopted only if the whole 95% interval for its gain is above zero.

| From | To | Found before | Found after | Only the first finds | Only the second finds | Gain in share found | 95% interval | Adopted |
|---|---|---|---|---|---|---|---|---|
| transfer fields | + customer fields | 245 | 358 | 48 | 161 | +10.9 points | +7.6 to +14.3 | Yes |
| + customer fields | + account behavior | 358 | 696 | 16 | 354 | +32.6 points | +28.9 to +36.2 | Yes |
| + account behavior | + network | 696 | 694 | 37 | 35 | -0.2 points | -1.9 to +1.5 | No |

Adopted model: Boosted trees: + account behavior (32 features, 137 trees).

Against the logistic regression on the same tuning days, the adopted model finds 696 to the regression's 599: +9.4 points, 95% interval +6.6 to +12.1.

## The rule variants

Peer means, spreads and each variant's terms come from the training days. The variant that finds most on the tuning days at 1,000 alerts a day is kept as the rule.

| Variant | Found on training days (of 1,800) | Found on tuning days (of 1,036) | Share found on tuning days |
|---|---|---|---|
| Classic: largest z-score of amount, sender volume, receiver volume | 7 | 2 | 0.2% |
| Largest z-score of all 34 quantities | 171 | 101 | 9.7% |
| Best single z-score, high or low side | 171 | 144 | 13.9% |
| Sum of the best z-scores, up to 5, high or low side | 389 | 263 | 25.4% |

Kept: Sum of the best z-scores, up to 5, high or low side. Its terms:

- different accounts the sender paid by ACH in 3 days (high side)
- accounts on that 5-day loop (high side)
- ACH transfers the sender made in 3 days (low side)
- currencies the receiver used in 3 days (low side)
- ACH transfers the receiver got in 3 days (low side)

A side check on where the terms are chosen. The same sum of five with its terms chosen on the tuning days instead finds 323 on the tuning days and 79 of 956 on the test days. Both tuning days are mid-week. The training days hold a weekend and two weekdays, and terms chosen there hold up on the test days.

## The grid

Fitted on days 2 to 4, trees stopped on day 5, compared on the tuning days.

| Depth | Smallest leaf weight | Trees | Average precision on the tuning days |
|---|---|---|---|
| 6 | 1 | 121 | 0.6147 |
| 6 | 5 | 118 | 0.6257 |
| 8 | 1 | 99 | 0.6147 |
| 8 | 5 | 108 | 0.6281 |

Chosen: depth 8, smallest leaf weight 5.

## The boosted models

| Model | Features | Trees |
|---|---|---|
| Boosted trees: transfer fields | 8 | 74 |
| Boosted trees: + customer fields | 14 | 297 |
| Boosted trees: + account behavior | 32 | 137 |
| Boosted trees: + network | 48 | 108 |

## Weighting the rare class (adopted model, tuning days)

| Weight on laundering | Value | Average precision | Found at the main budget | Brier score | Mean score | True rate |
|---|---|---|---|---|---|---|
| none | 1.0 | 0.6485 | 696 | 0.000522 | 0.00105 | 0.00107 |
| square root of the imbalance | 27.7 | 0.6316 | 694 | 0.001447 | 0.00479 | 0.00107 |
| the full imbalance | 765.4 | 0.6021 | 663 | 0.006749 | 0.01693 | 0.00107 |

A weight changes the scores, so the mean score moves away from the true rate. No weight is used.

## Learning from the warm-up days too (adopted model, tuning days)

| Trained on | Transfers | Laundering | Trees | Average precision | Found at the main budget |
|---|---|---|---|---|---|
| days 2 to 5 | 1,379,551 | 1,800 | 137 | 0.6485 | 696 |
| days 0 to 5 | 3,248,921 | 2,530 | 160 | 0.6255 | 669 |

## Test days by kind of laundering, at 1,000 alerts a day

| Laundering | Transfers | Rule finds | Regression finds | Adopted model finds | Share |
|---|---|---|---|---|---|
| Bipartite | 44 | 9 | 33 | 38 | 86% |
| Cycle | 55 | 18 | 36 | 47 | 85% |
| Fan-in | 57 | 3 | 43 | 49 | 86% |
| Fan-out | 65 | 59 | 62 | 65 | 100% |
| Gather-scatter | 127 | 49 | 104 | 121 | 95% |
| Not in a labeled scheme | 371 | 13 | 43 | 47 | 13% |
| Random | 43 | 8 | 34 | 38 | 88% |
| Scatter-gather | 110 | 50 | 97 | 103 | 94% |
| Stack | 84 | 6 | 62 | 74 | 88% |
| In a labeled scheme (all 8 types) | 585 | 202 | 471 | 535 | 91% |
| In a labeled scheme that began before the test days | 375 | 125 | 307 | 352 | 94% |
| In a labeled scheme that began on the test days | 210 | 77 | 164 | 183 | 87% |
| All laundering | 956 | 215 | 514 | 582 | 61% |

## F1 for the laundering class

The line is the one with the best F1 on the tuning days, then applied to the test days.

| Method | Best F1 on tuning days | F1 on test days at that line |
|---|---|---|
| Peer-group rule (z-scores) | 0.190 | 0.135 |
| Isolation forest (no labels) | 0.048 | 0.019 |
| Logistic regression | 0.547 | 0.461 |
| Boosted trees: transfer fields | 0.161 | 0.150 |
| Boosted trees: + customer fields | 0.272 | 0.246 |
| Boosted trees: + account behavior | 0.667 | 0.606 |
| Boosted trees: + network | 0.669 | 0.611 |
