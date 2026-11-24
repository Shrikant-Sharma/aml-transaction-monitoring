# Model document: transfer-level laundering score

Built on a public, synthetic benchmark. No real customer, bank or transfer is in it. Nothing here is deployed.

## 1. Purpose and use

Rank each day's transfers by how much they look like laundering, so that reviewers work the highest-scored ones first. The score is advice for a reviewer. It closes nothing and files nothing on its own.

## 2. Data

The IBM anti-money-laundering benchmark, HI-Small: 5,078,345 transfers, 5,177 labeled laundering (1 in 981). All but 1,108 of the transfers fall in ten regular days. Labels come from the simulator, so every laundering transfer is known. A real bank knows only what its reviewers confirmed.

| Days | Use |
|---|---|
| 0 and 1 | History only. Their look-back windows are short |
| 2 to 5 | Training. Day 5 also sets the number of trees |
| 6 and 7 | Tuning: settings, feature sets, calibration, the alert line, the monitoring references |
| 8 and 9 | Test. No choice for version 1 follows them. They were not sealed: every run of step 3 printed them while the project was built |
| 10 onward | Left out. Regular activity has stopped |

## 3. Method

Gradient-boosted trees (XGBoost), depth 8, no class weight. Features describe the transfer, the two customers and how active each account has been; every feature of a transfer uses only what happened up to the end of that transfer's day, and a test checks that.

Version 1 has 32 features. 16 network features (counterparties, loops, split-and-merge paths) were built and tested, and were left out: on the tuning days they added -0.2 points of laundering found, with a 95% interval from -1.9 to +1.5.

Version 2 adds one feature, transfers from this sender to this receiver on the 7 days before, after monitoring showed version 1 raising 2,409 alerts on a day set for 1,000 (section 6).

## 4. How it was tested

- Against simpler methods at the same number of alerts a day: a peer-group z-score rule, an isolation forest and a logistic regression.
- Intervals for the share of laundering found resample whole schemes, since transfers of one scheme are found or missed together.
- A richer feature set is adopted only if the interval for its gain on the tuning days stays above zero.
- Calibration is fitted on one tuning day and checked on the other, then checked again on the test days.
- The published scoring rules were also applied (step 6), to compare with published models.
- Unit tests: no feature looks ahead; every feature equals a slow recomputation; contributions add up to the score.

## 5. Results

Version 1, test days, 1,000 alerts a day: 582 of 956 laundering transfers found (60.9%; 95% interval 55.8% to 65.5%), precision 29.1%. The logistic regression found 514, the peer-group rule 215, the isolation forest 38.

By kind: 535 of 585 laundering transfers in labeled schemes were found, and 47 of 371 outside them.

Scored the published way (F1, mean of five seeds), with the same input fields the published models see: 62.6 to 65.9 on HI-Small and 26.5 to 35.3 on LI-Small across three feature sets. The best published graph neural network reports 68.2 and 33.1. results/published_protocol.md says what is the same and what is not.

## 6. What monitoring found

On day 8, the first unseen Friday, the fixed line raised 2,409 alerts and precision fell to 12.6%. The alert-volume check tripped that night. 64% of that day's alerts were pairs that had paid each other exactly 7 days earlier: weekly payers, which a 3-day memory sees as first-time pairs.

Version 2 gives the model the pair's history over the 7 days before. On the same day it raises 1,829 alerts and finds 300. Those days suggested the change, so the better check is the second portfolio: there, at 1,000 alerts a day, version 1 finds 249 and version 2 finds 252 of 668 (+0.4 points; 95% interval -1.6 to +2.6). That check was not blind: a wider candidate was tried on both portfolios in the same trial, and on the second it found 228.

## 7. Limits

- Synthetic data. The simulator's laundering follows eight scripted patterns. Real laundering adapts.
- Laundering outside the labeled schemes is mostly missed: 47 of 371 found.
- Ten days of data hold one weekly cycle. The training days hold no Friday with full history. A monthly pattern would be invisible.
- Features are built once a day, after the day closes. The model does not score a transfer at the moment it is made.
- Labels are complete and immediate here. With real labels, only worked alerts are labeled, and late.
- What the line misses can only be estimated from a sample: 346 of 859,364 transfers below the line are laundering. A plan of 3,000 reviews by score band estimates that count to within about 62%.
- A model trained on one portfolio loses ground on another: on the second portfolio it finds 243 where a model trained there finds 252.

## 8. Monitoring plan

| Check | Watches | Compared with | Limits | If it trips |
|---|---|---|---|---|
| Feed | Transfers received in the day | The median training day | Red below 0.3 times that, red above 3 times that | Stop. Publish no alerts for the day and call the data owner. The other checks are not read. |
| Alert volume | Alerts at the fixed line | Alerts a day on the tuning days | Amber below 0.5 times that, amber above 2 times that | Find out what the extra alerts have in common before the queue is worked. |
| Score shift | Population stability index of the model score | The tuning days | Amber above 0.1, red above 0.25 | Amber: watch. Red: review the model before relying on the line. |
| Feature shift | The largest population stability index among the model's features | The tuning days | Amber above 0.1, red above 0.25 | Amber: watch. Red: find the feature and its source. |
| Precision | Precision of the day's alerts, once reviewers have worked them | The lower end of the 95% interval for precision on the tuning days | Amber below that | Review what was missed and what was flagged. Two days in a row: revalidate. |

Revalidate when: a new portfolio, product or channel is scored; any red on two days in a row; precision amber on two days in a row.

Retraining: with the labels of days 6 and 7 added, version 2 finds 593 against 589 on days 8 and 9 (+0.4 points; 95% interval -1.0 to +1.9).

## 9. Versions

| Version | Features | Change | Why |
|---|---|---|---|
| 1 | 32 | First model | Chosen on the tuning days (step 3) |
| 2 | 33 | Adds the pair's history over the 7 days before | The alert spike on the first unseen Friday (step 5) |

