# Decisions

The choices in this project that were mine to make or to defend. Each entry gives the situation, the options on the table, the choice, the reason, and what happened. Every number is from my own run on Windows, unless it is marked as the reference run.

## 1. How to report version 2's fix, after my rerun did not reproduce it

**Date:** November 23, 2026

**Situation.** I reran the project on my own machine (Windows, Python 3.13, the pinned library versions). My features matched the reference run byte for byte, but the boosted models came out slightly different. The cause is the platform: XGBoost's Windows build and my processor's arithmetic give different trees from the same data and the same seed. A small diagnostic on made-up data showed it.

Most results moved by a few transfers. One did not. The reference run says version 2 (the pair's 7-day history) fixed the Friday alert spike and was confirmed on a second portfolio: laundering found went from 233 to 283 of 668, +7.5 points, interval +4.6 to +10.4. In my run the same check gave 249 to 252, +0.4 points, interval -1.6 to +2.6, and version 2 raised more Friday alerts than version 1 there (1,646 against 1,561).

**Options.**

| Option | What it would say | Cost | Gain |
|---|---|---|---|
| A. My single run as it is | The fix softened the Friday but showed no clear gain on the second portfolio | Nothing more to run | Simple and honest, but one draw |
| B. My run plus a check across seeds | The fix's gain as a range across training draws | About 20 minutes of running | Shows whether the result is steady |
| C. B, recorded as my own extension | The same as B, as my change to the project | Uses my one extension | B's evidence, from a problem I found myself |

**Choice.** C.

**Reason.** A gain from one trained model against another depends on the training draw as well as on the test days. The reference interval resamples laundering schemes on the test days but holds the trained models fixed, so it leaves the training draw out. One run cannot say whether a fix is real. Five seeds can show the spread.

**What happened.** I retrained version 1 and version 2 with five seeds on both portfolios, the same way the monitoring step trains them (`extensions/seed_check.py`). Laundering found on days 8 and 9 at 1,000 alerts a day:

| Portfolio | Seed | Version 1 | Version 2 | Gain | Friday alerts, version 1 | Friday alerts, version 2 |
|---|---|---|---|---|---|---|
| First (HI-Small), of 956 | Project's seed | 582 | 589 | +7 | 2,409 | 1,829 |
| | 1 | 578 | 599 | +21 | 2,183 | 1,295 |
| | 2 | 577 | 594 | +17 | 1,722 | 1,486 |
| | 3 | 576 | 612 | +36 | 2,118 | 1,026 |
| | 4 | 561 | 597 | +36 | 3,375 | 1,488 |
| | **Mean** | **574.8** | **598.2** | **+23.4 (+2.4 points)** | **2,361** | **1,425** |
| Second (LI-Small), of 668 | Project's seed | 249 | 252 | +3 | 1,561 | 1,646 |
| | 1 | 235 | 238 | +3 | 2,180 | 2,013 |
| | 2 | 242 | 256 | +14 | 1,586 | 1,573 |
| | 3 | 242 | 264 | +22 | 1,707 | 1,184 |
| | 4 | 233 | 268 | +35 | 1,358 | 1,256 |
| | **Mean** | **240.2** | **255.6** | **+15.4 (+2.3 points)** | **1,678** | **1,534** |

What it shows:

- On my machine, version 2 found at least as much laundering as version 1 on every draw, on both portfolios. On the second portfolio the gain ran from +3 to +35, a mean of 2.3 points. The reference's single draw claimed 7.5.
- On the first portfolio, version 2 lowered the Friday alerts on all five draws. On the second portfolio it lowered them clearly on two draws, about evenly on two, and raised them on one.
- My project-seed run was the weakest of my five draws. The reference's project-seed run was the strongest of five draws in the same check on Linux, where the second portfolio's gain ran from -8 to +50.
- Version 1's Friday spike itself varies with the draw: 1,722 to 3,375 alerts on a line set for 1,000.

So the 7-day history helps on average, and on this evidence it is a real fix, but a smaller and less steady one than a single run suggested. I report it as a range across seeds, and I no longer quote one interval for it.

Limit. Five seeds is a small number. The seeds and the second portfolio were both seen before this entry was written, so this is a check of steadiness, not a blind test.

## 2. Leaving the 16 network features out

**Date:** November 23, 2026

**Situation.** The design builds 16 network features: counterparties, money coming back, closed loops, split-and-merge paths. They are the features I expected to matter most for laundering. The design leaves them out of the model. After my own run I had to decide whether I stand by that.

**Options.**

| Option | Gain | Cost |
|---|---|---|
| A. Keep them in (48 features) | The richest picture of each transfer | 16 more features to build, test, monitor and explain, with no measured gain |
| B. Leave them out (32 features) | A simpler model that finds the same | The model cannot use loops or paths if they matter on other data |
| C. Decide by point score | Simple | A difference of a few transfers would decide the model |

**Choice.** B, by a rule: a feature set is adopted only if the whole 95% interval for its gain on the tuning days is above zero.

**Reason.** A point score can differ by chance. The interval resamples whole laundering schemes, so it shows how much of a gain is noise. Each extra feature is one more thing that can drift or break.

**What happened.** In my run, on the tuning days at 1,000 alerts a day, the model with network features found 694 and the model without found 696: -0.2 points, interval -1.9 to +1.5. On the test days it was 592 against 582: +1.0 points, interval -0.4 to +2.6. Neither interval is clear of zero. I kept them out.

Limit. Under the published rules, on six fields only, the network set scores a little higher (F1 64.4 against 62.6 on the first portfolio), inside the spread across seeds. The rule was written after the first results had been seen, so I say "I decided by a rule, not by eye", and never that the rule was set in advance.

## 3. A fixed score line, with the daily queue beside it

**Date:** November 23, 2026

**Situation.** A ranking has to become a yes or no each night. There are three ways to draw the line, and they behave differently on a bad day.

**Options.**

| Option | My run, test days | Gain | Cost |
|---|---|---|---|
| A. A fixed score line, set from capacity | 3,428 alerts, 610 found, precision 17.8% | The alert count becomes a warning when the scores change meaning | The queue overflows on a bad day |
| B. Each day's top 1,000 | 2,000 alerts, 582 found, precision 29.1% | Reviewers never overflow | A bad day is hidden, because the queue is always full |
| C. A cost rule: alert when the chance is at least 1 in R | At R = 20: about 2,600 alerts a day | Ties the line to what a catch is worth | Needs an agreed R, and the volume swings |

**Choice.** A for monitoring, with B reported beside it for workload.

**Reason.** I set the line from an assumed capacity of 1,000 alerts a day on the tuning days. That put it at a score of 0.0768, about a 1 in 15 chance of laundering. Frozen, it is the only same-night check that can show the scores have stopped meaning what they meant.

**What happened.** On the first unseen Friday the line raised 2,409 alerts, and the alert-volume check tripped that night. The score-shift check stayed under its limit (PSI 0.081). The feature-shift check went red on the Friday and on a normal Saturday. With a daily top 1,000, that Friday would have looked like any other day, at 27.7% precision.

Limit. 1,000 a day is an assumption, not a measured capacity. The reports show 250 to 5,000.

## 4. No class weight

**Date:** November 23, 2026

**Situation.** One transfer in about a thousand is laundering. The usual advice for a rare class is to weight it up in training. The design uses no weight. After my own run I had to decide whether I stand by that.

**Options.** Results on the tuning days, 1,036 laundering transfers, at 1,000 alerts a day. The true rate is 0.00107.

| Option | Found | Brier score | Mean score |
|---|---|---|---|
| A. No weight | 696 | 0.000522 | 0.00105 |
| B. The square root of the imbalance (27.7) | 694 | 0.001447 | 0.00479 |
| C. The full imbalance (765.4) | 663 | 0.006749 | 0.01693 |

**Choice.** A.

**Reason.** I judge the model at a fixed number of alerts a day, so only the ranking matters, and a weight did not improve the ranking. A weight also bends the scores away from probabilities, and the alert line and the cost rule both read the score as a probability.

**What happened.** The full weight found 33 fewer and pushed the mean score to about 16 times the true rate. With no weight the mean score sat within 2% of the true rate before any calibration. I kept no weight.

Limit. This was tested on one model and one portfolio's tuning days. A weight can still help a model that is judged at a fixed score line of 0.5.
