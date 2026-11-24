# Step 4: the alert line (version 1)

Chosen on the tuning days (6 and 7). Reported on the test days (8 and 9).

## A. Are the scores usable as probabilities?

Each method is fitted on one tuning day and checked on the other, both ways. Lower is better.

| Method | Log loss | Brier score |
|---|---|---|
| Raw scores | 0.002763 | 0.000522 |
| Platt scaling | 0.002765 | 0.000523 |
| Isotonic regression (kept) | 0.002737 | 0.000521 |

Laundering the probabilities expect against laundering seen, by probability band:

| Probability | Tuning days: transfers | Expected | Seen | Test days: transfers | Expected | Seen |
|---|---|---|---|---|---|---|
| 0 to 0.001 | 841,560 | 5 | 5 | 699,669 | 10 | 20 |
| 0.001 to 0.01 | 117,442 | 176 | 176 | 144,507 | 216 | 133 |
| 0.01 to 0.05 | 3,727 | 107 | 107 | 13,413 | 377 | 140 |
| 0.05 to 0.2 | 2,076 | 172 | 172 | 4,467 | 367 | 165 |
| 0.2 to 0.5 | 145 | 43 | 43 | 220 | 68 | 45 |
| 0.5 to 0.8 | 60 | 32 | 32 | 65 | 35 | 30 |
| 0.8 to 1 | 514 | 501 | 501 | 451 | 437 | 423 |

On the tuning days expected equals seen by construction: the calibrator was fitted there. The test days are the check.

| Test day | Laundering expected | Laundering seen |
|---|---|---|
| 8 | 1083 | 514 |
| 9 | 427 | 442 |

## B. The alert line

The line is the score that gives 1,000 alerts a day on the tuning days: **0.0768**. A transfer exactly at the line has about a 1 in 15 chance of being laundering.

| Alerts a day on tuning days | Line | Chance of laundering at the line | Tuning: found | Tuning: precision | Test: alerts a day | Test: found | Test: precision | Test: share found |
|---|---|---|---|---|---|---|---|---|
| 250 | 0.6289 | 81.8% | 489 | 97.8% | 218 | 413 | 94.7% | 43.2% |
| 500 | 0.1462 | 12.3% | 610 | 61.0% | 670 | 533 | 39.8% | 55.8% |
| 1,000 | 0.0768 | 6.7% | 696 | 34.8% | 1,714 | 610 | 17.8% | 63.8% |
| 2,000 | 0.0298 | 2.6% | 803 | 20.1% | 4,584 | 744 | 8.1% | 77.8% |
| 5,000 | 0.0054 | 0.3% | 869 | 8.7% | 12,654 | 825 | 3.3% | 86.3% |

The same choice read as a cost trade. If one missed laundering transfer is worth R reviews, alert when the probability is at least 1 in R. Isotonic regression gives probabilities in steps, so the number of alerts moves in jumps.

| R | Tuning: alerts a day | Tuning: found | Test: alerts a day | Test: found | Test: precision | Test: share found |
|---|---|---|---|---|---|---|
| 5 | 360 | 576 | 368 | 498 | 67.7% | 52.1% |
| 10 | 571 | 629 | 800 | 545 | 34.1% | 57.0% |
| 20 | 1,398 | 748 | 2,602 | 663 | 12.7% | 69.4% |
| 50 | 2,922 | 845 | 8,190 | 797 | 4.9% | 83.4% |
| 100 | 3,261 | 855 | 9,308 | 803 | 4.3% | 84.0% |
| 200 | 4,240 | 868 | 11,532 | 816 | 3.5% | 85.4% |

## C. The test days at the fixed line

| Day | Part | Transfers | Alerts at the line | Laundering the probabilities expect | Found | Precision | Share found | Top 1,000 of the day: found | Precision |
|---|---|---|---|---|---|---|---|---|---|
| 6 | Tuning | 482,751 | 966 | 340 | 330 | 34.2% | 66.4% | 331 | 33.1% |
| 7 | Tuning | 482,773 | 1,034 | 359 | 366 | 35.4% | 67.9% | 365 | 36.5% |
| 8 | Test | 654,467 | 2,409 | 493 | 303 | 12.6% | 58.9% | 277 | 27.7% |
| 9 | Test | 208,325 | 1,019 | 304 | 307 | 30.1% | 69.5% | 305 | 30.5% |

At the fixed line, both test days: 3,428 alerts, 610 of 956 laundering transfers found. Precision 17.8% (95% interval 16.6% to 19.1%). Share found 63.8% (95% interval 59.0% to 68.1%, resampling whole schemes). F1 0.278. The probabilities expected 798 laundering transfers among the alerts.

As a daily queue of the top 1,000: 2,000 alerts, 582 found. Precision 29.1% (27.2% to 31.1%). Share found 60.9% (55.8% to 65.5%). F1 0.394. If only a sample of those alerts were quality-checked, about 881 would pin precision to within 3 points.

The line was set to give 1,000 alerts a day. Step 5 picks up what happened on day 8.

## D. What the line misses, estimated from a sample

On the test days 859,364 transfers sit below the line and 346 of them are laundering, 1 in 2,484. In production that count is unknown. The plans below estimate it from a sample of reviews.

| Band below the line | Tuning: transfers | Tuning: laundering | Test: transfers | Test: laundering |
|---|---|---|---|---|
| Just below: up to 5,000 a day | 8,000 | 173 | 21,881 | 215 |
| Middle: up to 50,000 a day | 90,000 | 130 | 87,813 | 73 |
| Far below: the rest | 865,524 | 37 | 749,670 | 58 |

Three plans with 3,000 reviews each, every plan drawn 5,000 times from the test days:

| Plan | Reviews per band | Laundering in the sample, on average | Samples with none | Average estimate | Typical error | Interval covers the truth |
|---|---|---|---|---|---|---|
| Simple random sample below the line | 3,000 | 1.2 | 29% | 353 | ±313 (90%) | 96.8% |
| By score band, in proportion to band size | 76, 307, 2,617 | 1.2 | 29% | 348 | ±313 (90%) | 87.6% |
| By score band, planned on the tuning days | 407, 1,001, 1,592 | 5.0 | 1% | 349 | ±213 (62%) | 91.2% |

Typical error is the root mean squared error. The interval is a 95% interval, so it should cover the truth about 95% of the time.

The planned design at other sizes:

| Reviews | Per band | Typical error | Interval covers the truth | Typical interval |
|---|---|---|---|---|
| 1,000 | 179, 333, 488 | ±367 (106%) | 85.8% | 179 to 4,335 |
| 3,000 | 407, 1,001, 1,592 | ±213 (62%) | 91.2% | 187 to 1,588 |
| 10,000 | 1,202, 3,338, 5,460 | ±116 (33%) | 94.8% | 218 to 745 |
| 30,000 | 3,475, 10,013, 16,512 | ±66 (19%) | 96.0% | 255 to 541 |

One sample, drawn once with the fixed seed: 0, 1, 0 laundering transfers in the three bands. Estimated misses 88 (95% interval 48 to 1,344); the truth is 346. That puts the share found at 87% (31% to 93%); the truth is 63.8%.

## E. Champion and challenger

Both sides get 1,000 alerts a day on the same test days.

| Comparison | First finds | Second finds | Only the first | Only the second | Gain in share found | 95% interval | McNemar p |
|---|---|---|---|---|---|---|---|
| Peer-group rule against the adopted model | 215 | 582 | 23 | 390 | +38.4 points | +32.6 to +43.9 | 3.06e-87 |
| Logistic regression against the adopted model | 514 | 582 | 41 | 109 | +7.1 points | +4.1 to +10.0 | 2.61e-08 |
| Adopted model against the network model that was not adopted | 582 | 592 | 22 | 32 | +1.0 points | -0.4 to +2.6 | 0.22 |

The interval resamples whole schemes. The McNemar test treats every transfer as independent, so it overstates the evidence.

A live split test would send some accounts to each model and compare the precision of their alerts. To tell 29.9% from 34.8% (the regression and the adopted model on the tuning days) with 80% power at the 5% level takes 1,460 alerts per arm, about 3 days at half the budget each. One week of alerts could show a gap of 3.1 points or more.

