# Step 5: monitoring, a fix, and a second portfolio

Version 1 (32 features), its alert line (0.0768) and the rules were fixed before this step.

## The rules

| Check | Watches | Compared with | Limits | If it trips |
|---|---|---|---|---|
| Feed | transfers received in the day | the median training day | red below 0.3, red above 3.0 | Stop. Publish no alerts for the day and call the data owner. The other checks are not read. |
| Alert volume | alerts at the fixed line | alerts a day on the tuning days | amber below 0.5, amber above 2.0 | Find out what the extra alerts have in common before the queue is worked. |
| Score shift | population stability index of the model score | the tuning days | amber above 0.1, red above 0.25 | Amber: watch. Red: review the model before relying on the line. |
| Feature shift | the largest population stability index among the model's features | the tuning days | amber above 0.1, red above 0.25 | Amber: watch. Red: find the feature and its source. |
| Precision | precision of the day's alerts, once reviewers have worked them | the lower end of the 95% interval for precision on the tuning days | amber below 1.0 | Review what was missed and what was flagged. Two days in a row: revalidate. |

The median training day has 344,760 transfers. The line gave 1,000 alerts a day on the tuning days at a precision of 34.8%; the lower end of its 95% interval, 32.7%, is the precision floor.

## A. Version 1, day by day

Days 6 and 7 are the tuning days, where the line and the references were set. Days 8 and 9 are new. The four checks on the left can run the same night. The precision check needs the reviewers' labels and comes later.

| Day | Transfers | Feed | Alerts | Alert volume | Score shift (PSI) | | Largest feature shift (PSI) | | Feature that moved most | Found, of laundering | Precision | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 482,751 | ok | 966 | ok | 0.000 | ok | 0.000 | ok |  | 330 of 497 | 34.2% | ok |
| 7 | 482,773 | ok | 1,034 | ok | 0.000 | ok | 0.000 | ok |  | 366 of 539 | 35.4% | ok |
| 8 | 654,467 | ok | 2,409 | AMBER | 0.081 | ok | 0.421 | RED | transfers from this sender to this receiver on the 2 days before | 303 of 514 | 12.6% | AMBER |
| 9 | 208,325 | ok | 1,019 | ok | 0.092 | ok | 0.558 | RED | transfers the sender received today | 307 of 442 | 30.1% | AMBER |
| 10 | 396 | RED | 232 | not read | 9.597 | not read | 12.369 | not read |  | 215 of 232 | | not read |
| 11 | 281 | RED | 166 | not read | 9.672 | not read | 13.359 | not read |  | 158 of 170 | | not read |
| 12 | 184 | RED | 103 | not read | 9.571 | not read | 14.075 | not read |  | 97 of 106 | | not read |
| 13 | 121 | RED | 64 | not read | 9.550 | not read | 14.075 | not read |  | 60 of 70 | | not read |
| 14 | 46 | RED | 29 | not read | 10.883 | not read | 14.075 | not read |  | 28 of 28 | | not read |
| 15 | 46 | RED | 26 | not read | 10.616 | not read | 14.153 | not read |  | 26 of 26 | | not read |
| 16 | 23 | RED | 13 | not read | 10.593 | not read | 14.153 | not read |  | 13 of 15 | | not read |
| 17 | 11 | RED | 8 | not read | 10.628 | not read | 14.153 | not read |  | 8 of 8 | | not read |

After day 9 the file holds 1,108 transfers in all, 655 of them laundering. The feed check stops scoring on day 10. Had it not, the line would have raised 641 alerts on those days and found 605 of the 655.

Day 8: the line was set to give 1,000 alerts a day and gave 2,409. The feature that moved most that day: transfers from this sender to this receiver on the 2 days before (PSI 0.421).

Laundering the calibrated probabilities expect among the alerts, against what was found (known once labels are in):

| Day | Expected in the alerts | Found |
|---|---|---|
| 6 | 340 | 330 |
| 7 | 359 | 366 |
| 8 | 493 | 303 |
| 9 | 304 | 307 |

## B. What the alerts of day 8 have in common

Everything in this table except the last two columns can be computed the same night, before any label exists.

| Day | Alerts | By ACH | Pair had no transfer on the 2 days before | Pair had a transfer exactly 7 days before | Sender sent nothing in the 3 days before | False alerts | Of those, pair paid 7 days before |
|---|---|---|---|---|---|---|---|
| 7 | 1,034 | 100% | 97% | 3% | 58% | 668 | 3% |
| 8 | 2,409 | 100% | 99% | 64% | 81% | 2,106 | 73% |
| 9 | 1,019 | 100% | 98% | 4% | 55% | 712 | 4% |

Version 1 remembers 3 days. A pair that pays once a week looks like a first-time pair to it.

## C. Version 2: one more feature

Added: transfers from this sender to this receiver on the 7 days before (`pair_before_7d`). Same training days, same settings, 118 trees, 33 features. Its line for 1,000 alerts a day on the tuning days is 0.0792.

On the tuning days the new feature changes little: at 1,000 alerts a day version 1 finds 696 and version 2 finds 691 (-0.5 points, 95% interval -1.7 to +0.7). Neither tuning day is a Friday, so the tuning days cannot show what the feature is for. Its case rests on the days below.

Also tried: adding transfers the sender made on the 7 days before and transfers the receiver got on the 7 days before. On the tuning days that wider model finds 693 to version 2's 691 (+0.2 points, 95% interval -1.0 to +1.4). Left out. On days 8 and 9 it finds 596 to version 2's 589, and at its own line it raises 1,666 alerts on day 8.

The first portfolio again, both versions:

| Day | Transfers | Laundering | Version 1: alerts at its line | Found | Top 1,000: found | Version 2: alerts at its line | Found | Top 1,000: found |
|---|---|---|---|---|---|---|---|---|
| 6 | 482,751 | 497 | 966 | 330 | 331 | 989 | 332 | 332 |
| 7 | 482,773 | 539 | 1,034 | 366 | 365 | 1,011 | 359 | 359 |
| 8 | 654,467 | 514 | 2,409 | 303 | 277 | 1,829 | 300 | 284 |
| 9 | 208,325 | 442 | 1,019 | 307 | 305 | 958 | 304 | 305 |

Days 8 and 9 at 1,000 alerts a day: version 1 finds 582, version 2 finds 589 of 956 (+0.7 points, 95% interval -0.5 to +2.0). Version 2 at its line: 2,787 alerts, 604 found, precision 21.7%, share found 63.2%. As a daily queue: precision 29.4% (27.5% to 31.5%), share found 61.6% (56.5% to 66.1%). These two days are the ones that suggested the feature, so this is not a clean test of it. Section D is the better one.

Version 2 under the same rules. `pair_before_7d` is left out of the feature-shift check: on day 6 it has 6 days of history behind it and on day 7 it has 7, so on the reference days it is still filling up.

| Day | Transfers | Feed | Alerts | Alert volume | Score shift (PSI) | | Largest feature shift (PSI) | | Feature that moved most | Found, of laundering | Precision | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 482,751 | ok | 989 | ok | 0.001 | ok | 0.000 | ok |  | 332 of 497 | 33.6% | ok |
| 7 | 482,773 | ok | 1,011 | ok | 0.001 | ok | 0.000 | ok |  | 359 of 539 | 35.5% | ok |
| 8 | 654,467 | ok | 1,829 | ok | 0.149 | AMBER | 0.421 | RED | transfers from this sender to this receiver on the 2 days before | 300 of 514 | 16.4% | AMBER |
| 9 | 208,325 | ok | 958 | ok | 0.077 | ok | 0.558 | RED | transfers the sender received today | 304 of 442 | 31.7% | AMBER |

| Day | Laundering expected in version 2's alerts | Found |
|---|---|---|
| 6 | 342 | 332 |
| 7 | 353 | 359 |
| 8 | 409 | 300 |
| 9 | 289 | 304 |

## D. The check on a second portfolio

LI-Small: 6,924,049 transfers, 1 laundering in 1,942. Other banks and customers. It played no part in finding the problem or in designing the feature. Both versions are trained on its days 2 to 5 (1,880,369 transfers, 1,333 laundering) with the settings chosen on the first portfolio (94 and 60 trees), and each gets its own line on its tuning days.

| Day | Transfers | Laundering | Version 1: alerts at its line | Found | Top 1,000: found | Version 2: alerts at its line | Found | Top 1,000: found |
|---|---|---|---|---|---|---|---|---|
| 6 | 658,504 | 368 | 1,007 | 153 | 153 | 1,021 | 155 | 153 |
| 7 | 657,938 | 402 | 993 | 174 | 174 | 979 | 166 | 168 |
| 8 | 891,573 | 367 | 1,561 | 145 | 114 | 1,646 | 143 | 107 |
| 9 | 282,877 | 301 | 1,008 | 136 | 135 | 913 | 136 | 145 |

Days 8 and 9 at 1,000 alerts a day: version 1 finds 249, version 2 finds 252 of 668 (+0.4 points, 95% interval -1.6 to +2.6). At their own lines: version 1 raises 2,569 alerts and finds 281 (precision 10.9%); version 2 raises 2,559 and finds 279 (precision 10.9%). Average precision on those days: 0.238 and 0.237.

On this portfolio's tuning days, as on the first, the feature changes little (327 against 321; -0.8 points, 95% interval -2.9 to +1.3), and the wider model is no better than version 2 (321 against 317; -0.5 points, 95% interval -2.4 to +1.4).

This check was not blind. Version 2 and the wider model were both fitted on this portfolio in the same trial as on the first, so its test days had been seen for both before version 2 was fixed. The wider model fails here: on days 8 and 9 it finds 228 of 668, against version 1's 249 (-3.1 points, 95% interval -5.7 to -0.7), and at its own line it raises 1,868 alerts on day 8.

Where version 2 finds laundering here: 155 of 184 in labeled schemes, 97 of 484 outside them.

## E. Does a model carry over to a new portfolio?

Version 2 as trained on the first portfolio, applied to the second with its old line and the first portfolio's references:

| Day | Transfers | Feed | Alerts | Alert volume | Score shift (PSI) | | Largest feature shift (PSI) | | Feature that moved most | Found, of laundering | Precision | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 658,504 | ok | 1,068 | ok | 0.001 | ok | 0.057 | ok |  | 148 of 368 | 13.9% | AMBER |
| 7 | 657,938 | ok | 1,052 | ok | 0.005 | ok | 0.057 | ok |  | 169 of 402 | 16.1% | AMBER |
| 8 | 891,573 | ok | 2,236 | AMBER | 0.161 | AMBER | 0.418 | RED | transfers from this sender to this receiver on the 2 days before | 136 of 367 | 6.1% | AMBER |
| 9 | 282,877 | ok | 1,065 | ok | 0.090 | ok | 0.555 | RED | transfers the sender received today | 140 of 301 | 13.1% | AMBER |

The features that differ most between the two portfolios' tuning days:

| Feature | PSI |
|---|---|
| currency | 0.057 |
| number of accounts the receiver's owner holds | 0.026 |
| number of accounts the sender's owner holds | 0.010 |
| sender's customer type | 0.004 |
| size of the transfer | 0.002 |
| payment format | 0.001 |

The second portfolio's test days (1,174,450 transfers, 668 laundering):

| Option | Line | Alerts a day | Laundering found | Precision | Share found | 95% interval | Average precision |
|---|---|---|---|---|---|---|---|
| Trained on the first portfolio, old line | 0.0792 | 1,650 | 276 | 8.4% | 41.3% | 35.3% to 47.1% | 0.255 |
| Trained on the first portfolio, line re-set on the second | 0.0824 | 1,542 | 267 | 8.7% | 40.0% | 34.0% to 45.9% | 0.255 |
| Trained on the second portfolio, its own line | 0.0760 | 1,280 | 279 | 10.9% | 41.8% | 35.9% to 47.4% | 0.237 |

At the same 1,000 alerts a day, the model trained on the first portfolio finds 243 and the one trained on the second finds 252 (+1.3 points, 95% interval -1.7 to +4.6).

## F. Does retraining with two more days of labels help?

Version 2 learned from days 2 to 5. A second copy learned from days 2 to 7 (2,345,075 transfers, 2,836 laundering). Both get 1,000 alerts a day on days 8 and 9 of the first portfolio.

| | Frozen | Retrained |
|---|---|---|
| Laundering found | 589 | 593 |
| Found by this one only | 23 | 27 |
| Average precision | 0.584 | 0.592 |

Gain in share found: +0.4 points, 95% interval -1.0 to +1.9.

