# Why the alerts were raised

Version 2 raised 2,787 alerts on the test days at its fixed line. results/alert_reasons.csv has one row per alert with the three features that pushed its score up most.

The contributions are SHAP values taken from the trees themselves. For every alert they add up to the model's score on the log-odds scale; the largest gap found was 4.3e-06.

| Feature | Group | Average push on the alerts | Average push on all test-day transfers | Times among the top 3 reasons |
|---|---|---|---|---|
| payment format | transfer | 1.59 | 0.47 | 2,703 |
| transfers from this sender to this receiver in 3 days | behavior | 1.07 | 0.20 | 2,074 |
| size of the transfer against others in the same currency and payment format | transfer | 0.97 | 0.31 | 1,425 |
| transfers from this sender to this receiver on the 7 days before | added in version 2 | 0.80 | 1.65 | 788 |
| currencies the sender used in 3 days | behavior | 0.58 | 0.16 | 449 |
| ACH transfers the receiver got today | behavior | 0.51 | 0.14 | 59 |
| number of accounts the sender's owner holds | customer | 0.44 | 0.11 | 204 |
| ACH transfers the receiver got in 3 days | behavior | 0.33 | 0.22 | 323 |
| paid in one currency and received in another | transfer | 0.32 | 0.03 | 200 |
| transfers the sender made in 3 days | behavior | 0.23 | 0.48 | 0 |
| transfers the receiver got today | behavior | 0.21 | 0.08 | 0 |
| the two banks are in different countries | customer | 0.18 | 0.13 | 0 |
| currency | transfer | 0.17 | 0.12 | 1 |
| size of the transfer | transfer | 0.16 | 0.09 | 1 |
| number of accounts the receiver's owner holds | customer | 0.16 | 0.09 | 6 |

The push is in log-odds: 0.7 doubles the odds, 2.3 multiplies them by ten. It says what moved this model's score. It does not say what causes laundering.

