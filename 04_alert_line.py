"""Step 4. Set the alert line for version 1, check the scores as probabilities, and test how to measure what the line misses.

    python 04_alert_line.py            under 1 minute

Everything is chosen on the tuning days (6 and 7) and then reported on the test days (8 and 9).

  A. Calibration. Are the scores usable as probabilities? Raw scores, Platt scaling and isotonic regression are
     each fitted on one tuning day and checked on the other. The best one is kept.
  B. The alert line. One fixed score line that gives the main budget of alerts a day on the tuning days, and
     what that line costs and finds. A second table reads the same choice as a cost trade.
  C. The test days at that fixed line, day by day, next to the same days worked as a daily queue.
  D. What the line misses. In production nobody reviews below the line, so the misses are unknown and have to be
     estimated from a sample. Here the truth is known, so three sampling plans are tested against it.
  E. Champion and challenger. The adopted model against the peer-group rule, the logistic regression and the
     richer model that was not adopted, on the same test days at the same number of alerts. Plus the size a live
     split test would need.

Writes results/alert_line.json, results/alert_line.md and results/fitted/alert_line_v1.json.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, SEED, TEST_DAYS, TUNE_DAYS, WORK, note_time, read_json, write_json
from line import LABEL, choose_calibrator, fit_calibrator, line_for, probability_quality
from metrics import alert_counts, paired_gain, recall_interval_by_scheme, top_per_day
from model import apply_calibrator
from sampling import allocate, simulate, two_proportion_n

BUDGETS = (250, 500, 1000, 2000, 5000)
COST_RATIOS = (5, 10, 20, 50, 100, 200)          # how many reviews one missed laundering transfer is worth
BANDS_BELOW = (5000, 50000)                      # the bands below the line end at these many transfers a day (tuning-day cut points)
SAMPLE_SIZES = (1000, 3000, 10000, 30000)
MAIN_SAMPLE = 3000
REPEATS = 5000


def main():
    t0 = time.time()
    models = read_json(RESULTS / "models.json")
    adopted, main_budget = models["adopted"], models["main_budget"]
    scores = pd.read_parquet(WORK / "hi_scores.parquet")
    index = pd.read_parquet(WORK / "hi_index.parquet", columns=["y", "day", "scheme"])
    y, day, scheme = index["y"].to_numpy(), index["day"].to_numpy(), index["scheme"].to_numpy()
    s = scores[adopted].to_numpy()
    tune, test = np.isin(day, TUNE_DAYS), np.isin(day, TEST_DAYS)
    n_tune_days, n_test_days = len(TUNE_DAYS), len(TEST_DAYS)
    rng = np.random.default_rng(SEED)
    result = {"model": adopted, "version": 1, "main_budget": main_budget, "tuning_days": list(TUNE_DAYS), "test_days": list(TEST_DAYS)}

    # ------------------------------------------------------------ A. calibration
    best, table = choose_calibrator(s, y, day == TUNE_DAYS[0], day == TUNE_DAYS[1])
    calibrator = fit_calibrator(best, s[tune], y[tune])
    p = apply_calibrator(calibrator, s)
    edges = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0000001]
    reliability = {}
    for part, rows_of_part in (("tuning", tune), ("test", test)):
        reliability[part] = []
        for low, high in zip(edges[:-1], edges[1:]):
            rows = rows_of_part & (p >= low) & (p < high)
            reliability[part].append({"from": low, "to": min(high, 1.0), "transfers": int(rows.sum()), "expected": float(p[rows].sum()), "laundering": int(y[rows].sum())})
    result["calibration"] = {"fitted_on_one_tuning_day_checked_on_the_other": table, "kept": best,
                             "test_days_raw": probability_quality(s[test].astype(np.float64), y[test]),
                             "test_days_kept": probability_quality(p[test], y[test]),
                             "by_test_day": [{"day": int(d), "expected": float(p[day == d].sum()), "laundering": int(y[day == d].sum())} for d in TEST_DAYS],
                             "reliability": reliability}
    print(f"calibration: kept '{best}' ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ B. the alert line
    line = line_for(s[tune], main_budget, n_tune_days)
    lines = []
    for k in BUDGETS:
        cut = line_for(s[tune], k, n_tune_days)
        on_tune, on_test = alert_counts(s[tune] >= cut, y[tune]), alert_counts(s[test] >= cut, y[test])
        lines.append({"alerts_per_day_on_tuning_days": k, "line": cut, "probability_at_line": float(apply_calibrator(calibrator, np.array([cut]))[0]),
                      "tune": on_tune, "test": on_test, "test_alerts_per_day": on_test["alerts"] / n_test_days})
    costs = []
    for ratio in COST_RATIOS:
        alerted = p >= 1 / ratio
        on_tune, on_test = alert_counts(alerted[tune], y[tune]), alert_counts(alerted[test], y[test])
        costs.append({"reviews_one_miss_is_worth": ratio, "alert_when_probability_at_least": 1 / ratio,
                      "tune": on_tune, "test": on_test, "tune_alerts_per_day": on_tune["alerts"] / n_tune_days, "test_alerts_per_day": on_test["alerts"] / n_test_days})
    p_line = float(apply_calibrator(calibrator, np.array([line]))[0])
    result["line"] = {"line": line, "probability_at_line": p_line, "one_in": 1 / p_line, "by_budget": lines, "by_cost": costs}
    write_json(RESULTS / "fitted" / "alert_line_v1.json", {"version": 1, "model": adopted, "line": line, "alerts_per_day_on_tuning_days": main_budget,
                                                           "set_on_days": list(TUNE_DAYS), "calibrator": calibrator, "probability_at_line": p_line})

    # ------------------------------------------------------------ C. the test days at the fixed line, and as a daily queue
    alerted = s >= line
    by_day = []
    for d in list(TUNE_DAYS) + list(TEST_DAYS):
        rows = day == d
        c = alert_counts(alerted[rows], y[rows])
        queue = alert_counts(top_per_day(s[rows], day[rows], main_budget), y[rows])
        by_day.append({"day": int(d), "part": "tuning" if d in TUNE_DAYS else "test", "transfers": int(rows.sum()), "expected_from_probabilities": float(p[rows & alerted].sum()),
                       **c, "queue_found": queue["found"], "queue_precision": queue["precision"]})
    on_test = alert_counts(alerted[test], y[test])
    on_test["recall_low"], on_test["recall_high"] = recall_interval_by_scheme(alerted[test], y[test], scheme[test], seed=SEED)
    on_test["f1"] = 2 * on_test["found"] / (on_test["alerts"] + on_test["laundering"])
    on_test["expected_from_probabilities"] = float(p[test & alerted].sum())
    queued = top_per_day(s[test], day[test], main_budget)
    queue = alert_counts(queued, y[test])
    queue["recall_low"], queue["recall_high"] = recall_interval_by_scheme(queued, y[test], scheme[test], seed=SEED)
    queue["f1"] = 2 * queue["found"] / (queue["alerts"] + queue["laundering"])
    queue["alerts_to_review_for_precision_within_3_points"] = int(np.ceil(1.959964 ** 2 * queue["precision"] * (1 - queue["precision"]) / 0.03 ** 2))
    result["test_at_line"] = {"by_day": by_day, "both_test_days": on_test, "both_test_days_as_a_daily_queue": queue}

    # ------------------------------------------------------------ D. what the line misses, estimated from a sample
    cuts = [line] + [line_for(s[tune], k, n_tune_days) for k in BANDS_BELOW] + [-np.inf]
    names = [f"Just below: up to {BANDS_BELOW[0]:,} a day", f"Middle: up to {BANDS_BELOW[1]:,} a day", "Far below: the rest"]

    def bands(rows):
        N = np.array([int((rows & (s < cuts[h]) & (s >= cuts[h + 1])).sum()) for h in range(3)])
        M = np.array([int((rows & (s < cuts[h]) & (s >= cuts[h + 1]) & (y == 1)).sum()) for h in range(3)])
        return N, M

    N_plan, M_plan = bands(tune)                                    # the plan is made on the tuning days
    N, M = bands(test)                                              # and carried out on the test days
    plan = allocate(N_plan, M_plan / N_plan, MAIN_SAMPLE)
    designs = {
        "Simple random sample below the line": (np.array([N.sum()]), np.array([M.sum()]), np.array([MAIN_SAMPLE])),
        "By score band, in proportion to band size": (N, M, np.maximum(1, np.round(MAIN_SAMPLE * N / N.sum()).astype(int))),
        "By score band, planned on the tuning days": (N, M, plan),
    }
    result["below_the_line"] = {
        "bands": [{"band": names[h], "tuning_transfers": int(N_plan[h]), "tuning_laundering": int(M_plan[h]), "test_transfers": int(N[h]), "test_laundering": int(M[h])} for h in range(3)],
        "truth_on_test_days": {"transfers_below": int(N.sum()), "laundering_below": int(M.sum()), "one_in": float(N.sum() / M.sum())},
        "main_sample": MAIN_SAMPLE, "repeats": REPEATS,
        "designs": {name: simulate(*design, REPEATS, rng) for name, design in designs.items()},
    }
    planned = result["below_the_line"]["designs"]["By score band, planned on the tuning days"]         # the same run serves both tables
    result["below_the_line"]["by_sample_size"] = [planned if n == MAIN_SAMPLE else simulate(N, M, allocate(N_plan, M_plan / N_plan, n), REPEATS, rng) for n in SAMPLE_SIZES]
    # one sample, as a single analyst would have drawn it
    drawn = np.array([rng.hypergeometric(M[h], N[h] - M[h], plan[h]) for h in range(3)])
    total = sum(N[h] * rng.beta(drawn[h] + 0.5, plan[h] - drawn[h] + 0.5, size=20000) for h in range(3))
    est, low, high = float((drawn / plan * N).sum()), float(np.quantile(total, 0.025)), float(np.quantile(total, 0.975))
    found = on_test["found"]
    result["below_the_line"]["one_sample"] = {"reviews_per_band": plan.tolist(), "laundering_in_sample_per_band": drawn.tolist(), "estimated_misses": est, "low": low, "high": high,
                                              "share_found_estimate": found / (found + est), "share_found_low": found / (found + high), "share_found_high": found / (found + low),
                                              "share_found_truth": on_test["recall"]}
    print(f"sampling plans simulated ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ E. champion and challenger
    def alerts(name, rows):
        return top_per_day(scores[name].to_numpy()[rows], day[rows], main_budget)

    comparisons = []
    for champion, challenger, label in (("rule", adopted, "Peer-group rule against the adopted model"), ("linear", adopted, "Logistic regression against the adopted model"),
                                        (adopted, "network", "Adopted model against the network model that was not adopted")):
        if champion != challenger:
            comparisons.append({"comparison": label, "a": champion, "b": challenger,
                                **paired_gain(alerts(champion, test), alerts(challenger, test), y[test], scheme[test], seed=SEED)})
    p_champion = alert_counts(alerts("linear", tune), y[tune])["precision"]
    p_challenger = alert_counts(alerts(adopted, tune), y[tune])["precision"]
    per_arm = two_proportion_n(p_champion, p_challenger)
    week = int(7 * main_budget / 2)                                 # alerts per arm in one week, at half the budget each
    gap = next(g for g in np.arange(0.005, 0.5, 0.0005) if two_proportion_n(p_champion, min(p_champion + g, 0.999)) <= week)
    result["champion_challenger"] = {"on_test_days": comparisons,
                                     "live_split_test": {"champion_precision": p_champion, "challenger_precision": p_challenger, "alerts_per_arm": per_arm,
                                                         "days_at_half_the_budget_per_arm": float(np.ceil(per_arm / (main_budget / 2))),
                                                         "smallest_gap_seen_with_one_week": float(gap)}}

    write_json(RESULTS / "alert_line.json", result)
    write_report(result)
    note_time("04_alert_line", time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s. See results/alert_line.md")


def write_report(r):
    cal, line, at = r["calibration"], r["line"], r["test_at_line"]
    L = ["# Step 4: the alert line (version 1)", "", "Chosen on the tuning days (6 and 7). Reported on the test days (8 and 9).", "",
         "## A. Are the scores usable as probabilities?", "", "Each method is fitted on one tuning day and checked on the other, both ways. Lower is better.", "",
         "| Method | Log loss | Brier score |", "|---|---|---|"]
    L += [f"| {LABEL[k]}{' (kept)' if k == cal['kept'] else ''} | {v['log_loss']:.6f} | {v['brier']:.6f} |" for k, v in cal["fitted_on_one_tuning_day_checked_on_the_other"].items()]
    L += ["", "Laundering the probabilities expect against laundering seen, by probability band:", "",
          "| Probability | Tuning days: transfers | Expected | Seen | Test days: transfers | Expected | Seen |", "|---|---|---|---|---|---|---|"]
    for a, b in zip(cal["reliability"]["tuning"], cal["reliability"]["test"]):
        L.append(f"| {a['from']:g} to {a['to']:g} | {a['transfers']:,} | {a['expected']:.0f} | {a['laundering']:,} | {b['transfers']:,} | {b['expected']:.0f} | {b['laundering']:,} |")
    L += ["", "On the tuning days expected equals seen by construction: the calibrator was fitted there. The test days are the check.", "",
          "| Test day | Laundering expected | Laundering seen |", "|---|---|---|"]
    L += [f"| {d['day']} | {d['expected']:.0f} | {d['laundering']:,} |" for d in cal["by_test_day"]]
    L += ["", "## B. The alert line", "",
          f"The line is the score that gives {r['main_budget']:,} alerts a day on the tuning days: **{line['line']:.4f}**. "
          f"A transfer exactly at the line has about a 1 in {line['one_in']:.0f} chance of being laundering.", "",
          "| Alerts a day on tuning days | Line | Chance of laundering at the line | Tuning: found | Tuning: precision | Test: alerts a day | Test: found | Test: precision | Test: share found |", "|---|---|---|---|---|---|---|---|---|"]
    for b in line["by_budget"]:
        L.append(f"| {b['alerts_per_day_on_tuning_days']:,} | {b['line']:.4f} | {b['probability_at_line']:.1%} | {b['tune']['found']:,} | {b['tune']['precision']:.1%} | "
                 f"{b['test_alerts_per_day']:,.0f} | {b['test']['found']:,} | {b['test']['precision']:.1%} | {b['test']['recall']:.1%} |")
    L += ["", "The same choice read as a cost trade. If one missed laundering transfer is worth R reviews, alert when the probability is at least 1 in R. "
          "Isotonic regression gives probabilities in steps, so the number of alerts moves in jumps.", "",
          "| R | Tuning: alerts a day | Tuning: found | Test: alerts a day | Test: found | Test: precision | Test: share found |", "|---|---|---|---|---|---|---|"]
    for c in line["by_cost"]:
        L.append(f"| {c['reviews_one_miss_is_worth']} | {c['tune_alerts_per_day']:,.0f} | {c['tune']['found']:,} | {c['test_alerts_per_day']:,.0f} | {c['test']['found']:,} | {c['test']['precision']:.1%} | {c['test']['recall']:.1%} |")
    t, q = at["both_test_days"], at["both_test_days_as_a_daily_queue"]
    L += ["", "## C. The test days at the fixed line", "",
          f"| Day | Part | Transfers | Alerts at the line | Laundering the probabilities expect | Found | Precision | Share found | Top {r['main_budget']:,} of the day: found | Precision |", "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {d['day']} | {d['part'].capitalize()} | {d['transfers']:,} | {d['alerts']:,} | {d['expected_from_probabilities']:.0f} | {d['found']:,} | {d['precision']:.1%} | {d['recall']:.1%} | "
          f"{d['queue_found']:,} | {d['queue_precision']:.1%} |" for d in at["by_day"]]
    L += ["", f"At the fixed line, both test days: {t['alerts']:,} alerts, {t['found']:,} of {t['laundering']:,} laundering transfers found. "
          f"Precision {t['precision']:.1%} (95% interval {t['precision_low']:.1%} to {t['precision_high']:.1%}). "
          f"Share found {t['recall']:.1%} (95% interval {t['recall_low']:.1%} to {t['recall_high']:.1%}, resampling whole schemes). F1 {t['f1']:.3f}. "
          f"The probabilities expected {t['expected_from_probabilities']:.0f} laundering transfers among the alerts.", "",
          f"As a daily queue of the top {r['main_budget']:,}: {q['alerts']:,} alerts, {q['found']:,} found. "
          f"Precision {q['precision']:.1%} ({q['precision_low']:.1%} to {q['precision_high']:.1%}). Share found {q['recall']:.1%} ({q['recall_low']:.1%} to {q['recall_high']:.1%}). F1 {q['f1']:.3f}. "
          f"If only a sample of those alerts were quality-checked, about {q['alerts_to_review_for_precision_within_3_points']:,} would pin precision to within 3 points.", "",
          "The line was set to give 1,000 alerts a day. Step 5 picks up what happened on day 8.", ""]
    b = r["below_the_line"]
    truth = b["truth_on_test_days"]
    L += ["## D. What the line misses, estimated from a sample", "",
          f"On the test days {truth['transfers_below']:,} transfers sit below the line and {truth['laundering_below']:,} of them are laundering, 1 in {truth['one_in']:,.0f}. "
          "In production that count is unknown. The plans below estimate it from a sample of reviews.", "",
          "| Band below the line | Tuning: transfers | Tuning: laundering | Test: transfers | Test: laundering |", "|---|---|---|---|---|"]
    L += [f"| {x['band']} | {x['tuning_transfers']:,} | {x['tuning_laundering']:,} | {x['test_transfers']:,} | {x['test_laundering']:,} |" for x in b["bands"]]
    L += ["", f"Three plans with {b['main_sample']:,} reviews each, every plan drawn {b['repeats']:,} times from the test days:", "",
          "| Plan | Reviews per band | Laundering in the sample, on average | Samples with none | Average estimate | Typical error | Interval covers the truth |", "|---|---|---|---|---|---|---|"]
    for name, d in b["designs"].items():
        L.append(f"| {name} | {', '.join(f'{v:,}' for v in d['per_band'])} | {d['mean_laundering_in_sample']:.1f} | {d['share_with_no_laundering_in_sample']:.0%} | "
                 f"{d['mean_estimate']:.0f} | ±{d['rmse']:.0f} ({d['rmse'] / truth['laundering_below']:.0%}) | {d['coverage']:.1%} |")
    L += ["", "Typical error is the root mean squared error. The interval is a 95% interval, so it should cover the truth about 95% of the time.", "",
          "The planned design at other sizes:", "", "| Reviews | Per band | Typical error | Interval covers the truth | Typical interval |", "|---|---|---|---|---|"]
    L += [f"| {d['reviews']:,} | {', '.join(f'{v:,}' for v in d['per_band'])} | ±{d['rmse']:.0f} ({d['rmse'] / truth['laundering_below']:.0%}) | {d['coverage']:.1%} | {d['median_low']:,.0f} to {d['median_high']:,.0f} |"
          for d in b["by_sample_size"]]
    one = b["one_sample"]
    L += ["", f"One sample, drawn once with the fixed seed: {', '.join(str(v) for v in one['laundering_in_sample_per_band'])} laundering transfers in the three bands. "
          f"Estimated misses {one['estimated_misses']:,.0f} (95% interval {one['low']:,.0f} to {one['high']:,.0f}); the truth is {truth['laundering_below']:,}. "
          f"That puts the share found at {one['share_found_estimate']:.0%} ({one['share_found_low']:.0%} to {one['share_found_high']:.0%}); the truth is {one['share_found_truth']:.1%}.", ""]
    cc = r["champion_challenger"]
    L += ["## E. Champion and challenger", "", f"Both sides get {r['main_budget']:,} alerts a day on the same test days.", "",
          "| Comparison | First finds | Second finds | Only the first | Only the second | Gain in share found | 95% interval | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {c['comparison']} | {c['found_a']} | {c['found_b']} | {c['only_a']} | {c['only_b']} | {c['gain'] * 100:+.1f} points | {c['gain_low'] * 100:+.1f} to {c['gain_high'] * 100:+.1f} | {c['mcnemar_p']:.3g} |"
          for c in cc["on_test_days"]]
    live = cc["live_split_test"]
    L += ["", "The interval resamples whole schemes. The McNemar test treats every transfer as independent, so it overstates the evidence.", "",
          f"A live split test would send some accounts to each model and compare the precision of their alerts. "
          f"To tell {live['champion_precision']:.1%} from {live['challenger_precision']:.1%} (the regression and the adopted model on the tuning days) with 80% power at the 5% level "
          f"takes {live['alerts_per_arm']:,} alerts per arm, about {live['days_at_half_the_budget_per_arm']:.0f} days at half the budget each. "
          f"One week of alerts could show a gap of {live['smallest_gap_seen_with_one_week'] * 100:.1f} points or more.", ""]
    (RESULTS / "alert_line.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
