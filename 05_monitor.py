"""Step 5. Watch version 1 day by day, find what broke, fix it, and check the fix on a second portfolio.

    python 05_monitor.py            about 4 minutes on 2 cores

Version 1, its alert line and the rules in monitoring_rules.json are all fixed before this step runs.

  A. Version 1 on the first portfolio, day by day from day 6. Each day gets the checks a monitoring job can run that
     night (feed, alert volume, score shift, feature shift) and, later, the one that needs reviewers' labels
     (precision). After day 9 the file holds at most a few hundred transfers a day; that is treated as what it would be
     in production, a broken feed.
  B. The alert spike on day 8: what the extra alerts have in common. None of it needs a label.
  C. Version 2: version 1 plus one feature, the pair's history over the 7 days before. Same training days, same
     settings. Then the same days again. Those days suggested the change, so they are not a clean test of it.
  D. The check on a second portfolio. LI-Small played no part in finding the problem or in designing the feature.
     Both versions are trained there the same way and compared on its test days. The check was not blind: a wider
     candidate was tried on both portfolios in the same trial, and its results are reported beside version 2's.
  E. Does a model carry over? Version 2 trained on the first portfolio and applied to the second: as it is, with its
     line re-set there, and against a version 2 trained there.
  F. Retraining. Once the labels of days 6 and 7 are in, does a retrained version 2 do better on days 8 and 9?

Writes results/monitoring.json, results/monitoring.md and, under results/fitted/, model_v2.json,
model_v2_settings.json and alert_line_v2.json. The scores behind the numbers go to data/work/hi_scores_v2.parquet
and data/work/li_scores.parquet, so that every count can be recomputed without training anything.
"""
import gc
import time

import numpy as np
import pandas as pd

from common import FORMATS, LAST_REGULAR_DAY, RESULTS, ROOT, SEED, TEST_DAYS, TRAIN_DAYS, TUNE_DAYS, WEEK, WORK, note_time, read_json, wilson, write_json
from features import ADDED_LATER, MEANING
from line import choose_calibrator, fit_calibrator, line_for
from metrics import alert_counts, paired_gain, psi_against, psi_reference, ranking_quality, recall_interval_by_scheme, top_per_day
from model import apply_calibrator, fit_boosted, load_model, refit, scores_of

MONITORED = list(TUNE_DAYS) + list(TEST_DAYS)
# The feature-shift check compares a day with the tuning days. A feature that looks back 7 days has 6 days behind it
# on day 6 and 7 on day 7, so on those days it is still filling up and gives no usable reference. It is left out of
# the check until a full week of history exists for every reference day.
NOT_YET_COMPARABLE = ADDED_LATER


def load(portfolio, columns):
    X = pd.read_parquet(WORK / f"{portfolio}_features.parquet", columns=columns)
    index = pd.read_parquet(WORK / f"{portfolio}_index.parquet", columns=["y", "day", "scheme", "s", "r"])
    return X, index["y"].to_numpy(), index["day"].to_numpy(), index["scheme"].to_numpy(), index["s"].to_numpy().astype(np.int64), index["r"].to_numpy().astype(np.int64)


def train_version(X, y, day, columns, params):
    """The same recipe as step 3: fit on all training days but the last, let the last one set the number of trees, refit on all."""
    fit_rows, stop_rows, train = np.isin(day, TRAIN_DAYS[:-1]), day == TRAIN_DAYS[-1], np.isin(day, TRAIN_DAYS)
    _, trees = fit_boosted(X, y, fit_rows, stop_rows, columns, params)
    return refit(X, y, train, columns, params, trees), trees


def references(X, y, day, score, line, columns):
    """What the rules compare a day with. All of it comes from the training and tuning days."""
    tune = np.isin(day, TUNE_DAYS)
    on_tune = alert_counts(score[tune] >= line, y[tune])
    return {"numbers": {"median_training_day": float(np.median([int((day == d).sum()) for d in TRAIN_DAYS])), "alerts_per_day": on_tune["alerts"] / len(TUNE_DAYS),
                        "precision_on_tuning_days": on_tune["precision"], "precision_floor": wilson(on_tune["found"], on_tune["alerts"])[0]},
            "score": psi_reference(score[tune]), "features": {f: psi_reference(X[f].to_numpy()[tune]) for f in columns}}


def judge(rules, day_stats, numbers):
    """Read one day against the rules, in their order. Returns {rule name: status}."""
    by_name = {r["name"]: r for r in rules}
    status = {}
    ratio = day_stats["transfers"] / numbers["median_training_day"]
    status["feed"] = "red" if (ratio < by_name["feed"]["red_below"] or ratio > by_name["feed"]["red_above"]) else "ok"
    if status["feed"] == "red":
        return {**status, **{name: "not read" for name in ("alert volume", "score shift", "feature shift", "precision")}}
    ratio = day_stats["alerts"] / numbers["alerts_per_day"]
    status["alert volume"] = "amber" if (ratio < by_name["alert volume"]["amber_below"] or ratio > by_name["alert volume"]["amber_above"]) else "ok"
    for name, value in (("score shift", day_stats["score_psi"]), ("feature shift", day_stats["largest_feature_psi"])):
        status[name] = "red" if value > by_name[name]["red_above"] else "amber" if value > by_name[name]["amber_above"] else "ok"
    status["precision"] = "amber" if day_stats["precision"] < numbers["precision_floor"] else "ok"
    return status


def watch(X, y, day, score, line, days, columns, ref, rules, budget):
    """The checks for each day in `days`, and what the same day gives as a queue of the top `budget`."""
    out = []
    for d in days:
        rows = np.flatnonzero(day == d)
        if len(rows) == 0:
            continue
        counts = alert_counts(score[rows] >= line, y[rows])
        queue = alert_counts(top_per_day(score[rows], day[rows], budget), y[rows])
        shifts = {f: psi_against(ref["features"][f], X[f].to_numpy()[rows]) for f in columns if f not in NOT_YET_COMPARABLE}
        worst = max(shifts, key=shifts.get)
        stats = {"day": int(d), "transfers": int(len(rows)), "alerts": counts["alerts"], "score_psi": psi_against(ref["score"], score[rows]),
                 "largest_feature_psi": shifts[worst], "feature_with_largest_psi": worst, "meaning": MEANING[worst],
                 "laundering": counts["laundering"], "found": counts["found"], "precision": counts["precision"], "recall": counts["recall"],
                 "queue_found": queue["found"], "queue_precision": queue["precision"]}
        stats["status"] = judge(rules, stats, ref["numbers"])
        out.append(stats)
    return out


def what_the_alerts_share(X, y, day, s, r, alerted, d, n_accounts):
    """For the alerts of day d: how many are a pair with nothing on the 2 days before, and how many of those pairs
    did transact exactly a week earlier. Only the last line (false alerts) needs labels."""
    pair = s * n_accounts + r
    rows = (day == d) & alerted
    week_ago = np.unique(pair[(day == d - WEEK) & (s != r)])
    recent_senders = np.unique(s[(day >= d - 3) & (day < d)])
    seen = np.isin(pair[rows], week_ago)
    false = rows & (y == 0)
    return {"day": int(d), "alerts": int(rows.sum()),
            "by_ach": float((X["format"].to_numpy()[rows] == FORMATS.index("ACH")).mean()),
            "no_transfer_on_the_2_days_before": float((X["pair_before_3d"].to_numpy()[rows] == 0).mean()),
            "pair_paid_exactly_a_week_before": float(seen.mean()), "pairs_seen_a_week_before": int(seen.sum()),
            "sender_sent_nothing_in_the_3_days_before": float((~np.isin(s[rows], recent_senders)).mean()),
            "false_alerts": int(false.sum()), "false_alerts_pair_paid_a_week_before": float(np.isin(pair[false], week_ago).mean()) if false.any() else None}


def at_line(score, y, scheme, rows, line):
    alerted = score[rows] >= line
    c = alert_counts(alerted, y[rows])
    c["recall_low"], c["recall_high"] = recall_interval_by_scheme(alerted, y[rows], scheme[rows], seed=SEED)
    c["alerts_per_day"] = c["alerts"] / len(TEST_DAYS)
    return c


def by_day(score, y, day, line, budget):
    rows = []
    for d in MONITORED:
        m = day == d
        c, q = alert_counts(score[m] >= line, y[m]), alert_counts(top_per_day(score[m], day[m], budget), y[m])
        rows.append({"day": int(d), "transfers": int(m.sum()), "laundering": c["laundering"], "alerts": c["alerts"], "found": c["found"], "precision": c["precision"],
                     "queue_found": q["found"], "queue_precision": q["precision"]})
    return rows


def main():
    t0 = time.time()
    rules_file = read_json(ROOT / "monitoring_rules.json")
    rules = rules_file["checked_in_this_order"]
    v1 = read_json(RESULTS / "fitted" / "model_v1_settings.json")
    line1_file = read_json(RESULTS / "fitted" / "alert_line_v1.json")
    cols1, line1, budget = v1["features"], line1_file["line"], line1_file["alerts_per_day_on_tuning_days"]
    cols2, wide = cols1 + ADDED_LATER[:1], cols1 + ADDED_LATER
    params = {"max_depth": v1["max_depth"], "min_child_weight": v1["min_child_weight"]}
    result = {"rules_version": rules_file["version"], "budget": budget, "version_1": {"features": len(cols1), "line": line1},
              "added_in_version_2": [{"feature": f, "meaning": MEANING[f]} for f in ADDED_LATER[:1]],
              "tried_and_left_out": [{"feature": f, "meaning": MEANING[f]} for f in ADDED_LATER[1:]],
              "left_out_of_the_feature_shift_check": NOT_YET_COMPARABLE}

    # ============================================================ the first portfolio
    X, y, day, scheme, s, r = load("hi", wide)
    n_accounts = int(max(s.max(), r.max())) + 1
    tune, test = np.isin(day, TUNE_DAYS), np.isin(day, TEST_DAYS)

    # ------------------------------------------------------------ A. version 1, day by day
    score1 = scores_of(load_model(RESULTS / "fitted" / "model_v1.json"), X, cols1)
    ref1 = references(X, y, day, score1, line1, cols1)
    result["reference"] = ref1["numbers"]
    result["version_1"]["first_portfolio"] = watch(X, y, day, score1, line1, range(TUNE_DAYS[0], int(day.max()) + 1), cols1, ref1, rules, budget)
    tail = day > LAST_REGULAR_DAY
    result["after_day_9"] = {"transfers": int(tail.sum()), "laundering": int(y[tail].sum()), "alerts": int((score1[tail] >= line1).sum()),
                             "found": int(((score1[tail] >= line1) & (y[tail] == 1)).sum())}
    p1 = apply_calibrator(line1_file["calibrator"], score1)
    result["version_1"]["expected_against_seen"] = [{"day": int(d), "expected_in_alerts": float(p1[(day == d) & (score1 >= line1)].sum()),
                                                     "found": int(((day == d) & (score1 >= line1) & (y == 1)).sum())} for d in MONITORED]
    print(f"A. version 1 watched ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ B. what the alerts of each day have in common
    result["what_the_alerts_share"] = [what_the_alerts_share(X, y, day, s, r, score1 >= line1, d, n_accounts) for d in MONITORED if d - WEEK >= 0]

    # ------------------------------------------------------------ C. version 2: one more feature, the same recipe
    model2, trees2 = train_version(X, y, day, cols2, params)
    model2.save_model(RESULTS / "fitted" / "model_v2.json")
    write_json(RESULTS / "fitted" / "model_v2_settings.json", {"version": 2, "features": cols2, "added_to_version_1": ADDED_LATER[:1], "trees": trees2,
                                                               **{k: v1[k] for k in ("max_depth", "min_child_weight", "learning_rate", "subsample", "colsample_bytree", "tree_method", "max_bin")},
                                                               "trained_on_days": list(TRAIN_DAYS), "seed": SEED})
    score2 = scores_of(model2, X, cols2)
    kind, table = choose_calibrator(score2, y, day == TUNE_DAYS[0], day == TUNE_DAYS[1])
    calibrator2 = fit_calibrator(kind, score2[tune], y[tune])
    line2 = line_for(score2[tune], budget, len(TUNE_DAYS))
    p2 = apply_calibrator(calibrator2, score2)
    write_json(RESULTS / "fitted" / "alert_line_v2.json", {"version": 2, "line": line2, "alerts_per_day_on_tuning_days": budget, "set_on_days": list(TUNE_DAYS),
                                                           "calibrator": calibrator2, "probability_at_line": float(apply_calibrator(calibrator2, np.array([line2]))[0])})
    ref2 = references(X, y, day, score2, line2, cols2)
    model_wide, trees_wide = train_version(X, y, day, wide, params)
    score_wide = scores_of(model_wide, X, wide)
    line_wide = line_for(score_wide[tune], budget, len(TUNE_DAYS))

    def queue(score, rows):
        return top_per_day(score[rows], day[rows], budget)

    on2 = at_line(score2, y, scheme, test, line2)
    q2 = alert_counts(queue(score2, test), y[test])
    q2["recall_low"], q2["recall_high"] = recall_interval_by_scheme(queue(score2, test), y[test], scheme[test], seed=SEED)
    result["version_2"] = {
        "features": len(cols2), "trees": trees2, "line": line2, "calibrator": kind, "calibration_table": table, "reference": ref2["numbers"],
        "tuning_days_v1_against_v2": paired_gain(queue(score1, tune), queue(score2, tune), y[tune], scheme[tune], seed=SEED),
        "tuning_days_v2_against_wider": {"trees": trees_wide, **paired_gain(queue(score2, tune), queue(score_wide, tune), y[tune], scheme[tune], seed=SEED)},
        "wider_on_test_days": {"line": line_wide, "by_day": by_day(score_wide, y, day, line_wide, budget),
                               **paired_gain(queue(score2, test), queue(score_wide, test), y[test], scheme[test], seed=SEED)},
        "first_portfolio_v1": by_day(score1, y, day, line1, budget), "first_portfolio_v2": by_day(score2, y, day, line2, budget),
        "first_portfolio_watch": watch(X, y, day, score2, line2, MONITORED, cols2, ref2, rules, budget),
        "test_days_v1_against_v2": paired_gain(queue(score1, test), queue(score2, test), y[test], scheme[test], seed=SEED),
        "test_days_at_line": on2, "test_days_as_a_daily_queue": q2,
        "average_precision": {"v1_tuning": ranking_quality(score1[tune], y[tune])["average_precision"], "v2_tuning": ranking_quality(score2[tune], y[tune])["average_precision"],
                              "v1_test": ranking_quality(score1[test], y[test])["average_precision"], "v2_test": ranking_quality(score2[test], y[test])["average_precision"]},
        "expected_against_seen": [{"day": int(d), "expected_in_alerts": float(p2[(day == d) & (score2 >= line2)].sum()),
                                   "found": int(((day == d) & (score2 >= line2) & (y == 1)).sum())} for d in MONITORED]}
    print(f"C. version 2 trained and watched ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------ F. retraining version 2 with the labels of days 6 and 7
    wider = np.isin(day, TRAIN_DAYS) | tune
    again = scores_of(refit(X, y, wider, cols2, params, trees2), X.loc[test], cols2)
    result["retraining"] = {"trained_on": f"days {TRAIN_DAYS[0]} to {TUNE_DAYS[-1]}", "trained_on_transfers": int(wider.sum()), "trained_on_laundering": int(y[wider].sum()),
                            "average_precision_frozen": ranking_quality(score2[test], y[test])["average_precision"],
                            "average_precision_retrained": ranking_quality(again, y[test])["average_precision"],
                            **paired_gain(queue(score2, test), top_per_day(again, day[test], budget), y[test], scheme[test], seed=SEED)}
    print(f"F. retraining compared ({time.time() - t0:.0f} s)", flush=True)
    retrained_all = np.full(len(y), np.nan, dtype=np.float32)       # the retrained copy is only scored on the test days
    retrained_all[test] = again
    pd.DataFrame({"v1": score1, "v2": score2, "v2_wider": score_wide, "v2_retrained": retrained_all}).to_parquet(WORK / "hi_scores_v2.parquet", index=False)
    del X, score1, score_wide, again, p1, p2, model_wide
    gc.collect()

    # ============================================================ the second portfolio
    X, y, day, scheme, s, r = load("li", wide)
    tune, test = np.isin(day, TUNE_DAYS), np.isin(day, TEST_DAYS)
    result["second_portfolio_size"] = {"transfers": int(len(y)), "laundering": int(y.sum()), "one_in": round(len(y) / y.sum()),
                                       "training": {"transfers": int(np.isin(day, TRAIN_DAYS).sum()), "laundering": int(y[np.isin(day, TRAIN_DAYS)].sum())},
                                       "tuning": {"transfers": int(tune.sum()), "laundering": int(y[tune].sum())},
                                       "test": {"transfers": int(test.sum()), "laundering": int(y[test].sum())}}

    # ------------------------------------------------------------ D. the second portfolio: both versions, and the wider candidate, trained there
    local = {}
    for name, columns in (("v1", cols1), ("v2", cols2), ("wider", wide)):
        model, trees = train_version(X, y, day, columns, params)
        score = scores_of(model, X, columns)
        local[name] = {"score": score, "trees": trees, "line": line_for(score[tune], budget, len(TUNE_DAYS))}
        print(f"D. second portfolio, {name}: {trees} trees ({time.time() - t0:.0f} s)", flush=True)

    def queue(score, rows):                                         # the same helper, now on the second portfolio's days
        return top_per_day(score[rows], day[rows], budget)

    result["second_portfolio_check"] = {
        "trees": {name: v["trees"] for name, v in local.items()}, "lines": {name: v["line"] for name, v in local.items()},
        "v1_by_day": by_day(local["v1"]["score"], y, day, local["v1"]["line"], budget), "v2_by_day": by_day(local["v2"]["score"], y, day, local["v2"]["line"], budget),
        "test_days_v1_against_v2": paired_gain(queue(local["v1"]["score"], test), queue(local["v2"]["score"], test), y[test], scheme[test], seed=SEED),
        "tuning_days_v1_against_v2": paired_gain(queue(local["v1"]["score"], tune), queue(local["v2"]["score"], tune), y[tune], scheme[tune], seed=SEED),
        "tuning_days_v2_against_wider": paired_gain(queue(local["v2"]["score"], tune), queue(local["wider"]["score"], tune), y[tune], scheme[tune], seed=SEED),
        "wider_on_test_days": {"by_day": by_day(local["wider"]["score"], y, day, local["wider"]["line"], budget),
                               "against_v1": paired_gain(queue(local["v1"]["score"], test), queue(local["wider"]["score"], test), y[test], scheme[test], seed=SEED),
                               "against_v2": paired_gain(queue(local["v2"]["score"], test), queue(local["wider"]["score"], test), y[test], scheme[test], seed=SEED)},
        "average_precision_on_test_days": {name: ranking_quality(v["score"][test], y[test])["average_precision"] for name, v in local.items() if name != "wider"},
        "test_days_at_line": {name: at_line(local[name]["score"], y, scheme, test, local[name]["line"]) for name in ("v1", "v2")}}
    in_scheme = scheme[test] >= 0
    alerted = queue(local["v2"]["score"], test)
    result["second_portfolio_check"]["v2_by_kind"] = {
        "in_a_labeled_scheme": {"laundering": int((y[test][in_scheme] == 1).sum()), "found": int((alerted[in_scheme] & (y[test][in_scheme] == 1)).sum())},
        "not_in_a_labeled_scheme": {"laundering": int((y[test][~in_scheme] == 1).sum()), "found": int((alerted[~in_scheme] & (y[test][~in_scheme] == 1)).sum())}}

    # ------------------------------------------------------------ E. does version 2 carry over from the first portfolio?
    carried = scores_of(model2, X, cols2)
    result["carried_over"] = {"watch": watch(X, y, day, carried, line2, MONITORED, cols2, ref2, rules, budget)}
    shifts = {f: psi_against(ref2["features"][f], X[f].to_numpy()[tune]) for f in cols2 if f not in NOT_YET_COMPARABLE}
    result["carried_over"]["largest_feature_shifts"] = [{"feature": f, "meaning": MEANING[f], "psi": v} for f, v in sorted(shifts.items(), key=lambda kv: -kv[1])[:6]]
    reset = line_for(carried[tune], budget, len(TUNE_DAYS))
    options = [("Trained on the first portfolio, old line", carried, line2), ("Trained on the first portfolio, line re-set on the second", carried, reset),
               ("Trained on the second portfolio, its own line", local["v2"]["score"], local["v2"]["line"])]
    result["carried_over"]["options"] = [{"option": name, "line": cut, "average_precision": ranking_quality(sc[test], y[test])["average_precision"], **at_line(sc, y, scheme, test, cut)}
                                         for name, sc, cut in options]
    result["carried_over"]["same_alerts_carried_against_local"] = paired_gain(queue(carried, test), queue(local["v2"]["score"], test), y[test], scheme[test], seed=SEED)
    print(f"E. carry-over measured ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame({"v1_trained_here": local["v1"]["score"], "v2_trained_here": local["v2"]["score"], "v2_wider_trained_here": local["wider"]["score"],
                  "v2_carried_over": carried}).to_parquet(WORK / "li_scores.parquet", index=False)

    write_json(RESULTS / "monitoring.json", result)
    write_report(result, rules)
    note_time("05_monitor", time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s. See results/monitoring.md")


# -------------------------------------------------------------------- the report
MARK = {"ok": "ok", "amber": "AMBER", "red": "RED", "not read": "not read"}
WATCH_HEAD = ["| Day | Transfers | Feed | Alerts | Alert volume | Score shift (PSI) | | Largest feature shift (PSI) | | Feature that moved most | Found, of laundering | Precision | |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def watch_rows(days):
    rows = []
    for d in days:
        st = d["status"]
        left = (f"| {d['day']} | {d['transfers']:,} | {MARK[st['feed']]} | {d['alerts']:,} | {MARK[st['alert volume']]} | {d['score_psi']:.3f} | {MARK[st['score shift']]} | "
                f"{d['largest_feature_psi']:.3f} | {MARK[st['feature shift']]} | {d['meaning'] if st['feed'] != 'red' and d['largest_feature_psi'] > 0.1 else ''} | {d['found']:,} of {d['laundering']:,} | ")
        rows.append(left + (f"{d['precision']:.1%} | {MARK[st['precision']]} |" if st["feed"] != "red" else "| not read |"))
    return rows


def gain_text(g):
    return f"{g['gain'] * 100:+.1f} points, 95% interval {g['gain_low'] * 100:+.1f} to {g['gain_high'] * 100:+.1f}"


def side_by_side(a, b):
    head = ["| Day | Transfers | Laundering | Version 1: alerts at its line | Found | Top 1,000: found | Version 2: alerts at its line | Found | Top 1,000: found |", "|---|---|---|---|---|---|---|---|---|"]
    return head + [f"| {x['day']} | {x['transfers']:,} | {x['laundering']:,} | {x['alerts']:,} | {x['found']:,} | {x['queue_found']:,} | {z['alerts']:,} | {z['found']:,} | {z['queue_found']:,} |" for x, z in zip(a, b)]


def write_report(r, rules):
    ref, v1, v2 = r["reference"], r["version_1"], r["version_2"]
    budget = r["budget"]
    L = ["# Step 5: monitoring, a fix, and a second portfolio", "",
         f"Version 1 ({v1['features']} features), its alert line ({v1['line']:.4f}) and the rules were fixed before this step.", "",
         "## The rules", "", "| Check | Watches | Compared with | Limits | If it trips |", "|---|---|---|---|---|"]
    for rule in rules:
        limits = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in rule.items() if k.startswith(("red", "amber")))
        L.append(f"| {rule['name'].capitalize()} | {rule['watch']} | {rule['compare_with']} | {limits} | {rule['action']} |")
    L += ["", f"The median training day has {ref['median_training_day']:,.0f} transfers. The line gave {ref['alerts_per_day']:,.0f} alerts a day on the tuning days "
          f"at a precision of {ref['precision_on_tuning_days']:.1%}; the lower end of its 95% interval, {ref['precision_floor']:.1%}, is the precision floor.", "",
          "## A. Version 1, day by day", "",
          "Days 6 and 7 are the tuning days, where the line and the references were set. Days 8 and 9 are new. "
          "The four checks on the left can run the same night. The precision check needs the reviewers' labels and comes later.", ""]
    L += WATCH_HEAD + watch_rows(v1["first_portfolio"])
    tail = r["after_day_9"]
    spike = next(d for d in v1["first_portfolio"] if d["day"] == TEST_DAYS[0])
    L += ["", f"After day 9 the file holds {tail['transfers']:,} transfers in all, {tail['laundering']:,} of them laundering. "
          f"The feed check stops scoring on day 10. Had it not, the line would have raised {tail['alerts']:,} alerts on those days and found {tail['found']:,} of the {tail['laundering']:,}.", "",
          f"Day {spike['day']}: the line was set to give {ref['alerts_per_day']:,.0f} alerts a day and gave {spike['alerts']:,}. "
          f"The feature that moved most that day: {spike['meaning']} (PSI {spike['largest_feature_psi']:.3f}).", "",
          "Laundering the calibrated probabilities expect among the alerts, against what was found (known once labels are in):", "",
          "| Day | Expected in the alerts | Found |", "|---|---|---|"]
    L += [f"| {d['day']} | {d['expected_in_alerts']:.0f} | {d['found']:,} |" for d in v1["expected_against_seen"]]
    L += ["", f"## B. What the alerts of day {TEST_DAYS[0]} have in common", "",
          "Everything in this table except the last two columns can be computed the same night, before any label exists.", "",
          "| Day | Alerts | By ACH | Pair had no transfer on the 2 days before | Pair had a transfer exactly 7 days before | Sender sent nothing in the 3 days before | False alerts | Of those, pair paid 7 days before |",
          "|---|---|---|---|---|---|---|---|"]
    for a in r["what_the_alerts_share"]:
        L.append(f"| {a['day']} | {a['alerts']:,} | {a['by_ach']:.0%} | {a['no_transfer_on_the_2_days_before']:.0%} | {a['pair_paid_exactly_a_week_before']:.0%} | "
                 f"{a['sender_sent_nothing_in_the_3_days_before']:.0%} | {a['false_alerts']:,} | {a['false_alerts_pair_paid_a_week_before']:.0%} |")
    L += ["", "Version 1 remembers 3 days. A pair that pays once a week looks like a first-time pair to it.", "",
          "## C. Version 2: one more feature", "",
          f"Added: {r['added_in_version_2'][0]['meaning']} (`{r['added_in_version_2'][0]['feature']}`). Same training days, same settings, {v2['trees']} trees, {v2['features']} features. "
          f"Its line for {budget:,} alerts a day on the tuning days is {v2['line']:.4f}.", ""]
    g, w, wt = v2["tuning_days_v1_against_v2"], v2["tuning_days_v2_against_wider"], v2["wider_on_test_days"]
    wide_day_8 = next(d for d in wt["by_day"] if d["day"] == TEST_DAYS[0])
    L += [f"On the tuning days the new feature changes little: at {budget:,} alerts a day version 1 finds {g['found_a']} and version 2 finds {g['found_b']} ({gain_text(g)}). "
          "Neither tuning day is a Friday, so the tuning days cannot show what the feature is for. Its case rests on the days below.", "",
          f"Also tried: adding {' and '.join(x['meaning'] for x in r['tried_and_left_out'])}. On the tuning days that wider model finds {w['found_b']} to version 2's {w['found_a']} ({gain_text(w)}). Left out. "
          f"On days 8 and 9 it finds {wt['found_b']} to version 2's {wt['found_a']}, and at its own line it raises {wide_day_8['alerts']:,} alerts on day 8.", "",
          "The first portfolio again, both versions:", ""]
    L += side_by_side(v2["first_portfolio_v1"], v2["first_portfolio_v2"])
    t, q, at = v2["test_days_v1_against_v2"], v2["test_days_as_a_daily_queue"], v2["test_days_at_line"]
    L += ["", f"Days 8 and 9 at {budget:,} alerts a day: version 1 finds {t['found_a']}, version 2 finds {t['found_b']} of {t['laundering']} ({gain_text(t)}). "
          f"Version 2 at its line: {at['alerts']:,} alerts, {at['found']:,} found, precision {at['precision']:.1%}, share found {at['recall']:.1%}. "
          f"As a daily queue: precision {q['precision']:.1%} ({q['precision_low']:.1%} to {q['precision_high']:.1%}), share found {q['recall']:.1%} ({q['recall_low']:.1%} to {q['recall_high']:.1%}). "
          "These two days are the ones that suggested the feature, so this is not a clean test of it. Section D is the better one.", "",
          f"Version 2 under the same rules. `{NOT_YET_COMPARABLE[0]}` is left out of the feature-shift check: on day 6 it has 6 days of history behind it and on day 7 it has 7, "
          "so on the reference days it is still filling up.", ""]
    L += WATCH_HEAD + watch_rows(v2["first_portfolio_watch"])
    L += ["", "| Day | Laundering expected in version 2's alerts | Found |", "|---|---|---|"]
    L += [f"| {d['day']} | {d['expected_in_alerts']:.0f} | {d['found']:,} |" for d in v2["expected_against_seen"]]
    size, c = r["second_portfolio_size"], r["second_portfolio_check"]
    L += ["", "## D. The check on a second portfolio", "",
          f"LI-Small: {size['transfers']:,} transfers, 1 laundering in {size['one_in']:,}. Other banks and customers. It played no part in finding the problem or in designing the feature. "
          f"Both versions are trained on its days 2 to 5 ({size['training']['transfers']:,} transfers, {size['training']['laundering']:,} laundering) with the settings chosen on the first portfolio "
          f"({c['trees']['v1']} and {c['trees']['v2']} trees), and each gets its own line on its tuning days.", ""]
    L += side_by_side(c["v1_by_day"], c["v2_by_day"])
    t, a1, a2, k = c["test_days_v1_against_v2"], c["test_days_at_line"]["v1"], c["test_days_at_line"]["v2"], c["v2_by_kind"]
    tn, wd, wv = c["tuning_days_v1_against_v2"], c["tuning_days_v2_against_wider"], c["wider_on_test_days"]["against_v1"]
    wide_day_8_here = next(d for d in c["wider_on_test_days"]["by_day"] if d["day"] == TEST_DAYS[0])
    L += ["", f"Days 8 and 9 at {budget:,} alerts a day: version 1 finds {t['found_a']}, version 2 finds {t['found_b']} of {t['laundering']} ({gain_text(t)}). "
          f"At their own lines: version 1 raises {a1['alerts']:,} alerts and finds {a1['found']:,} (precision {a1['precision']:.1%}); version 2 raises {a2['alerts']:,} and finds {a2['found']:,} (precision {a2['precision']:.1%}). "
          f"Average precision on those days: {c['average_precision_on_test_days']['v1']:.3f} and {c['average_precision_on_test_days']['v2']:.3f}.", "",
          f"On this portfolio's tuning days, as on the first, the feature changes little ({tn['found_a']} against {tn['found_b']}; {gain_text(tn)}), "
          f"and the wider model is no better than version 2 ({wd['found_a']} against {wd['found_b']}; {gain_text(wd)}).", "",
          "This check was not blind. Version 2 and the wider model were both fitted on this portfolio in the same trial as on the first, "
          "so its test days had been seen for both before version 2 was fixed. "
          f"The wider model fails here: on days 8 and 9 it finds {wv['found_b']} of {wv['laundering']}, against version 1's {wv['found_a']} ({gain_text(wv)}), "
          f"and at its own line it raises {wide_day_8_here['alerts']:,} alerts on day 8.", "",
          f"Where version 2 finds laundering here: {k['in_a_labeled_scheme']['found']} of {k['in_a_labeled_scheme']['laundering']} in labeled schemes, "
          f"{k['not_in_a_labeled_scheme']['found']} of {k['not_in_a_labeled_scheme']['laundering']} outside them.", "",
          "## E. Does a model carry over to a new portfolio?", "",
          "Version 2 as trained on the first portfolio, applied to the second with its old line and the first portfolio's references:", ""]
    co = r["carried_over"]
    L += WATCH_HEAD + watch_rows(co["watch"])
    L += ["", "The features that differ most between the two portfolios' tuning days:", "", "| Feature | PSI |", "|---|---|"]
    L += [f"| {x['meaning']} | {x['psi']:.3f} |" for x in co["largest_feature_shifts"]]
    L += ["", f"The second portfolio's test days ({size['test']['transfers']:,} transfers, {size['test']['laundering']:,} laundering):", "",
          "| Option | Line | Alerts a day | Laundering found | Precision | Share found | 95% interval | Average precision |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {o['option']} | {o['line']:.4f} | {o['alerts_per_day']:,.0f} | {o['found']:,} | {o['precision']:.1%} | {o['recall']:.1%} | {o['recall_low']:.1%} to {o['recall_high']:.1%} | {o['average_precision']:.3f} |"
          for o in co["options"]]
    g = co["same_alerts_carried_against_local"]
    L += ["", f"At the same {budget:,} alerts a day, the model trained on the first portfolio finds {g['found_a']} and the one trained on the second finds {g['found_b']} ({gain_text(g)}).", ""]
    t = r["retraining"]
    L += ["## F. Does retraining with two more days of labels help?", "",
          f"Version 2 learned from days 2 to 5. A second copy learned from {t['trained_on']} ({t['trained_on_transfers']:,} transfers, {t['trained_on_laundering']:,} laundering). "
          f"Both get {budget:,} alerts a day on days 8 and 9 of the first portfolio.", "",
          "| | Frozen | Retrained |", "|---|---|---|",
          f"| Laundering found | {t['found_a']} | {t['found_b']} |", f"| Found by this one only | {t['only_a']} | {t['only_b']} |",
          f"| Average precision | {t['average_precision_frozen']:.3f} | {t['average_precision_retrained']:.3f} |", "",
          f"Gain in share found: {gain_text(t)}.", ""]
    (RESULTS / "monitoring.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
