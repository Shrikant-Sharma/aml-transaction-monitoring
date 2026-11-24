"""Step 3. Score the same days several ways, from a peer-group rule to boosted trees, and decide which model to keep.
This is version 1 of the model. It chooses from the 48 candidate features; the one feature added later is not used here.

    python 03_models.py            about 8 minutes on 2 cores

What it does, in order:
  1. The peer-group rule: z-scores against peers. Four variants, from the classic three quantities to the best
     five of 34. Peer means, spreads and each variant's terms come from the training days; the variant that finds
     most on the tuning days is kept as "the rule".
  2. An isolation forest on all 48 candidate features. It never sees a label.
  3. A logistic regression on all 48. The plain supervised model.
  4. Gradient-boosted trees (XGBoost). A small grid picks depth and leaf size on the tuning days; the number of
     trees is set by early stopping on day 5. Then one model per feature set, each set adding a group:
     transfer, +customer, +behavior, +network.
  5. The decision. A richer feature set is adopted only if, on the tuning days, it finds more laundering at the
     same number of alerts and the 95% interval for the gain stays above zero.
  6. Two checks on the adopted model: weighting the rare class up, and learning from the two warm-up days too.
  7. Everything is measured on the tuning days and on the test days. No choice is made on the test days.

Writes results/models.json, results/models.md, results/fitted/rule.json, results/fitted/model_v1.json,
results/fitted/model_v1_settings.json and data/work/hi_scores.parquet (one score per transfer and method).
"""
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import (FIRST_DATE, LAST_REGULAR_DAY, RESULTS, SEED, TEST_DAYS, TRAIN_DAYS, TUNE_DAYS, WARMUP_DAYS, WORK, note_time, write_json)
from features import CANDIDATES, CATEGORICAL, GROUPS, MEANING
from metrics import at_budget, best_f1_line, f1_at_line, paired_gain, ranking_quality, recall_interval_by_scheme, top_per_day
from model import FIXED, MOST_TREES, STOP_AFTER, fit_boosted, refit, scores_of

BUDGETS = (250, 500, 1000, 2000, 5000)          # alerts a day the reviewers could take
MAIN_BUDGET = 1000
FEATURE_SETS = {
    "transfer": GROUPS["transfer"],
    "customer": GROUPS["transfer"] + GROUPS["customer"],
    "behavior": GROUPS["transfer"] + GROUPS["customer"] + GROUPS["behavior"],
    "network": CANDIDATES,
}
GRID = [{"max_depth": 6, "min_child_weight": 1}, {"max_depth": 6, "min_child_weight": 5},
        {"max_depth": 8, "min_child_weight": 1}, {"max_depth": 8, "min_child_weight": 5}]

# how the logistic regression and the rule see each feature
FLAGS = ["to_self", "same_bank", "cross_currency", "same_owner", "cross_border", "back_3d", "loop_3d", "loop_5d"]
ONE_HOT = CATEGORICAL + ["hour"]
SIZES = ["amount_log", "amount_vs_peers"]
COUNTS = [f for f in CANDIDATES if f not in FLAGS + ONE_HOT + SIZES]
RULE_QUANTITIES = ["amount_log"] + COUNTS       # what a peer-group rule can watch: the amount and every count
CLASSIC = ["amount_log", "s_sent_1d", "r_recv_1d"]
MOST_TERMS = 5

ORDER = ("rule", "forest", "linear", "transfer", "customer", "behavior", "network")
NAMES = {"rule": "Peer-group rule (z-scores)", "forest": "Isolation forest (no labels)", "linear": "Logistic regression",
         "transfer": "Boosted trees: transfer fields", "customer": "Boosted trees: + customer fields",
         "behavior": "Boosted trees: + account behavior", "network": "Boosted trees: + network"}


# ---------------------------------------------------------------- the peer-group rule
def quantity(X, name):
    """One watched quantity and the peer group of every transfer for it.

    The amount is compared within its currency and payment format. A count is compared within the customer type
    of the account it describes and the payment format."""
    fmt = X["format"].to_numpy().astype(np.int64)
    if name == "amount_log":
        return X[name].to_numpy(), X["currency"].to_numpy().astype(np.int64) * 10 + fmt
    side = "receiver_type" if name.startswith("r_") else "sender_type"
    return np.log1p(X[name].to_numpy()), (X[side].to_numpy().astype(np.int64) + 1) * 10 + fmt


def fit_rule(X, rows):
    """Mean and spread of every watched quantity within each peer group, on the training rows."""
    fitted = {}
    for name in RULE_QUANTITIES:
        value, group = quantity(X, name)
        v, g = value[rows].astype(np.float64), group[rows]
        size = int(group.max()) + 1
        n = np.bincount(g, minlength=size)
        mean = np.bincount(g, weights=v, minlength=size) / np.maximum(n, 1)
        sd = np.sqrt(np.maximum(np.bincount(g, weights=v * v, minlength=size) / np.maximum(n, 1) - mean ** 2, 0))
        few = n < 30                                               # too few peers: compare with everyone
        mean[few], sd[few] = v.mean(), v.std()
        fitted[name] = {"mean": mean.tolist(), "sd": sd.tolist(), "n": n.tolist()}
    return fitted


def z_scores(X, fitted, names, rows):
    """z-score of each named quantity against its peer group, for the given rows. One column per name."""
    out = np.zeros((len(rows), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        value, group = quantity(X, name)
        mean, sd = np.array(fitted[name]["mean"]), np.array(fitted[name]["sd"])
        g = group[rows]
        out[:, j] = np.where(sd[g] > 1e-9, (value[rows] - mean[g]) / np.maximum(sd[g], 1e-9), 0)
    return out


def choose_rule(Z, y, day, jitter):
    """Four ways to turn z-scores into one rule score. Returns each variant's terms and what it finds on the days it is given."""
    def found(score):
        return at_budget(score + jitter, y, day, MAIN_BUDGET)["found"]

    column = {name: j for j, name in enumerate(RULE_QUANTITIES)}
    variants = {}
    classic = [column[n] for n in CLASSIC]
    variants["Classic: largest z-score of amount, sender volume, receiver volume"] = {
        "terms": [(n, 1) for n in CLASSIC], "combine": "largest", "found": found(Z[:, classic].max(axis=1))}
    variants[f"Largest z-score of all {len(RULE_QUANTITIES)} quantities"] = {
        "terms": [(n, 1) for n in RULE_QUANTITIES], "combine": "largest", "found": found(Z.max(axis=1))}
    singles = {(name, sign): found(sign * Z[:, j]) for name, j in column.items() for sign in (1, -1)}
    best = max(singles, key=singles.get)
    variants["Best single z-score, high or low side"] = {"terms": [best], "combine": "sum", "found": singles[best]}
    terms, total, best_found = [], np.zeros(len(Z), dtype=np.float64), -1
    for _ in range(MOST_TERMS):                                    # add the z-score that helps most, until none helps
        trial = {(name, sign): found(total + sign * Z[:, j]) for name, j in column.items() for sign in (1, -1) if (name, sign) not in terms}
        pick = max(trial, key=trial.get)
        if trial[pick] <= best_found:
            break
        terms.append(pick)
        best_found = trial[pick]
        total = total + pick[1] * Z[:, column[pick[0]]]
    variants[f"Sum of the best z-scores, up to {MOST_TERMS}, high or low side"] = {"terms": terms, "combine": "sum", "found": best_found}
    return variants


def rule_score(X, fitted, terms, combine, rows, jitter):
    names = [name for name, _ in terms]
    Z = z_scores(X, fitted, names, rows) * np.array([sign for _, sign in terms], dtype=np.float32)
    score = Z.max(axis=1) if combine == "largest" else Z.sum(axis=1, dtype=np.float64)
    return score.astype(np.float64) + jitter


# ---------------------------------------------------------------- logistic regression
def linear_model():
    """Categories one-hot, counts on a log scale, everything on one scale, then a plain logistic regression."""
    prepare = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ONE_HOT),
        ("counts", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), COUNTS),
        ("sizes", StandardScaler(), SIZES),
        ("flags", "passthrough", FLAGS)], sparse_threshold=0)
    return make_pipeline(prepare, LogisticRegression(C=1.0, max_iter=1000))


def in_parts(score_part, X, rows, parts=20):
    """Score the given rows a slice at a time, to keep memory flat. Rows not asked for stay empty."""
    out = np.full(len(X), np.nan, dtype=np.float32)
    for part in np.array_split(np.flatnonzero(rows), parts):
        out[part] = score_part(X.iloc[part])
    return out


def measure(score, y, day, scheme, rows):
    """All the measures for one score on one set of days."""
    s, yy, dd = score[rows], y[rows], day[rows]
    out = ranking_quality(s, yy)
    out["by_budget"] = {}
    for k in BUDGETS:
        counts = at_budget(s, yy, dd, k)
        counts["recall_low"], counts["recall_high"] = recall_interval_by_scheme(top_per_day(s, dd, k), yy, scheme[rows], seed=SEED)
        out["by_budget"][str(k)] = counts
    return out


def main():
    t0 = time.time()
    X = pd.read_parquet(WORK / "hi_features.parquet", columns=CANDIDATES)
    index = pd.read_parquet(WORK / "hi_index.parquet")
    y = index["y"].to_numpy()
    day = index["day"].to_numpy()
    scheme = index["scheme"].to_numpy()
    warmup = np.isin(day, WARMUP_DAYS)
    train = np.isin(day, TRAIN_DAYS)
    fit_rows = np.isin(day, TRAIN_DAYS[:-1])
    stop_rows = day == TRAIN_DAYS[-1]
    tune = np.isin(day, TUNE_DAYS)
    test = np.isin(day, TEST_DAYS)
    later = day >= TUNE_DAYS[0]                                    # everything after the training days
    parts = (("warmup", warmup), ("train", train), ("tune", tune), ("test", test), ("after_day_9", day > LAST_REGULAR_DAY))
    result = {"days": {"warmup": list(WARMUP_DAYS), "train": list(TRAIN_DAYS), "tune": list(TUNE_DAYS), "test": list(TEST_DAYS)},
              "rows": {name: {"transfers": int(m.sum()), "laundering": int(y[m].sum()), "one_in": round(m.sum() / max(y[m].sum(), 1))} for name, m in parts},
              "budgets": list(BUDGETS), "main_budget": MAIN_BUDGET, "features": {k: len(v) for k, v in FEATURE_SETS.items()},
              "seed": SEED, "versions": {"xgboost": xgb.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    scores = {}

    # 1. the peer-group rule
    jitter = np.random.default_rng(SEED).random(len(X)) * 1e-6     # equal rule scores are ordered at random, not by file order
    train_rows, tune_rows = np.flatnonzero(train), np.flatnonzero(tune)
    fitted_rule = fit_rule(X, train_rows)
    variants = choose_rule(z_scores(X, fitted_rule, RULE_QUANTITIES, train_rows), y[train], day[train], jitter[train_rows])   # terms come from the training days
    for v in variants.values():                                    # the variants are then compared on the tuning days
        tuned = rule_score(X, fitted_rule, v["terms"], v["combine"], tune_rows, jitter[tune_rows])
        v["found_on_tuning_days"] = at_budget(tuned, y[tune], day[tune], MAIN_BUDGET)["found"]
    kept = max(variants, key=lambda v: variants[v]["found_on_tuning_days"])
    later_rows = np.flatnonzero(later)
    rule = np.full(len(X), np.nan)
    rule[later_rows] = rule_score(X, fitted_rule, variants[kept]["terms"], variants[kept]["combine"], later_rows, jitter[later_rows])
    scores["rule"] = rule
    laundering_tune = int(y[tune].sum())
    result["rule"] = {"quantities_watched": len(RULE_QUANTITIES), "kept": kept, "laundering_on_training_days": int(y[train].sum()),
                      "variants": [{"variant": name, "found_on_training_days": v["found"], "found_on_tuning_days": v["found_on_tuning_days"],
                                    "share_found": v["found_on_tuning_days"] / laundering_tune,
                                    "terms": [{"quantity": q, "meaning": MEANING[q], "side": "high" if sign > 0 else "low"} for q, sign in v["terms"]] if len(v["terms"]) <= MOST_TERMS else None}
                                   for name, v in variants.items()]}
    # a side check on where the terms are chosen: the same greedy choice made on the tuning days instead
    chosen_late = choose_rule(z_scores(X, fitted_rule, RULE_QUANTITIES, tune_rows), y[tune], day[tune], jitter[tune_rows])
    late = chosen_late[next(name for name in chosen_late if name.startswith("Sum of the best"))]
    test_rows = np.flatnonzero(test)
    late_on_test = at_budget(rule_score(X, fitted_rule, late["terms"], "sum", test_rows, jitter[test_rows]), y[test], day[test], MAIN_BUDGET)
    result["rule"]["terms_chosen_on_tuning_days_instead"] = {"found_on_tuning_days": late["found"], "found_on_test_days": late_on_test["found"],
                                                             "laundering_on_test_days": late_on_test["laundering"]}
    del chosen_late
    write_json(RESULTS / "fitted" / "rule.json", {"kept": kept, "combine": variants[kept]["combine"],
                                                  "terms": [{"quantity": q, "sign": sign} for q, sign in variants[kept]["terms"]],
                                                  "fitted": {q: fitted_rule[q] for q, _ in variants[kept]["terms"]}})
    print(f"rule: kept '{kept}' ({variants[kept]['found_on_tuning_days']} of {laundering_tune} found on the tuning days at {MAIN_BUDGET:,} alerts a day) ({time.time() - t0:.0f} s)", flush=True)
    del tuned

    # 2. a forest that sees no labels
    rng = np.random.default_rng(SEED)
    sample = rng.choice(np.flatnonzero(train), size=200_000, replace=False)
    forest = IsolationForest(n_estimators=200, max_samples=4096, random_state=SEED, n_jobs=2).fit(X.loc[sample, CANDIDATES])
    scores["forest"] = in_parts(lambda part: -forest.score_samples(part[CANDIDATES]), X, later)
    print(f"isolation forest scored ({time.time() - t0:.0f} s)", flush=True)
    del forest

    # 3. a logistic regression
    linear = linear_model().fit(X.loc[train, CANDIDATES], y[train])
    scores["linear"] = in_parts(lambda part: linear.predict_proba(part[CANDIDATES])[:, 1], X, later)
    result["linear"] = {"columns_after_one_hot": int(linear[-1].coef_.shape[1]), "iterations": int(linear[-1].n_iter_[0])}
    print(f"logistic regression scored ({time.time() - t0:.0f} s)", flush=True)
    del linear

    # 4. boosted trees: a small grid on the full feature set
    grid_rows = []
    for params in GRID:
        model, trees = fit_boosted(X, y, fit_rows, stop_rows, CANDIDATES, params)
        ap = ranking_quality(scores_of(model, X.loc[tune], CANDIDATES), y[tune])["average_precision"]
        grid_rows.append({**params, "trees": trees, "average_precision_on_tuning_days": ap})
        print(f"grid {params}: {trees} trees, average precision on the tuning days {ap:.4f} ({time.time() - t0:.0f} s)", flush=True)
    chosen = max(grid_rows, key=lambda g: g["average_precision_on_tuning_days"])
    params = {"max_depth": chosen["max_depth"], "min_child_weight": chosen["min_child_weight"]}
    result["grid"] = grid_rows
    result["chosen"] = {**params, **{k: v for k, v in FIXED.items() if k not in ("n_jobs", "eval_metric")}, "most_trees": MOST_TREES, "stop_after": STOP_AFTER}

    #    then one model per feature set, with those settings
    result["models"], fitted_models, speed = {}, {}, {}
    for name, columns in FEATURE_SETS.items():
        t1 = time.time()
        _, trees = fit_boosted(X, y, fit_rows, stop_rows, columns, params)
        fitted_models[name] = refit(X, y, train, columns, params, trees)
        fit_seconds = time.time() - t1
        t1 = time.time()
        scores[name] = scores_of(fitted_models[name], X, columns)
        speed[name] = {"seconds_to_fit": round(fit_seconds), "transfers_scored_a_second": round(len(X) / (time.time() - t1))}
        result["models"][name] = {"features": len(columns), "trees": trees}
        print(f"model '{name}': {len(columns)} features, {trees} trees ({time.time() - t0:.0f} s)", flush=True)

    # 5. the decision: a richer set is adopted only if its gain on the tuning days is clear of zero
    def alerts(name, rows):
        return top_per_day(scores[name][rows], day[rows], MAIN_BUDGET)

    adopted, ladder = "transfer", []
    for name in ("customer", "behavior", "network"):
        gain = paired_gain(alerts(adopted, tune), alerts(name, tune), y[tune], scheme[tune], seed=SEED)
        take = gain["gain_low"] > 0
        ladder.append({"from": adopted, "to": name, **gain, "adopted": bool(take)})
        adopted = name if take else adopted
    result["ladder"], result["adopted"] = ladder, adopted
    columns = FEATURE_SETS[adopted]
    trees_adopted = result["models"][adopted]["trees"]
    fitted_models[adopted].save_model(RESULTS / "fitted" / "model_v1.json")
    write_json(RESULTS / "fitted" / "model_v1_settings.json", {"version": 1, "feature_set": adopted, "features": columns, "trees": trees_adopted, **result["chosen"],
                                                               "trained_on_days": list(TRAIN_DAYS), "seed": SEED})
    result["boosting_against_regression"] = paired_gain(alerts("linear", tune), alerts(adopted, tune), y[tune], scheme[tune], seed=SEED)
    print(f"adopted: '{adopted}' ({len(columns)} features) ({time.time() - t0:.0f} s)", flush=True)
    del fitted_models

    # 6a. does weighting the rare class up help?
    natural = (y[train] == 0).sum() / (y[train] == 1).sum()
    result["class_weight"] = []
    for label, weight in (("none", 1.0), ("square root of the imbalance", float(np.sqrt(natural))), ("the full imbalance", float(natural))):
        s = scores[adopted] if weight == 1.0 else scores_of(refit(X, y, train, columns, params, trees_adopted, weight), X, columns)
        result["class_weight"].append({"weight": label, "value": round(weight, 1),
                                       "average_precision_on_tuning_days": ranking_quality(s[tune], y[tune])["average_precision"],
                                       "found_at_main_budget": at_budget(s[tune], y[tune], day[tune], MAIN_BUDGET)["found"],
                                       "brier_on_tuning_days": float(brier_score_loss(y[tune], s[tune])),
                                       "mean_score": float(s[tune].mean()), "true_rate": float(y[tune].mean())})
        print(f"class weight {label}: average precision {result['class_weight'][-1]['average_precision_on_tuning_days']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    # 6b. does it help to learn from the two warm-up days too? Their 3-day windows are not full.
    wide_fit, wide_all = warmup | fit_rows, warmup | train
    _, trees = fit_boosted(X, y, wide_fit, stop_rows, columns, params)
    s = scores_of(refit(X, y, wide_all, columns, params, trees), X.loc[tune], columns)
    result["warmup_check"] = [
        {"trained_on": f"days {TRAIN_DAYS[0]} to {TRAIN_DAYS[-1]}", "transfers": int(train.sum()), "laundering": int(y[train].sum()), "trees": trees_adopted,
         "average_precision_on_tuning_days": ranking_quality(scores[adopted][tune], y[tune])["average_precision"],
         "found_at_main_budget": at_budget(scores[adopted][tune], y[tune], day[tune], MAIN_BUDGET)["found"]},
        {"trained_on": f"days {WARMUP_DAYS[0]} to {TRAIN_DAYS[-1]}", "transfers": int(wide_all.sum()), "laundering": int(y[wide_all].sum()), "trees": trees,
         "average_precision_on_tuning_days": ranking_quality(s, y[tune])["average_precision"],
         "found_at_main_budget": at_budget(s, y[tune], day[tune], MAIN_BUDGET)["found"]}]
    print(f"warm-up check done ({time.time() - t0:.0f} s)", flush=True)

    # 7. measure everything on the tuning days and on the test days
    result["tune"], result["test"] = {}, {}
    for name in ORDER:
        s = scores[name]
        result["tune"][name] = measure(s, y, day, scheme, tune)
        result["test"][name] = measure(s, y, day, scheme, test)
        line, f1 = best_f1_line(s[tune], y[tune])
        result["tune"][name]["best_f1"] = f1
        result["test"][name]["f1_at_the_tuning_line"] = f1_at_line(s[test], y[test], line)

    #    where laundering is found on the test days, by scheme type and by when the scheme began
    shown = ("rule", "linear", adopted)
    alerted = {name: alerts(name, test) for name in shown}
    yt = y[test] == 1
    kind = index["scheme_type"].to_numpy()[test]
    kind = np.where(kind == "", "Not in a labeled scheme", kind)
    began = pd.to_datetime(pd.Series(index["scheme_start"].to_numpy()[test]).replace("", pd.NaT), format="%Y/%m/%d %H:%M")
    began_day = (began.dt.normalize() - pd.Timestamp(FIRST_DATE)).dt.days.to_numpy()
    groups = [(k, yt & (kind == k)) for k in sorted(set(kind[yt]))]
    groups += [("In a labeled scheme (all 8 types)", yt & (kind != "Not in a labeled scheme")),
               ("In a labeled scheme that began before the test days", yt & (began_day < TEST_DAYS[0])),
               ("In a labeled scheme that began on the test days", yt & (began_day >= TEST_DAYS[0])),
               ("All laundering", yt)]
    result["test_by_kind"] = {k: {"laundering": int(m.sum()), **{name: int((a & m).sum()) for name, a in alerted.items()}} for k, m in groups}

    pd.DataFrame({name: scores[name] for name in ORDER}).to_parquet(WORK / "hi_scores.parquet", index=False)
    write_json(RESULTS / "models.json", result)
    write_report(result)
    note_time("03_models", time.time() - t0, boosted_models=speed)
    print(f"done in {time.time() - t0:.0f} s. See results/models.md")


def write_report(r):
    adopted, main = r["adopted"], str(r["main_budget"])
    lines = ["# Step 3: the rule, the forest, the regression and the boosted models", "",
             "A replay on a public, synthetic benchmark. No real customer or bank is in it.", "",
             "## The days", "",
             "| Part | Days | Transfers | Laundering | One in |", "|---|---|---|---|---|"]
    label = {"warmup": "Warm-up (history only)", "train": "Training", "tune": "Tuning", "test": "Test", "after_day_9": "After day 9 (left out)"}
    for part in ("warmup", "train", "tune", "test", "after_day_9"):
        days = "10 to 17" if part == "after_day_9" else f"{r['days'][part][0]} to {r['days'][part][-1]}"
        x = r["rows"][part]
        lines.append(f"| {label[part]} | {days} | {x['transfers']:,} | {x['laundering']:,} | {x['one_in']:,} |")
    lines += ["", "Day 0 is September 1, 2022. Most features look back 3 days, so days 0 and 1 have less history behind them than the rest. "
              "They feed the history of later days and no model learns from their rows. Regular activity stops after day 9. "
              "The later days hold only the last legs of laundering schemes, so they are scored and left out of every result.", "",
              f"## Test days at {r['main_budget']:,} alerts a day", "",
              "| Method | Alerts | Laundering found | Precision | 95% interval | Share of laundering found | 95% interval |", "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        c = r["test"][name]["by_budget"][main]
        lines.append(f"| {NAMES[name]}{' (adopted)' if name == adopted else ''} | {c['alerts']:,} | {c['found']:,} | {c['precision']:.1%} | {c['precision_low']:.1%} to {c['precision_high']:.1%} | "
                     f"{c['recall']:.1%} | {c['recall_low']:.1%} to {c['recall_high']:.1%} |")
    lines += ["", "The precision interval is a Wilson interval. The interval for the share found resamples whole schemes, because transfers of one scheme are found or missed together.", ""]
    for part, title in (("test", "## Test days at other numbers of alerts a day"), ("tune", "## Tuning days")):
        lines += [title, "", f"Laundering found, as a share of all laundering on those days ({r['rows'][part]['laundering']:,} laundering transfers in {r['rows'][part]['transfers']:,}).", "",
                  "| Method | " + " | ".join(f"{k:,} a day" for k in r["budgets"]) + " | Average precision | ROC AUC |",
                  "|---|" + "---|" * (len(r["budgets"]) + 2)]
        for name in ORDER:
            m = r[part][name]
            lines.append(f"| {NAMES[name]} | " + " | ".join(f"{m['by_budget'][str(k)]['recall']:.1%}" for k in r["budgets"]) + f" | {m['average_precision']:.3f} | {m['roc_auc']:.3f} |")
        lines.append("")
    lines += ["## Does each richer feature set pay for itself?", "",
              f"Judged on the tuning days at {r['main_budget']:,} alerts a day. A set is adopted only if the whole 95% interval for its gain is above zero.", "",
              "| From | To | Found before | Found after | Only the first finds | Only the second finds | Gain in share found | 95% interval | Adopted |", "|---|---|---|---|---|---|---|---|---|"]
    for g in r["ladder"]:
        lines.append(f"| {NAMES[g['from']].split(': ')[1]} | {NAMES[g['to']].split(': ')[1]} | {g['found_a']} | {g['found_b']} | {g['only_a']} | {g['only_b']} | "
                     f"{g['gain'] * 100:+.1f} points | {g['gain_low'] * 100:+.1f} to {g['gain_high'] * 100:+.1f} | {'Yes' if g['adopted'] else 'No'} |")
    b = r["boosting_against_regression"]
    lines += ["", f"Adopted model: {NAMES[adopted]} ({r['models'][adopted]['features']} features, {r['models'][adopted]['trees']} trees).", "",
              f"Against the logistic regression on the same tuning days, the adopted model finds {b['found_b']} to the regression's {b['found_a']}: "
              f"{b['gain'] * 100:+.1f} points, 95% interval {b['gain_low'] * 100:+.1f} to {b['gain_high'] * 100:+.1f}.", "",
              "## The rule variants", "",
              f"Peer means, spreads and each variant's terms come from the training days. The variant that finds most on the tuning days at {r['main_budget']:,} alerts a day is kept as the rule.", "",
              f"| Variant | Found on training days (of {r['rule']['laundering_on_training_days']:,}) | Found on tuning days (of {r['rows']['tune']['laundering']:,}) | Share found on tuning days |", "|---|---|---|---|"]
    lines += [f"| {v['variant']} | {v['found_on_training_days']} | {v['found_on_tuning_days']} | {v['share_found']:.1%} |" for v in r["rule"]["variants"]]
    lines += ["", f"Kept: {r['rule']['kept']}. Its terms:", ""]
    kept = next(v for v in r["rule"]["variants"] if v["variant"] == r["rule"]["kept"])
    lines += [f"- {t['meaning']} ({t['side']} side)" for t in (kept["terms"] or [])] or ["- every watched quantity, high side"]
    side = r["rule"]["terms_chosen_on_tuning_days_instead"]
    lines += ["", f"A side check on where the terms are chosen. The same sum of five with its terms chosen on the tuning days instead finds {side['found_on_tuning_days']} on the tuning days "
              f"and {side['found_on_test_days']} of {side['laundering_on_test_days']} on the test days. Both tuning days are mid-week. "
              "The training days hold a weekend and two weekdays, and terms chosen there hold up on the test days."]
    lines += ["", "## The grid", "", f"Fitted on days {r['days']['train'][0]} to {r['days']['train'][-2]}, trees stopped on day {r['days']['train'][-1]}, compared on the tuning days.", "",
              "| Depth | Smallest leaf weight | Trees | Average precision on the tuning days |", "|---|---|---|---|"]
    lines += [f"| {g['max_depth']} | {g['min_child_weight']} | {g['trees']} | {g['average_precision_on_tuning_days']:.4f} |" for g in r["grid"]]
    lines += ["", f"Chosen: depth {r['chosen']['max_depth']}, smallest leaf weight {r['chosen']['min_child_weight']}.", "",
              "## The boosted models", "", "| Model | Features | Trees |", "|---|---|---|"]
    lines += [f"| {NAMES[n]} | {m['features']} | {m['trees']} |" for n, m in r["models"].items()]
    lines += ["", "## Weighting the rare class (adopted model, tuning days)", "",
              "| Weight on laundering | Value | Average precision | Found at the main budget | Brier score | Mean score | True rate |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {c['weight']} | {c['value']} | {c['average_precision_on_tuning_days']:.4f} | {c['found_at_main_budget']} | {c['brier_on_tuning_days']:.6f} | {c['mean_score']:.5f} | {c['true_rate']:.5f} |"
              for c in r["class_weight"]]
    lines += ["", "A weight changes the scores, so the mean score moves away from the true rate. No weight is used.", "",
              "## Learning from the warm-up days too (adopted model, tuning days)", "",
              "| Trained on | Transfers | Laundering | Trees | Average precision | Found at the main budget |", "|---|---|---|---|---|---|"]
    lines += [f"| {w['trained_on']} | {w['transfers']:,} | {w['laundering']:,} | {w['trees']} | {w['average_precision_on_tuning_days']:.4f} | {w['found_at_main_budget']} |" for w in r["warmup_check"]]
    lines += ["", f"## Test days by kind of laundering, at {r['main_budget']:,} alerts a day", "",
              "| Laundering | Transfers | Rule finds | Regression finds | Adopted model finds | Share |", "|---|---|---|---|---|---|"]
    lines += [f"| {k.capitalize() if k.isupper() else k} | {v['laundering']:,} | {v['rule']:,} | {v['linear']:,} | {v[adopted]:,} | {v[adopted] / max(v['laundering'], 1):.0%} |" for k, v in r["test_by_kind"].items()]
    lines += ["", "## F1 for the laundering class", "", "The line is the one with the best F1 on the tuning days, then applied to the test days.", "",
              "| Method | Best F1 on tuning days | F1 on test days at that line |", "|---|---|---|"]
    lines += [f"| {NAMES[n]} | {r['tune'][n]['best_f1']:.3f} | {r['test'][n]['f1_at_the_tuning_line']:.3f} |" for n in ORDER]
    (RESULTS / "models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
