"""Step 6. Score the model the way the published papers score theirs, so the numbers can sit side by side.

    python 06_published_protocol.py            about 36 minutes on 2 cores

The published rules, read from the paper (arXiv 2306.11586v3) and its code (github.com/IBM/Multi-GNN):
  - Whole days are split into train, validation and test, as close to 60/20/20 by transfers as whole days allow.
    For both portfolios that is days 0 to 5, days 6 and 7, and day 8 onward.
  - The test part includes the days after day 9, where regular activity has stopped and most transfers are laundering.
  - The measure is F1 for the laundering class. Settings are chosen on the validation days. Five seeds are averaged.
  - The published models see six fields: sender, receiver, time, amount received, currency received, payment format.

What this script does with those rules:
  - Trains on days 2 to 5 (the first two days stay history only, as in step 3), stops the trees on the validation
    days, sets the alert line at the best F1 on the validation days, and reports F1 on the test days.
  - Runs four feature sets. Three are built only from the six published fields: account behavior (22 features),
    account behavior with the 16 network features (38), and account behavior with the 7-day pair history that
    version 2 added (23). The fourth is version 2 itself (33 features), which also uses the customer file, the
    bank and the currency paid.
  - Uses the depth and leaf size chosen in step 3. Nothing is tuned again here.

Writes results/published_protocol.json and results/published_protocol.md.
"""
import gc
import itertools
import time

import numpy as np
import pandas as pd

from common import PORTFOLIOS, RAW, RAW_COLUMNS, RESULTS, SEED, WARMUP_DAYS, WORK, note_time, read_json, write_json
from features import CANDIDATES, CURRENCIES, GROUPS
from metrics import best_f1_line
from model import fit_boosted, scores_of

SEEDS = [SEED + k for k in range(5)]
PUBLISHED = {            # arXiv 2306.11586v3, Table 2: F1 for the laundering class in percent, mean and spread over five seeds
    "LightGBM + graph features": {"hi": (62.86, 0.25), "li": (20.83, 1.50)},
    "XGBoost + graph features": {"hi": (63.23, 0.17), "li": (27.30, 0.33)},
    "GIN (graph neural network)": {"hi": (28.70, 1.13), "li": (7.90, 2.78)},
    "PNA (graph neural network)": {"hi": (56.77, 2.41), "li": (14.85, 1.46)},
    "Multi-GIN+EU (graph neural network)": {"hi": (64.79, 1.22), "li": (26.88, 6.63)},
    "Multi-PNA+EU (graph neural network, best published)": {"hi": (68.16, 2.65), "li": (33.07, 2.63)},
}
# Features that need a field the published models do not see.
NOT_PUBLISHED_FIELDS = GROUPS["customer"] + ["same_bank", "cross_currency", "s_currencies_3d", "r_currencies_3d"]
SAME_FIELDS = [f for f in CANDIDATES if f not in NOT_PUBLISHED_FIELDS]     # 38: everything that needs only the published fields
SAME_PLAIN = [f for f in SAME_FIELDS if f not in GROUPS["network"]]        # 22: without the network group
SAME_WEEK = SAME_PLAIN + ["pair_before_7d"]                                # 23: with the 7-day pair history of version 2
RECEIVED = {"amount_log": "amount_log_received", "amount_vs_peers": "amount_vs_peers_received", "currency": "currency_received"}
LABELS = ("same_plain", "same", "same_week", "v2")
SHORT = {"same_plain": "Same fields: account behavior", "same": "Same fields: + network", "same_week": "Same fields: + 7-day pair history", "v2": "Version 2"}


def published_split(block):
    """The split rule of the published code: whole 24-hour blocks, as close to 60/20/20 as possible."""
    counts = np.bincount(block)
    best, best_error = None, np.inf
    for i, j in itertools.combinations(range(len(counts)), 2):
        parts = np.array([counts[:i].sum(), counts[i:j].sum(), counts[j:].sum()]) / counts.sum()
        error = np.max(np.abs(parts - np.array([0.6, 0.2, 0.2])) / np.array([0.6, 0.2, 0.2]))
        if error < best_error:
            best, best_error = (i, j), error
    return best


def f1_parts(alerted, y):
    found, alerts, total = int((alerted & (y == 1)).sum()), int(alerted.sum()), int((y == 1).sum())
    return {"alerts": alerts, "found": found, "laundering": total, "precision": found / max(alerts, 1), "recall": found / max(total, 1),
            "f1": 2 * found / max(alerts + total, 1)}


def as_received(names):
    """The same features, with amount and currency taken as received. Those are the fields the published models see."""
    return [RECEIVED.get(f, f) for f in names]


def run_portfolio(portfolio, params, version_2, t0):
    """Everything for one portfolio, inside one function so that its memory is given back before the next one."""
    needed = list(dict.fromkeys(SAME_FIELDS + ["pair_before_7d"] + version_2))
    X = pd.read_parquet(WORK / f"{portfolio}_features.parquet", columns=needed)
    index = pd.read_parquet(WORK / f"{portfolio}_index.parquet", columns=["y", "day", "minute"])
    y, day = index["y"].to_numpy(), index["day"].to_numpy()

    # the published split: 24-hour blocks counted from the first transfer
    minutes = day.astype(np.int64) * 1440 + index["minute"].to_numpy()
    block = (minutes - minutes.min()) // 1440
    i, j = published_split(block)
    train_days, valid_days, test_days = list(range(i)), list(range(i, j)), list(range(j, int(block.max()) + 1))
    train = (block < i) & ~np.isin(day, WARMUP_DAYS)
    valid = (block >= i) & (block < j)
    test = block >= j
    regular_test = test & (block <= 9)
    later = valid | test
    info = {"blocks_are_calendar_days": bool((block == day).all()), "first_transfer_minute_of_day_0": int(minutes.min()),
            "train_days": train_days, "validation_days": valid_days, "test_days": test_days,
            "shares": {"train": float((block < i).mean()), "validation": float(valid.mean()), "test": float(test.mean())},
            "rows": {"train_used": int(train.sum()), "validation": int(valid.sum()), "test": int(test.sum()), "test_days_8_and_9": int(regular_test.sum())},
            "laundering": {"train_used": int(y[train].sum()), "validation": int(y[valid].sum()), "test": int(y[test].sum()),
                           "test_days_8_and_9": int(y[regular_test].sum()), "test_after_day_9": int(y[test & ~regular_test].sum())},
            "transfers_after_day_9": int((test & ~regular_test).sum())}
    print(f"{portfolio}: train days {train_days}, validation {valid_days}, test {test_days[0]} to {test_days[-1]}; "
          f"shares {info['shares']['train']:.1%} / {info['shares']['validation']:.1%} / {info['shares']['test']:.1%}", flush=True)

    # amount and currency as received, for the "same fields" sets
    raw = pd.read_csv(RAW / PORTFOLIOS[portfolio]["transactions"], header=0, names=RAW_COLUMNS, usecols=["amt_recv", "cur_recv", "fmt"],
                      dtype={"cur_recv": "category", "fmt": "category"})
    received = raw["amt_recv"].to_numpy()
    group = raw["cur_recv"].cat.codes.to_numpy().astype(np.int64) * 100 + raw["fmt"].cat.codes.to_numpy()
    medians = pd.Series(received[train]).groupby(group[train]).median()
    peer = pd.Series(group).map(medians[medians > 0]).to_numpy(dtype=float)
    X["amount_log_received"] = np.log10(1 + received).astype(np.float32)
    X["amount_vs_peers_received"] = np.where(np.isfinite(peer) & (received > 0), np.log10(np.maximum(received, 1e-9) / peer), 0).astype(np.float32)
    X["currency_received"] = pd.Categorical(raw["cur_recv"], categories=CURRENCIES).codes.astype(np.float32)
    del raw, group, peer, received, index, minutes
    gc.collect()

    info["runs"] = {}
    for label, columns in (("same_plain", as_received(SAME_PLAIN)), ("same", as_received(SAME_FIELDS)), ("same_week", as_received(SAME_WEEK)), ("v2", version_2)):
        runs = []
        for seed in SEEDS:
            model, trees = fit_boosted(X, y, train, valid, columns, params, seed=seed)
            score = np.zeros(len(X), dtype=np.float32)
            score[later] = scores_of(model, X.loc[later, columns], columns)
            line, valid_f1 = best_f1_line(score[valid], y[valid])
            run = {"seed": seed, "trees": trees, "line": line, "validation_f1": valid_f1,
                   "test": f1_parts(score[test] >= line, y[test]),
                   "test_days_8_and_9": f1_parts(score[regular_test] >= line, y[regular_test]),
                   "test_after_day_9": f1_parts(score[test & ~regular_test] >= line, y[test & ~regular_test]),
                   "test_at_half": f1_parts(score[test] >= 0.5, y[test])}
            runs.append(run)
            del model
            print(f"  {portfolio} {label} seed {seed}: {trees} trees, F1 on the published test days {run['test']['f1']:.4f} "
                  f"(days 8 and 9 only {run['test_days_8_and_9']['f1']:.4f}; line at 0.5 {run['test_at_half']['f1']:.4f}) ({time.time() - t0:.0f} s)", flush=True)
        summary = {}
        for part in ("test", "test_days_8_and_9", "test_after_day_9", "test_at_half"):
            values = np.array([r[part]["f1"] for r in runs]) * 100
            summary[part] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "lowest": float(values.min()), "highest": float(values.max())}
        info["runs"][label] = {"features": len(columns), "by_seed": runs, "f1_percent": summary}
    return info


def main():
    t0 = time.time()
    settings = read_json(RESULTS / "fitted" / "model_v2_settings.json")
    params = {"max_depth": settings["max_depth"], "min_child_weight": settings["min_child_weight"]}
    version_2 = settings["features"]
    result = {"published": {name: {p: {"mean": m, "sd": s} for p, (m, s) in row.items()} for name, row in PUBLISHED.items()},
              "seeds": SEEDS, "settings": params, "same_fields": SAME_FIELDS, "fields_left_out_of_same": NOT_PUBLISHED_FIELDS,
              "feature_counts": {"same_plain": len(SAME_PLAIN), "same": len(SAME_FIELDS), "same_week": len(SAME_WEEK), "v2": len(version_2)}, "portfolios": {}}
    for portfolio in ("hi", "li"):
        result["portfolios"][portfolio] = run_portfolio(portfolio, params, version_2, t0)
        gc.collect()
    write_json(RESULTS / "published_protocol.json", result)
    write_report(result)
    note_time("06_published_protocol", time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s. See results/published_protocol.md")


def write_report(r):
    hi, li = r["portfolios"]["hi"], r["portfolios"]["li"]
    counts = r["feature_counts"]

    def cell(portfolio, label, part):
        v = r["portfolios"][portfolio]["runs"][label]["f1_percent"][part]
        return f"{v['mean']:.2f} ± {v['sd']:.2f}"

    lines = ["# Step 6: scored the way the published papers score", "",
             "F1 for the laundering class, in percent. Mean and spread over five seeds. Higher is better.", "",
             "| Model | Source | HI-Small | LI-Small |", "|---|---|---|---|"]
    for name, row in r["published"].items():
        lines.append(f"| {name} | Published | {row['hi']['mean']:.2f} ± {row['hi']['sd']:.2f} | {row['li']['mean']:.2f} ± {row['li']['sd']:.2f} |")
    for label in LABELS:
        lines.append(f"| Boosted trees. {SHORT[label]} ({counts[label]} features) | This run | {cell('hi', label, 'test')} | {cell('li', label, 'test')} |")
    lines += ["", "Published numbers: Egressy and others, Provably Powerful Graph Neural Networks for Directed Multigraphs, arXiv 2306.11586v3, Table 2.", "",
              "## What is the same and what is not", "",
              "| | Published | This run |", "|---|---|---|",
              f"| Split | Whole days, closest to 60/20/20 | The same rule: train days {hi['train_days'][0]} to {hi['train_days'][-1]}, validation days {hi['validation_days'][0]} and {hi['validation_days'][-1]}, test day {hi['test_days'][0]} onward |",
              "| Measure | F1 for the laundering class on the test days | The same |",
              "| Days after day 9 | In the test part | In the test part here. Left out everywhere else in this project |",
              "| Chosen on the validation days | Settings, loss weights, the training round to keep | The number of trees and the alert line |",
              "| Seeds | Five, averaged | Five, averaged |",
              "| Fields the model sees | Sender, receiver, time, amount received, currency received, payment format | \"Same fields\": those six. Version 2: also the customer file, the bank and the currency paid |",
              "| What a test transfer may use | The whole graph, later transfers included | Only transfers up to the end of its own day |",
              "| Training rows | Days 0 to 5 | Days 2 to 5. Days 0 and 1 are history only |",
              "| Features | Learned by the network, or the publisher's graph features | The features of features.py |",
              "", "## The test part, split in two", "",
              "The published test part mixes two very different periods. Days 8 and 9 are regular days. After day 9 the files hold only the last legs of laundering schemes.", "",
              "| Portfolio | Period | Transfers | Laundering | " + " | ".join(SHORT[label] for label in LABELS) + " |", "|---|---|---|---|" + "---|" * len(LABELS)]
    for portfolio, info, title in (("hi", hi, "HI-Small"), ("li", li, "LI-Small")):
        rows, laun = info["rows"], info["laundering"]
        for period, part, n_rows, n_laun in (("Day 8 onward (published test)", "test", rows["test"], laun["test"]),
                                             ("Days 8 and 9 only", "test_days_8_and_9", rows["test_days_8_and_9"], laun["test_days_8_and_9"]),
                                             ("After day 9 only", "test_after_day_9", info["transfers_after_day_9"], laun["test_after_day_9"])):
            lines.append(f"| {title} | {period} | {n_rows:,} | {n_laun:,} | " + " | ".join(cell(portfolio, label, part) for label in LABELS) + " |")
    lines += ["", "## With the line at 0.5", "",
              "The published code calls a transfer laundering when the model's score for laundering is the larger one, which is a line at 0.5 on scores "
              "shaped by a loss weight tuned on the validation days. A model with no such weight needs its line set on the validation days. "
              "With the line simply left at 0.5:", "",
              "| Portfolio | " + " | ".join(SHORT[label] for label in LABELS) + " |", "|---|" + "---|" * len(LABELS),
              "| HI-Small | " + " | ".join(cell("hi", label, "test_at_half") for label in LABELS) + " |",
              "| LI-Small | " + " | ".join(cell("li", label, "test_at_half") for label in LABELS) + " |",
              "", "## Each seed", "",
              "| Portfolio | Features | Seed | Trees | Line | Alerts | Found | Laundering | Precision | Share found | F1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for portfolio, title in (("hi", "HI-Small"), ("li", "LI-Small")):
        for label in LABELS:
            for run in r["portfolios"][portfolio]["runs"][label]["by_seed"]:
                t = run["test"]
                lines.append(f"| {title} | {SHORT[label]} ({counts[label]}) | {run['seed']} | {run['trees']} | {run['line']:.3f} | {t['alerts']:,} | {t['found']:,} | {t['laundering']:,} | "
                             f"{t['precision']:.1%} | {t['recall']:.1%} | {t['f1'] * 100:.2f} |")
    (RESULTS / "published_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
