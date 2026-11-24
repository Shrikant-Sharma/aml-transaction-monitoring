"""Step 7. Reasons for the alerts, the charts, the model document and the headline numbers.

    python 07_report.py            about 1 minute

  1. Reasons. For every alert version 2 raises on the test days: the three features that pushed its score up most.
     The contributions are SHAP values computed from the trees themselves; they add up to the score exactly.
  2. What drives the model: the average size of each feature's contribution, on the alerts and on all test-day transfers.
  3. Five charts in results/charts/.
  4. results/model_document.md: purpose, data, method, tests, results, limits, monitoring plan.
  5. results/summary.json: the headline numbers, each under the step it comes from.
"""
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FORMATS, RESULTS, SEED, TEST_DAYS, WORK, note_time, read_json, write_json
from features import CURRENCIES, CUSTOMER_TYPES, GROUPS, MEANING
from model import apply_calibrator, contributions, load_model, log_odds, scores_of

CHARTS = RESULTS / "charts"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# One color per thing, the same in every chart. Checked with a palette validator for color-blind readers;
# every series also carries a written label, so color is never the only signal.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COLOR = {"v1": BLUE, "v2": ORANGE, "linear": AQUA, "rule": YELLOW, "forest": MAGENTA}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
})
FLAGS = {"to_self", "same_bank", "cross_currency", "same_owner", "cross_border", "back_3d", "loop_3d", "loop_5d"}
SAME_FIELD_SETS = ("same_plain", "same", "same_week")           # the feature sets of step 6 that use only the published fields


# -------------------------------------------------------------------- reasons
def say(feature, value, amount, currency):
    """One feature of one transfer, in words."""
    if feature == "amount_log":
        return f"size of the transfer: {amount:,.2f} {currency}"
    if feature == "amount_vs_peers":
        return f"{10 ** value:,.1f} times the usual amount for its currency and payment format"
    if feature == "format":
        return f"payment format: {FORMATS[int(value)]}"
    if feature == "currency":
        return f"currency: {CURRENCIES[int(value)]}"
    if feature in ("sender_type", "receiver_type"):
        who = CUSTOMER_TYPES[int(value)] if 0 <= int(value) < len(CUSTOMER_TYPES) else "not in the customer file"
        return f"{MEANING[feature]}: {who}"
    if feature == "hour":
        return f"time of day: {int(value):02d}:00"
    if feature in FLAGS:
        return f"{MEANING[feature]}: {'yes' if value else 'no'}"
    return f"{MEANING[feature]}: {int(value):,}"


def reasons(result):
    settings = read_json(RESULTS / "fitted" / "model_v2_settings.json")
    frozen = read_json(RESULTS / "fitted" / "alert_line_v2.json")
    columns, line = settings["features"], frozen["line"]
    model = load_model(RESULTS / "fitted" / "model_v2.json")
    X = pd.read_parquet(WORK / "hi_features.parquet", columns=columns)
    index = pd.read_parquet(WORK / "hi_index.parquet", columns=["y", "day", "fmt", "cur_paid", "amt_paid"])
    test = np.flatnonzero(np.isin(index["day"].to_numpy(), TEST_DAYS))
    score = scores_of(model, X.iloc[test], columns)
    alerts, alert_scores = test[score >= line], score[score >= line]
    part = contributions(model, X.iloc[alerts], columns)
    gap = float(np.abs(part.sum(axis=1) - log_odds(model, X.iloc[alerts], columns)).max())
    assert gap < 1e-3, gap                                          # the contributions must add up to the score
    values, y = X.iloc[alerts].to_numpy(), index["y"].to_numpy()[alerts]
    amount, currency = index["amt_paid"].to_numpy()[alerts], index["cur_paid"].to_numpy()[alerts]
    probability = apply_calibrator(frozen["calibrator"], alert_scores)
    top3 = np.argsort(-part[:, :-1], axis=1)[:, :3]
    with open(RESULTS / "alert_reasons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_in_file", "day", "amount", "currency", "payment_format", "score", "probability", "laundering_known_later", "reason_1", "reason_2", "reason_3"])
        for i, row in enumerate(alerts):
            why = [say(columns[j], values[i, j], amount[i], currency[i]) if part[i, j] > 0 else "" for j in top3[i]]
            w.writerow([int(row) + 2, int(index["day"].iloc[row]), f"{amount[i]:.2f}", currency[i], index["fmt"].iloc[row], f"{alert_scores[i]:.4f}", f"{probability[i]:.4f}", int(y[i])] + why)
    in_top3 = np.bincount(top3.ravel(), minlength=len(columns))
    sample = np.random.default_rng(SEED).choice(test, size=200_000, replace=False)
    everywhere = np.abs(contributions(model, X.iloc[sample], columns)[:, :-1]).mean(axis=0)
    on_alerts = np.abs(part[:, :-1]).mean(axis=0)
    group_of = {f: g for g, names in GROUPS.items() for f in names}
    table = [{"feature": f, "meaning": MEANING[f], "group": group_of.get(f, "added in version 2"), "mean_size_on_alerts": float(on_alerts[j]),
              "mean_size_on_all_test_transfers": float(everywhere[j]), "among_the_top_3_reasons": int(in_top3[j])} for j, f in enumerate(columns)]
    table.sort(key=lambda t: -t["mean_size_on_alerts"])
    result["reasons"] = {"alerts": int(len(alerts)), "laundering_among_them": int(y.sum()), "largest_gap_between_contributions_and_score": gap,
                         "sample_of_all_test_transfers": int(len(sample)), "features": table}
    lines = ["# Why the alerts were raised", "",
             f"Version 2 raised {len(alerts):,} alerts on the test days at its fixed line. results/alert_reasons.csv has one row per alert with the three features that pushed its score up most.", "",
             "The contributions are SHAP values taken from the trees themselves. For every alert they add up to the model's score "
             f"on the log-odds scale; the largest gap found was {gap:.1e}.", "",
             "| Feature | Group | Average push on the alerts | Average push on all test-day transfers | Times among the top 3 reasons |", "|---|---|---|---|---|"]
    lines += [f"| {t['meaning']} | {t['group']} | {t['mean_size_on_alerts']:.2f} | {t['mean_size_on_all_test_transfers']:.2f} | {t['among_the_top_3_reasons']:,} |" for t in table[:15]]
    lines += ["", "The push is in log-odds: 0.7 doubles the odds, 2.3 multiplies them by ten. It says what moved this model's score. It does not say what causes laundering.", ""]
    (RESULTS / "reasons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- charts
def clean(ax, grid_axis="y", baseline=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(baseline)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def titled(fig, title, subtitle, legend_rows=0):
    """Title and subtitle at fixed distances from the top edge, whatever the height of the figure. Returns the top of the plot area."""
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.10 / height, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.40 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    return 1 - (0.72 + 0.30 * legend_rows) / height


def chart_methods(models):
    budgets = models["budgets"]
    series = [("v1", models["adopted"], "Boosted trees (version 1)"), ("linear", "linear", "Logistic regression"),
              ("rule", "rule", "Peer-group rule (z-scores)"), ("forest", "forest", "Isolation forest")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    x = np.arange(len(budgets))
    for key, name, label in series:
        share = [100 * models["test"][name]["by_budget"][str(k)]["recall"] for k in budgets]
        ax.plot(x, share, color=COLOR[key], linewidth=2, solid_capstyle="round", zorder=3, label=label)
        ax.scatter(x, share, s=42, color=COLOR[key], edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.annotate(f"{label}  {share[-1]:.0f}%", (x[-1], share[-1]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{k:,}" for k in budgets])
    ax.set_xlim(-0.25, len(budgets) - 1 + 1.75)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Alerts a day")
    clean(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, handlelength=1.6)
    top = titled(fig, "Share of laundering found on the two test days", f"{models['rows']['test']['laundering']:,} laundering transfers in {models['rows']['test']['transfers']:,}. Every method gets the same number of alerts a day.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "methods.png")
    plt.close(fig)


def chart_ladder(models):
    names = {"transfer": "Transfer fields (8)", "customer": "+ customer fields (14)", "behavior": "+ account behavior (32)", "network": "+ network (48)"}
    found = {"transfer": models["ladder"][0]["found_a"], **{g["to"]: g["found_b"] for g in models["ladder"]}}
    total = models["rows"]["tune"]["laundering"]
    fig, ax = plt.subplots(figsize=(9.2, 3.5), dpi=200)
    order = list(names)[::-1]
    for i, key in enumerate(order):
        taken = key == "transfer" or next(g for g in models["ladder"] if g["to"] == key)["adopted"]
        ax.barh(i, found[key], height=0.42, color=BLUE if taken else "#86b6ef", zorder=3)
        note = ""
        if key != "transfer":
            g = next(g for g in models["ladder"] if g["to"] == key)
            note = f"   {g['gain'] * 100:+.1f} points ({g['gain_low'] * 100:+.1f} to {g['gain_high'] * 100:+.1f})" + ("" if g["adopted"] else ", not adopted")
        ax.text(found[key] + total * 0.012, i, f"{found[key]:,}{note}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)), [names[k] for k in order], fontsize=9.5, color=INK)
    ax.set_xlim(0, total * 1.42)
    ax.set_xticks([0, 250, 500, 750, 1000], ["0", "250", "500", "750", "1,000"])
    ax.set_xlabel(f"Laundering transfers found, of {total:,}", loc="left")
    clean(ax, grid_axis="x", baseline=False)
    top = titled(fig, "Does each richer feature set pay for itself?", f"Tuning days, {models['main_budget']:,} alerts a day. Beside each bar: its gain over the set above it, with a 95% interval.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "ladder.png")
    plt.close(fig)


def chart_friday(monitoring):
    panels = [("First portfolio (HI-Small)", monitoring["version_2"]["first_portfolio_v1"], monitoring["version_2"]["first_portfolio_v2"]),
              ("Second portfolio (LI-Small), each version trained there", monitoring["second_portfolio_check"]["v1_by_day"], monitoring["second_portfolio_check"]["v2_by_day"])]
    weekday = {6: "Wed", 7: "Thu", 8: "Fri", 9: "Sat"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200, sharey=True)
    highest = max(max(d["alerts"] for d in a + b) for _, a, b in panels)
    for ax, (title, a, b) in zip(axes, panels):
        x = np.arange(len(a))
        for offset, rows, key, label in ((-0.17, a, "v1", "Version 1 (3-day memory)"), (0.17, b, "v2", "Version 2 (adds the 7-day pair history)")):
            ax.bar(x + offset, [d["alerts"] for d in rows], width=0.3, color=COLOR[key], zorder=3, label=label)
            for xi, d in zip(x, rows):
                ax.text(xi + offset, d["alerts"] + highest * 0.012, f"{d['alerts']:,}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x, [f"Day {d['day']}\n{weekday.get(d['day'], '')}" for d in a])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(0, highest * 1.1)
        ax.set_yticks([0, 1000, 2000, 3000], ["0", "1,000", "2,000", "3,000"])
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    top = titled(fig, "Alerts a day at the fixed line", f"Each line was set to give {monitoring['budget']:,} alerts a day on days 6 and 7. Days 8 and 9 are new to the model.", legend_rows=1)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.006, top + 0.34 / fig.get_figheight()), ncol=2, frameon=False, fontsize=9, labelcolor=INK, handlelength=1.2, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "friday.png")
    plt.close(fig)


def chart_by_kind(models):
    adopted = models["adopted"]
    kinds = [(k, v) for k, v in models["test_by_kind"].items() if not k.startswith(("In a labeled", "All"))]
    kinds.sort(key=lambda kv: kv[1][adopted] / kv[1]["laundering"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
    for i, (k, v) in enumerate(kinds):
        share = 100 * v[adopted] / v["laundering"]
        ax.barh(i, share, height=0.5, color=BLUE, zorder=3)
        ax.text(share + 1.5, i, f"{v[adopted]} of {v['laundering']}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(kinds)), [k.capitalize() if k.isupper() else k for k, _ in kinds], fontsize=9.5, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    clean(ax, grid_axis="x", baseline=False)
    top = titled(fig, "What version 1 finds, by kind of laundering", f"Test days, {models['main_budget']:,} alerts a day. The publisher labels eight scheme types; the rest belongs to no labeled scheme.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(CHARTS / "by_kind.png")
    plt.close(fig)


def chart_published(published):
    counts = published["feature_counts"]
    short = {"same_plain": "account behavior", "same": "+ network", "same_week": "+ 7-day pair history", "v2": "version 2 (also uses the customer file)"}
    rows = [(name, "published", row) for name, row in published["published"].items()]
    rows += [(f"This project, same fields: {short[k]} ({counts[k]})" if k != "v2" else f"This project, {short[k]} ({counts[k]})", "ours",
              {p: {"mean": published["portfolios"][p]["runs"][k]["f1_percent"]["test"]["mean"], "sd": published["portfolios"][p]["runs"][k]["f1_percent"]["test"]["sd"]} for p in ("hi", "li")})
             for k in ("same_plain", "same", "same_week", "v2")]
    rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), dpi=200, sharey=True)
    for ax, key, title in zip(axes, ("hi", "li"), ("HI-Small", "LI-Small")):
        for i, (name, kind, row) in enumerate(rows):
            color = BLUE if kind == "ours" else MUTED
            m, sd = row[key]["mean"], row[key]["sd"]
            ax.plot([m - sd, m + sd], [i, i], color=color, linewidth=2, solid_capstyle="round", zorder=3)
            ax.scatter([m], [i], s=52, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
            ax.text(m + max(sd, 1.5) + 1.6, i, f"{m:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xlabel("F1 for the laundering class, percent")
        clean(ax, grid_axis="x", baseline=False)
    axes[0].set_yticks(range(len(rows)), [name for name, _, _ in rows], fontsize=8.5, color=INK)
    handles = [plt.Line2D([], [], color=MUTED, marker="o", linewidth=2, markersize=6, label="Published (Egressy and others, 2024)"),
               plt.Line2D([], [], color=BLUE, marker="o", linewidth=2, markersize=6, label="This project")]
    top = titled(fig, "Scored the published way", "F1 for the laundering class on the published test days. Dot: mean of five seeds. Bar: one standard deviation each side.", legend_rows=1)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, top + 0.34 / fig.get_figheight()), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, top), w_pad=2.6)
    fig.savefig(CHARTS / "published.png")
    plt.close(fig)


# -------------------------------------------------------------------- the model document and the summary
def summary_of(models, line, mon, published, result):
    adopted, main = models["adopted"], str(models["main_budget"])
    test = models["test"]
    v2, second, carried = mon["version_2"], mon["second_portfolio_check"], mon["carried_over"]
    spike1 = next(d for d in mon["version_1"]["first_portfolio"] if d["day"] == TEST_DAYS[0])
    spike2 = next(d for d in v2["first_portfolio_v2"] if d["day"] == TEST_DAYS[0])
    li1 = next(d for d in second["v1_by_day"] if d["day"] == TEST_DAYS[0])
    li2 = next(d for d in second["v2_by_day"] if d["day"] == TEST_DAYS[0])
    li_wide = next(d for d in second["wider_on_test_days"]["by_day"] if d["day"] == TEST_DAYS[0])
    share = next(a for a in mon["what_the_alerts_share"] if a["day"] == TEST_DAYS[0])
    kinds = models["test_by_kind"]
    s = {
        "data": {"by_day": read_json(WORK / "hi_manifest.json")["by_day"], "second_portfolio_by_day": read_json(WORK / "li_manifest.json")["by_day"],
                 "transfers": sum(models["rows"][p]["transfers"] for p in models["rows"]), "laundering": sum(models["rows"][p]["laundering"] for p in models["rows"]),
                 "test_transfers": models["rows"]["test"]["transfers"], "test_laundering": models["rows"]["test"]["laundering"],
                 "second_portfolio_transfers": mon["second_portfolio_size"]["transfers"], "second_portfolio_one_in": mon["second_portfolio_size"]["one_in"]},
        "step_3_test_days_at_1000_a_day": {name: {"found": test[name]["by_budget"][main]["found"], "share_found": test[name]["by_budget"][main]["recall"],
                                                  "precision": test[name]["by_budget"][main]["precision"]} for name in ("rule", "forest", "linear", adopted, "network")},
        "step_3_adopted": {"feature_set": adopted, "features": models["models"][adopted]["features"], "trees": models["models"][adopted]["trees"],
                           "network_gain_on_tuning_days": {k: models["ladder"][-1][k] for k in ("found_a", "found_b", "gain", "gain_low", "gain_high", "adopted")},
                           "in_labeled_schemes": kinds["In a labeled scheme (all 8 types)"], "not_in_a_labeled_scheme": kinds["Not in a labeled scheme"]},
        "step_4_line": {"line": line["line"]["line"], "one_in": line["line"]["one_in"], "calibrator": line["calibration"]["kept"],
                        "test_at_line": {k: line["test_at_line"]["both_test_days"][k] for k in ("alerts", "found", "precision", "recall")},
                        "below_the_line_truth": line["below_the_line"]["truth_on_test_days"],
                        "sampling_typical_error": {name: d["rmse"] for name, d in line["below_the_line"]["designs"].items()},
                        "sampling_coverage": {name: d["coverage"] for name, d in line["below_the_line"]["designs"].items()}},
        "step_5_day_8": {"alerts_version_1": spike1["alerts"], "alerts_expected": mon["reference"]["alerts_per_day"], "found_version_1": spike1["found"],
                         "share_of_alerts_whose_pair_paid_a_week_before": share["pair_paid_exactly_a_week_before"],
                         "share_of_false_alerts_whose_pair_paid_a_week_before": share["false_alerts_pair_paid_a_week_before"],
                         "alerts_version_2": spike2["alerts"], "found_version_2": spike2["found"]},
        "step_5_version_2_first_portfolio_test_days": {"found_at_1000_a_day_v1": v2["test_days_v1_against_v2"]["found_a"], "found_at_1000_a_day_v2": v2["test_days_v1_against_v2"]["found_b"],
                                                      "laundering": v2["test_days_v1_against_v2"]["laundering"]},
        "step_5_second_portfolio_check": {"day_8_alerts_v1": li1["alerts"], "day_8_alerts_v2": li2["alerts"],
                                          **{k: second["test_days_v1_against_v2"][k] for k in ("found_a", "found_b", "laundering", "gain", "gain_low", "gain_high")},
                                          "wider_candidate_found": second["wider_on_test_days"]["against_v1"]["found_b"], "wider_candidate_day_8_alerts": li_wide["alerts"]},
        "step_5_carried_over": {k: carried["same_alerts_carried_against_local"][k] for k in ("found_a", "found_b", "gain", "gain_low", "gain_high")},
        "step_5_retraining": {k: mon["retraining"][k] for k in ("found_a", "found_b", "gain", "gain_low", "gain_high")},
        "step_7_reasons": {k: result["reasons"][k] for k in ("alerts", "laundering_among_them")},
        "step_6_published_protocol_f1_percent": {p: {k: published["portfolios"][p]["runs"][k]["f1_percent"]["test"] for k in ("same_plain", "same", "same_week", "v2")} for p in ("hi", "li")},
    }
    return s


def limits_in_words(rule):
    """The limits of one monitoring rule. Feed, alert volume and precision are read as a multiple of what they are compared with; the two shift checks are plain values."""
    as_multiple = rule["name"] in ("feed", "alert volume", "precision")
    parts = []
    for key, value in rule.items():
        if key.startswith(("red", "amber")):
            level, side = key.split("_")
            if not as_multiple:
                parts.append(f"{level} {side} {value:g}")
            elif value == 1:
                parts.append(f"{level} {side} that")
            else:
                parts.append(f"{level} {side} {value:g} times that")
    return ", ".join(parts).capitalize()


def model_document(models, line, mon, published, result):
    adopted, main = models["adopted"], str(models["main_budget"])
    t = models["test"]
    v2, cc = mon["version_2"], mon["second_portfolio_check"]
    g = cc["test_days_v1_against_v2"]
    spike = next(d for d in mon["version_1"]["first_portfolio"] if d["day"] == TEST_DAYS[0])
    fixed = next(d for d in v2["first_portfolio_v2"] if d["day"] == TEST_DAYS[0])
    share = next(a for a in mon["what_the_alerts_share"] if a["day"] == TEST_DAYS[0])
    k = models["test_by_kind"]
    net = models["ladder"][-1]
    truth = line["below_the_line"]["truth_on_test_days"]
    planned = line["below_the_line"]["designs"]["By score band, planned on the tuning days"]
    co = mon["carried_over"]["same_alerts_carried_against_local"]
    rt = mon["retraining"]
    f1 = published["portfolios"]
    best = max(published["published"].values(), key=lambda row: row["hi"]["mean"])
    same = {p: (min(f1[p]["runs"][k]["f1_percent"]["test"]["mean"] for k in SAME_FIELD_SETS), max(f1[p]["runs"][k]["f1_percent"]["test"]["mean"] for k in SAME_FIELD_SETS)) for p in ("hi", "li")}
    rules = read_json(RESULTS.parent / "monitoring_rules.json")
    L = ["# Model document: transfer-level laundering score", "",
         "Built on a public, synthetic benchmark. No real customer, bank or transfer is in it. Nothing here is deployed.", "",
         "## 1. Purpose and use", "",
         "Rank each day's transfers by how much they look like laundering, so that reviewers work the highest-scored ones first. "
         "The score is advice for a reviewer. It closes nothing and files nothing on its own.", "",
         "## 2. Data", "",
         f"The IBM anti-money-laundering benchmark, HI-Small: {sum(models['rows'][p]['transfers'] for p in models['rows']):,} transfers, "
         f"{sum(models['rows'][p]['laundering'] for p in models['rows']):,} labeled laundering (1 in {round(sum(models['rows'][p]['transfers'] for p in models['rows']) / sum(models['rows'][p]['laundering'] for p in models['rows'])):,}). "
         f"All but {models['rows']['after_day_9']['transfers']:,} of the transfers fall in ten regular days. "
         "Labels come from the simulator, so every laundering transfer is known. A real bank knows only what its reviewers confirmed.", "",
         "| Days | Use |", "|---|---|", "| 0 and 1 | History only. Their look-back windows are short |", "| 2 to 5 | Training. Day 5 also sets the number of trees |",
         "| 6 and 7 | Tuning: settings, feature sets, calibration, the alert line, the monitoring references |", "| 8 and 9 | Test. No choice for version 1 follows them. They were not sealed: every run of step 3 printed them while the project was built |",
         "| 10 onward | Left out. Regular activity has stopped |", "",
         "## 3. Method", "",
         f"Gradient-boosted trees (XGBoost), depth {models['chosen']['max_depth']}, no class weight. Features describe the transfer, the two customers and how active each account has been; "
         "every feature of a transfer uses only what happened up to the end of that transfer's day, and a test checks that.", "",
         f"Version 1 has {models['models'][adopted]['features']} features. 16 network features (counterparties, loops, split-and-merge paths) were built and tested, and were left out: "
         f"on the tuning days they added {net['gain'] * 100:+.1f} points of laundering found, with a 95% interval from {net['gain_low'] * 100:+.1f} to {net['gain_high'] * 100:+.1f}.", "",
         f"Version 2 adds one feature, {MEANING['pair_before_7d']}, after monitoring showed version 1 raising {spike['alerts']:,} alerts on a day set for {mon['reference']['alerts_per_day']:,.0f} (section 6).", "",
         "## 4. How it was tested", "",
         "- Against simpler methods at the same number of alerts a day: a peer-group z-score rule, an isolation forest and a logistic regression.",
         "- Intervals for the share of laundering found resample whole schemes, since transfers of one scheme are found or missed together.",
         "- A richer feature set is adopted only if the interval for its gain on the tuning days stays above zero.",
         "- Calibration is fitted on one tuning day and checked on the other, then checked again on the test days.",
         "- The published scoring rules were also applied (step 6), to compare with published models.",
         "- Unit tests: no feature looks ahead; every feature equals a slow recomputation; contributions add up to the score.", "",
         "## 5. Results", "",
         f"Version 1, test days, {models['main_budget']:,} alerts a day: {t[adopted]['by_budget'][main]['found']} of {models['rows']['test']['laundering']} laundering transfers found "
         f"({t[adopted]['by_budget'][main]['recall']:.1%}; 95% interval {t[adopted]['by_budget'][main]['recall_low']:.1%} to {t[adopted]['by_budget'][main]['recall_high']:.1%}), "
         f"precision {t[adopted]['by_budget'][main]['precision']:.1%}. The logistic regression found {t['linear']['by_budget'][main]['found']}, the peer-group rule {t['rule']['by_budget'][main]['found']}, "
         f"the isolation forest {t['forest']['by_budget'][main]['found']}.", "",
         f"By kind: {k['In a labeled scheme (all 8 types)'][adopted]} of {k['In a labeled scheme (all 8 types)']['laundering']} laundering transfers in labeled schemes were found, "
         f"and {k['Not in a labeled scheme'][adopted]} of {k['Not in a labeled scheme']['laundering']} outside them.", "",
         f"Scored the published way (F1, mean of five seeds), with the same input fields the published models see: {same['hi'][0]:.1f} to {same['hi'][1]:.1f} on HI-Small and "
         f"{same['li'][0]:.1f} to {same['li'][1]:.1f} on LI-Small across three feature sets. The best published graph neural network reports {best['hi']['mean']:.1f} and {best['li']['mean']:.1f}. "
         "results/published_protocol.md says what is the same and what is not.", "",
         "## 6. What monitoring found", "",
         f"On day 8, the first unseen Friday, the fixed line raised {spike['alerts']:,} alerts and precision fell to {spike['precision']:.1%}. The alert-volume check tripped that night. "
         f"{share['pair_paid_exactly_a_week_before']:.0%} of that day's alerts were pairs that had paid each other exactly 7 days earlier: weekly payers, which a 3-day memory sees as first-time pairs.", "",
         f"Version 2 gives the model the pair's history over the 7 days before. On the same day it raises {fixed['alerts']:,} alerts and finds {fixed['found']:,}. "
         f"Those days suggested the change, so the better check is the second portfolio: there, at {mon['budget']:,} alerts a day, version 1 finds {g['found_a']} and version 2 finds {g['found_b']} of {g['laundering']} "
         f"({g['gain'] * 100:+.1f} points; 95% interval {g['gain_low'] * 100:+.1f} to {g['gain_high'] * 100:+.1f}). "
         f"That check was not blind: a wider candidate was tried on both portfolios in the same trial, and on the second it found {cc['wider_on_test_days']['against_v1']['found_b']}.", "",
         "## 7. Limits", "",
         "- Synthetic data. The simulator's laundering follows eight scripted patterns. Real laundering adapts.",
         f"- Laundering outside the labeled schemes is mostly missed: {k['Not in a labeled scheme'][adopted]} of {k['Not in a labeled scheme']['laundering']} found.",
         "- Ten days of data hold one weekly cycle. The training days hold no Friday with full history. A monthly pattern would be invisible.",
         "- Features are built once a day, after the day closes. The model does not score a transfer at the moment it is made.",
         "- Labels are complete and immediate here. With real labels, only worked alerts are labeled, and late.",
         f"- What the line misses can only be estimated from a sample: {truth['laundering_below']:,} of {truth['transfers_below']:,} transfers below the line are laundering. "
         f"A plan of {planned['reviews']:,} reviews by score band estimates that count to within about {planned['rmse'] / truth['laundering_below']:.0%}.",
         f"- A model trained on one portfolio loses ground on another: on the second portfolio it finds {co['found_a']} where a model trained there finds {co['found_b']}.", "",
         "## 8. Monitoring plan", "",
         "| Check | Watches | Compared with | Limits | If it trips |", "|---|---|---|---|---|"]
    for rule in rules["checked_in_this_order"]:
        L.append(f"| {rule['name'].capitalize()} | {rule['watch'].capitalize()} | {rule['compare_with'].capitalize()} | {limits_in_words(rule)} | {rule['action']} |")
    L += ["", "Revalidate when: " + "; ".join(rules["revalidate_when"]) + ".", "",
          f"Retraining: with the labels of days 6 and 7 added, version 2 finds {rt['found_b']} against {rt['found_a']} on days 8 and 9 "
          f"({rt['gain'] * 100:+.1f} points; 95% interval {rt['gain_low'] * 100:+.1f} to {rt['gain_high'] * 100:+.1f}).", "",
          "## 9. Versions", "", "| Version | Features | Change | Why |", "|---|---|---|---|",
          f"| 1 | {models['models'][adopted]['features']} | First model | Chosen on the tuning days (step 3) |",
          f"| 2 | {v2['features']} | Adds the pair's history over the 7 days before | The alert spike on the first unseen Friday (step 5) |", ""]
    (RESULTS / "model_document.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    CHARTS.mkdir(parents=True, exist_ok=True)
    models, line, mon = read_json(RESULTS / "models.json"), read_json(RESULTS / "alert_line.json"), read_json(RESULTS / "monitoring.json")
    published = read_json(RESULTS / "published_protocol.json")
    result = {}
    reasons(result)
    print(f"reasons written ({time.time() - t0:.0f} s)", flush=True)
    chart_methods(models)
    chart_ladder(models)
    chart_friday(mon)
    chart_by_kind(models)
    chart_published(published)
    print(f"charts drawn ({time.time() - t0:.0f} s)", flush=True)
    model_document(models, line, mon, published, result)
    summary = summary_of(models, line, mon, published, result)
    write_json(RESULTS / "summary.json", summary)
    write_json(RESULTS / "reasons.json", result["reasons"])
    note_time("07_report", time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s. See results/model_document.md, results/reasons.md and results/charts/")


if __name__ == "__main__":
    main()
