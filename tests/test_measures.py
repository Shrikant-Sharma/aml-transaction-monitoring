"""Tests for the measures, the sampling code and the model helpers. No data file is needed.

    python tests/test_measures.py

Each check compares a function with a slower or independent way of getting the same answer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import wilson  # noqa: E402
from metrics import (alert_counts, best_f1_line, f1_at_line, paired_gain, psi, recall_interval_by_scheme, top_per_day)  # noqa: E402
from model import apply_calibrator, contributions, log_odds, refit, scores_of  # noqa: E402
from sampling import allocate, simulate, two_proportion_n  # noqa: E402

FAILED = []


def check(ok, text):
    print(f"{'PASS' if ok else 'FAIL'}  {text}")
    if not ok:
        FAILED.append(text)


def main():
    rng = np.random.default_rng(11)
    n = 20000
    y = (rng.random(n) < 0.02).astype(np.int8)
    score = (rng.random(n) * 0.5 + y * rng.random(n) * 0.6).astype(np.float32)
    day = rng.integers(0, 4, n)
    scheme = np.where(rng.random(n) < 0.6, rng.integers(0, 60, n), -1)

    # the alert queue
    picked = top_per_day(score, day, 150)
    per_day = [int(picked[day == d].sum()) for d in range(4)]
    clean = all(score[(day == d) & picked].min() >= score[(day == d) & ~picked].max() for d in range(4))
    check(per_day == [150] * 4 and clean, "top_per_day takes exactly 150 a day, and nothing left out scores higher than anything taken")
    counts = alert_counts(picked, y)
    check(counts["found"] == int((picked & (y == 1)).sum()) and abs(counts["recall"] - counts["found"] / y.sum()) < 1e-12, "alert_counts agrees with counting by hand")

    # F1
    line, best = best_f1_line(score, y)
    brute = max(f1_score(y, score >= t) for t in np.unique(score)[::40])
    exact = f1_score(y, score >= line)
    check(abs(best - exact) < 1e-12 and best >= brute - 1e-12, f"best_f1_line: its F1 ({best:.4f}) equals scikit-learn's at that line and no other line tried beats it")
    check(abs(f1_at_line(score, y, 0.5) - f1_score(y, score >= 0.5)) < 1e-12, "f1_at_line equals scikit-learn's F1")

    # intervals
    low, high = wilson(5, 10)
    check(abs(low - 0.2366) < 5e-4 and abs(high - 0.7634) < 5e-4 and wilson(0, 10)[0] == 0 and abs(wilson(0, 10)[1] - 0.2775) < 5e-4,
          "wilson matches the textbook values for 5 of 10 and 0 of 10")
    low, high = recall_interval_by_scheme(picked, y, scheme, seed=1)
    check(low < counts["recall"] < high, f"the scheme-resampled interval ({low:.3f} to {high:.3f}) holds the share found ({counts['recall']:.3f})")
    alone = recall_interval_by_scheme(picked, y, np.full(n, -1), seed=1)
    simple = wilson(counts["found"], int(y.sum()))
    check(abs(alone[0] - simple[0]) < 0.02 and abs(alone[1] - simple[1]) < 0.02, "with no schemes at all it comes out close to the plain Wilson interval")

    # paired comparison
    other = top_per_day(score + rng.normal(0, 0.05, n).astype(np.float32), day, 150)
    g = paired_gain(picked, other, y, scheme, seed=1)
    check(g["found_b"] - g["found_a"] == g["only_b"] - g["only_a"] and abs(g["gain"] - (g["found_b"] - g["found_a"]) / y.sum()) < 1e-12 and g["gain_low"] <= g["gain"] <= g["gain_high"],
          "paired_gain: the gain equals the difference in counts and sits inside its interval")
    same = paired_gain(picked, picked, y, scheme, seed=1)
    check(same["gain"] == 0 and same["gain_low"] == 0 and same["gain_high"] == 0 and same["mcnemar_p"] == 1.0, "paired_gain of a queue against itself is exactly zero")

    # stability index
    a = rng.normal(0, 1, 50000)
    check(psi(a, rng.normal(0, 1, 50000)) < 0.01 < 0.1 < psi(a, rng.normal(0.5, 1, 50000)) < psi(a, rng.normal(1, 1, 50000)), "psi is near zero for the same population and grows with the shift")
    flag = (rng.random(50000) < 0.01).astype(float)
    check(psi(flag, flag) == 0 and psi(flag, (rng.random(50000) < 0.2).astype(float)) > 0.25, "psi works on a yes/no feature")

    # sampling
    check(two_proportion_n(0.5, 0.6) == 388, "two_proportion_n gives the textbook 388 per arm for 50% against 60%")
    plan = allocate(np.array([8000, 90000, 860000]), np.array([0.014, 0.0013, 0.00012]), 3000)
    check(plan.sum() == 3000 and plan.min() >= 100, f"allocate spends exactly the budget and leaves no band empty ({plan.tolist()})")
    sim = simulate(np.array([100000]), np.array([2000]), np.array([2000]), 4000, np.random.default_rng(3))
    check(0.93 <= sim["coverage"] <= 0.97 and abs(sim["mean_estimate"] - 2000) < 30, f"simulate: the 95% interval covers the truth {sim['coverage']:.1%} of the time on an easy case, and the estimate is unbiased")

    # calibration helpers
    s = np.array([0.0, 0.1, 0.4, 0.9])
    iso = {"kind": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.2, 1.0]}
    check(np.allclose(apply_calibrator(iso, s), [0.0, 0.04, 0.16, 0.84]) and np.allclose(apply_calibrator({"kind": "none"}, s), s), "apply_calibrator: isotonic is straight lines between its steps, and 'none' leaves scores alone")
    check(np.allclose(apply_calibrator({"kind": "platt", "a": 1.0, "b": 0.0}, s[1:]), s[1:], atol=1e-6), "apply_calibrator: Platt with slope 1 and shift 0 changes nothing")

    # the model's own explanations add up
    X = pd.DataFrame(rng.normal(size=(4000, 6)).astype(np.float32), columns=list("abcdef"))
    target = ((X["a"] + X["b"] * X["c"] + rng.normal(0, 0.5, 4000)) > 1.5).to_numpy().astype(np.int8)
    model = refit(X, target, np.ones(4000, dtype=bool), list("abcdef"), {"max_depth": 4, "min_child_weight": 1}, 60)
    parts = contributions(model, X, list("abcdef"))
    margin = log_odds(model, X, list("abcdef"))
    back = 1 / (1 + np.exp(-margin))
    check(parts.shape == (4000, 7) and np.abs(parts.sum(axis=1) - margin).max() < 1e-3, "contributions: the features and the base value add up to the log-odds of every row")
    check(np.abs(back - scores_of(model, X, list("abcdef"))).max() < 1e-5, "log_odds turned back into a probability equals the model's score")

    print("\nAll measure tests passed." if not FAILED else f"\n{len(FAILED)} check(s) failed.")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
