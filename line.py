"""The alert line and the calibration of scores. Small functions shared by steps 4 and 5."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from model import apply_calibrator

KINDS = ("none", "platt", "isotonic")
LABEL = {"none": "Raw scores", "platt": "Platt scaling", "isotonic": "Isotonic regression"}


def logit(score):
    clipped = np.clip(score.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def fit_calibrator(kind, score, y):
    """A map from raw score to probability. 'none' leaves the score as it is, 'platt' fits a logistic curve on the
    log-odds of the score, 'isotonic' fits the best rising step function."""
    if kind == "none":
        return {"kind": "none"}
    if kind == "platt":
        model = LogisticRegression(C=1e6, max_iter=1000).fit(logit(score).reshape(-1, 1), y)
        return {"kind": "platt", "a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(score.astype(np.float64), y)
    return {"kind": "isotonic", "x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()}


def probability_quality(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "mean_probability": float(p.mean()), "true_rate": float(y.mean())}


def choose_calibrator(score, y, first, second):
    """Fit each kind on one set of rows and check it on the other, both ways. Returns the kind with the lowest
    log loss and the table behind the choice."""
    cross = {kind: [] for kind in KINDS}
    for fit_rows, check_rows in ((first, second), (second, first)):
        for kind in KINDS:
            calibrator = fit_calibrator(kind, score[fit_rows], y[fit_rows])
            cross[kind].append(probability_quality(apply_calibrator(calibrator, score[check_rows]), y[check_rows]))
    table = {kind: {"log_loss": float(np.mean([c["log_loss"] for c in cross[kind]])), "brier": float(np.mean([c["brier"] for c in cross[kind]]))} for kind in KINDS}
    return min(KINDS, key=lambda kind: table[kind]["log_loss"]), table


def line_for(scores_on_the_tuning_days, alerts_per_day, n_days):
    """The score line that gives this many alerts a day, on average, over the tuning days."""
    return float(np.sort(scores_on_the_tuning_days)[::-1][alerts_per_day * n_days - 1])
