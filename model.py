"""The boosted-tree model: its fixed settings and the few functions every script uses to fit, score and explain it."""
import numpy as np
import xgboost as xgb

from common import SEED

# Settings that are not searched. Depth and leaf size are chosen by the small grid in 03_models.py.
FIXED = dict(learning_rate=0.08, subsample=0.8, colsample_bytree=0.7, tree_method="hist", max_bin=128, n_jobs=2, eval_metric="aucpr")
MOST_TREES = 600
STOP_AFTER = 40                                 # stop when the held-back day has not improved for this many trees


def fit_boosted(X, y, fit_rows, stop_rows, columns, params, weight=1.0, seed=SEED):
    """Fit on fit_rows and let stop_rows decide the number of trees. Returns the model and that number."""
    model = xgb.XGBClassifier(n_estimators=MOST_TREES, early_stopping_rounds=STOP_AFTER, scale_pos_weight=weight, random_state=seed, **FIXED, **params)
    model.fit(X.loc[fit_rows, columns], y[fit_rows], eval_set=[(X.loc[stop_rows, columns], y[stop_rows])], verbose=False)
    return model, int(model.best_iteration) + 1


def refit(X, y, rows, columns, params, trees, weight=1.0, seed=SEED):
    """Fit a fixed number of trees on the given rows."""
    model = xgb.XGBClassifier(n_estimators=trees, scale_pos_weight=weight, random_state=seed, **FIXED, **params)
    model.fit(X.loc[rows, columns], y[rows])
    return model


def scores_of(model, X, columns):
    """The model's score for each row: a number between 0 and 1, higher means more like laundering."""
    return model.predict_proba(X[columns])[:, 1].astype(np.float32)


def load_model(path):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def contributions(model, X, columns):
    """How much each feature moved each row's score, on the log-odds scale (SHAP values from the trees themselves).
    One column per feature, then one last column with the base value. Each row sums to the row's log-odds."""
    return model.get_booster().predict(xgb.DMatrix(X[columns]), pred_contribs=True)


def log_odds(model, X, columns):
    return model.get_booster().predict(xgb.DMatrix(X[columns]), output_margin=True)


def apply_calibrator(calibrator, score):
    """Turn raw model scores into calibrated probabilities with a stored calibrator (see 04_alert_line.py)."""
    score = np.asarray(score, dtype=np.float64)
    if calibrator["kind"] == "none":
        return score
    if calibrator["kind"] == "platt":
        clipped = np.clip(score, 1e-7, 1 - 1e-7)
        return 1 / (1 + np.exp(-(calibrator["a"] * np.log(clipped / (1 - clipped)) + calibrator["b"])))
    return np.interp(score, calibrator["x"], calibrator["y"])                 # isotonic: straight lines between the fitted steps
