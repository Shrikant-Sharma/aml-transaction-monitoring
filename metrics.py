"""The measures used in every report. Each is a few lines so it can be read and checked by hand."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from common import wilson


def top_per_day(score, day, k):
    """True for the k highest-scored transfers of each day: the alert queue when reviewers can take k a day."""
    picked = np.zeros(len(score), dtype=bool)
    for d in np.unique(day):
        rows = np.flatnonzero(day == d)
        if len(rows) <= k:
            picked[rows] = True
        else:
            picked[rows[np.argpartition(-score[rows], k - 1)[:k]]] = True
    return picked


def alert_counts(alerted, y):
    """Alerts, laundering found, precision and recall for one set of alerts."""
    alerts = int(alerted.sum())
    found = int((alerted & (y == 1)).sum())
    total = int((y == 1).sum())
    lo, hi = wilson(found, alerts)
    return {"alerts": alerts, "found": found, "laundering": total,
            "precision": found / alerts if alerts else 0.0, "precision_low": lo, "precision_high": hi,
            "recall": found / total if total else 0.0}


def at_budget(score, y, day, k):
    return alert_counts(top_per_day(score, day, k), y)


def ranking_quality(score, y):
    """Average precision (the area under the precision-recall curve) and the area under the ROC curve."""
    return {"average_precision": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score))}


def best_f1_line(score, y):
    """The score line that gives the highest F1 for the laundering class, and that F1."""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order] == 1)
    alerts = np.arange(1, len(order) + 1)
    f1 = 2 * hits / (alerts + hits[-1])
    last_of_tie = np.r_[score[order][1:] != score[order][:-1], True]      # a line can only sit between different scores
    best = np.flatnonzero(last_of_tie)[np.argmax(f1[last_of_tie])]
    return float(score[order][best]), float(f1[best])


def f1_at_line(score, y, line):
    alerted = score >= line
    found = int((alerted & (y == 1)).sum())
    return 2 * found / max(int(alerted.sum()) + int((y == 1).sum()), 1)


def recall_interval_by_scheme(alerted, y, scheme, draws=2000, seed=0):
    """A 95% interval for recall that respects schemes: transfers of one scheme are found or missed together more
    often than chance, so whole schemes are resampled, not single transfers. Laundering outside any labeled scheme
    counts as a scheme of one."""
    rows = np.flatnonzero(y == 1)
    unit = scheme_units(y, scheme)                                         # every stray transfer is its own unit
    n_units = unit.max() + 1
    found_per = np.bincount(unit, weights=alerted[rows].astype(float), minlength=n_units)
    size_per = np.bincount(unit, minlength=n_units).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_units, size=(draws, n_units))
    recalls = found_per[pick].sum(axis=1) / size_per[pick].sum(axis=1)
    return float(np.quantile(recalls, 0.025)), float(np.quantile(recalls, 0.975))


def scheme_units(y, scheme):
    """For the laundering transfers, in order: which scheme each belongs to. Laundering outside any labeled scheme
    is a scheme of one."""
    unit = scheme[y == 1].astype(np.int64).copy()
    single = unit < 0
    unit[single] = -1 - np.arange(single.sum())
    return np.unique(unit, return_inverse=True)[1]


def paired_gain(alerted_a, alerted_b, y, scheme, draws=4000, seed=0):
    """Two alert sets of the same size on the same transfers: how much more laundering does B find than A?

    Returns the counts, the gain as a share of all laundering, a 95% interval for the gain that resamples whole
    schemes, and the exact McNemar p-value on the laundering only one of the two found. The p-value treats
    transfers as independent, which they are not, so the interval is the one to trust."""
    found_a, found_b = alerted_a[y == 1], alerted_b[y == 1]
    unit = scheme_units(y, scheme)
    n_units = unit.max() + 1
    per_a = np.bincount(unit, weights=found_a.astype(float), minlength=n_units)
    per_b = np.bincount(unit, weights=found_b.astype(float), minlength=n_units)
    size = np.bincount(unit, minlength=n_units).astype(float)
    pick = np.random.default_rng(seed).integers(0, n_units, size=(draws, n_units))
    gains = (per_b[pick].sum(axis=1) - per_a[pick].sum(axis=1)) / size[pick].sum(axis=1)
    only_a, only_b = int((found_a & ~found_b).sum()), int((~found_a & found_b).sum())
    return {"found_a": int(found_a.sum()), "found_b": int(found_b.sum()), "both": int((found_a & found_b).sum()),
            "only_a": only_a, "only_b": only_b, "laundering": int(len(found_a)), "schemes": int(n_units),
            "gain": float((found_b.sum() - found_a.sum()) / len(found_a)),
            "gain_low": float(np.quantile(gains, 0.025)), "gain_high": float(np.quantile(gains, 0.975)),
            "mcnemar_p": float(binomtest(only_b, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0}


def psi_reference(expected, bins=10):
    """The reference side of a population stability index: bin edges from the reference sample and its share per bin."""
    expected = np.asarray(expected, dtype=np.float64)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                                     # one value dominates: compare the share of that value
        common = float(np.median(expected))
        return {"common": common, "shares": [float((expected == common).mean()), float((expected != common).mean())]}
    edges[0], edges[-1] = -np.inf, np.inf
    return {"edges": edges.tolist(), "shares": (np.histogram(expected, edges)[0] / len(expected)).tolist()}


def psi_against(reference, actual):
    """Population stability index of a new sample against a stored reference.
    Common reading: under 0.1 stable, 0.1 to 0.25 worth a look, over 0.25 shifted."""
    actual = np.asarray(actual, dtype=np.float64)
    if "common" in reference:
        a = np.array([(actual == reference["common"]).mean(), (actual != reference["common"]).mean()])
    else:
        a = np.histogram(actual, np.array(reference["edges"]))[0] / len(actual)
    e, a = np.clip(np.array(reference["shares"]), 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, bins=10):
    """Population stability index between two samples of one quantity."""
    return psi_against(psi_reference(expected, bins), actual)
