"""Sampling plans and sample sizes: how many reviews, where to spend them, and how far to trust the answer."""
import numpy as np
from scipy.stats import norm


def allocate(N, rate, n, floor=100):
    """Split n reviews over the bands. Each band gets a floor; the rest goes where the count is least certain
    (band size times the spread of a yes/no outcome at the planned rate)."""
    weight = N * np.sqrt(rate * (1 - rate))
    share = floor + (n - floor * len(N)) * weight / weight.sum()
    out = np.floor(share).astype(int)
    out[np.argmax(share - out)] += n - out.sum()
    return out


def simulate(N, M, n, repeats, rng, draws=1000):
    """Draw the sample `repeats` times from bands of size N holding M laundering transfers, n reviews per band.
    Each time: the estimate of the total, and a 95% interval from a Beta(k + 1/2, n - k + 1/2) per band."""
    n = np.minimum(n, N)
    k = np.column_stack([rng.hypergeometric(M[h], N[h] - M[h], n[h], size=repeats) for h in range(len(N))])
    estimate = (k / n * N).sum(axis=1)
    low, high = np.empty(repeats), np.empty(repeats)
    for start in range(0, repeats, 250):
        part = k[start:start + 250]
        total = np.zeros((len(part), draws))
        for h in range(len(N)):
            total += N[h] * rng.beta(part[:, [h]] + 0.5, n[h] - part[:, [h]] + 0.5, size=(len(part), draws))
        low[start:start + 250], high[start:start + 250] = np.quantile(total, [0.025, 0.975], axis=1)
    truth = M.sum()
    return {"reviews": int(n.sum()), "per_band": n.tolist(), "mean_estimate": float(estimate.mean()), "rmse": float(np.sqrt(((estimate - truth) ** 2).mean())),
            "coverage": float(((low <= truth) & (truth <= high)).mean()), "median_low": float(np.median(low)), "median_high": float(np.median(high)),
            "share_with_no_laundering_in_sample": float((k.sum(axis=1) == 0).mean()), "mean_laundering_in_sample": float(k.sum(axis=1).mean())}


def two_proportion_n(p1, p2, alpha=0.05, power=0.8):
    """Alerts per arm for a two-sided test of two independent proportions."""
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pooled = (p1 + p2) / 2
    return int(np.ceil((z_a * np.sqrt(2 * pooled * (1 - pooled)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2))
