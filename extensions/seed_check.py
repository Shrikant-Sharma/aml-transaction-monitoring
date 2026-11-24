"""Is version 2's gain steady, or does it depend on the random draw?

    python extensions/seed_check.py . [seeds...]            about 20 minutes, from the project folder

Trains version 1 and version 2 exactly as 05_monitor.py does (same days, same settings, same early stopping), once
per seed, on both portfolios. For each: laundering found on days 8 and 9 at 1,000 alerts a day, and alerts on the
Friday at the version's own line. Changes nothing in the project: it reads data/work and results/fitted only.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from common import RESULTS, TEST_DAYS, TRAIN_DAYS, TUNE_DAYS, read_json  # noqa: E402
from features import ADDED_LATER  # noqa: E402
from line import line_for  # noqa: E402
from metrics import top_per_day  # noqa: E402
from model import fit_boosted, refit, scores_of  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("monitor", ROOT / "05_monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)

seeds = [int(s) for s in sys.argv[2:]] or [20261122, 1, 2, 3, 4]
v1 = read_json(RESULTS / "fitted" / "model_v1_settings.json")
cols1 = v1["features"]
cols2 = cols1 + ADDED_LATER[:1]
params = {"max_depth": v1["max_depth"], "min_child_weight": v1["min_child_weight"]}
budget = 1000
t0 = time.time()
print("portfolio  seed      v1 found  v2 found  gain   v1 Friday  v2 Friday", flush=True)
for portfolio in ("hi", "li"):
    X, y, day, scheme, s, r = monitor.load(portfolio, cols2)
    tune, test = np.isin(day, TUNE_DAYS), np.isin(day, TEST_DAYS)
    fit_rows, stop_rows, train = np.isin(day, TRAIN_DAYS[:-1]), day == TRAIN_DAYS[-1], np.isin(day, TRAIN_DAYS)
    for seed in seeds:
        out = {}
        for name, columns in (("v1", cols1), ("v2", cols2)):
            _, trees = fit_boosted(X, y, fit_rows, stop_rows, columns, params, seed=seed)
            score = scores_of(refit(X, y, train, columns, params, trees, seed=seed), X, columns)
            line = line_for(score[tune], budget, len(TUNE_DAYS))
            found = int((top_per_day(score[test], day[test], budget) & (y[test] == 1)).sum())
            out[name] = (found, int((score[day == 8] >= line).sum()))
        print(f"{portfolio:9}  {seed:<8}  {out['v1'][0]:8}  {out['v2'][0]:8}  {out['v2'][0] - out['v1'][0]:+4}   {out['v1'][1]:9,}  {out['v2'][1]:9,}"
              f"   ({time.time() - t0:.0f} s)", flush=True)
