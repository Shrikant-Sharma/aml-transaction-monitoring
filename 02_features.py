"""Step 2. Turn the raw transfers of each portfolio into one table of features.

    python 02_features.py          both portfolios (about 2 minutes on 2 cores)
    python 02_features.py hi       only the portfolio the model is built on

Writes, under data/work/:
    <portfolio>_features.parquet   one row per transfer, 51 features (see features.py)
    <portfolio>_index.parquet      label, day, sender, receiver and, for labeled schemes, the scheme and its type
and results/fitted/peer_medians.json, the one quantity that is fitted here: the median amount per
currency and payment format on the training days of the first portfolio. The second portfolio reuses it.
"""
import gc
import sys
import time

import pandas as pd

from common import FIRST_DATE, PORTFOLIOS, RAW, RESULTS, TRAIN_DAYS, WORK, note_time, read_accounts, read_json, read_patterns, read_transactions, sha256_of, write_json
from features import ALL_FEATURES, build_features, fit_peer_medians


def attach_schemes(tx, portfolio):
    """Mark the transfers that belong to a scheme the publisher labeled by type."""
    pat = read_patterns(portfolio)
    key_cols = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_paid", "cur_paid", "fmt"]
    pat["amt_paid"] = pat["amt_paid"].astype(float)
    pat = pat.drop_duplicates(key_cols)
    first_seen = pat.groupby("scheme")["ts"].min().rename("scheme_start")
    pat = pat.join(first_seen, on="scheme")
    merged = tx[key_cols].merge(pat[key_cols + ["scheme", "scheme_type", "scheme_start"]], on=key_cols, how="left")
    return merged["scheme"].fillna(-1).astype("int32").to_numpy(), merged["scheme_type"].fillna("").to_numpy(), merged["scheme_start"].fillna("").to_numpy()


def main():
    which = sys.argv[1:] or list(PORTFOLIOS)
    WORK.mkdir(parents=True, exist_ok=True)
    for portfolio in which:
        t0 = time.time()
        tx, keys = read_transactions(portfolio)
        accounts = read_accounts(portfolio, keys)
        print(f"{portfolio}: {len(tx):,} transfers, {len(keys):,} accounts, {int(tx['y'].sum()):,} laundering (1 in {len(tx) / tx['y'].sum():,.0f})")
        medians_path = RESULTS / "fitted" / "peer_medians.json"
        if portfolio == "hi":
            medians = fit_peer_medians(tx, TRAIN_DAYS)
            write_json(medians_path, medians)
        else:
            medians = read_json(medians_path)
        features = build_features(tx, accounts, medians, verbose=True)
        features.to_parquet(WORK / f"{portfolio}_features.parquet", index=False)
        scheme, scheme_type, scheme_start = attach_schemes(tx, portfolio)
        index = pd.DataFrame({"y": tx["y"], "day": tx["day"], "minute": tx["minute"], "s": tx["s"], "r": tx["r"], "fmt": tx["fmt"], "cur_paid": tx["cur_paid"],
                              "amt_paid": tx["amt_paid"], "scheme": scheme, "scheme_type": scheme_type, "scheme_start": scheme_start})
        index.to_parquet(WORK / f"{portfolio}_index.parquet", index=False)
        in_schemes = int((index["scheme"] >= 0).sum())
        write_json(WORK / f"{portfolio}_manifest.json", {
            "portfolio": portfolio, "source_file": PORTFOLIOS[portfolio]["transactions"],
            "source_sha256": sha256_of(RAW / PORTFOLIOS[portfolio]["transactions"]),
            "transfers": len(tx), "accounts": len(keys), "laundering": int(tx["y"].sum()),
            "laundering_in_labeled_schemes": in_schemes, "labeled_schemes": int(index.loc[index["scheme"] >= 0, "scheme"].nunique()),
            "features": ALL_FEATURES, "peer_medians_from": "hi, training days",
            "by_day": [{"day": int(d), "weekday": (pd.Timestamp(FIRST_DATE) + pd.Timedelta(days=int(d))).day_name(),
                        "transfers": int((index["day"] == d).sum()), "laundering": int(index.loc[index["day"] == d, "y"].sum())} for d in sorted(index["day"].unique())],
        })
        print(f"{portfolio}: {len(ALL_FEATURES)} features written in {time.time() - t0:.0f} s; {in_schemes:,} laundering transfers sit in labeled schemes\n")
        note_time(f"02_features_{portfolio}", time.time() - t0, transfers=len(tx))
        del tx, keys, accounts, features, index                          # free the memory before the next portfolio
        gc.collect()


if __name__ == "__main__":
    main()
