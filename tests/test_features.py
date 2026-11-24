"""Tests for the features. No data file is needed: they run on a small made-up set of transfers.

    python tests/test_features.py

  1. No look-ahead. The features of every transfer up to day d must be the same whether or not the later days
     exist. If any feature used the future, cutting the future off would change it.
  2. A slow second opinion. Every feature of every transfer is recomputed straight from its definition with
     plain loops and sets, and compared with the fast version.
  3. Shape. 51 columns, in the documented order, no missing values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FORMATS, LONG_WINDOW, WEEK, WINDOW  # noqa: E402
from features import ALL_FEATURES, CURRENCIES, CUSTOMER_TYPES, build_features, fit_peer_medians  # noqa: E402


def toy(seed=7, n_accounts=300, n_days=10, n_transfers=7500):
    """Random transfers among a few hundred accounts, with some loops and split-and-merge chains planted."""
    rng = np.random.default_rng(seed)
    bank = rng.integers(0, 12, n_accounts)
    accounts = pd.DataFrame({
        "owner": [f"E{o}" for o in rng.integers(0, 200, n_accounts)],
        "customer_type": rng.choice(CUSTOMER_TYPES + [None], n_accounts),
        "country": [f"C{b % 4}" for b in bank],
    })
    accounts.loc[rng.random(n_accounts) < 0.05, ["owner", "country"]] = None          # a few accounts missing from the accounts file
    accounts["owner_accounts"] = accounts.groupby("owner")["owner"].transform("size")
    s = rng.integers(0, n_accounts, n_transfers)
    r = rng.integers(0, n_accounts, n_transfers)
    same = rng.random(n_transfers) < 0.08
    r[same] = s[same]                                                                 # transfers to self
    # make activity lumpy, so that pairs repeat and counts differ between accounts
    busy = rng.integers(0, 40, n_transfers // 2)
    s[: len(busy)] = busy
    r[len(busy) // 2: len(busy)] = rng.integers(0, 40, len(busy) - len(busy) // 2)
    day = rng.integers(0, n_days, n_transfers)
    fmt = rng.choice(FORMATS, n_transfers, p=[0.4, 0.05, 0.1, 0.15, 0.15, 0.05, 0.1])
    cur = rng.choice(CURRENCIES[:4], n_transfers, p=[0.1, 0.1, 0.1, 0.7])
    cur_recv = cur.copy()
    swap = rng.random(n_transfers) < 0.03
    cur_recv[swap] = rng.choice(CURRENCIES[4:7], swap.sum())
    rows = [pd.DataFrame({"s": s, "r": r, "day": day, "fmt": fmt, "cur_paid": cur, "cur_recv": cur_recv})]
    planted = []
    for k in range(25):                                                               # loops of 3 to 5 accounts, by ACH
        members = rng.choice(n_accounts, rng.integers(3, 6), replace=False)
        start = rng.integers(0, n_days)
        for j, a in enumerate(members):
            planted.append((a, members[(j + 1) % len(members)], min(n_days - 1, start + rng.integers(0, 3)), "ACH"))
    for k in range(25):                                                               # one source, several middle accounts, one end
        source, end = rng.choice(n_accounts, 2, replace=False)
        start = rng.integers(0, n_days)
        for m in rng.choice(n_accounts, rng.integers(2, 5), replace=False):
            if m in (source, end):
                continue
            planted.append((source, m, min(n_days - 1, start + rng.integers(0, 2)), "ACH"))
            planted.append((m, end, min(n_days - 1, start + rng.integers(0, 3)), "ACH"))
    p = pd.DataFrame(planted, columns=["s", "r", "day", "fmt"])
    p["cur_paid"] = p["cur_recv"] = "US Dollar"
    rows.append(p)
    tx = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)   # file order is not day order
    tx["amt_paid"] = np.round(np.exp(rng.normal(6, 2, len(tx))), 2)
    tx.loc[rng.random(len(tx)) < 0.01, "amt_paid"] = 0.0
    tx["minute"] = rng.integers(0, 1440, len(tx))
    tx["from_bank"] = [f"{bank[a]:03d}" for a in tx["s"]]
    tx["to_bank"] = [f"{bank[a]:03d}" for a in tx["r"]]
    tx["day"] = tx["day"].astype("int16")
    return tx, accounts


def reach(edges_from, start):
    """All accounts that can be reached from `start` by following edges (start itself only if a path returns to it)."""
    seen, stack = set(), list(edges_from.get(start, ()))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(edges_from.get(node, ()))
    return seen


def slow_features(tx, accounts, medians):
    """Every feature of every transfer, straight from its definition. Slow on purpose."""
    s, r, day = tx["s"].to_numpy(), tx["r"].to_numpy(), tx["day"].to_numpy()
    fmt, cp, cr = tx["fmt"].to_numpy(), tx["cur_paid"].to_numpy(), tx["cur_recv"].to_numpy()
    amount = tx["amt_paid"].to_numpy()
    real = s != r
    ach = fmt == "ACH"
    def plain(column):                                               # missing values as None, whatever pandas calls them
        return [None if pd.isna(v) else v for v in accounts[column]]

    owner, ctype, country = plain("owner"), plain("customer_type"), plain("country")
    owned = accounts["owner_accounts"].fillna(0).to_numpy()
    out = {name: np.zeros(len(tx), dtype=np.float64) for name in ALL_FEATURES}

    def graph_of(rows):
        forward, backward = {}, {}
        for i in rows:
            forward.setdefault(s[i], set()).add(r[i])
            backward.setdefault(r[i], set()).add(s[i])
        return forward, backward

    for d in sorted(set(day)):
        today = np.flatnonzero(day == d)
        win = np.flatnonzero((day >= d - WINDOW + 1) & (day <= d) & real)
        before = np.flatnonzero((day >= d - WINDOW + 1) & (day < d) & real)
        long = np.flatnonzero((day >= d - LONG_WINDOW + 1) & (day <= d) & real)
        week = np.flatnonzero((day >= d - WEEK) & (day < d) & real)
        day_rows = np.flatnonzero((day == d) & real)
        pays, paid_by = graph_of(win)                                    # all formats
        a_pays, a_paid_by = graph_of(win[ach[win]])                      # by ACH, 3 days
        l_pays, l_paid_by = graph_of(long[ach[long]])                    # by ACH, 5 days
        for i in today:
            a, b = s[i], r[i]
            out["amount_log"][i] = np.log10(1 + amount[i])
            med = medians.get(f"{cp[i]}|{fmt[i]}")
            out["amount_vs_peers"][i] = np.log10(amount[i] / med) if (med and amount[i] > 0) else 0.0
            out["format"][i] = FORMATS.index(fmt[i])
            out["currency"][i] = CURRENCIES.index(cp[i])
            out["hour"][i] = tx["minute"].iloc[i] // 60
            out["to_self"][i] = a == b
            out["same_bank"][i] = tx["from_bank"].iloc[i] == tx["to_bank"].iloc[i]
            out["cross_currency"][i] = cp[i] != cr[i]
            out["sender_type"][i] = CUSTOMER_TYPES.index(ctype[a]) if ctype[a] in CUSTOMER_TYPES else -1
            out["receiver_type"][i] = CUSTOMER_TYPES.index(ctype[b]) if ctype[b] in CUSTOMER_TYPES else -1
            out["same_owner"][i] = (owner[a] is not None) and (owner[a] == owner[b]) and (a != b)
            out["sender_owner_accounts"][i] = owned[a]
            out["receiver_owner_accounts"][i] = owned[b]
            out["cross_border"][i] = (country[a] is not None) and (country[b] is not None) and (country[a] != country[b])
            # behavior
            out["s_sent_1d"][i] = (s[day_rows] == a).sum()
            out["s_sent_3d"][i] = (s[win] == a).sum()
            out["s_sent_ach_1d"][i] = ((s[day_rows] == a) & ach[day_rows]).sum()
            out["s_sent_ach_3d"][i] = ((s[win] == a) & ach[win]).sum()
            out["s_recv_1d"][i] = (r[day_rows] == a).sum()
            out["s_recv_3d"][i] = (r[win] == a).sum()
            out["s_recv_ach_3d"][i] = ((r[win] == a) & ach[win]).sum()
            out["r_recv_1d"][i] = (r[day_rows] == b).sum()
            out["r_recv_3d"][i] = (r[win] == b).sum()
            out["r_recv_ach_1d"][i] = ((r[day_rows] == b) & ach[day_rows]).sum()
            out["r_recv_ach_3d"][i] = ((r[win] == b) & ach[win]).sum()
            out["r_sent_1d"][i] = (s[day_rows] == b).sum()
            out["r_sent_3d"][i] = (s[win] == b).sum()
            out["r_sent_ach_3d"][i] = ((s[win] == b) & ach[win]).sum()
            out["s_currencies_3d"][i] = len(set(cp[win][s[win] == a]) | set(cr[win][r[win] == a]))
            out["r_currencies_3d"][i] = len(set(cp[win][s[win] == b]) | set(cr[win][r[win] == b]))
            out["pair_count_3d"][i] = ((s[win] == a) & (r[win] == b)).sum()
            out["pair_before_3d"][i] = ((s[before] == a) & (r[before] == b)).sum()
            out["pair_before_7d"][i] = ((s[week] == a) & (r[week] == b)).sum()
            out["s_sent_before_7d"][i] = (s[week] == a).sum()
            out["r_recv_before_7d"][i] = (r[week] == b).sum()
            # network
            out["s_receivers_3d"][i] = len(pays.get(a, ()))
            out["s_senders_3d"][i] = len(paid_by.get(a, ()))
            out["r_senders_3d"][i] = len(paid_by.get(b, ()))
            out["r_receivers_3d"][i] = len(pays.get(b, ()))
            out["s_receivers_ach_3d"][i] = len(a_pays.get(a, ()))
            out["s_senders_ach_3d"][i] = len(a_paid_by.get(a, ()))
            out["r_senders_ach_3d"][i] = len(a_paid_by.get(b, ()))
            out["r_receivers_ach_3d"][i] = len(a_pays.get(b, ()))
            if a != b:
                out["back_3d"][i] = a in pays.get(b, ())
                out["triangle_3d"][i] = len(a_pays.get(b, set()) & a_paid_by.get(a, set()))
                for name, forward, backward in (("3d", a_pays, a_paid_by), ("5d", l_pays, l_paid_by)):
                    if b in reach(forward, a) and a in reach(forward, b):
                        group = (reach(forward, a) & reach(backward, a)) | {a}
                        out[f"loop_{name}"][i] = 1
                        out[f"loop_size_{name}"][i] = len(group)
                if b in a_pays.get(a, ()):                               # the pair is an ACH edge of the window
                    ends = a_pays.get(b, set()) - {a}
                    out["split_paths_3d"][i] = max([len(a_pays[a] & a_paid_by[x]) for x in ends], default=1)
                    sources = a_paid_by.get(a, set()) - {b}
                    out["merge_paths_3d"][i] = max([len(a_pays[src] & a_paid_by[b]) for src in sources], default=1)
    return pd.DataFrame(out, columns=ALL_FEATURES)


def main():
    tx, accounts = toy()
    medians = fit_peer_medians(tx, range(0, 4))
    fast = build_features(tx, accounts, medians)
    failures = 0

    # 3. shape
    ok = list(fast.columns) == ALL_FEATURES and len(ALL_FEATURES) == 51 and not fast.isna().any().any() and (fast.dtypes == "float32").all()
    print(f"{'PASS' if ok else 'FAIL'}  51 features in the documented order, no missing values")
    failures += not ok

    # 1. no look-ahead
    n_days = int(tx["day"].max()) + 1
    for d in range(n_days - 1):
        part = tx[tx["day"] <= d]
        cut = build_features(part.reset_index(drop=True), accounts, medians)
        same = np.array_equal(cut.to_numpy(), fast.loc[part.index].to_numpy())
        print(f"{'PASS' if same else 'FAIL'}  days 0 to {d}: the same features with and without the later days ({len(part):,} transfers)")
        failures += not same

    # 2. the slow second opinion
    slow = slow_features(tx, accounts, medians)
    for name in ALL_FEATURES:
        close = np.allclose(fast[name].to_numpy(dtype=np.float64), slow[name].to_numpy(), rtol=1e-5, atol=1e-5)
        if not close:
            worst = int(np.argmax(np.abs(fast[name].to_numpy(dtype=np.float64) - slow[name].to_numpy())))
            print(f"FAIL  {name}: row {worst} fast {fast[name].iloc[worst]} slow {slow[name].iloc[worst]}")
            failures += 1
    seen = {name: int((slow[name] != 0).sum()) for name in ("loop_3d", "loop_5d", "triangle_3d", "back_3d", "same_owner", "cross_currency")}
    split_seen, merge_seen = int((slow["split_paths_3d"] >= 2).sum()), int((slow["merge_paths_3d"] >= 2).sum())
    covered = all(v > 0 for v in seen.values()) and split_seen > 0 and merge_seen > 0
    failures += not covered
    print(f"{'PASS' if failures == 0 else 'FAIL'}  all 51 features equal a slow recomputation on {len(tx):,} transfers")
    print(f"{'PASS' if covered else 'FAIL'}  the made-up data exercises the rare features: " + ", ".join(f"{k} {v}" for k, v in seen.items())
          + f", split paths of 2 or more {split_seen}, merge paths of 2 or more {merge_seen}")
    print("\nAll feature tests passed." if failures == 0 else f"\n{failures} check(s) failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
