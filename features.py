"""Features for one portfolio of transfers.

The monitoring runs once a day, after the day has closed. So a transfer made on day d
may be described by anything that happened up to the end of day d, and by nothing after it.
Every function here keeps to that rule; tests/test_features.py checks it.

Four groups, from plain to rich. Version 1 of the model chose from these 48:

  transfer   what is on the transfer itself (amount, payment format, currency, time)
  customer   who owns the two accounts (customer type, same owner, same country)
  behavior   how busy each account has been, today and over the last 3 days
  network    who pays whom: how many counterparties, money coming back, loops, split-and-merge paths

Three more were built after the monitoring step showed what a 3-day memory misses. Version 2 uses the first.
The other two were tried with it and left out (see 05_monitor.py):

  pair_before_7d      how often this sender paid this receiver on the 7 days before today
  s_sent_before_7d    how many transfers the sender made on those 7 days
  r_recv_before_7d    how many transfers the receiver got on those 7 days
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from common import FORMATS, LONG_WINDOW, WEEK, WINDOW

CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble", "Rupee",
              "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"]
CUSTOMER_TYPES = ["Corporation", "Partnership", "Sole Proprietorship", "Individual", "Country", "Direct"]

GROUPS = {
    "transfer": ["amount_log", "amount_vs_peers", "format", "currency", "hour", "to_self", "same_bank", "cross_currency"],
    "customer": ["sender_type", "receiver_type", "same_owner", "sender_owner_accounts", "receiver_owner_accounts", "cross_border"],
    "behavior": ["s_sent_1d", "s_sent_3d", "s_sent_ach_1d", "s_sent_ach_3d", "s_recv_1d", "s_recv_3d", "s_recv_ach_3d", "s_currencies_3d",
                 "r_recv_1d", "r_recv_3d", "r_recv_ach_1d", "r_recv_ach_3d", "r_sent_1d", "r_sent_3d", "r_sent_ach_3d", "r_currencies_3d",
                 "pair_count_3d", "pair_before_3d"],
    "network": ["s_receivers_3d", "s_senders_3d", "r_senders_3d", "r_receivers_3d",
                "s_receivers_ach_3d", "s_senders_ach_3d", "r_senders_ach_3d", "r_receivers_ach_3d",
                "back_3d", "triangle_3d", "loop_3d", "loop_size_3d", "loop_5d", "loop_size_5d", "split_paths_3d", "merge_paths_3d"],
}
CANDIDATES = [name for group in GROUPS.values() for name in group]       # the 48 features version 1 chose from
ADDED_LATER = ["pair_before_7d", "s_sent_before_7d", "r_recv_before_7d"]  # built after monitoring; version 2 uses the first
ALL_FEATURES = CANDIDATES + ADDED_LATER
CATEGORICAL = ["format", "currency", "sender_type", "receiver_type"]

# What each feature means, in the words used in the reports and the alert reasons.
MEANING = {
    "amount_log": "size of the transfer",
    "amount_vs_peers": "size of the transfer against others in the same currency and payment format",
    "format": "payment format",
    "currency": "currency",
    "hour": "time of day",
    "to_self": "sender and receiver are the same account",
    "same_bank": "both accounts are at the same bank",
    "cross_currency": "paid in one currency and received in another",
    "sender_type": "sender's customer type",
    "receiver_type": "receiver's customer type",
    "same_owner": "both accounts have the same owner",
    "sender_owner_accounts": "number of accounts the sender's owner holds",
    "receiver_owner_accounts": "number of accounts the receiver's owner holds",
    "cross_border": "the two banks are in different countries",
    "s_sent_1d": "transfers the sender made today",
    "s_sent_3d": "transfers the sender made in 3 days",
    "s_sent_ach_1d": "ACH transfers the sender made today",
    "s_sent_ach_3d": "ACH transfers the sender made in 3 days",
    "s_recv_1d": "transfers the sender received today",
    "s_recv_3d": "transfers the sender received in 3 days",
    "s_recv_ach_3d": "ACH transfers the sender received in 3 days",
    "s_currencies_3d": "currencies the sender used in 3 days",
    "r_recv_1d": "transfers the receiver got today",
    "r_recv_3d": "transfers the receiver got in 3 days",
    "r_recv_ach_1d": "ACH transfers the receiver got today",
    "r_recv_ach_3d": "ACH transfers the receiver got in 3 days",
    "r_sent_1d": "transfers the receiver made today",
    "r_sent_3d": "transfers the receiver made in 3 days",
    "r_sent_ach_3d": "ACH transfers the receiver made in 3 days",
    "r_currencies_3d": "currencies the receiver used in 3 days",
    "pair_count_3d": "transfers from this sender to this receiver in 3 days",
    "pair_before_3d": "transfers from this sender to this receiver on the 2 days before",
    "s_receivers_3d": "different accounts the sender paid in 3 days",
    "s_senders_3d": "different accounts that paid the sender in 3 days",
    "r_senders_3d": "different accounts that paid the receiver in 3 days",
    "r_receivers_3d": "different accounts the receiver paid in 3 days",
    "s_receivers_ach_3d": "different accounts the sender paid by ACH in 3 days",
    "s_senders_ach_3d": "different accounts that paid the sender by ACH in 3 days",
    "r_senders_ach_3d": "different accounts that paid the receiver by ACH in 3 days",
    "r_receivers_ach_3d": "different accounts the receiver paid by ACH in 3 days",
    "back_3d": "the receiver also paid the sender within 3 days",
    "triangle_3d": "third accounts that close a three-step ACH loop back to the sender",
    "loop_3d": "sender and receiver sit on a closed ACH loop within 3 days",
    "loop_size_3d": "accounts on that 3-day loop",
    "loop_5d": "sender and receiver sit on a closed ACH loop within 5 days",
    "loop_size_5d": "accounts on that 5-day loop",
    "split_paths_3d": "middle accounts the sender uses to reach one same end account (ACH, 3 days)",
    "merge_paths_3d": "middle accounts through which one same source reaches this receiver (ACH, 3 days)",
    "pair_before_7d": "transfers from this sender to this receiver on the 7 days before",
    "s_sent_before_7d": "transfers the sender made on the 7 days before",
    "r_recv_before_7d": "transfers the receiver got on the 7 days before",
}


def fit_peer_medians(tx, days):
    """Median amount for each currency and payment format, taken on the given days. A fitted quantity, like a model weight."""
    part = tx[tx["day"].isin(days)]
    med = part.groupby(["cur_paid", "fmt"])["amt_paid"].median()
    return {f"{c}|{f}": float(v) for (c, f), v in med.items() if v > 0}


def _window_table(index, day, mask, n_accounts, n_days, window):
    """Counts per account and day, then summed over the trailing `window` days. Returns (today, window)."""
    flat = np.bincount(index[mask].astype(np.int64) * n_days + day[mask], minlength=n_accounts * n_days)
    today = flat.reshape(n_accounts, n_days).astype(np.int32)
    total = np.cumsum(today, axis=1)
    win = total.copy()
    win[:, window:] -= total[:, :-window]
    return today, win


def _lookup(sorted_keys, values, keys):
    """values[i] for the entry of sorted_keys equal to keys[i], or 0 where there is none."""
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=np.float32)
    pos = np.searchsorted(sorted_keys, keys)
    pos[pos == len(sorted_keys)] = 0
    hit = sorted_keys[pos] == keys
    return np.where(hit, values[pos], 0).astype(np.float32)


def _loops(edge_keys, n_accounts):
    """Strongly connected groups of the graph: accounts that can each reach the others and be reached back."""
    graph = sparse.csr_matrix((np.ones(len(edge_keys), dtype=np.int8), (edge_keys // n_accounts, edge_keys % n_accounts)),
                              shape=(n_accounts, n_accounts))
    _, label = connected_components(graph, directed=True, connection="strong")
    return graph, label, np.bincount(label)


def _split_and_merge(graph, n_accounts):
    """For every edge of the graph, two counts about two-step paths s -> w -> x (x different from s).

    split[s -> w]: the largest number of different middle accounts through which s reaches one same x, among the x that w pays.
    merge[w -> x]: the largest number of different middle accounts through which one same s reaches x, among the s that pay w.
    A sender who splits money over several accounts that all pay the same end account shows a high count on both legs.
    Returns the edge keys in sorted order and the two arrays.
    """
    graph.sort_indices()
    indptr, indices = graph.indptr, graph.indices
    src = np.repeat(np.arange(n_accounts), np.diff(indptr))             # s of every edge, in sorted edge order
    mid = indices                                                       # w of every edge
    edge_keys = src.astype(np.int64) * n_accounts + mid
    fan = np.diff(indptr)[mid]                                          # how many accounts w pays
    total = int(fan.sum())
    split = np.ones(len(src), dtype=np.float32)
    merge = np.ones(len(src), dtype=np.float32)
    if total == 0:
        return edge_keys, split, merge
    first = np.repeat(np.arange(len(src)), fan)                         # index of the edge s -> w for every path
    offset = np.arange(total) - np.repeat(np.cumsum(fan) - fan, fan)
    second = indptr[mid[first]] + offset                                # index of the edge w -> x for every path
    end = indices[second]
    keep = end != src[first]                                            # a path that returns to s is a loop, counted elsewhere
    first, second, end = first[keep], second[keep], end[keep]
    pair = src[first].astype(np.int64) * n_accounts + end               # (s, x)
    _, inverse, count = np.unique(pair, return_inverse=True, return_counts=True)
    middles = count[inverse].astype(np.float32)                         # middle accounts between s and x, for every path
    np.maximum.at(split, first, middles)
    np.maximum.at(merge, second, middles)
    return edge_keys, split, merge


def build_features(tx, accounts, peer_medians, verbose=False):
    """One row of features per transfer, in the order of `tx`. All values are float32."""
    n = len(tx)
    n_accounts = len(accounts)
    s = tx["s"].to_numpy()
    r = tx["r"].to_numpy()
    day = tx["day"].to_numpy().astype(np.int64)
    n_days = int(day.max()) + 1
    fmt = pd.Categorical(tx["fmt"], categories=FORMATS).codes.astype(np.int64)
    ach = fmt == FORMATS.index("ACH")
    real = s != r                                                       # transfers between two different accounts
    out = {name: np.zeros(n, dtype=np.float32) for name in ALL_FEATURES}

    # ---- transfer
    amount = tx["amt_paid"].to_numpy()
    out["amount_log"] = np.log10(1 + amount).astype(np.float32)
    peer_key = (tx["cur_paid"] + "|" + tx["fmt"]).map(peer_medians).to_numpy(dtype=float)
    out["amount_vs_peers"] = np.where(np.isfinite(peer_key) & (amount > 0), np.log10(np.maximum(amount, 1e-9) / peer_key), 0).astype(np.float32)
    out["format"] = fmt.astype(np.float32)
    out["currency"] = pd.Categorical(tx["cur_paid"], categories=CURRENCIES).codes.astype(np.float32)
    out["hour"] = (tx["minute"].to_numpy() // 60).astype(np.float32)
    out["to_self"] = (~real).astype(np.float32)
    out["same_bank"] = (tx["from_bank"].to_numpy() == tx["to_bank"].to_numpy()).astype(np.float32)
    out["cross_currency"] = (tx["cur_paid"].to_numpy() != tx["cur_recv"].to_numpy()).astype(np.float32)

    # ---- customer
    ctype = pd.Categorical(accounts["customer_type"], categories=CUSTOMER_TYPES).codes
    owner = pd.factorize(accounts["owner"])[0]
    country = pd.factorize(accounts["country"])[0]
    owned = accounts["owner_accounts"].fillna(0).to_numpy(dtype=np.float32)
    out["sender_type"] = ctype[s].astype(np.float32)
    out["receiver_type"] = ctype[r].astype(np.float32)
    out["same_owner"] = ((owner[s] == owner[r]) & (owner[s] >= 0) & real).astype(np.float32)
    out["sender_owner_accounts"] = owned[s]
    out["receiver_owner_accounts"] = owned[r]
    out["cross_border"] = ((country[s] != country[r]) & (country[s] >= 0) & (country[r] >= 0)).astype(np.float32)

    # ---- behavior: counts per account, today and over the window (self transfers left out)
    for name, index, mask in (("s_sent", s, real), ("s_sent_ach", s, real & ach), ("r_recv", r, real), ("r_recv_ach", r, real & ach)):
        today, win = _window_table(index, day, mask, n_accounts, n_days, WINDOW)
        out[name + "_1d"] = today[index, day].astype(np.float32)
        out[name + "_3d"] = win[index, day].astype(np.float32)
        if name == "s_sent":                                           # the same table also says what the receiver sent
            out["r_sent_1d"] = today[r, day].astype(np.float32)
            out["r_sent_3d"] = win[r, day].astype(np.float32)
        elif name == "s_sent_ach":
            out["r_sent_ach_3d"] = win[r, day].astype(np.float32)
        elif name == "r_recv":                                         # and what the sender received
            out["s_recv_1d"] = today[s, day].astype(np.float32)
            out["s_recv_3d"] = win[s, day].astype(np.float32)
        else:
            out["s_recv_ach_3d"] = win[s, day].astype(np.float32)
    del today, win

    # ---- the rest needs the set of transfers in the window, so it goes day by day
    order = np.argsort(day, kind="stable")
    start = np.searchsorted(day[order], np.arange(n_days + 1))
    cur_paid = pd.Categorical(tx["cur_paid"], categories=CURRENCIES).codes.astype(np.int64)
    cur_recv = pd.Categorical(tx["cur_recv"], categories=CURRENCIES).codes.astype(np.int64)
    n_cur = len(CURRENCIES) + 1
    for d in range(n_days):
        rows = order[start[d]:start[d + 1]]                             # today's transfers
        if len(rows) == 0:
            continue
        win = order[start[max(0, d - WINDOW + 1)]:start[d + 1]]         # the window, today included
        before = order[start[max(0, d - WINDOW + 1)]:start[d]]          # the window without today
        long = order[start[max(0, d - LONG_WINDOW + 1)]:start[d + 1]]
        week = order[start[max(0, d - WEEK)]:start[d]]                  # the 7 days before today
        win, before, long, week = win[real[win]], before[real[before]], long[real[long]], week[real[week]]
        key_today = s[rows].astype(np.int64) * n_accounts + r[rows]
        back_today = r[rows].astype(np.int64) * n_accounts + s[rows]

        # currencies each account touched in the window
        touched = np.unique(np.concatenate([s[win].astype(np.int64) * n_cur + cur_paid[win], r[win].astype(np.int64) * n_cur + cur_recv[win]]))
        n_currencies = np.bincount(touched // n_cur, minlength=n_accounts)
        out["s_currencies_3d"][rows] = n_currencies[s[rows]]
        out["r_currencies_3d"][rows] = n_currencies[r[rows]]

        # pairs and counterparties
        edges, per_edge = np.unique(s[win].astype(np.int64) * n_accounts + r[win], return_counts=True)
        out["pair_count_3d"][rows] = _lookup(edges, per_edge, key_today)
        edges_before, per_before = np.unique(s[before].astype(np.int64) * n_accounts + r[before], return_counts=True)
        out["pair_before_3d"][rows] = _lookup(edges_before, per_before, key_today)
        edges_week, per_week = np.unique(s[week].astype(np.int64) * n_accounts + r[week], return_counts=True)
        out["pair_before_7d"][rows] = _lookup(edges_week, per_week, key_today)
        out["s_sent_before_7d"][rows] = np.bincount(s[week], minlength=n_accounts)[s[rows]]
        out["r_recv_before_7d"][rows] = np.bincount(r[week], minlength=n_accounts)[r[rows]]
        paid = np.bincount(edges // n_accounts, minlength=n_accounts)    # different accounts each account paid
        got = np.bincount(edges % n_accounts, minlength=n_accounts)      # different accounts that paid each account
        out["s_receivers_3d"][rows] = paid[s[rows]]
        out["s_senders_3d"][rows] = got[s[rows]]
        out["r_senders_3d"][rows] = got[r[rows]]
        out["r_receivers_3d"][rows] = paid[r[rows]]
        out["back_3d"][rows] = _lookup(edges, np.ones(len(edges), dtype=np.float32), back_today) * real[rows]

        # the ACH graph of the window
        ach_edges = np.unique(s[win[ach[win]]].astype(np.int64) * n_accounts + r[win[ach[win]]])
        paid = np.bincount(ach_edges // n_accounts, minlength=n_accounts)
        got = np.bincount(ach_edges % n_accounts, minlength=n_accounts)
        out["s_receivers_ach_3d"][rows] = paid[s[rows]]
        out["s_senders_ach_3d"][rows] = got[s[rows]]
        out["r_senders_ach_3d"][rows] = got[r[rows]]
        out["r_receivers_ach_3d"][rows] = paid[r[rows]]
        graph, label, size = _loops(ach_edges, n_accounts)
        on_loop = (label[s[rows]] == label[r[rows]]) & real[rows]
        out["loop_3d"][rows] = on_loop
        out["loop_size_3d"][rows] = np.where(on_loop, size[label[s[rows]]], 0)
        two_step = (graph.astype(np.int32) @ graph.astype(np.int32)).tocsr()
        out["triangle_3d"][rows] = np.asarray(two_step[r[rows], s[rows]]).ravel() * real[rows]
        path_edges, split, merge = _split_and_merge(graph, n_accounts)
        out["split_paths_3d"][rows] = _lookup(path_edges, split, key_today)
        out["merge_paths_3d"][rows] = _lookup(path_edges, merge, key_today)

        # loops over the longer window
        long_edges = np.unique(s[long[ach[long]]].astype(np.int64) * n_accounts + r[long[ach[long]]])
        _, label, size = _loops(long_edges, n_accounts)
        on_loop = (label[s[rows]] == label[r[rows]]) & real[rows]
        out["loop_5d"][rows] = on_loop
        out["loop_size_5d"][rows] = np.where(on_loop, size[label[s[rows]]], 0)
        if verbose:
            print(f"  day {d:2d}: {len(rows):9,d} transfers, {len(edges):9,d} pairs in the window, {len(ach_edges):7,d} by ACH", flush=True)
    return pd.DataFrame(out, columns=ALL_FEATURES, copy=False)           # copy=False: keep the columns as they are, do not build a second table
