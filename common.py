"""Paths, constants and small helpers shared by every script."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"

SEED = 20261122

# The two portfolios. HI-Small is where the model is built and tested.
# LI-Small is a second portfolio, with about half the laundering rate, used only in the revalidation step.
PORTFOLIOS = {
    "hi": {"transactions": "HI-Small_Trans.csv", "accounts": "HI-Small_accounts.csv", "patterns": "HI-Small_Patterns.txt"},
    "li": {"transactions": "LI-Small_Trans.csv", "accounts": "LI-Small_accounts.csv", "patterns": "LI-Small_Patterns.txt"},
}

# Days are counted from the first date in the file (September 1, 2022 is day 0).
# Regular activity runs for ten days. After that the file holds only the last legs of
# laundering schemes, so those days are scored but left out of every result.
FIRST_DATE = "2022-09-01"
WARMUP_DAYS = (0, 1)                   # history only: the 3-day windows are not full yet, so no model learns from these rows
TRAIN_DAYS = (2, 3, 4, 5)              # the models learn here (day 5 also sets the number of trees)
TUNE_DAYS = (6, 7)                     # settings, the calibration and the alert line are chosen here
TEST_DAYS = (8, 9)                     # touched once, at the end
LAST_REGULAR_DAY = 9

WINDOW = 3                             # days of history behind most features, today included
LONG_WINDOW = 5                        # days of history behind the two "money came back" features
WEEK = 7                               # days before today behind the weekly pair history added in version 2

RAW_COLUMNS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def note_time(step, seconds, **more):
    """Record how long a step took, in results/timings.json. Timings depend on the machine, so they are kept apart
    from the results, which do not."""
    path = RESULTS / "timings.json"
    timings = read_json(path) if path.exists() else {}
    timings[step] = {"seconds": round(seconds), **more}
    write_json(path, timings)


def read_transactions(portfolio):
    """The raw transfers of one portfolio, with integer account numbers and a day counter.

    s and r are the sender and the receiver as integers (one number per bank-and-account pair).
    Rows keep the order of the file, so row i here is line i + 2 of the CSV.
    """
    path = RAW / PORTFOLIOS[portfolio]["transactions"]
    tx = pd.read_csv(path, header=0, names=RAW_COLUMNS,
                     dtype={"from_bank": str, "to_bank": str, "from_acct": str, "to_acct": str, "y": "int8"})
    ts = pd.to_datetime(tx["ts"], format="%Y/%m/%d %H:%M")
    tx["day"] = (ts.dt.normalize() - pd.Timestamp(FIRST_DATE)).dt.days.astype("int16")
    tx["minute"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int16")
    # Bank numbers are zero-padded in the transfers and plain in the accounts file, so compare them as numbers.
    from_key = tx["from_bank"].astype("int64").astype(str) + "|" + tx["from_acct"]
    to_key = tx["to_bank"].astype("int64").astype(str) + "|" + tx["to_acct"]
    codes, keys = pd.factorize(pd.concat([from_key, to_key], ignore_index=True))
    n = len(tx)
    tx["s"] = codes[:n].astype("int32")
    tx["r"] = codes[n:].astype("int32")
    return tx, pd.Series(keys, name="key")


def read_accounts(portfolio, keys):
    """One row per account in `keys` order: who owns it, what kind of customer, which country the bank is in."""
    acc = pd.read_csv(RAW / PORTFOLIOS[portfolio]["accounts"], dtype=str)
    acc.columns = ["bank_name", "bank_id", "account", "owner", "owner_name"]
    acc["key"] = acc["bank_id"].astype("int64").astype(str) + "|" + acc["account"]
    acc = acc.drop_duplicates("key").set_index("key")
    acc["customer_type"] = acc["owner_name"].str.replace(r"\s*#\d+$", "", regex=True)
    acc["country"] = acc["bank_name"].str.replace(r"\s*Bank\s*#\d+$", "", regex=True)
    acc["owner_accounts"] = acc.groupby("owner")["owner"].transform("size")
    return acc.reindex(keys.values)[["owner", "customer_type", "country", "owner_accounts"]].reset_index(drop=True)


def read_patterns(portfolio):
    """The laundering schemes the publisher labeled by type. Returns one row per transfer in a scheme."""
    rows, scheme, kind = [], -1, None
    with open(RAW / PORTFOLIOS[portfolio]["patterns"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                scheme += 1
                kind = line.split(" - ", 1)[1].split(":")[0].strip()
            elif line and not line.startswith("END"):
                rows.append([scheme, kind] + line.split(","))
    out = pd.DataFrame(rows, columns=["scheme", "scheme_type"] + RAW_COLUMNS)
    return out


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a share k of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
