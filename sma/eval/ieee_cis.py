"""IEEE-CIS Fraud loader + per-transaction artifact builder (4b finance).

Real Kaggle transaction-fraud data. Same shape as the healthcare loader: a flat
CSV-row artifact for the generic structured adapter (the honest 'before') and a
plain attr=value text for baselines. The 339 anonymized V columns and the label
are NEVER encoded.
"""
from __future__ import annotations

import csv
import pathlib
import random
from dataclasses import dataclass

# Drop ids, raw timestamp, the LABEL, and all opaque engineered columns (V*, id_*).
_DROP_EXACT = {"TransactionID", "TransactionDT", "isFraud"}


def _keep(col: str) -> bool:
    return col not in _DROP_EXACT and not col.startswith("V") and not col.startswith("id_")


@dataclass(frozen=True)
class Txn:
    tid: str
    fields: dict[str, str]
    label: str  # "fraud" vs "legit"


def _csv_path() -> pathlib.Path:
    return pathlib.Path("data/raw/ieee_cis/train_transaction.csv")


def load_transactions(sample: int = 1500, seed: int = 7, balanced: bool = True,
                      scan_cap: int = 120000) -> list[Txn]:
    """Stream up to scan_cap rows, collect a balanced fraud/legit sample
    (fraud is ~3.5%, so a balanced set makes retrieval-by-analogy meaningful)."""
    half = sample // 2
    fraud: list[Txn] = []
    legit: list[Txn] = []
    with _csv_path().open() as fh:
        r = csv.DictReader(fh)
        for i, row in enumerate(r):
            if i >= scan_cap:
                break
            fields = {k: v for k, v in row.items() if _keep(k) and v not in ("", "NaN")}
            t = Txn(row["TransactionID"], fields, "fraud" if row["isFraud"] == "1" else "legit")
            (fraud if t.label == "fraud" else legit).append(t)
            if balanced and len(fraud) >= half and len(legit) >= half:
                break
    rng = random.Random(seed)
    if balanced:
        out = fraud[:half] + legit[:half]
    else:
        out = (fraud + legit)[:sample]
    rng.shuffle(out)
    return out


def row_csv(t: Txn) -> str:
    keys = sorted(t.fields)
    return ",".join(keys) + "\n" + ",".join(t.fields[k] for k in keys) + "\n"


def row_text(t: Txn) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(t.fields.items()))
