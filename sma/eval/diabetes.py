"""Diabetes-130 loader + per-encounter artifact builder (4b cross-domain).

Real UCI EHR. We build a flat CSV-row artifact for the GENERIC structured
adapter (the honest 'before': flat triples, no higher-order relations) and a
plain attr=value text for the lexical/dense baselines. The readmission label is
NEVER encoded (leakage guard).
"""
from __future__ import annotations

import csv
import pathlib
import random
from dataclasses import dataclass

# ids, mostly-missing columns, and the LABEL are excluded from the encoding.
DROP = {"encounter_id", "patient_nbr", "weight", "payer_code", "readmitted"}


@dataclass(frozen=True)
class Encounter:
    eid: str
    fields: dict[str, str]   # cleaned attribute -> value (no '?'/'' )
    label: str               # "early" (readmitted <30 days) vs "not"


def _csv_path() -> pathlib.Path:
    return next(pathlib.Path("data/raw/diabetes130").rglob("diabetic_data.csv"))


def load_encounters(sample: int = 1500, seed: int = 7, balanced: bool = True) -> list[Encounter]:
    rows = list(csv.DictReader(_csv_path().open()))
    rng = random.Random(seed)
    rng.shuffle(rows)

    def to_enc(r: dict) -> Encounter:
        fields = {k: v for k, v in r.items() if k not in DROP and v not in ("?", "")}
        label = "early" if r["readmitted"] == "<30" else "not"
        return Encounter(r["encounter_id"], fields, label)

    encs = [to_enc(r) for r in rows]
    if not balanced:
        return encs[:sample]
    # balance the two classes so retrieval-by-analogy has signal (early is ~11%)
    early = [e for e in encs if e.label == "early"]
    notr = [e for e in encs if e.label == "not"]
    half = sample // 2
    out = early[:half] + notr[:half]
    rng.shuffle(out)
    return out


def row_csv(enc: Encounter) -> str:
    """Two-line CSV (header + one row) for the structured adapter -> flat
    (attribute row value) triples."""
    keys = sorted(enc.fields)
    return ",".join(keys) + "\n" + ",".join(enc.fields[k] for k in keys) + "\n"


def row_text(enc: Encounter) -> str:
    """attr=value text for BM25 / dense baselines."""
    return " ".join(f"{k}={v}" for k, v in sorted(enc.fields.items()))
