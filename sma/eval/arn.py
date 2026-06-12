"""ARN dataset helpers."""

from __future__ import annotations

import csv
import pathlib


DEFAULT_ARN_PATH = pathlib.Path(
    "data/raw/arn/Analogical Reasoning on Narratives (ARN) dataset.xlsx - Sheet1.csv"
)

ARN_REQUIRED_COLUMNS = {
    "id",
    "proverb",
    "query_narrative",
    "first_choice",
    "second_choice",
    "distractor_similarity",
    "analogy_level",
    "correct_answer",
}


def validate_columns(columns: list[str]) -> bool:
    return ARN_REQUIRED_COLUMNS.issubset(set(columns))


def load_arn_rows(path: str | pathlib.Path = DEFAULT_ARN_PATH, limit: int | None = None) -> list[dict[str, str]]:
    source = pathlib.Path(path)
    with source.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not validate_columns(reader.fieldnames):
            raise ValueError(f"ARN CSV has unexpected columns: {reader.fieldnames}")
        rows = []
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows


def arn_choice_corpus(path: str | pathlib.Path = DEFAULT_ARN_PATH, limit: int = 12) -> tuple[str, str]:
    """Return a raw text corpus of answer choices plus one suggested query."""

    rows = load_arn_rows(path, limit=limit)
    blocks: list[str] = []
    suggested_query = ""
    for row in rows:
        if not suggested_query:
            suggested_query = row["query_narrative"]
        correct = row["correct_answer"].strip()
        for choice_number, column in (("1", "first_choice"), ("2", "second_choice")):
            label = "correct" if choice_number == correct else "distractor"
            blocks.append(
                "\n".join(
                    [
                        f"ARN id={row['id']} choice={choice_number} label={label}",
                        f"proverb: {row['proverb']}",
                        f"analogy_level: {row['analogy_level']}",
                        row[column],
                    ]
                )
            )
    return "\n\n".join(blocks), suggested_query
