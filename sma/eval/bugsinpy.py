"""BugsInPy metadata parsing helpers."""

from __future__ import annotations

import pathlib


def discover_bug_metadata(root: str | pathlib.Path) -> list[pathlib.Path]:
    return sorted(pathlib.Path(root).glob("projects/*/bugs/*/bug.info"))

