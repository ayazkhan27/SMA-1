"""LogHub acquisition metadata helpers."""

from __future__ import annotations

import json
import pathlib


def load_manifest(path: str | pathlib.Path = "data/manifests/datasets.json") -> dict:
    return json.loads(pathlib.Path(path).read_text())


def loghub_files(path: str | pathlib.Path = "data/manifests/datasets.json") -> dict:
    manifest = load_manifest(path)
    return manifest["loghub_raw"]["files"]

