#!/usr/bin/env python3
"""Checksum-verified dataset downloader for SMA-1 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request


def md5_file(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def iter_files(manifest: dict):
    for dataset_name, dataset in manifest.items():
        for filename, spec in dataset.get("files", {}).items():
            yield dataset_name, filename, spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/datasets.json")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    out_root = pathlib.Path(args.out)
    selected = set(args.only)

    for dataset_name, filename, spec in iter_files(manifest):
        if selected and dataset_name not in selected and filename not in selected:
            continue
        out = out_root / dataset_name / filename
        print(f"{dataset_name}: {filename}")
        print(f"  url: {spec['url']}")
        print(f"  out: {out}")
        if args.dry_run:
            continue
        if not out.exists():
            download(spec["url"], out)
        expected = spec.get("md5")
        if expected:
            actual = md5_file(out)
            if actual != expected:
                print(f"checksum mismatch for {out}: expected {expected}, got {actual}", file=sys.stderr)
                return 2
            print("  checksum: ok")
        else:
            print("  checksum: skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

