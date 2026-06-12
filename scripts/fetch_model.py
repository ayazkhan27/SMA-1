#!/usr/bin/env python3
"""Download the default quantized local orchestrator model."""

from __future__ import annotations

import argparse
import pathlib


DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit("Install with `pip install -e '.[local-llm]'` first.") from exc
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.file,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
