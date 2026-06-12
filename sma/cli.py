"""CLI for SMA MVP."""

from __future__ import annotations

import argparse
import json
import sys

from sma.agent.service import MemoryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sma")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("adapter_id")
    enc.add_argument("artifact_file")
    enc.add_argument("--store", default="data/processed/store")

    ret = sub.add_parser("retrieve")
    ret.add_argument("case_id")
    ret.add_argument("--k", type=int, default=10)
    ret.add_argument("--store", default="data/processed/store")

    mp = sub.add_parser("map")
    mp.add_argument("base_id")
    mp.add_argument("target_id")
    mp.add_argument("--scorer", default="ses")
    mp.add_argument("--store", default="data/processed/store")

    report = sub.add_parser("report")
    report.add_argument("--out", default="reports/report.html")

    sub.add_parser("ui")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "report":
        from sma.eval.report import main as report_main

        raise SystemExit(report_main(["--out", args.out]))
    if args.cmd == "ui":
        from sma.ui.app import main as ui_main

        raise SystemExit(ui_main([]))

    service = MemoryService(getattr(args, "store", "data/processed/store"))
    if args.cmd == "encode":
        artifact = open(args.artifact_file, encoding="utf-8").read()
        print(json.dumps(service.encode(artifact, args.adapter_id), indent=2))
    elif args.cmd == "retrieve":
        print(json.dumps(service.retrieve(case_id=args.case_id, k=args.k), indent=2))
    elif args.cmd == "map":
        print(json.dumps(service.map(args.base_id, args.target_id, args.scorer), indent=2))
    else:
        print(f"unknown command: {args.cmd}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

