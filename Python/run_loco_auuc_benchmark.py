#!/usr/bin/env python3
"""Run LOCO-AUUC benchmark (Registry learners + online PFI). Use --quick for CI."""

from __future__ import annotations

import argparse
from pathlib import Path

from loco_auuc.benchmark import run_full_benchmark, write_benchmark_outputs

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs" / "latex"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 learners only")
    ap.add_argument("--out", type=Path, default=ART)
    args = ap.parse_args()
    result = run_full_benchmark(quick=args.quick)
    write_benchmark_outputs(result, args.out)
    DOCS.mkdir(parents=True, exist_ok=True)
    from loco_auuc.benchmark import write_monitoring_paper

    write_monitoring_paper(result, DOCS / "loco_auuc_monitoring.tex")
    print("Wrote", args.out / "loco_auuc_benchmark.json")
    print("Wrote", args.out / "loco_auuc_benchmark.tex")
    print("Wrote", DOCS / "loco_auuc_monitoring.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
