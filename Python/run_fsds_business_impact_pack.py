#!/usr/bin/env python3
"""One command: all GTM demos + uplift bench + dashboard PNG."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=PY)


def main() -> int:
    demos = [
        ["python3", "demo_fsds_multi_agent_economics.py"],
        ["python3", "demo_fsds_multi_agent_debate_early_stop.py"],
        ["python3", "demo_fsds_langgraph_hook.py"],
        ["python3", "demo_federated_legal_agent_blocks.py"],
        ["python3", "run_uplift_subset_benchmark.py", "--n-perm", "32"],
        ["python3", "plot_fsds_business_impact_dashboard_en.py"],
    ]
    for cmd in demos:
        _run(cmd)
    print("\nDone. See artifacts/fsds_business_impact_dashboard_en.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
