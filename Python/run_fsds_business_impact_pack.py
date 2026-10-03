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
        ["python3", "plot_fsds_scenario_slides_deck.py"],
        ["python3", "export_fsds_scenario_slides_pptx.py"],
        ["python3", "synthesize_fsds_economics_report.py"],
    ]
    for cmd in demos:
        _run(cmd)
    print("\nDone:")
    print("  artifacts/fsds_business_impact_dashboard_en.png")
    print("  artifacts/fsds_scenario_slides/fsds_scenario_deck_{en,zh,bilingual}.pptx")
    print("  docs/FSDS_BUSINESS_IMPACT_EXECUTIVE_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
