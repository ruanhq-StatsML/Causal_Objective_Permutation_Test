#!/usr/bin/env python3
"""Package Online PFI delivery: run demo, copy artifacts, write MANIFEST."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEL = ROOT / "delivery" / "online_pfi"
PY = Path(__file__).resolve().parent


def main() -> int:
    subprocess.run([sys.executable, str(PY / "demo_fsds_online_pfi_viz.py")], check=True, cwd=PY)

    DEL.mkdir(parents=True, exist_ok=True)
    copies = [
        (ROOT / "artifacts" / "07_online_pfi_dashboard.png", DEL / "07_online_pfi_dashboard.png"),
        (ROOT / "artifacts" / "online_pfi_summary.json", DEL / "online_pfi_report.json"),
        (ROOT / "docs" / "FSDS_ONLINE_PFI_DELIVERY_PACKAGE.md", DEL / "README_DELIVERY.md"),
        (ROOT / "docs" / "FSDS_ONLINE_PFI.md", DEL / "FSDS_ONLINE_PFI.md"),
        (ROOT / "FSDS_online_PFI_delivery.tex", DEL / "FSDS_online_PFI_delivery.tex"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)

    manifest = {
        "package": "FSDS_Online_PFI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "Python/package_online_pfi_delivery.py",
        "core_modules": [
            "Python/fsds_sot/online_pfi.py",
            "Python/fsds_sot/plot_online_pfi.py",
            "Python/fsds_sot/attribution.py",
            "Python/fsds_sot/closed_loop.py",
        ],
        "artifacts_in_this_folder": [p.name for p in DEL.iterdir() if p.is_file()],
        "logic_summary": (
            "Rolling ref vs live → RF domain + permutation VIMP (online PFI) "
            "+ MMD/AUC + onlineRFPerm p-value → decay reference → SoT/closed-loop actuators"
        ),
    }
    (DEL / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"delivery_dir": str(DEL), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
