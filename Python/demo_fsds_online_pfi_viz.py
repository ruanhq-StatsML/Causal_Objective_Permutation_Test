#!/usr/bin/env python3
"""Demo: online PFI stream + 4-panel visualization on agentic windows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fsds_sot.agentic_dgp import generate_agentic_batch, episodes_to_arrays
from fsds_sot.online_pfi import default_agentic_feature_names, run_online_pfi_stream
from fsds_sot.plot_online_pfi import plot_online_pfi_dashboard

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main() -> int:
    ref_eps = generate_agentic_batch(60, seed=1, live=False)
    X_ref, _, _, _ = episodes_to_arrays(ref_eps)
    p = X_ref.shape[1]

    # Simulate production: stable → drift ramps in
    windows = []
    for i, drift in enumerate([0.0, 0.3, 0.8, 1.2, 1.4, 1.0, 0.5]):
        live = generate_agentic_batch(35, seed=10 + i, live=drift > 0, drift_strength=drift + 0.5)
        X_w, _, _, _ = episodes_to_arrays(live)
        windows.append(X_w)

    report = run_online_pfi_stream(
        X_ref,
        windows,
        feature_names=default_agentic_feature_names(p),
        decay_alpha=0.88,
        seed=2026,
    )
    png = plot_online_pfi_dashboard(
        report,
        ART / "07_online_pfi_dashboard.png",
        title="Online PFI — agentic trace windows (FSDS)",
    )

    summary = {
        "where_in_repo": {
            "stream_logic": "Python/fsds_sot/online_pfi.py",
            "visualization": "Python/fsds_sot/plot_online_pfi.py",
            "batch_snapshot_analogue": "Python/fsds_sot/attribution.py (Panel 4 in viz.py)",
            "rap_branch_onlineRFPerm": "commit 9a6149a Python/online_drift_detectors.py (RAP, not merged here)",
        },
        "stream_steps": len(report.steps),
        "final_step": report.steps[-1].to_dict() if report.steps else {},
        "artifact_png": str(png),
    }
    (ART / "online_pfi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
