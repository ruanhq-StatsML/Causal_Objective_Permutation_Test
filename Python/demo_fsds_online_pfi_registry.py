#!/usr/bin/env python3
"""Demo: Model Registry refit → predict → delta (canonical online PFI logic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fsds_sot.agentic_dgp import generate_agentic_batch, episodes_to_arrays
from fsds_sot.online_pfi_registry import run_online_registry_stream

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "delivery" / "online_pfi"


def main() -> int:
    ref = generate_agentic_batch(50, seed=1, live=False)
    X_ref, Y_ref, _, _ = episodes_to_arrays(ref)
    windows = []
    for i, drift in enumerate([0.0, 0.5, 1.0, 1.4]):
        live = generate_agentic_batch(40, seed=10 + i, live=drift > 0, drift_strength=drift + 0.5)
        X_w, Y_w, _, _ = episodes_to_arrays(live)
        windows.append((X_w, Y_w))

    report = run_online_registry_stream(X_ref, Y_ref, windows, n_perm=30, seed=2026)
    out = {
        "software": {
            "model_registry": "Python/model_registry_class.py",
            "permute_refit_test": "Python/DRPerm.py",
            "r_risk_loco": "Python/R_risk_loco.py",
            "online_stream_wrapper": "Python/fsds_sot/online_pfi_registry.py",
            "note": "fsds_sot/online_pfi.py is sklearn RF-PFI dashboard only; NOT Model Registry.",
        },
        "stream": report.to_dict(),
    }
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "online_pfi_registry_report.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
