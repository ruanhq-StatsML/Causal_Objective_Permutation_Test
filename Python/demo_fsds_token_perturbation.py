#!/usr/bin/env python3
"""Token mask/replace → re-encode → MMD + PO-risk audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PY = Path(__file__).resolve().parent
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

from fsds_sot.agentic_dgp import generate_agentic_batch  # noqa: E402
from fsds_sot.token_perturbation_audit import run_token_perturbation_audit  # noqa: E402

ROOT = PY.parent
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main() -> int:
    seed = 7451
    ref = generate_agentic_batch(100, seed=seed, live=False)
    live = generate_agentic_batch(100, seed=seed + 9, live=True, drift_strength=1.5)
    report = run_token_perturbation_audit(ref, live, seed=seed, mask_frac=0.3)
    payload = report.to_dict()
    top_pos = payload["token_position_loco"][:5]
    top_grp = payload["prefix_group_perturbations"][:3]
    payload["summary"] = {
        "baseline_mmd2": payload["baseline"]["mmd2"],
        "baseline_po_risk": payload["baseline"]["po_risk_observed"],
        "top_token_masks_by_delta_mmd": top_pos,
        "top_prefix_masks_by_delta_mmd": top_grp,
    }
    out = ART / "token_perturbation_report.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
