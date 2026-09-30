#!/usr/bin/env python3
"""Demo: three dataloaders → ref/live → modality + fine-grain attribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PY = Path(__file__).resolve().parent
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

from fsds_sot.agentic_dgp import generate_agentic_batch  # noqa: E402
from fsds_sot.modality_attribution import run_modality_attribution  # noqa: E402

ROOT = PY.parent
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main() -> int:
    seed = 7451
    ref = generate_agentic_batch(120, seed=seed, live=False)
    live = generate_agentic_batch(120, seed=seed + 1, live=True, drift_strength=1.4)

    report = run_modality_attribution(
        ref,
        live,
        seed=seed,
        include_registry_deltas=True,
    )
    payload = report.to_dict()

    # Rank modalities by global-slice VIMP mass (concat model)
    slice_rank = sorted(
        payload["fine_grain"].keys(),
        key=lambda m: sum(f["vimp"] for f in payload["fine_grain"][m]),
        reverse=True,
    )
    payload["modality_rank_by_global_vimp"] = slice_rank

    out = ART / "modality_attribution_report.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out), "modality_rank": slice_rank}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
