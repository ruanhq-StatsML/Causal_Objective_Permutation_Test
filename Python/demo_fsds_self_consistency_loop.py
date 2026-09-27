#!/usr/bin/env python3
"""Self-consistency closed-loop: dispersion-driven sampling until stable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from demo_fsds_sot import synthetic_sot_batch
from fsds_sot.applications import adaptive_sample_count, build_self_consistency_trace
from fsds_sot.closed_loop import run_self_iteration, summarize_loop


def main() -> int:
    rng = np.random.default_rng(7)
    B, p, d = 5, 32, 20
    X_ref, _, _ = synthetic_sot_batch(80, B, p, seed=1, drift_strength=0.0)

    def chains_for_window(disp_scale: float) -> np.ndarray:
        base = rng.normal(size=d)
        n = adaptive_sample_count(
            np.stack([base + rng.normal(scale=disp_scale, size=d) for _ in range(4)]),
            n_min=3,
            n_max=12,
        )
        return np.stack([base + rng.normal(scale=disp_scale, size=d) for _ in range(n)])

    disp_scales = [0.55, 0.35, 0.18, 0.08, 0.03]
    windows = []
    segments_meta = []
    for i, scale in enumerate(disp_scales):
        chains = chains_for_window(scale)
        seg = build_self_consistency_trace(chains)
        windows.append(
            synthetic_sot_batch(80, B, p, seed=20 + i, drift_strength=scale * 0.5)[0]
        )
        if i == 0:
            seg0 = seg
        segments_meta.append(
            {
                "n_chains": chains.shape[0],
                "dispersion": float(seg.trace_features[-2]),
                "adaptive_n_star": adaptive_sample_count(chains),
            }
        )

    state, outcomes = run_self_iteration(
        X_ref,
        windows,
        seg0.embeddings,
        seg0.quality,
        stabilize_threshold=0.17,
    )
    summary = summarize_loop(state, outcomes)

    out = {
        "pattern": "self_consistency",
        "windows_meta": segments_meta,
        "loop": summary.to_dict(),
        "final_n_recommendation": segments_meta[summary.iterations - 1]["adaptive_n_star"]
        if summary.iterations
        else None,
    }
    print(json.dumps(out, indent=2))
    return 0 if summary.stabilized else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
