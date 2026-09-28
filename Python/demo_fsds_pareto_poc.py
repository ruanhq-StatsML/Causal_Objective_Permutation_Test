#!/usr/bin/env python3
"""Standalone Pareto POC sweep (JSON to stdout)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fsds_sot.agentic_dgp import _branch_need_vector, generate_agentic_batch, episodes_to_arrays
from fsds_sot.pareto_poc import run_pareto_sweep


def main() -> int:
    ref = generate_agentic_batch(100, seed=11, live=False)
    live = generate_agentic_batch(100, seed=22, live=True, drift_strength=1.4)
    eval_eps = generate_agentic_batch(60, seed=33, live=True, drift_strength=1.4)
    X0, _, _, _ = episodes_to_arrays(ref)
    X1, _, _, _ = episodes_to_arrays(live)
    _, _, Z, Q = episodes_to_arrays(eval_eps)
    need = _branch_need_vector(eval_eps[0].branch_names)
    out = run_pareto_sweep(Z, Q, need, X0, X1, seed=2026, success_epsilon=0.03)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
