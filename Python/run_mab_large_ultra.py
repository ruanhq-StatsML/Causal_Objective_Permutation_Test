"""Large-scale MAB: dim=500, n=100k, batch=500, ref=10k on nonlinear_messy_ultra."""
from __future__ import annotations

import argparse
import copy
import os

from config_MAB import CONFIG
from run_mab_benchmark import run_benchmark

LARGE_DGP = {
    "name": "nonlinear_messy_ultra_large_sm05",
    "dgp": "nonlinear_messy_ultra",
    "shift_magnitude": 0.5,
    "shift_point_index": 50000,
    "shift_interval": 12000,
    "mini_shift_interval": 2400,
    "covariate_drift_strength": 0.45,
}

EPSILON_GRID = [
    0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025,
    0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070, 0.075,
    0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130, 0.150,
    0.170, 0.200, 0.220, 0.250, 0.280, 0.300, 0.350, 0.400, 0.450,
    0.500, 0.550, 0.600, 0.650, 0.700,
]


def build_config() -> dict:
    cfg = copy.deepcopy(CONFIG)
    cfg["data"].update(
        {
            "total_samples": 100000,
            "ref_samples": 10000,
            "batch_size": 500,
            "feature_dim": 500,
            "dgp": "nonlinear_messy_ultra",
            "shift_magnitude": 0.5,
            "shift_point_index": 50000,
            "shift_interval": 12000,
            "mini_shift_interval": 2400,
            "covariate_drift_strength": 0.45,
        }
    )
    cfg["dgp_grid"] = [LARGE_DGP]
    cfg["policies"]["Epsilon_Greedy"]["epsilon"] = EPSILON_GRID
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale ultra messy MAB benchmark")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["Epsilon_Greedy", "UCB", "LinUCB_Momentum"],
    )
    parser.add_argument("--output-prefix", default="mab_large")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = build_config()
    print(
        "[mab_large] dim=500 n=100000 batch=500 ref=10000 "
        f"policies={args.policies} workers={args.workers}",
        flush=True,
    )
    metrics, summary = run_benchmark(
        cfg,
        quick=False,
        dgp_names=[LARGE_DGP["name"]],
        policy_names=args.policies,
        output_prefix=args.output_prefix,
        plot=False,
        workers=max(1, args.workers),
        resume=args.resume,
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
