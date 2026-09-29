"""CI figure for online PO-risk.

The risk is ``drperm_po_risk``: ``MODEL_REGISTRY`` ``model_m`` / ``model_e``
cross-fit, then ``mean(tau ** 2)``. The original batch is ``T = 0``. Every
sample after it that sits in the recent window is ``T = 1``. The band is
``online_bootstrap_ci``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mma_wrapper import online_bootstrap_ci, plot_online_porisk


def concept_stream(n: int, p: int, delta: float, seed: int):
    """First half is the original batch. The second half carries the concept shift."""
    rng = np.random.default_rng(seed)
    corr = np.eye(p)
    x = rng.multivariate_normal(np.zeros(p), corr, size=n)
    y = x @ np.ones(p) + rng.normal(size=n)
    beta = np.zeros(p)
    beta[: min(4, p)] = np.arange(min(4, p)) * float(delta)
    y[n // 2:] = y[n // 2:] + x[n // 2:] @ beta
    return x, y


def main() -> None:
    x_shift, y_shift = concept_stream(64, 4, delta=0.8, seed=3)
    x_null, y_null = concept_stream(64, 4, delta=0.0, seed=4)
    shifted = online_bootstrap_ci(x_shift, y_shift, ref_n=16, recent_n=10, step=8, n_boot=12, seed=3)
    null = online_bootstrap_ci(x_null, y_null, ref_n=16, recent_n=10, step=8, n_boot=12, seed=4)
    for name, rows in (("concept", shifted), ("null", null)):
        for row in rows:
            print(
                f"{name} m={row['m']} PO-risk={row['porisk']:.4f} "
                f"CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
            )
    out = Path("/opt/cursor/artifacts/online_porisk_ci.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_online_porisk(
        [("concept drift", shifted), ("no concept drift", null)],
        str(out),
        enter_m=40,
    )
    payload = {"concept": shifted, "null": null}
    Path("/opt/cursor/artifacts/online_porisk_ci.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    repo = Path(__file__).resolve().parent / "results"
    (repo / "online_porisk_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("CI_OK", out)


if __name__ == "__main__":
    main()
