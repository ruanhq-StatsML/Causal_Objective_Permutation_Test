"""Online PFI and online PO-risk, as already implemented in ``mma_wrapper``.

``online_pfi`` shuffles one token inside the fixed-reference plus trailing
window and records the MMD drop. ``online_bootstrap_ci`` scores that same
window with the DRPerm PO-risk (``model_m`` / ``model_e``, later samples
``T = 1``) and the stratified bootstrap percentile band.
"""

from mma_wrapper import (
    drperm_po_risk,
    online_bootstrap_ci,
    online_pfi,
    plot_online_porisk,
)

__all__ = [
    "drperm_po_risk",
    "online_bootstrap_ci",
    "online_pfi",
    "plot_online_porisk",
    "monitor",
]


def monitor(tokens, y, grid, ref_n: int = 16, recent_n: int = 8, step: int = 8,
            n_boot: int = 8, seed: int = 2026):
    """One pass: online PFI on the tokens, online PO-risk CI on ``y``."""
    pfi = online_pfi(tokens, grid, ref_n=ref_n, recent_n=recent_n, step=step, seed=seed)
    porisk = online_bootstrap_ci(
        tokens.reshape(tokens.shape[0], -1) if tokens.ndim > 2 else tokens,
        y,
        ref_n=ref_n,
        recent_n=recent_n,
        step=step,
        n_boot=n_boot,
        seed=seed,
    )
    return {"pfi": pfi, "porisk": porisk}
