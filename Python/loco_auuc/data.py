"""Array splits for uplift monitoring (REF train/eval + LIVE batch)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_ref_live(
    X_ref: np.ndarray,
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    X_live: np.ndarray,
    t_live: np.ndarray,
    y_live: np.ndarray,
    *,
    eval_frac: float = 0.25,
    seed: int = 42,
):
    """Hold out part of REF for AUUC; LIVE stays untouched (deployment window)."""
    X_tr, X_ev, t_tr, t_ev, y_tr, y_ev = train_test_split(
        X_ref,
        t_ref,
        y_ref,
        test_size=eval_frac,
        stratify=t_ref,
        random_state=seed,
    )
    return (X_tr, t_tr, y_tr), (X_ev, t_ev, y_ev), (X_live, t_live, y_live)
