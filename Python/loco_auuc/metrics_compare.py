"""Compare our AUUC integral to scikit-uplift when available."""

from __future__ import annotations

import numpy as np

from .metrics import auuc, qini_curve


def auuc_integral_report(y, treatment, uplift) -> dict:
    """
    Document the integral: AUUC = (1/n) * int_0^1 gain(f) df
    where gain(f) is cumulative Qini at population fraction f.
    """
    frac, gain = qini_curve(y, treatment, uplift)
    area = np.trapezoid(gain, frac) if hasattr(np, "trapezoid") else np.trapz(gain, frac)
    n = len(y)
    raw = area / max(n, 1)
    from .metrics import auuc as auuc_primary

    ours = auuc_primary(y, treatment, uplift)
    out = {
        "n": n,
        "auuc_primary": float(ours),
        "auuc_gain_integral": float(raw),
        "raw_area": float(area),
        "gain_at_10pct": float(np.interp(0.1, frac, gain)),
        "gain_at_50pct": float(np.interp(0.5, frac, gain)),
        "gain_at_100pct": float(gain[-1]) if len(gain) else 0.0,
    }
    try:
        from sklift.metrics import uplift_auc_score, qini_auc_score

        t = np.asarray(treatment).astype(int).ravel()
        y = np.asarray(y).astype(float).ravel()
        u = np.asarray(uplift, dtype=float).ravel()
        # sklift API: (y_true, uplift, treatment)
        out["sklift_uplift_auc"] = float(uplift_auc_score(y, u, t))
        out["sklift_qini_auc"] = float(qini_auc_score(y, u, t))
        out["delta_vs_sklift_uplift"] = float(out["sklift_uplift_auc"] - ours)
    except Exception as e:
        out["sklift_error"] = str(e)
    return out
