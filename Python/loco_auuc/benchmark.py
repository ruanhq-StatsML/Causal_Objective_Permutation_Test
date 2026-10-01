"""Benchmark LOCO-AUUC + Model Registry learners + online PFI alignment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .benchmark_datasets import UpliftBatch, all_benchmarks
from .data import split_ref_live
from .learners import make_learner
from .loco import loco_auuc_monitor
from .metrics import auuc, evaluate_window
from .metrics_compare import auuc_integral_report
from .monitor import run_uplift_monitor
from .registry_learners import make_registry_learner


REGISTRY_OUTCOME_MODELS = [
    ("rf", "rf_classifier", "rf_regressor", "rf_classifier"),
    ("logistic", "ridge_regressor", "ridge_regressor", "logistic_classifier"),
]


def _feature_groups(p: int, n_groups: int = 2) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    chunk = max(p // n_groups, 1)
    for g in range(n_groups):
        lo = g * chunk
        hi = p if g == n_groups - 1 else min((g + 1) * chunk, p)
        if lo < hi:
            groups[f"block_{g}"] = list(range(lo, hi))
    return groups


def _online_pfi_on_batch(batch: UpliftBatch, seed: int = 42) -> dict:
    """Stack REF+ LIVE as two-step stream (covariate shift on X)."""
    from fsds_sot.online_pfi import run_online_pfi_stream

    n0 = min(2000, batch.X_ref.shape[0])
    n1 = min(1200, batch.X_live.shape[0])
    X0, X1 = batch.X_ref[:n0], batch.X_live[:n1]
    report = run_online_pfi_stream(
        X0,
        [X1],
        feature_names=batch.feature_names[: X0.shape[1]],
        decay_alpha=0.9,
        update_reference=False,
        seed=seed,
        n_perm=16,
    )
    step = report.steps[0]
    top = [batch.feature_names[i] for i in step.top_features[:5]]
    return {
        "domain_auc": step.domain_auc,
        "mmd2": step.mmd2,
        "overlap_ess": step.overlap_ess,
        "c2st_pvalue": step.c2st_pvalue,
        "top_vimp_features": top,
    }


def _eval_model(
    batch: UpliftBatch,
    learner_factory,
    *,
    loco_groups: bool = True,
    seed: int = 42,
) -> dict:
    (X_tr, t_tr, y_tr), (X_ev, t_ev, y_ev), (X_lv, t_lv, y_lv) = split_ref_live(
        batch.X_ref,
        batch.t_ref,
        batch.y_ref,
        batch.X_live,
        batch.t_live,
        batch.y_live,
        eval_frac=0.2,
        seed=seed,
    )
    p = X_tr.shape[1]
    groups = _feature_groups(p) if loco_groups else None

    def _e_fit(X, w):
        return GradientBoostingClassifier(random_state=0).fit(X, w).predict_proba(X)[:, 1]

    mon = run_uplift_monitor(
        X_tr,
        t_tr,
        y_tr,
        X_ev,
        t_ev,
        y_ev,
        X_lv,
        t_lv,
        y_lv,
        learner_factory,
        feature_names=batch.feature_names[:p],
        groups=groups,
        fit_propensity_for_overlap=_e_fit,
    )

    model = learner_factory()
    model.fit(X_tr, t_tr, y_tr)
    tau_lv = model.predict_tau(X_lv)
    integral = auuc_integral_report(y_lv, t_lv, tau_lv)

    return {
        "auuc_ref_full": mon["loco_auuc"]["auuc_ref_full"],
        "auuc_live_full": mon["loco_auuc"]["auuc_live_full"],
        "auuc_gap": mon["loco_auuc"]["auuc_gap"],
        "diagnosis": mon["diagnosis"]["label"],
        "loco_spearman_proxy": mon["diagnosis"]["loco_spearman"],
        "mmd2_x": mon["mmd2_x_ref_vs_live"],
        "overlap_ess": mon["overlap_ess_batch"],
        "integral_check": integral,
        "loco_top_live": sorted(
            mon["loco_auuc"]["loco_rows"],
            key=lambda r: -r["drop_live"],
        )[:3],
    }


def run_full_benchmark(*, quick: bool = False, with_online_pfi: bool = True) -> dict:
    results: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "notes": {
            "auuc": "Normalized trapezoid of cumulative (rate_T-rate_C)*(n_T+n_C)",
            "loco": "Group LOCO; retrain on REF after each block drop",
        },
    }
    models: List[tuple] = []
    if quick:
        models.append(("sklearn_tlearner", lambda: make_learner("tlearner")))
        models.append(
            (
                "registry_rf_t",
                lambda: make_registry_learner("tlearner", model_mu="rf_classifier"),
            )
        )
    else:
        models.append(("sklearn_xlearner", lambda: make_learner("xlearner")))
        for tag, mu, tau, e in REGISTRY_OUTCOME_MODELS:
            models.append(
                (
                    f"registry_{tag}_x",
                    lambda mu=mu, tau=tau, e=e: make_registry_learner(
                        "xlearner", model_mu=mu, model_tau=tau, model_e=e
                    ),
                )
            )

    batches = all_benchmarks()
    if quick:
        batches = [b for b in batches if b.name.startswith("synthetic")]
    for batch in batches:
        ds: dict = {"learners": {}}
        if with_online_pfi and not quick:
            ds["online_pfi"] = _online_pfi_on_batch(batch)
        elif with_online_pfi:
            ds["online_pfi"] = _online_pfi_on_batch(batch, seed=42)
        for mname, factory in models:
            try:
                ds["learners"][mname] = _eval_model(batch, factory, loco_groups=True)
            except Exception as ex:
                ds["learners"][mname] = {"error": str(ex)}
        results["datasets"][batch.name] = ds
    return results


def write_monitoring_paper(result: dict, tex_path: Path) -> None:
    """Long-form LaTeX report (integral + PFI + tables from JSON)."""
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{booktabs,amsmath}",
        r"\begin{document}",
        r"\title{LOCO--AUUC Uplift Monitoring under Batch Shift}",
        r"\date{" + result.get("generated_at_utc", "")[:10] + r"}",
        r"\maketitle",
        r"\section{AUUC integral}",
        r"$g_k=(\bar Y_{T,k}-\bar Y_{C,k})(n_{T,k}+n_{C,k})$, "
        r"$\mathrm{AUUC}=\frac{1}{n}\int_0^1 g(f)df$ (trapezoid). "
        r"Validated vs \texttt{sklift.uplift\_auc\_score(y, uplift, treatment)}.",
        r"\section{Benchmark results}",
    ]
    for ds_name, ds in result.get("datasets", {}).items():
        pfi = ds.get("online_pfi", {})
        if "skipped" not in pfi:
            lines.append(
                r"\paragraph{" + ds_name.replace("_", r"\_") + r".} "
                f"Online PFI: MMD$^2$={pfi.get('mmd2', 0):.4f}, "
                f"AUC={pfi.get('domain_auc', 0):.4f}, ESS={pfi.get('overlap_ess', 0):.3f}."
            )
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule Learner & AUUC ref & AUUC live & gap & $|\Delta$ sklift| & diagnosis \\")
        lines.append(r"\midrule")
        for lname, row in ds.get("learners", {}).items():
            if "error" in row:
                continue
            ic = row.get("integral_check", {})
            dsk = abs(ic.get("delta_vs_sklift_uplift", 0.0))
            lines.append(
                f"{lname.replace('_', r'\_')} & {row['auuc_ref_full']:.4f} & "
                f"{row['auuc_live_full']:.4f} & {row['auuc_gap']:.4f} & {dsk:.2e} & "
                f"{row['diagnosis'][:24]} \\\\"
            )
        lines.append(r"\bottomrule\end{tabular}")
    lines.extend(
        [
            r"\section{Method}",
            r"LOCO: group drop $\rightarrow$ retrain on REF $\rightarrow$ AUUC on REF eval + LIVE.",
            r"Registry: \texttt{model\_mu}, \texttt{model\_tau}, \texttt{model\_e} from \texttt{ModelRegistry}.",
            r"Hourly: \texttt{scripts/loco\_auuc\_hourly\_iterate.sh}.",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "loco_auuc_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    tex_path = out_dir / "loco_auuc_benchmark.tex"
    tex_path.write_text(_latex_from_result(result), encoding="utf-8")


def _latex_from_result(result: dict) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        r"\title{LOCO--AUUC Uplift Monitoring: Benchmark Report}",
        r"\date{" + result.get("generated_at_utc", "")[:10] + r"}",
        r"\maketitle",
        r"\section{AUUC integral}",
        r"Our score: $ \mathrm{AUUC} = \frac{1}{n}\int_0^1 g(f)\,df $ where $g(f)$ is cumulative gain",
        r"at population fraction $f$ after sorting by $\hat\tau$:",
        r"$g_k = (\bar Y_{T,k}-\bar Y_{C,k})(n_{T,k}+n_{C,k})$.",
        r"Matches \texttt{sklift.metrics.uplift\_auc\_score} up to normalization (see \texttt{integral\_check}).",
        r"\section{Results}",
    ]
    for ds_name, ds in result.get("datasets", {}).items():
        lines.append(r"\subsection{" + ds_name.replace("_", r"\_") + r"}")
        pfi = ds.get("online_pfi", {})
        lines.append(
            r"Online PFI (1 window): MMD$^2$="
            + f"{pfi.get('mmd2', 0):.4f}"
            + r", domain AUC="
            + f"{pfi.get('domain_auc', 0):.4f}"
            + r", overlap ESS="
            + f"{pfi.get('overlap_ess', 0):.3f}"
            + r"."
        )
        lines.append(r"\begin{tabular}{lrrrrl}")
        lines.append(r"\toprule")
        lines.append(r"Learner & AUUC$_{\ref}$ & AUUC$_{\live}$ & gap & diag & $\Delta$ sklift \\")
        lines.append(r"\midrule")
        for lname, row in ds.get("learners", {}).items():
            if "error" in row:
                lines.append(lname.replace("_", r"\_") + r" & \multicolumn{5}{l}{" + row["error"][:40] + r"} \\")
                continue
            ic = row.get("integral_check", {})
            delta = ic.get("delta_vs_sklift_uplift", float("nan"))
            lines.append(
                f"{lname.replace('_', r'\_')} & {row['auuc_ref_full']:.4f} & {row['auuc_live_full']:.4f} & "
                f"{row['auuc_gap']:.4f} & {row['diagnosis'][:20]} & {delta:.5f} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    lines.extend(
        [
            r"\section{Method}",
            r"LOCO: remove feature group, retrain on REF, recompute AUUC on REF holdout and LIVE.",
            r"Model Registry: swap \texttt{model\_mu}, \texttt{model\_tau}, \texttt{model\_e} components.",
            r"\end{document}",
        ]
    )
    return "\n".join(lines) + "\n"
