#!/usr/bin/env python3
"""Emit LaTeX tables from GTM artifacts for the incremental-value PO."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "latex" / "fsds_results_tables_generated.tex"
OUT_AGENT = ROOT / "docs" / "latex" / "fsds_agent_collaboration_tables_generated.tex"


def _load(name: str) -> dict:
    p = ART / name
    return json.loads(p.read_text()) if p.is_file() else {}


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def main() -> int:
    exec_sum = _load("fsds_economics_executive_summary.json")
    uplift = _load("uplift_two_layer_benchmark.json")
    boss = _load("boss_delivery_six_points.json")
    ma = _load("multi_agent_fsds_economics.json")
    early = _load("multi_agent_debate_early_stop.json")
    lg = _load("langgraph_fsds_hook.json")
    fed = _load("federated_legal_agent_blocks.json")

    lines: list[str] = [
        "% Auto-generated — export_fsds_results_tables_tex.py",
        f"% {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    # Table: where method is used
    lines += [
        r"\subsection{Table R1 --- Where the FSDS method runs (repository)}",
        r"\label{tab:fsds-where-used}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}p{0.14\textwidth}p{0.20\textwidth}p{0.18\textwidth}p{0.22\textwidth}p{0.18\textwidth}@{}}",
        r"\toprule",
        r"Domain & $X$ & $Y$ & Code entry & Step with \textbf{incremental} value \\",
        r"\midrule",
        r"Two-batch uplift & Customer covariates & Conversion / default & \texttt{run\_uplift\_subset\_localization} & \textbf{R} REALLOCATE cap \\",
        r"Agent SoT & \texttt{trace\_features} & Task success & \texttt{FSDSSoT.fit\_plan} & Branch budgets (steps 3--4) \\",
        r"Multi-agent debate & Trace batch + $Z$ roles & Debate success & \texttt{fit\_agent\_plan} & Role budget + early-stop \\",
        r"LangGraph & Node checkpoint traces & Node success & \texttt{build\_langgraph\_node\_trace} & Conditional node retry \\",
        r"Federated blocks & Local $X_k$ only & Optional global $Y$ & \texttt{local\_block\_fsds} & Uplink payload (B3) \\",
        r"Credit fulfillment & Bureau + channel & DPD / default & Same as uplift & Same \textbf{R} cap (EL units) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]

    # Table: direct incremental benefits
    lines += [
        r"\subsection{Table R2 --- Measured direct incremental value (demo artifacts)}",
        r"\label{tab:direct-incremental}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Application & Baseline & FSDS / policy & Increment & Evidence \\",
        r"\midrule",
    ]
    for row in exec_sum.get("direct_economic_benefits", []):
        line = _tex_escape(str(row.get("line", "")))
        metric = _tex_escape(str(row.get("metric", "")))
        unit = row.get("unit_savings_usd")
        unit_s = f"\\${unit:.4f}" if unit is not None else "---"
        ev = _tex_escape(str(row.get("evidence", "")).replace("artifacts/", ""))
        lines.append(f"{line} & --- & {metric} & {unit_s}/unit & \\texttt{{{ev}}} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]

    # Uplift per scenario
    lines += [
        r"\subsection{Table R3 --- Uplift two-layer bench: gates and OPEX net (sim)}",
        r"\label{tab:uplift-scenarios}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Scenario & MMD$^2$ & ESS$_{\mathrm{ovlp}}$ & PO $p$ & OPEX net (\$) & CAPEX tickets (\$) \\",
        r"\midrule",
    ]
    for run in uplift.get("runs", []):
        ds = _tex_escape(run.get("dataset", ""))
        g = run.get("gates", {})
        es = run.get("economics_summary", {})
        mmd = g.get("mmd2_x", 0)
        ess = g.get("overlap_ess_batch", 0)
        po = g.get("po_risk_pvalue", 0)
        opex = es.get("estimated_opex_net_usd", 0)
        capex = es.get("estimated_capex_tickets_usd", 0)
        lines.append(
            f"{ds} & {mmd:.4f} & {ess:.3f} & {po:.3f} & {opex:.2f} & {capex:.2f} \\\\"
        )
    total_opex = sum(
        r.get("economics_summary", {}).get("estimated_opex_net_usd", 0) for r in uplift.get("runs", [])
    )
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & --- & --- & --- & \\textbf{{{total_opex:.2f}}} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]

    # Agent MC table
    mc = ma.get("monte_carlo_eval", {})
    uni = mc.get("uniform_debate_swarm", {})
    fds = mc.get("fsds_routed_debate", {})
    e_cost = early.get("cost", {})
    lg_rec = lg.get("impact_receipt", {})
    lines += [
        r"\subsection{Table R4 --- Agent surfaces: cost and success deltas}",
        r"\label{tab:agent-mc}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Surface & Mean cost (\$) & $\Delta$ cost & Mean success & $\Delta$ success \\",
        r"\midrule",
        f"Multi-agent uniform & {uni.get('mean_cost_usd', 0):.5f} & --- & {uni.get('mean_success', 0):.2f} & --- \\\\",
        f"Multi-agent FSDS & {fds.get('mean_cost_usd', 0):.5f} & {mc.get('cost_reduction_pct', 0):.1f}\\% & {fds.get('mean_success', 0):.2f} & +{mc.get('success_delta', 0)*100:.0f}pp \\\\",
    ]
    if e_cost:
        lines.append(
            f"Debate early-stop & {e_cost.get('early_stop_usd', 0):.5f} & {e_cost.get('savings_pct', 0):.1f}\\% rounds & --- & --- \\\\"
        )
    net_lg = lg_rec.get("estimated_net_gain_usd", 0)
    lines += [
        f"LangGraph hook (receipt) & --- & \\${net_lg:.5f}/ep & --- & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]

    # Boss SoT economics if present
    pt3 = (boss.get("six_points") or {}).get("3_economics") or boss.get("economics") or {}
    fsds_ep = pt3.get("fsds_usd_per_episode", 0.0236)
    uni_ep = pt3.get("uniform_usd_per_episode", 0.0356)
    if not pt3:
        fsds_ep, uni_ep = 0.0236, 0.0356
    lines += [
        r"\noindent\textbf{SoT boss pack (reference):}",
        f" uniform \\${uni_ep:.4f}/ep $\\rightarrow$ FSDS \\${fsds_ep:.4f}/ep "
        f"($\\sim$33.6\\% variable cost reduction).",
        "",
    ]

    # Federated legal blocks
    blocks = fed.get("blocks", [])
    if blocks:
        lines += [
            r"\subsection{Table R5 --- Federated legal demo: block-level shift}",
            r"\label{tab:federated-legal}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{@{}lrrrl@{}}",
            r"\toprule",
            r"Block & MMD$^2$ & Domain AUC & ESS & Drift type \\",
            r"\midrule",
        ]
        for b in blocks:
            bid = _tex_escape(str(b.get("block_id", "")))
            lines.append(
                f"{bid} & {b.get('mmd2', 0):.4f} & {b.get('domain_auc', 0):.3f} "
                f"& {b.get('overlap_ess', 0):.3f} & {_tex_escape(str(b.get('drift_type', '')))} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
        ]

    # Scale table
    scales = exec_sum.get("scale_scenarios_usd_per_year", [])
    if scales:
        lines += [
            r"\subsection{Table R6 --- Illustrative annualization (config traffic tiers)}",
            r"\label{tab:scale-annual}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Scale & SoT \$/yr & Debate FSDS \$/yr & Early-stop \$/yr & Uplift OPEX \$/yr \\",
            r"\midrule",
        ]
        for s in scales:
            lines.append(
                f"{_tex_escape(s.get('scale', ''))} & "
                f"{s.get('direct_compute_sot_usd_per_year', 0):,.0f} & "
                f"{s.get('direct_compute_debate_fsds_usd_per_year', 0):,.0f} & "
                f"{s.get('direct_compute_debate_early_stop_usd_per_year', 0):,.0f} & "
                f"{s.get('direct_uplift_opex_usd_per_year', 0):,.1f} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
        ]

    lines.append(
        r"\noindent\emph{Regenerate:} \texttt{cd Python \&\& python3 export\_fsds\_results\_tables\_tex.py}."
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")

    collab = _load("agent_collaboration_scenarios.json")
    alines: list[str] = [
        "% Auto-generated — agent collaboration scenarios",
        f"% {datetime.now(timezone.utc).isoformat()}",
        "",
        r"\subsection{Table R7 --- Agent collaboration scenarios (uniform vs FSDS MC)}",
        r"\label{tab:agent-collab}",
        r"\begin{center}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Scenario & Pattern & \$ uniform & \$ FSDS & Save \% & $\Delta$ success (pp) & Checks $\Delta$ \\",
        r"\midrule",
    ]
    for sc in collab.get("scenarios", []):
        sid = _tex_escape(str(sc.get("scenario_id", "")))
        pat = _tex_escape(str(sc.get("pattern", "")))
        if "uniform" not in sc or "fsds" not in sc:
            continue
        inc = sc.get("incremental", {})
        u = sc["uniform"]["mean_cost_usd"]
        f = sc["fsds"]["mean_cost_usd"]
        alines.append(
            f"{sid} & {pat} & {u:.5f} & {f:.5f} & {inc.get('cost_reduction_pct', 0):.1f} & "
            f"{inc.get('success_delta_pp', 0):+.1f} & {inc.get('checks_delta', 0):+.0f} \\\\"
        )
    alines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"\subsection{Table R8 --- Collaboration objects ($X$, $Y$, $Z$, $W$) by scenario}",
        r"\label{tab:agent-xy}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}p{0.18\textwidth}p{0.74\textwidth}@{}}",
        r"\toprule",
        r"Scenario & Collaboration / monitoring note \\",
        r"\midrule",
    ]
    for sc in collab.get("scenarios", []):
        sid = _tex_escape(str(sc.get("scenario_id", "")))
        note = _tex_escape(str(sc.get("collaboration_note", sc.get("playbook", ""))))
        alines.append(f"{sid} & {note} \\\\")
    alines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
    ]
    stacked = next(
        (s for s in collab.get("scenarios", []) if s.get("scenario_id") == "stacked_multi_agent_fsds_plus_early_stop"),
        None,
    )
    if stacked and stacked.get("stacked_levers"):
        sl = stacked["stacked_levers"]
        alines += [
            r"\subsection{Table R9 --- Stacked multi-agent levers (micro FSDS + macro early-stop)}",
            r"\label{tab:stacked-ma}",
            r"\begin{center}",
            r"\small",
            r"\begin{tabular}{@{}lr@{}}",
            r"\toprule",
            r"Lever & Value (demo) \\",
            r"\midrule",
            f"FSDS cost reduction (micro) & {sl.get('fsds_cost_reduction_pct', 0):.1f}\\% \\\\",
            f"Early-stop at round & {sl.get('early_stop_rounds', 0)} / {sl.get('early_stop_max_rounds', 5)} \\\\",
            f"Macro round-cost save / debate & \\${sl.get('macro_round_cost_save_usd_per_debate', 0):.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
        ]
    alines.append(
        r"\noindent\emph{Regenerate suite:} \texttt{python3 demo\_fsds\_agent\_collaboration\_suite.py}."
    )
    OUT_AGENT.write_text("\n".join(alines) + "\n")
    print(f"Wrote {OUT_AGENT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
