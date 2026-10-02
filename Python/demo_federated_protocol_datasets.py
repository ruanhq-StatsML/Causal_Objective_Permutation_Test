#!/usr/bin/env python3
"""Multi-dataset federated FSDS protocol + communication efficiency + LaTeX export."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fsds_sot.federated_protocol import (  # noqa: E402
    FederatedAttributionProtocol,
    FederatedNode,
    FederatedProtocolConfig,
    partition_feature_blocks,
)

ART = ROOT.parent / "artifacts"
DOCS = ROOT.parent / "docs" / "latex"

FLOAT_BYTES = 8  # float64 accounting


def _ref_live(
    X: np.ndarray,
    *,
    live_frac: float = 0.4,
    shift_cols: slice,
    shift_mag: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_ref, X_live = train_test_split(X, test_size=live_frac, random_state=seed)
    X_live = X_live.copy()
    if shift_cols.stop > shift_cols.start:
        X_live[:, shift_cols] += shift_mag
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(X_ref)
    X_live = scaler.transform(X_live)
    return X_ref, X_live


def _comm_stats(
    n_ref: int,
    n_live: int,
    p: int,
    n_nodes: int,
    block_sizes: List[int],
) -> Dict[str, float]:
    """Uplink floats vs naive centralization of X_ref and X_live."""
    uplink = sum(3 * bk + 3 for bk in block_sizes)
    raw = (n_ref + n_live) * p
    return {
        "uplink_floats": float(uplink),
        "raw_matrix_floats": float(raw),
        "efficiency_ratio": float(raw / max(uplink, 1)),
        "uplink_kib": uplink * FLOAT_BYTES / 1024,
        "raw_mib": raw * FLOAT_BYTES / (1024 * 1024),
    }


def _top_feature_index(name: str) -> Optional[int]:
    if name.startswith("f") and name[1:].isdigit():
        return int(name[1:])
    return None


def _run_dataset(
    name: str,
    X: np.ndarray,
    feature_names: List[str],
    *,
    shift_cols: slice,
    shift_mag: float,
    seed: int,
    n_nodes: int = 5,
) -> Dict[str, Any]:
    X_ref, X_live = _ref_live(X, shift_cols=shift_cols, shift_mag=shift_mag, seed=seed)
    blocks = partition_feature_blocks(X.shape[1], n_nodes)
    block_sizes = [len(b) for b in blocks]
    nodes = [
        FederatedNode(f"node_{i}", blk, feature_names)
        for i, blk in enumerate(blocks)
    ]
    proto = FederatedAttributionProtocol(
        nodes,
        FederatedProtocolConfig(secure_aggregation=False, fdr_alpha=0.05),
    )
    result = proto.run(X_ref, X_live)
    actions = result.closed_loop.actions
    top1 = result.global_ranking[0][0] if result.global_ranking else ""
    tidx = _top_feature_index(top1)
    shift_hit = (
        tidx is not None
        and shift_cols.start <= tidx < shift_cols.stop
        if shift_cols.stop > shift_cols.start
        else None
    )
    return {
        "dataset": name,
        "n_features": X.shape[1],
        "n_ref": X_ref.shape[0],
        "n_live": X_live.shape[0],
        "n_nodes": len(nodes),
        "shift_cols": [shift_cols.start, shift_cols.stop],
        "top1_feature": top1,
        "top1_in_shift_block": shift_hit,
        "top_global_features": result.global_ranking[:8],
        "n_fdr_significant": len(result.closed_loop.significant),
        "action_counts": dict(Counter(actions.values())),
        "sample_actions": {f: actions[f] for f in result.closed_loop.significant[:6]},
        "drift_types_among_significant": {
            f: result.closed_loop.drift_types[f]
            for f in result.closed_loop.significant[:6]
        },
        "server_notes": result.server_notes,
        "communication": _comm_stats(
            X_ref.shape[0], X_live.shape[0], X.shape[1], len(nodes), block_sizes
        ),
    }


def _load_ionosphere() -> Tuple[np.ndarray, List[str]]:
    bunch = fetch_openml("ionosphere", version=1, as_frame=False, parser="auto")
    X = np.asarray(bunch.data, dtype=float)
    return X, [f"f{j}" for j in range(X.shape[1])]


def _load_heart() -> Tuple[np.ndarray, List[str]]:
    bunch = fetch_openml("heart-statlog", version=1, as_frame=False, parser="auto")
    X = np.asarray(bunch.data, dtype=float)
    return X, [f"f{j}" for j in range(X.shape[1])]


def _tex_dict(d: Dict[str, int]) -> str:
    return ", ".join(f"\\textsc{{{k.replace('_', '\\\\_')}}}={v}" for k, v in d.items())


def write_protocol_results_tex(runs: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "% Auto-generated — demo_federated_protocol_datasets.py",
        "\\section{Empirical validation across tabular benchmarks}",
        "",
        "\\paragraph{Summary (communication + localization).}",
        "Each run simulates vertical FL: REF/LIVE split, covariate injection on a contiguous feature block, "
        "then the three-phase protocol with BH-FDR ($\\alpha{=}0.05$). "
        "\\textbf{Eff} is the ratio of raw centralized matrix elements $(n_{\\mathrm{ref}}+n_{\\mathrm{live}})\\times p$ "
        "to total uplink scalars $\\sum_k(3|F_k|+3)$. "
        "\\textbf{Top-1 hit} indicates whether the highest global attribution score falls inside the injected shift block.",
        "",
        "\\begin{center}",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrrl@{}}",
        "\\toprule",
        "Dataset & $p$ & nodes & FDR sig & Eff & uplink KiB & top-1 & hit \\\\",
        "\\midrule",
    ]
    for r in runs:
        ds = r["dataset"].replace("_", "\\_")
        comm = r["communication"]
        hit = "--"
        if r["top1_in_shift_block"] is True:
            hit = "yes"
        elif r["top1_in_shift_block"] is False:
            hit = "no"
        top1 = r.get("top1_feature", "").replace("_", "\\_")
        lines.append(
            f"{ds} & {r['n_features']} & {r['n_nodes']} & {r['n_fdr_significant']} & "
            f"{comm['efficiency_ratio']:.0f}$\\times$ & {comm['uplink_kib']:.2f} & "
            f"\\texttt{{{top1}}} & {hit} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
            "",
        ]
    )

    for r in runs:
        ds = r["dataset"].replace("_", "\\_")
        comm = r["communication"]
        lines.append(f"\\subparagraph{{{ds}.}}")
        lines.append(
            f"$n_{{\\mathrm{{ref}}}}+n_{{\\mathrm{{live}}}}={r['n_ref']}+{r['n_live']}$, "
            f"shift columns $[{r['shift_cols'][0]},\\,{r['shift_cols'][1]})$. "
            f"Communication: {comm['uplink_floats']:.0f} floats uplink ({comm['uplink_kib']:.2f} KiB) vs "
            f"{comm['raw_matrix_floats']:.0f} raw matrix floats ({comm['raw_mib']:.3f} MiB if centralized)---"
            f"\\textbf{{{comm['efficiency_ratio']:.0f}$\\times$}} fewer scalars over the wire. "
            f"FDR significant: {r['n_fdr_significant']}; actions: {_tex_dict(r['action_counts'])}."
        )
        if r["top_global_features"]:
            top = r["top_global_features"][:5]
            fmt = ", ".join(
                "\\texttt{" + t[0].replace("_", "\\_") + "}" + f" ({t[1]:.2f})" for t in top
            )
            lines.append(f"Top global scores: {fmt}.")
        if r["sample_actions"]:
            lines.append("\\begin{itemize}\\itemsep1pt")
            for f, act in r["sample_actions"].items():
                fn = f.replace("_", "\\_")
                dt = r["drift_types_among_significant"].get(f, "?")
                act_tex = act.replace("_", "\\_")
                lines.append(
                    f"\\item \\texttt{{{fn}}}: type={dt}, action=\\textsc{{{act_tex}}}."
                )
            lines.append("\\end{itemize}")
        lines.append("")

    lines.append(
        "\\paragraph{Monitoring metrics (targets).} "
        "Drift-type accuracy under injection, FDR $\\approx \\alpha$, top-$k$ hit rate in shifted blocks, "
        "action precision, one-window closed-loop latency. Secure aggregation is orthogonal to Eff and left to production."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    bc = load_breast_cancer()
    wine = load_wine()
    dia = load_diabetes()
    X_ion, names_ion = _load_ionosphere()
    X_hrt, names_hrt = _load_heart()

    specs = [
        ("WDBC_breast_cancer", bc.data.astype(float), [f"f{j}" for j in range(bc.data.shape[1])],
         slice(10, 16), 2.5, 42, 5),
        ("UCI_wine", wine.data.astype(float), [f"f{j}" for j in range(wine.data.shape[1])],
         slice(4, 8), 1.8, 7, 3),
        ("UCI_diabetes", dia.data.astype(float), [f"f{j}" for j in range(dia.data.shape[1])],
         slice(3, 7), 2.0, 11, 3),
        ("UCI_ionosphere", X_ion, names_ion, slice(12, 20), 1.5, 3, 5),
        ("UCI_heart_statlog", X_hrt, names_hrt, slice(4, 9), 2.2, 17, 3),
    ]

    runs = [
        _run_dataset(name, X, names, shift_cols=sc, shift_mag=mag, seed=seed, n_nodes=n_nodes)
        for name, X, names, sc, mag, seed, n_nodes in specs
    ]

    ART.mkdir(exist_ok=True)
    json_path = ART / "federated_protocol_benchmark.json"
    with open(json_path, "w") as f:
        json.dump({"runs": runs}, f, indent=2)

    tex_results = DOCS / "federated_fsds_protocol_results.tex"
    write_protocol_results_tex(runs, tex_results)

    print("Dataset summary:")
    for r in runs:
        c = r["communication"]
        print(
            f"  {r['dataset']}: Eff={c['efficiency_ratio']:.0f}x, "
            f"FDR={r['n_fdr_significant']}, top1={r['top1_feature']}, hit={r['top1_in_shift_block']}"
        )
    print("Wrote", json_path)
    print("Wrote", tex_results)
    print("Compile: docs/latex/federated_fsds_comprehensive_standalone.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
