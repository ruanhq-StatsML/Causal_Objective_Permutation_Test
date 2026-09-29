"""MCTS attribution, then a gate across ToT, SoT, and CoT.

One row is one finished search. ToT records child values and visit shares.
SoT records the three child values. CoT records the root value.

The concept plane is ``po_risk`` (mean of tau squared). The covariate plane is
MMD leave-one-column-out. Overlap is the mean of ``2 * min(e, 1-e)`` from the
logistic propensity. The gate follows the ToT policy table: a value-head
concept signal stays on ToT and re-calibrates; a visit-share covariate signal
stays on ToT and changes the branching factor; overlap below 0.3 hands the
next request to CoT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcts_porisk_attribution import (  # noqa: E402
    VALUE_COLS,
    _matrix,
    _rows,
    po_risk,
    porisk_loco,
)
from mma_wrapper import MMD  # noqa: E402

SOT_COLS = ["q_A", "q_B", "q_C"]
OVERLAP_FLOOR = 0.3


def overlap_score(X, W) -> float:
    """Mean of ``2 * min(e, 1-e)``. A shared support scores near 1."""
    W = np.asarray(W, dtype=int).ravel()
    e = LogisticRegression(max_iter=500).fit(X, W).predict_proba(X)[:, 1]
    e = np.clip(e, 0.02, 0.98)
    return float(np.mean(2.0 * np.minimum(e, 1.0 - e)))


def mmd_loco(X, W, names: list) -> dict:
    score = MMD()
    drops = score.MMD_LOCO(X, W)
    order = np.argsort(-drops)
    return {
        "mmd": float(score(X, W)),
        "loco": [(names[int(j)], float(drops[int(j)])) for j in order],
    }


def agent_report(name: str, X, Y, W, cols: list) -> dict:
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 1:
        full_po = po_risk(X, Y, W)
        full_mmd = float(MMD()(X, W))
        po = {"porisk": full_po, "loco": [(cols[0], full_po)]}
        mm = {"mmd": full_mmd, "loco": [(cols[0], full_mmd)]}
    else:
        po = porisk_loco(X, Y, W, cols)
        mm = mmd_loco(X, W, cols)
    return {
        "agent": name,
        "overlap": overlap_score(X, W),
        "porisk": po["porisk"],
        "po_top": po["loco"][0][0],
        "po_drop": po["loco"][0][1],
        "mmd": mm["mmd"],
        "mmd_top": mm["loco"][0][0],
        "mmd_drop": mm["loco"][0][1],
        "po": po,
        "mm": mm,
    }


def gate(tot: dict) -> dict:
    """Pick the agent for the next request from the ToT report."""
    if tot["overlap"] < OVERLAP_FLOOR:
        action, owner = "fall back to the chain", "CoT"
    elif tot["porisk"] > 1e-4 and tot["po_top"].startswith("q_"):
        action, owner = "re-calibrate the value head", "ToT"
    elif tot["mmd"] > 0.05 and tot["mmd_top"].startswith("p_"):
        action, owner = "adjust the branching factor", "ToT"
    else:
        action, owner = "re-plan the skeleton", "SoT"
    return {"agent": owner, "action": action}


def _print_agent(rep: dict) -> None:
    print(
        f"  {rep['agent']:3s}  overlap={rep['overlap']:.3f}  "
        f"PO-risk={rep['porisk']:.6f} LOCO {rep['po_top']} {rep['po_drop']:+.6f}  "
        f"MMD={rep['mmd']:+.4f} LOCO {rep['mmd_top']} {rep['mmd_drop']:+.4f}"
    )


def _batch(rows, cols):
    return _matrix(rows, cols)


def scenario_value_head() -> dict:
    """Shared searches. The new batch's node value picks up a slope on q_B."""
    rows = _rows(160, bias_b=0.0, c_uct=1.3, seed=20)
    w = np.random.default_rng(200).integers(0, 2, size=len(rows))
    value = _batch(rows, ["value"])[:, 0]
    q_b = _batch(rows, ["q_B"])[:, 0]
    y = value + w * 4.0 * (q_b - q_b.mean())
    tot = agent_report("ToT", _batch(rows, VALUE_COLS), y, w, VALUE_COLS)
    sot = agent_report("SoT", _batch(rows, SOT_COLS), y, w, SOT_COLS)
    cot = agent_report("CoT", value[:, None], y, w, ["value"])
    # Recalibrating removes the extra slope. The same searches, Y = node value.
    calmed = porisk_loco(_batch(rows, VALUE_COLS), value, w, VALUE_COLS)
    return {
        "title": "value head picks up q_B",
        "tot": tot, "sot": sot, "cot": cot,
        "choice": gate(tot),
        "after": calmed["porisk"],
    }


def scenario_branching() -> dict:
    """The live batch searches with a larger UCT constant. Y stays the node value."""
    ref = _rows(80, bias_b=0.0, c_uct=1.3, seed=0)
    live = _rows(80, bias_b=0.0, c_uct=2.8, seed=1)
    rows = ref + live
    w = np.array([0] * len(ref) + [1] * len(live))
    y = _batch(rows, ["value"])[:, 0]
    tot = agent_report("ToT", _batch(rows, VALUE_COLS), y, w, VALUE_COLS)
    sot = agent_report("SoT", _batch(rows, SOT_COLS), y, w, SOT_COLS)
    cot = agent_report("CoT", y[:, None], y, w, ["value"])
    return {
        "title": "UCT constant 1.3 then 2.8",
        "tot": tot, "sot": sot, "cot": cot,
        "choice": gate(tot),
        "after": None,
    }


def scenario_separated() -> dict:
    """A backup bias at B moves the whole cloud. Overlap collapses."""
    ref = _rows(80, bias_b=0.0, c_uct=1.3, seed=0)
    live = _rows(80, bias_b=1.6, c_uct=1.3, seed=1)
    rows = ref + live
    w = np.array([0] * len(ref) + [1] * len(live))
    y = _batch(rows, ["value"])[:, 0]
    tot = agent_report("ToT", _batch(rows, VALUE_COLS), y, w, VALUE_COLS)
    sot = agent_report("SoT", _batch(rows, SOT_COLS), y, w, SOT_COLS)
    cot = agent_report("CoT", y[:, None], y, w, ["value"])
    return {
        "title": "backup bias 1.6 at B",
        "tot": tot, "sot": sot, "cot": cot,
        "choice": gate(tot),
        "after": None,
    }


def _print_scenario(sc: dict) -> None:
    print(sc["title"])
    _print_agent(sc["tot"])
    _print_agent(sc["sot"])
    _print_agent(sc["cot"])
    choice = sc["choice"]
    print(f"  gate -> {choice['agent']}  {choice['action']}")
    if sc["after"] is not None:
        print(f"  after re-calibration  PO-risk={sc['after']:.6f}")


def main() -> None:
    print("agents: ToT = search table, SoT = child values, CoT = root value")
    value_head = scenario_value_head()
    branching = scenario_branching()
    separated = scenario_separated()
    _print_scenario(value_head)
    _print_scenario(branching)
    _print_scenario(separated)

    assert value_head["choice"]["agent"] == "ToT"
    assert value_head["choice"]["action"] == "re-calibrate the value head"
    assert value_head["tot"]["po_top"] == "q_B"
    assert value_head["after"] < value_head["tot"]["porisk"] / 10.0
    assert branching["choice"]["agent"] == "ToT"
    assert branching["choice"]["action"] == "adjust the branching factor"
    assert branching["tot"]["mmd_top"].startswith("p_")
    assert separated["tot"]["overlap"] < OVERLAP_FLOOR
    assert separated["choice"]["agent"] == "CoT"
    print("MCTS_MULTIAGENT_OK")


if __name__ == "__main__":
    main()
