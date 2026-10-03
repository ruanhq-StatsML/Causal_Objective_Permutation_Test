#!/usr/bin/env python3
"""Export tonight-landing checklist: scenarios with direct vs deferred economic benefit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CFG = ROOT / "config" / "fsds_economics_assumptions.json"
OUT = ART / "fsds_tonight_landing_checklist.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _annual(save_per_unit: float, units_per_month: float) -> float:
    return round(save_per_unit * units_per_month * 12, 2)


def main() -> int:
    cfg = _load(CFG)
    agent = cfg.get("agent_unit_economics_from_demos", {})
    traffic = cfg.get("traffic", {})
    uplift_cfg = cfg.get("uplift", {})

    boss = _load(ART / "boss_delivery_six_points.json")
    debate = _load(ART / "multi_agent_fsds_economics.json")
    early = _load(ART / "multi_agent_debate_early_stop.json")
    lang = _load(ART / "langgraph_fsds_hook.json")
    uplift_bench = _load(ART / "uplift_two_layer_benchmark.json")

    so_save = float(agent.get("so_direct_save_usd_per_ep", 0.012))
    debate_save = float(agent.get("debate_fsds_save_usd_per_debate", 0.0081))
    early_save = float(agent.get("debate_early_stop_save_usd_per_debate", 0.0088))

    uplift_runs = uplift_bench.get("runs", [])
    opex_total = sum(
        r.get("economics_summary", {}).get("estimated_opex_net_usd", 0) for r in uplift_runs
    )
    opex_avg = round(opex_total / len(uplift_runs), 2) if uplift_runs else 0.0

    tiers = [
        {
            "tier": "T0",
            "label_zh": "今晚可接：已有 LLM/API 账单或 agent trace",
            "label_en": "Land tonight on existing API billing or trace logs",
            "scenarios": [
                {
                    "id": "agent_sot_closed_loop",
                    "title_zh": "单 agent SoT：关键路径预算 + impact_receipt",
                    "title_en": "Agent SoT critical-path budget + impact receipt",
                    "direct_economic_benefit": True,
                    "benefit_type": "OPEX_compute",
                    "unit_save_usd": so_save,
                    "demo_pct": agent.get("so_cost_reduction_pct"),
                    "evidence_artifact": "artifacts/boss_delivery_six_points.json",
                    "validate_cmd": "cd Python && python3 -c \"from fsds_sot.closed_loop import suggest_intervention\"",
                    "prod_hook": [
                        "REF/LIVE 窗口：每 episode 写 trace_features + success",
                        "Sidecar: fit_agent_plan → suggest_intervention(..., attach_impact_receipt=True)",
                        "Scheduler 按 intervention.budgets 改 tier/token，日志落 impact_receipt",
                    ],
                    "metrics_24h": [
                        "sum(impact_receipt.estimated_net_gain_usd)",
                        "mean(usd_per_episode[1]) vs baseline[0]",
                    ],
                    "annual_usd_if_traffic": {
                        k: _annual(so_save, v.get("agent_episodes_per_month", 0))
                        for k, v in traffic.items()
                    },
                },
                {
                    "id": "multi_agent_fsds_debate",
                    "title_zh": "多 agent 辩论：按 role 重分配 token/tier",
                    "title_en": "Multi-agent debate: FSDS role-level budget",
                    "direct_economic_benefit": True,
                    "benefit_type": "OPEX_compute",
                    "unit_save_usd": debate_save,
                    "demo_pct": debate.get("economics_from_plan", {}).get("cost_savings_pct")
                    or agent.get("debate_cost_reduction_pct"),
                    "evidence_artifact": "artifacts/multi_agent_fsds_economics.json",
                    "validate_cmd": "cd Python && python3 demo_fsds_multi_agent_economics.py",
                    "prod_hook": [
                        "每轮存 role embedding (pro/con/judge)",
                        "fit_agent_plan on LIVE drift → budgets per role",
                        "impact_receipt 进 finance/FinOps 流水",
                    ],
                    "metrics_24h": ["debates_with_receipt / total_debates", "mean cost_savings_pct"],
                    "annual_usd_if_traffic": {
                        k: _annual(debate_save, v.get("debates_per_month", 0))
                        for k, v in traffic.items()
                    },
                },
                {
                    "id": "debate_early_stop",
                    "title_zh": "辩论 early-stop：离散度达标停轮",
                    "title_en": "Debate early-stop on dispersion gate",
                    "direct_economic_benefit": True,
                    "benefit_type": "OPEX_compute",
                    "unit_save_usd": early_save,
                    "demo_pct": early.get("debate_cost_savings_pct"),
                    "evidence_artifact": "artifacts/multi_agent_debate_early_stop.json",
                    "validate_cmd": "cd Python && python3 demo_fsds_multi_agent_debate_early_stop.py",
                    "prod_hook": [
                        "debate_early_stop_round(round_role_embs, dispersion_target=...)",
                        "与 FSDS 预算叠加（macro 轮次 + micro token）",
                    ],
                    "metrics_24h": ["mean rounds saved", "early_stop_rate"],
                    "annual_usd_if_traffic": {
                        k: _annual(early_save, v.get("debates_per_month", 0))
                        for k, v in traffic.items()
                    },
                },
                {
                    "id": "langgraph_checkpoint_fsds",
                    "title_zh": "LangGraph：checkpoint embedding 只重跑漂移节点",
                    "title_en": "LangGraph: retry drifted nodes only",
                    "direct_economic_benefit": True,
                    "benefit_type": "OPEX_compute",
                    "unit_save_usd": float(
                        lang.get("impact_receipt", {}).get("estimated_net_gain_usd", 0.015)
                    ),
                    "evidence_artifact": "artifacts/langgraph_fsds_hook.json",
                    "validate_cmd": "cd Python && python3 demo_fsds_langgraph_hook.py",
                    "prod_hook": [
                        "checkpoint 写 {node_id, embedding, W}",
                        "build_langgraph_node_trace + fit_agent_plan",
                        "conditional edge 读 impact_receipt.shift_summary.mmd2",
                    ],
                    "metrics_24h": ["partial_graph_replays_avoided", "tool_call retries only"],
                    "annual_usd_if_traffic": {
                        k: _annual(so_save, v.get("agent_episodes_per_month", 0))
                        for k, v in traffic.items()
                    },
                },
            ],
        },
        {
            "tier": "T1",
            "label_zh": "今晚可跑第一窗：需真实 margin / 治疗成本 + REF/LIVE 导出",
            "label_en": "First monitor window tonight: real margin/treat cost + REF/LIVE export",
            "scenarios": [
                {
                    "id": "uplift_cap_opex",
                    "title_zh": "Uplift 两批：REALLOCATE 封顶（OPEX，非立刻重训）",
                    "title_en": "Two-batch uplift: REALLOCATE caps (OPEX, defer retrain)",
                    "direct_economic_benefit": True,
                    "benefit_type": "OPEX_treatment_spend",
                    "config_keys": [
                        "uplift.margin_usd_per_conversion",
                        "uplift.treatment_cost_usd",
                        "uplift.treat_rate",
                    ],
                    "config_current": uplift_cfg,
                    "bench_opex_net_total_usd_sim": opex_total,
                    "bench_opex_net_avg_per_window_sim": opex_avg,
                    "evidence_artifact": "artifacts/uplift_two_layer_benchmark.json",
                    "validate_cmd": "cd Python && python3 run_uplift_subset_benchmark.py --n-perm 32",
                    "prod_hook": [
                        "导出 Hillstrom 式 CSV：X, Y, T + REF/LIVE 切窗",
                        "run_uplift_subset_localization → business_rules + impact_receipts",
                        "营销执行层只应用 econ_bucket=opex 的 cap；RELEARN=CAPEX 工单",
                    ],
                    "metrics_24h": [
                        "rules_applied with receipt rule_id",
                        "estimated_opex_net_usd (real treat volume)",
                    ],
                    "annual_usd_if_traffic": {
                        k: round(
                            opex_avg * v.get("uplift_monitor_windows_per_year", 12),
                            2,
                        )
                        for k, v in traffic.items()
                    },
                },
            ],
        },
        {
            "tier": "T2",
            "label_zh": "战略 / 合规：通常不是今晚 OPEX，但可今晚接监控收据",
            "label_en": "Strategic/compliance: rarely tonight OPEX; can wire receipts tonight",
            "scenarios": [
                {
                    "id": "federated_legal_blocks",
                    "title_zh": "联邦 legal/finance/product block 漂移",
                    "title_en": "Federated legal block drift monitors",
                    "direct_economic_benefit": False,
                    "benefit_type": "revenue_unlock_or_risk_avoidance",
                    "strategic_sim_quarter_usd": cfg.get("strategic_sim", {}).get(
                        "revenue_unlock_usd_quarter"
                    ),
                    "evidence_artifact": "artifacts/federated_legal_agent_blocks.json",
                    "validate_cmd": "cd Python && python3 demo_federated_legal_agent_blocks.py",
                    "prod_hook": [
                        "各 silo 只上传 block 级统计 + impact_receipt",
                        "全局 campaign 按 block 暂停而非整站关停",
                    ],
                    "metrics_24h": ["blocks_with_shift", "campaigns_unpaused_vs_full_stop"],
                },
            ],
        },
    ]

    not_tonight = [
        "全量重训 uplift 模型（RELEARN）—— CAPEX；需 PO/AUUC_ovlp 阶梯试跑后再批预算",
        "无 trace 时的纯论文仿真—— 不产生账单 delta",
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readme_zh": "T0=接 API/trace 即省钱；T1=填 config + 一窗 uplift；T2=合规叙事+收据",
        "readme_en": "T0=savings on API/trace; T1=config + one uplift window; T2=compliance receipts",
        "boss_sot_reference": boss.get("six_points", {}).get("5_economics", {}),
        "tiers": tiers,
        "explicitly_not_tonight_opex": not_tonight,
        "one_command_repro": "cd Python && python3 run_fsds_business_impact_pack.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"T0 scenarios: {len(tiers[0]['scenarios'])} direct-OPEX tracks")
    print(f"T1 uplift sim avg/window: ${opex_avg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
