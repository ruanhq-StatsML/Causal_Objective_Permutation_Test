#!/usr/bin/env python3
"""One-page slide per scenario (data digestion + economics), EN and ZH PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "fsds_scenario_slides"
W, H = 13.333, 7.5
BLUE, LBLUE = "#2563eb", "#dbeafe"
GREEN, LGREEN = "#059669", "#ecfdf5"
ORANGE, LORANGE = "#ea580c", "#ffedd5"
PURPLE, LPURPLE = "#7c3aed", "#ede9fe"
DARK, GRAY = "#0f172a", "#64748b"

SCENARIOS = [
    {
        "id": "01_cross_border_copilot",
        "title_en": "Scenario 1 · Cross-border sales copilot (federated blocks)",
        "title_zh": "场景 1 · 跨境销售 Copilot（联邦特征块）",
        "digest_en": (
            "REF/LIVE on trace + block features\n"
            "• Silos: Legal | Finance | Product (vertical partitions)\n"
            "• Local: MMD², domain AUC(W), ESS, top VIMP per block\n"
            "• Uplink: O(|F_k|) envelope — no raw X centralization\n"
            "• Server: merge rank → drift_type → OFS / narrow legal path\n"
            "• Agents: multi-role debate; block shift ≠ global kill switch"
        ),
        "digest_zh": (
            "REF/LIVE 双批次 + 分块特征\n"
            "• 三 silo：法务 | 财务 | 产品（垂直切分）\n"
            "• 本地：MMD²、域 AUC(W)、ESS、块内 VIMP\n"
            "• 上行：O(|F_k|) 包络，不集中原始 X\n"
            "• 服务端：合并排序 → drift 类型 → 块级 OFS / 收紧法务链\n"
            "• 多 agent：块 drift ≠ 全站下线"
        ),
        "econ_en": (
            "Direct OPEX\n"
            "• Target legal block only → less rework & re-debate\n"
            "• Multi-agent ~39% $/debate; early-stop ~20% rounds (demos)\n"
            "Incremental (sim)\n"
            "• Revenue unlock ~$125k/quarter vs hard shutdown\n"
            "• Block receipts → audit (MMD, rule_id)\n"
            "Justify: demo_federated_legal_agent_blocks.py"
        ),
        "econ_zh": (
            "直接 OPEX\n"
            "• 只动法务块 → 少 rework / 少无效辩论\n"
            "• 多 agent ~39% $/debate；早停 ~20% 轮次（demo）\n"
            "增量（sim）\n"
            "• 可控上线 vs 硬停：叙事 ~$125k/季度\n"
            "• 块级 receipt → 合规可签字\n"
            "依据：demo_federated_legal_agent_blocks.py"
        ),
        "fc": LPURPLE,
        "ec": PURPLE,
    },
    {
        "id": "02_bank_insurance_fl",
        "title_en": "Scenario 2 · Bank / insurance federated monitoring",
        "title_zh": "场景 2 · 银行 / 保险联邦监控",
        "digest_en": (
            "Data digestion without pooling X\n"
            "• Each legal entity: local REF/LIVE on feature block F_k\n"
            "• Batch label W; outcome Y for optional DRPerm / PO-risk\n"
            "• Upload: 3|F_k|+3 scalars/round (protocol bench)\n"
            "• Server: Fisher/Stouffer + online FDR on block hypotheses\n"
            "• Typing: covariate vs concept vs compound per block"
        ),
        "digest_zh": (
            "不并表 X 的数据消化\n"
            "• 各法人：块 F_k 上本地 REF/LIVE\n"
            "• 批次 W；可选 Y 做 DRPerm / PO-risk\n"
            "• 上行：每轮 3|F_k|+3 标量（协议 bench）\n"
            "• 服务端：p 值合并 + 在线 FDR\n"
            "• 块级：协变量 / 概念 / 复合 drift 分型"
        ),
        "econ_en": (
            "Direct (infra)\n"
            "• Comm Eff 48×–1410× vs full matrix uplink\n"
            "• e.g. Spambase: 1.45 KiB vs ~2 MiB / window\n"
            "Incremental\n"
            "• Run monitoring where centralization is banned\n"
            "• Top-1 block hit under injection (bench)\n"
            "Justify: federated_fsds_protocol_results.tex"
        ),
        "econ_zh": (
            "直接（基础设施）\n"
            "• 通信 Eff 48×–1410× vs 传全矩阵\n"
            "• 例 Spambase：1.45 KiB vs ~2 MiB / 窗\n"
            "增量\n"
            "• 禁并表仍可监控 → 项目可立项\n"
            "• 注入实验 top-1 块命中（bench）\n"
            "依据：federated_fsds_protocol_results.tex"
        ),
        "fc": LBLUE,
        "ec": BLUE,
    },
    {
        "id": "03_group_uplift",
        "title_en": "Scenario 3 · Group uplift / campaign (two-batch)",
        "title_zh": "场景 3 · 集团 Uplift / 投流（两批次）",
        "digest_en": (
            "REF calibrates τ̂; LIVE scored frozen\n"
            "• Gates: SRM, MMD²(X), ESS on ê(W|X), domain AUC\n"
            "• L1: AUUC_global vs AUUC_ovlp → mix vs ranking\n"
            "• L2: slice ATE; PO-risk on REF∪LIVE (concept gate)\n"
            "• Output: business_rules P1–P5 + impact_receipts\n"
            "• BU / quintile = LOGO groups (federated read)"
        ),
        "digest_zh": (
            "REF 校准 τ̂；LIVE 冻结打分\n"
            "• 闸门：SRM、MMD²(X)、ESS、域 AUC\n"
            "• L1：AUUC_global vs AUUC_ovlp → 混部 vs 排序\n"
            "• L2：分 slice ATE；PO-risk 概念门控\n"
            "• 输出：business_rules + impact_receipts\n"
            "• 事业部 / 分位 = LOGO 组（联邦视角）"
        ),
        "econ_en": (
            "Direct OPEX (sim)\n"
            "• Cap bad quintiles: ~$818/window avg (4 benches)\n"
            "• Total sim OPEX ~$3.3k across scenarios\n"
            "CAPEX separate: RELEARN tickets ~$10k (not in OPEX net)\n"
            "Incremental: defer retrain until cap trial fails + PO reject\n"
            "Justify: run_uplift_subset_benchmark.py"
        ),
        "econ_zh": (
            "直接 OPEX（sim）\n"
            "• 差分位 cap：四场景均值 ~$818/窗\n"
            "• 合计 sim OPEX ~$3.3k\n"
            "CAPEX 单列：RELEARN ~$10k（不进 OPEX net）\n"
            "增量：cap 试验失败 + PO 拒绝才 relearn\n"
            "依据：run_uplift_subset_benchmark.py"
        ),
        "fc": LORANGE,
        "ec": ORANGE,
    },
    {
        "id": "04_agent_platform",
        "title_en": "Scenario 4 · Agent platform (multi-agent + LangGraph)",
        "title_zh": "场景 4 · Agent 平台（多角色 + LangGraph）",
        "digest_en": (
            "Segment = role utterance OR graph checkpoint embed\n"
            "• REF/LIVE on episode trace features (tool, depth, mix)\n"
            "• Object 2: L, tier, checks from shift×quality×need\n"
            "• Debate: round-centroid dispersion → early-stop\n"
            "• LangGraph: re-run drifted nodes (e.g. tool_call) only\n"
            "• Closed loop: suggest_intervention + impact_receipt"
        ),
        "digest_zh": (
            "段 = 角色发言 或 图 checkpoint 向量\n"
            "• REF/LIVE 轨迹特征（工具、深度、混部）\n"
            "• Object 2：shift×quality×need → L/tier/checks\n"
            "• 辩论：轮次 centroid dispersion → 早停\n"
            "• LangGraph：仅重跑 drift 节点（如 tool_call）\n"
            "• 闭环：intervention + impact_receipt"
        ),
        "econ_en": (
            "Direct OPEX (demos)\n"
            "• SoT ~34% $/ep (~$0.012/ep)\n"
            "• Debate FSDS ~39% $/debate; +~20% round save\n"
            "• LangGraph ~$0.015/ep plan savings\n"
            "Mid scale: ~$72k/yr SoT + ~$16k/yr debate stack\n"
            "Justify: multi_agent_fsds_economics.json, langgraph_fsds_hook.json"
        ),
        "econ_zh": (
            "直接 OPEX（demo）\n"
            "• SoT ~34% $/ep（~$0.012/ep）\n"
            "• 辩论 FSDS ~39% $/debate + 早停 ~20% 轮次\n"
            "• LangGraph ~$0.015/ep 量级\n"
            "中型流量：SoT ~$72k/年 + 辩论 ~$16k/年\n"
            "依据：multi_agent / langgraph JSON"
        ),
        "fc": LGREEN,
        "ec": GREEN,
    },
    {
        "id": "05_bpo_hybrid",
        "title_en": "Scenario 5 · BPO / human–agent hybrid QA",
        "title_zh": "场景 5 · BPO / 人机协同质检",
        "digest_en": (
            "Digest handoffs, not full transcripts (federated option)\n"
            "• Segment: agent span vs human span embeddings\n"
            "• REF/LIVE on shift of handoff features\n"
            "• FSDS localizes which segments moved\n"
            "• QA samples shifted handoffs only\n"
            "• Tenant/site = block; no raw dialog cross-share"
        ),
        "digest_zh": (
            "消化 handoff 特征，非全量 transcript（可联邦）\n"
            "• 段：agent  span vs 人工 span embedding\n"
            "• REF/LIVE 看 handoff shift\n"
            "• FSDS 定位哪些段发生漂移\n"
            "• 督导仅抽检 shift 段\n"
            "• 租户/职场 = 块；不跨域共享对话原文"
        ),
        "econ_en": (
            "Direct OPEX\n"
            "• Supervisor FTE ∝ review volume\n"
            "• Shift-targeted sampling ↓ reviews at same quality SLA\n"
            "Incremental\n"
            "• Receipt justifies sampling to compliance\n"
            "Quantify: (full review rate − FSDS rate) × cost/hour × volume\n"
            "Justify: same receipt schema as agent closed_loop"
        ),
        "econ_zh": (
            "直接 OPEX\n"
            "• 督导 FTE ∝ 复核量\n"
            "• 只审 shift 段 → 同等 SLA 下少人时\n"
            "增量\n"
            "• receipt 支撑抽检合规叙事\n"
            "量化：（全量复核率 − FSDS 抽检率）× 人时单价 × 量\n"
            "依据：closed_loop impact_receipt 同构"
        ),
        "fc": LBLUE,
        "ec": GRAY,
    },
    {
        "id": "06_feature_store_ofs",
        "title_en": "Scenario 6 · Feature store / online OFS (federated blocks)",
        "title_zh": "场景 6 · 特征仓 / 在线 OFS（联邦块）",
        "digest_en": (
            "Block arrival stream → local FSDS each group\n"
            "• Client k: s_j, drift_type_j per feature/group\n"
            "• Covariate/compound → candidate retire; concept → no auto-drop\n"
            "• Server: incremental LOGO when new block joins\n"
            "• Pair with ESS gate (reuse uplift overlap logic)\n"
            "• Online FDR on merged block p-value stream (roadmap)"
        ),
        "digest_zh": (
            "特征组流入 → 各块本地 FSDS\n"
            "• Client k：每特征/组 s_j、drift_type_j\n"
            "• 协变量/复合 → candidate retire；纯概念 → 不自动删\n"
            "• 新块加入：服务端增量 LOGO\n"
            "• 与 ESS 闸门共用（uplift overlap 逻辑）\n"
            "• 块 p 值流上在线 FDR（路线）"
        ),
        "econ_en": (
            "Direct\n"
            "• Retire wrong features → less downstream train/score cost\n"
            "• Minimal uplink vs shipping full matrices for central FSDS\n"
            "Incremental\n"
            "• Faster safe OFS in regulated multi-team stores\n"
            "Justify: FSDS_FEDERATED_FEATURE_BLOCKS.md + block demo JSON"
        ),
        "econ_zh": (
            "直接\n"
            "• 退坏特征 → 下游训练/推理省算力\n"
            "• 相对集中式 FSDS 少传矩阵\n"
            "增量\n"
            "• 多团队特征仓合规退役加速\n"
            "依据：FSDS_FEDERATED_FEATURE_BLOCKS.md"
        ),
        "fc": LPURPLE,
        "ec": PURPLE,
    },
]


def _draw_slide(sc: dict, lang: str) -> None:
    is_zh = lang == "zh"
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    title = sc["title_zh"] if is_zh else sc["title_en"]
    ax.text(W / 2, H - 0.4, title, ha="center", fontsize=13, fontweight="bold", color=DARK)

    lbl_digest = "数据消化 Data digestion" if is_zh else "Data digestion"
    lbl_econ = "经济效益 Justification" if is_zh else "Economic justification"

    ax.text(0.35, H - 1.05, lbl_digest, fontsize=10, fontweight="bold", color=BLUE)
    ax.text(W / 2 + 0.15, H - 1.05, lbl_econ, fontsize=10, fontweight="bold", color=GREEN)

    digest = sc["digest_zh"] if is_zh else sc["digest_en"]
    econ = sc["econ_zh"] if is_zh else sc["econ_en"]

    for x, text, fc, ec in [
        (0.25, digest, sc["fc"], sc["ec"]),
        (W / 2 + 0.05, econ, LGREEN if not is_zh else LGREEN, GREEN),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.55),
                W / 2 - 0.35,
                H - 1.75,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.2,
                edgecolor=ec,
                facecolor=fc,
            )
        )
        ax.text(x + 0.15, H - 1.35, text, ha="left", va="top", fontsize=7.8, color=DARK, linespacing=1.25)

    foot = (
        "Receipt: MMD · AUC · p-value · rule_id · economics (sim)  |  "
        "Reproduce: run_fsds_business_impact_pack.py"
    )
    ax.text(W / 2, 0.25, foot, ha="center", fontsize=7, color=GRAY)

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "zh" if is_zh else "en"
    path = OUT / f"slide_{sc['id']}_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", path)


def main() -> int:
    for sc in SCENARIOS:
        _draw_slide(sc, "en")
        _draw_slide(sc, "zh")
    print("Done:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
