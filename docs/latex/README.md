# FSDS monitoring LaTeX (full bundle)

## One PDF — everything

```bash
cd docs/latex
pdflatex fsds_monitoring_full_standalone.tex
pdflatex fsds_monitoring_full_standalone.tex
```

**Master file:** [`fsds_monitoring_full_standalone.tex`](fsds_monitoring_full_standalone.tex)

**Single file (all `\input` inlined, copy-paste friendly):** [`fsds_monitoring_full_monolithic.tex`](fsds_monitoring_full_monolithic.tex)

- **Part I:** Two-batch uplift (`uplift_fsds_two_batch_formulation.tex`, benchmark tables, subset localization results, subset insights)
- **Part II:** Federated cross-block FSDS (communication efficiency, protocol, 5-dataset results)

## Partial compilations

| Document | Contents |
|----------|----------|
| [`uplift_fsds_two_batch_standalone.tex`](uplift_fsds_two_batch_standalone.tex) | Uplift only |
| [`federated_fsds_comprehensive_standalone.tex`](federated_fsds_comprehensive_standalone.tex) | Federated only |

## Regenerate auto-generated `.tex`

```bash
cd Python
python3 run_uplift_subset_benchmark.py      # uplift_fsds_benchmark_results.tex, subset_localization_results.tex
python3 demo_federated_protocol_datasets.py # federated_fsds_protocol_results.tex
python3 run_loco_auuc_benchmark.py          # uplift_fsds_benchmark_results.tex (LOCO tables)
python3 run_fsds_business_impact_pack.py    # GTM demos + fsds_business_impact_dashboard_en.png
```

## OPEX / CAPEX cost-sharing PO (English)

```bash
cd docs/latex && pdflatex fsds_opex_capex_cost_share_po.tex
```

File: [`fsds_opex_capex_cost_share_po.tex`](fsds_opex_capex_cost_share_po.tex) — discrete demo step-sizes and two-ledger governance.

## Concrete pipeline justification (X, Y, algorithm, measured delta)

[`fsds_concrete_pipeline_justification.tex`](fsds_concrete_pipeline_justification.tex) — step ledger with Hillstrom/synthetic numbers from benchmark JSON.

## Treasury + credit-risk fulfillment PO (English)

```bash
cd docs/latex && pdflatex fsds_treasury_credit_risk_po.tex
```

File: [`fsds_treasury_credit_risk_po.tex`](fsds_treasury_credit_risk_po.tex) — checking-account reconciliation (retained vs cash in), `impact_receipt` GL mapping, and credit origination fulfillment via the same two-batch uplift spine.

## Embedding mapping tables + MMD computation (EN + 中文 LaTeX)

```bash
cd docs/latex && pdflatex fsds_embedding_mapping_mmd_en.tex
xelatex fsds_embedding_mapping_mmd_zh.tex
```

- [`fsds_embedding_mapping_mmd_en.tex`](fsds_embedding_mapping_mmd_en.tex) — ReAct / multi-agent / RAG / LangGraph maps; eq. MMD$^2$, shift, $L_b$
- [`fsds_embedding_mapping_mmd_zh.tex`](fsds_embedding_mapping_mmd_zh.tex) — 中文版（ctex）
- Companion: [`fsds_embedding_logic_en.tex`](fsds_embedding_logic_en.tex), [`../FSDS_EMBEDDING_LOGIC_ZH.md`](../FSDS_EMBEDDING_LOGIC_ZH.md)

## Full conclusion PO (English) — recommended executive PDF

```bash
cd Python && python3 export_fsds_results_tables_tex.py && python3 export_fsds_full_conclusion_monolithic.py
cd ../docs/latex && pdflatex fsds_full_conclusion_po_en.tex
# single file:
pdflatex fsds_full_conclusion_po_en_monolithic.tex
```

- [`fsds_full_conclusion_po_en.tex`](fsds_full_conclusion_po_en.tex) — **C1–C8 conclusions**, incremental table, appendices R1–R10b
- [`fsds_full_conclusion_po_en_monolithic.tex`](fsds_full_conclusion_po_en_monolithic.tex) — tables inlined

## Full scenario PO (English, uplift + agent collaboration, tables R1–R9)

**Recommended full PDF:**

```bash
cd Python
python3 run_fsds_business_impact_pack.py
python3 demo_fsds_agent_collaboration_suite.py
python3 export_fsds_results_tables_tex.py
cd ../docs/latex && pdflatex fsds_full_scenarios_po_en.tex
```

- [`fsds_full_scenarios_po_en.tex`](fsds_full_scenarios_po_en.tex) — polished narratives + `\input` R1–R6 and R7–R9
- [`fsds_full_scenarios_po_en_monolithic.tex`](fsds_full_scenarios_po_en_monolithic.tex) — **single-file 全量** (tables inlined; `export_fsds_full_po_monolithic.py`)
- [`fsds_agent_collaboration_tables_generated.tex`](fsds_agent_collaboration_tables_generated.tex) — agent MC suite tables
- [`fsds_results_tables_generated.tex`](fsds_results_tables_generated.tex) — uplift + executive rollup tables

## FSDS method + incremental value PO (shorter, R1–R6 only)

```bash
cd Python && python3 export_fsds_results_tables_tex.py
cd ../docs/latex && pdflatex fsds_method_incremental_value_po_en.tex
```

- [`fsds_method_incremental_value_po_en.tex`](fsds_method_incremental_value_po_en.tex)

## Application X/Y and step-level benefit (English formulation + 中文)

**Full English formulation:**

```bash
cd docs/latex && pdflatex fsds_xy_benefit_formulation_en.tex && pdflatex fsds_xy_benefit_formulation_en.tex
```

- [`fsds_xy_benefit_formulation_en.tex`](fsds_xy_benefit_formulation_en.tex) — domains, $(X,Y,T,W,Z)$, gates vs direct benefit, credit mapping, audit join

**Compact ledger (one-pass summary table):**

```bash
cd docs/latex && pdflatex fsds_application_xy_benefit_ledger.tex
```

- [`fsds_application_xy_benefit_ledger.tex`](fsds_application_xy_benefit_ledger.tex)
- 中文逐步说明: [`../FSDS_WHERE_X_Y_POSITIVE_BENEFIT_ZH.md`](../FSDS_WHERE_X_Y_POSITIVE_BENEFIT_ZH.md)
