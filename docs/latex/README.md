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
