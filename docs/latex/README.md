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
```
