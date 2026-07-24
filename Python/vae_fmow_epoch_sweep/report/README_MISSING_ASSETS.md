# Missing report assets (not recoverable from transcript)

Agent: `bc-91559e09-3b34-4059-9df1-1c848dfcbc0b` (`https://cursor.com/agents/bc-91559e09-3b34-4059-9df1-1c848dfcbc0b`)

These files were generated on the agent VM under `Python/vae_fmow_epoch_sweep/report/`
(see transcript: `run_vae_results_report.py` → “Report written to vae_fmow_epoch_sweep/report/”).

## Generated on agent VM (original report)

| File | Role | Recoverable from transcript? |
|------|------|------------------------------|
| `index.html` | HTML summary table + image embeds | Regenerated from CSV (heatmap column empty) |
| `summary_table.csv` | Numeric table | Yes (recovered; same as sweep summary CSV) |
| `metric_bars.png` | Val loss / OOB / Top10% bars | Regenerated from CSV |
| `loss_curves.png` | Train/val loss curves per config | **No** — needs each `*/summary_vae.json` training arrays |
| `heatmap_grid.png` | 5-config sample heatmap collage | **No** — needs `*/sample_heatmap.png` binaries |
| `vimp_top_dims_ep50.png` | Top latent dims for ep50 | **No** — needs `ep50/vimp_latent_dims.csv` |

## Related per-run assets that lived on the agent VM

Under `Python/vae_fmow_epoch_sweep/{ep25,ep50,ep75,ep100,early_stop}/`:

- `summary_vae.json` (full; only `early_stop` partially present in transcript)
- `vimp_latent_dims.csv` (~3.5KB each; content never read into transcript)
- `sample_heatmap.png`
- `vae_model.pth`, `z_train_vae.pt`, `z_eval_vae.pt`, `importance_vae.npy`, etc.
- `train_*.log`

## Artifact uploads

No evidence in the transcript that report PNG/HTML were uploaded as Cursor artifacts or committed to git.
The only “Add files via upload” mention found was git commit `bda7545` for `Python/R_risk_loco.py`, unrelated to this report.
Report assets remained on the agent VM filesystem only.
