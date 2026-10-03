# Scenario slides & GTM pack — v2

## New in v2

| Item | Purpose |
|------|---------|
| `config/fsds_economics_assumptions.json` | Edit margin, traffic tiers, federated egress — re-run pack |
| `Python/fsds_economics_config.py` | Shared loader for uplift rules + executive rollup |
| `export_fsds_scenario_slides_pptx.py` | **EN / ZH / bilingual** PPTX (title + assumptions + 6 scenarios) |
| Executive rollup | Per-scale **`direct_uplift_opex_usd_per_year`**, federated egress $/yr |
| `run_fsds_business_impact_pack.py` | Now includes scenario PNGs + PPTX + synthesize |

## One command

```bash
cd Python && python3 run_fsds_business_impact_pack.py
```

## Decks

- `artifacts/fsds_scenario_slides/fsds_scenario_deck_en.pptx`
- `artifacts/fsds_scenario_slides/fsds_scenario_deck_zh.pptx`
- `artifacts/fsds_scenario_slides/fsds_scenario_deck_bilingual.pptx` (12 content slides + assumptions)

## Customize economics

1. Edit `config/fsds_economics_assumptions.json`
2. Re-run pack (uplift bench picks up margin/treat_cost for rule tags)
