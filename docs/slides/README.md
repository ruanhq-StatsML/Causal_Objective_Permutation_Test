# FSDS scenario one-pagers (slides)

| File | Purpose |
|------|---------|
| [FSDS_SCENARIOS_ONE_PAGER_EN.md](FSDS_SCENARIOS_ONE_PAGER_EN.md) | 6 scenarios × 1 page — English |
| [FSDS_SCENARIOS_ONE_PAGER_ZH.md](FSDS_SCENARIOS_ONE_PAGER_ZH.md) | 6 scenarios × 1 page — 中文 |

Regenerate PNG deck (12 images, EN + ZH):

```bash
cd Python && python3 plot_fsds_scenario_slides_deck.py
```

Output: `artifacts/fsds_scenario_slides/slide_{01-06}_{en|zh}.png`

**v2:** PPTX decks + `config/fsds_economics_assumptions.json` — see [CHANGELOG_V2.md](CHANGELOG_V2.md).

```bash
cd Python && python3 export_fsds_scenario_slides_pptx.py
```
