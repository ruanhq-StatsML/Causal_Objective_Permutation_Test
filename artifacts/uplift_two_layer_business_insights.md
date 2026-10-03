# Two-layer uplift localization — benchmark business insights

Generated: 2026-10-03 02:32 UTC. Script: `Python/run_uplift_subset_benchmark.py`.

Layers: **L1** = AUUC / LOCO / pairwise AUUC (ranking). **L2** = empirical ATE + mean τ̂ (effect).

## hillstrom

- **Gates:** MMD²=0.0016, ESS_overlap=0.990, PO-risk p=0.7273 (accept).
- **Strategy:** Regime=x_stable_gap_large_check_loco_and_po_risk; diagnose=concept_drift_or_tau_change; PO-risk p=0.727. Strategy: LOCO–AUUC → L1 pairwise AUUC; if PO reject add L2 ATE REALLOCATE.

### Layer 1 (ranking)
- Global ranking: AUUC_ref=-0.0449, AUUC_live=0.0154, gap=-0.0603.
- Top LOCO (L1 feature): 'block_0' drop_live=-0.0167 — ranking on LIVE leans on this input.
- Top pairwise AUUC (L1 users): domain_quintile_3 vs domain_quintile_4 Δ=-0.1260; cap worse-ranking slice 'domain_quintile_3'.
- Quintile spread: worst=domain_quintile_3 AUUC=-0.0557; best=domain_quintile_4 AUUC=0.0703.

### Layer 2 (uplift / effect)
- Layer 2 (effect): PO-risk not flagged — use empirical ATE / mean τ̂ as diagnostic only, not primary REALLOCATE on ATE pairs.
- Highest observed uplift slice: domain_quintile_3 empirical_ATE=0.1101, mean_tau_hat=0.0831.
- Top pairwise ATE (L2): domain_quintile_3 vs domain_quintile_5 ΔATE=0.1043; shift spend toward 'domain_quintile_3'.
- Top pairwise mean τ̂ (model level): domain_quintile_2 vs domain_quintile_5 Δ=0.0219.

### Business rules (ops)
- **[REALLOCATE]** (P1): Cap uplift in domain quintile 'domain_quintile_3' (AUUC_live=-0.0557, n=240); continue in 'domain_quintile_4' where ranking remains stronger.
  - *Economics (sim):* net impact **$546.00**, saved spend **$546.00**
- **[REALLOCATE]** (P2): Uplift ranking on LIVE depends on group 'block_0': do not drop this block from the model without replacement; if trimming segments, keep users where this group's LOCO signal is stable (LOCO AUUC drop_live=-0.0167).
- **[REALLOCATE]** (P2): Uplift ranking on LIVE depends on group 'block_1': do not drop this block from the model without replacement; if trimming segments, keep users where this group's LOCO signal is stable (LOCO AUUC drop_live=-0.0033).
- **[RELEARN]** (P2): X stable but AUUC gap large: extend labels and schedule prod uplift model relearn after REALLOCATE trial.
- **[MONITOR]** (P4): Layer-2 diagnostic (PO-risk not reject): largest slice ATE contrast domain_quintile_3 vs domain_quintile_5 (ΔATE=0.1043) — do not REALLOCATE on ATE alone; use Layer-1 AUUC rules first.

- **Scenario OPEX (sim):** net **$546.00**, saved spend **$546.00** (1 cap/realloc rules).
- **RELEARN CAPEX tickets (separate):** **$2500.00** (1 ticket(s), not in OPEX net).
