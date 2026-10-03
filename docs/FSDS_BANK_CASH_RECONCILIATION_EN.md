# FSDS → bank checking account (cash reconciliation)

See **Chinese master**: `docs/FSDS_BANK_CASH_RECONCILIATION_ZH.md`.

**Core:** Ending balance minus counterfactual balance ≈ customer **cash_in** + vendor/API **retained** (avoided debits) − monitor **cash_out** − approved RELEARN **cash_out**.

**Code:** `fsds_sot/bank_cash_bridge.py`, `reconcile_fsds_bank_cash.py` → `artifacts/fsds_bank_cash_bridge.json`.

**T0:** Map `kpi_before_after.usd_per_episode` to GL `6100` retained vs LLM invoice CSV.  
**T1:** Map `estimated_saved_spend_usd` to GL `6200` retained; CAPEX tickets to `1700` cash_out.  
**Prepaid API:** Savings may skip a top-up before checking balance moves.

**Acceptance:** Match `baseline_api_usd − actual` to receipt `rule_ids` within tolerance at month close.
