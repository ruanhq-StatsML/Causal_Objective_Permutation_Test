# Evening summary: epsilon-greedy explore traffic & dimension sensitivity

## Setup

| Experiment | DGP | Settings |
|---|---|---|
| High explore @ d=30 | `nonlinear_messy` (mid-stream shift) | n=20k / ref=5k / batch=25; also n=40k / batch=50 |
| Dim sensitivity | `nonlinear_messy_ultra` (messiest) | n=15k / ref=5k / batch=100; d ∈ {10…500} |

---

## 1) More exploration traffic → robustness (d=30)

**Main pattern (consistent across 3-rep / 5-rep / n=40k):**

- **Mean regret** is usually best at moderate ε ≈ **0.15–0.20** (sometimes 0.05 on the longer n=40k stream).
- **Higher ε (≥0.4)** raises average regret but **clearly lowers repeat CV** (std/mean across seeds).
  - Example (5-rep): low-ε CV ≈ 0.10 vs high-ε CV ≈ 0.03.
  - Path volatility (`path_cumregret_std_mean`) also tends to shrink as ε increases.

**Tradeoff:** more explore traffic buys **seed-to-seed stability** at the cost of **extra regret**. For robustness reporting, ε ≈ 0.3–0.5 is the sweet band; ε ≥ 0.6 is very stable but wastes regret.

Key files:
- `mab_scale_d30_higheps_variance_vs_explore.csv`
- `mab_scale_d30_higheps_r5_variance_vs_explore.csv`
- `mab_scale_d30_dense_higheps_variance_vs_explore.csv`
- `mab_scale_d30_n40k_higheps_variance_vs_explore.csv`

---

## 2) Sensitivity vs dimension (ultra-messy)

Grid: d = 10,20,30,40,50,75,100,150,200,300,400,500 · ε ∈ {0.05,0.10,0.20,0.30,0.40,0.50} · 3 repeats.

**Takeaways:**

- **Best mean regret** is almost always a **low-ε** policy (ε=0.05 or 0.10) across nearly all d.
- **Most robust (min CV)** policies are typically **higher ε** (0.3–0.5), again across most d — same explore↔stability tradeoff as above.
- Dimension effect is **not monotone**:
  - Low ε (0.05): regret weakly rises with d (corr ≈ +0.30); very noisy at extremes (d=10 CV=0.25, d=500 CV=0.41).
  - Mid/high ε (0.2–0.4): regret often **falls** with d (corr ≈ −0.4 to −0.7) — exploration helps more when the arm pool is harder in high-d ultra-messy regimes.
- Absolute regret on ultra-messy is much larger than on the simpler messy d=30 runs (hundreds vs ~70), as expected.

Key files:
- `mab_eps_dim_sensitivity_dim_sensitivity.csv` / `.png`
- `mab_eps_dim_sensitivity_summary.csv`
- `mab_eps_dim_sensitivity_metrics.csv`

---

## Bottom line

1. **More explore traffic ⇒ more robust (lower CV), worse mean regret.**
2. That tradeoff **holds across dimension** on the messiest DGP; best-mean stays low-ε, best-robust stays high-ε.
3. For d=30 calibration, prefer **ε ≈ 0.15–0.20** for performance, **ε ≈ 0.4–0.5** if you care about seed stability.
