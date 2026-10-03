# Feature-store attribution layers

Relational schema (order facts):

```text
user --places--> order --for--> item --sold_by--> merchant
                              W=0/1 at merchant (existing vs new batch)
```

| Layer | Grain | Role in delivery |
|-------|--------|------------------|
| **Order** | one conversion row | Raw attrs: gmv, quantity, discount_rate, delivery_mins, user_rating, basket_size |
| **Merchant** | rollup of orders | Primary FSDS axis: rich + rolling aggregations → LOGO-MMD + hierarchical tree (L1→L3) |
| **User** | rollup of orders | User-side feature store (`aggregate_to_user`); same ref/live batch via merchant W on orders |
| **Cross-level** | merchant L1 + product vertical L2 | `cross_level.py`: GMV shift localized to category vertical |

Scripts:

- `merchant_prototype.py` — data gen + PO-risk path (concept drift)
- `logo_mmd.py` — covariate shift + LOGO-MMD at merchant grain
- `multilevel_localization.py` — raw attr → agg family → leaf (BB-FDR tree)
- `cross_level.py` — two-level merchant + vertical table
