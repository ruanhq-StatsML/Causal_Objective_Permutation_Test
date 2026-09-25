# Meta-Learner Objective Function Test via Pseudo-Outcome Learner and R-Learner
This repository provides the Python \& R implementation of "Permutation Test via Causal Inference Objective Functions". In this project, we formulate the distribution shift detection problem as a causal inference problem(regarding the existing batch of data as control group/batch and the newly coming batch of data as treatment group/batch) via various forms of the objective functions followed by the permute-then-refit procedure. This repo include permutation-based distribution-shift testing procedures built on doubly robust pseudo-outcome learners(PO-risk) and R-learners(R-risk). This implementation allows flexible specification of nuisance estimation models, enabling users to flexibly choose among different versions of the propensity score model(logistic regression, tree-based learners - Random Forest, XGBoost, CatBoost and MLP) and the outcome model(linear regression, tree-based learners - Random Forest, XGBoost, CatBoost  and MLP).

### Motivation: Meta-Learner and Distribution Shift
We leverage the meta-learner(causal forest, doubly-robust pseudo-outcome learner and the R-learner) to quantify the distance between the two batches of datasets, then the variable importance for this version of meta-learner is a proxy for the distribution shift - from another aspect, so called OOD Variable Importance. The illustration of the feature selection as the validity of this proposed framework in both Covariate Shift(P(X) shift) and Concept Drift(P(Y|X) shift).

- For Covariate Shift(the feature selection consistency in terms of kendall's tau correlation is leveraged for efficiency of the ranking of the feature importance in the covariate shift).
<img width="900" height="970" alt="vecshift_lambda_cor06_polished" src="https://github.com/user-attachments/assets/0275705c-7a57-4edb-9832-6356cc2c541b" />

- For Concept Drift(the feature selection consistency in terms of kendall's tau correlation is leveraged for efficiency of the ranking of the feature importance in the concept drift).
<img width="900" height="970" alt="uq_vimpood_cd_on_cs_allmethods_polished" src="https://github.com/user-attachments/assets/65925cae-d308-4651-8ba1-979d122e5dae" />

- This framework is highly flexible that can be extended into multiple versions of the variable importances as well as different notions of the meta learners(ranging from R-learner, PO-learner, X-learner to U-learner etc) - as demonstrated below
<img width="2046" height="1128" alt="metaLearnerFlexibility" src="https://github.com/user-attachments/assets/74936d34-1e49-4c09-a13f-e1a35be929fc" />


- The overall efficiency for this procedure is highly efficient both in terms of the and the

### Motivation: Causal Objective Function and Hypothesis Testing
The causal learner + permute-then-refit procedure can be leveraged for the hypothesis testing procedure with very solid theoretical guarantee - enabling the tractable theoretical guarantee.


### Python version:
```python
!pip install causal_objective_perm
from causal_objective_perm import RRPerm, DRPerm
rng = np.random.default_rng(2023)
n = 400
p = 16
X = rng.normal(size = (n, p))
Y = 10.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(scale = 1.0, size = n)
W = rng.binomial(1, 0.5, size = n)
output = RRPerm(X, Y, W, n_splits = 5, model_m = 'rf_regressor', model_e = 'rf_classifier')
output#FALSE

output = DRPerm(X, Y, W, seed = 2026)
output#FALSE

```

### R version:
#### Installation

```R
install.packages(c("devtools", "roxygen2", "testthat", "grf", "MASS"))
devtools::install_local("path/to/CFPerm")
n <- 400
p <- 16
X <- matrix(rnorm(n*p, 0, 1), n, p)
Y <- 10.0 * X[, 1] + 2.0 * X[, 2] + 1.0 * X[, 3] + rnorm(n, 0, 1)
W <- rbinom(n, 1, 0.5)
output <- RRPerm(X, Y, W, n_splits = 5, m_model = 'rf_regression', e_model = 'rf_classification')
```





## Skill: `rap_clever_covariate_guide` (Reasoning-via-Planning integration)

`Python/rap_clever_covariate_guide.py` is a small, dependency-light (numpy +
stdlib) MVP skill that carries the causal-inference signals in this repo into
the *semantic* layer of an LLM planner (RAP — Reasoning via Planning / MCTS).

It fuses two signals:

1. **Clever covariate** `H = (Y - e) / (e (1 - e))` — the TMLE clever covariate.
   Given the planner's prior `e` that a step succeeds and the observed outcome
   `Y`, `H` gives the *direction and magnitude* of the surprise for that step.
2. **Online rolling empirical p-value** — the same permute-then-refit online
   monitor this repo already uses for streaming distribution-shift detection
   (`empirical_pval` / `anomaly_score = 1 - p` in `mab_benchmark_core.py`,
   the OnlineRFPerm-style statistic). It gives the *statistical confidence*
   that the surprise is a genuine regime change rather than noise.

The skill renders these into a natural-language RAP planning prompt with an
explicit adjustment hint (`aggressive` / `conservative` / `keep`) and a
regime-shift alarm (`p < alpha`) that recommends re-planning, so the LLM
*reads* the anomaly signal and adapts its next expansion.

```python
from rap_clever_covariate_guide import RAPCleverCovariateSkill, RAPWithCovariate

skill = RAPCleverCovariateSkill(threshold=0.5, alpha=0.05)
res = skill.invoke(state="current reasoning state...", prior=0.3, outcome=1,
                   history=["step1...", "step2..."])
print(res["hint"])   # aggressive | conservative | keep
print(res["replan"]) # True when the rolling p-value crosses alpha
print(res["prompt"]) # full RAP planning prompt with the H + p-value blocks

# Drop into an MCTS expansion loop:
planner = RAPWithCovariate(llm, skill)
out = planner.expand(node, prior=0.3, outcome=1)  # -> children + hint + replan
```

### Why this has production value

The clever covariate supplies **numeric direction** and the online rolling
p-value supplies **statistical significance gating**. Injecting the fused
signal into the planning prompt (rather than only into numeric UCB/pruning)
buys:

- **Lower token cost.** The planner escalates exploration only when a deviation
  is statistically notable, instead of overreacting to a single noisy outcome
  (large `|H|` but non-significant `p` -> `keep`) — fewer wasted expansions.
- **Higher success rate / faster recovery.** A sustained, significant anomaly
  raises the alarm and steers the planner toward safe, proven branches and
  re-planning, so it escapes bad branches faster.
- **Near-zero marginal engineering cost.** Pure prompt engineering on top of
  statistics the pipeline already computes — no fine-tuning or extra training.
- **Complementary "numeric + semantic" guidance.** The same drift signal that
  drives numeric UCB injection now also reaches the action-generation layer,
  giving double-sided control over exploration.

Run the tests / demo:

```bash
cd Python
python -m pytest test_rap_clever_covariate_guide.py -q
python rap_clever_covariate_guide.py   # prints an example prompt
```

## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
