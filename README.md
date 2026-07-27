# Causal Objective Permutation Test via Doubly Robust Pseudo-Outcome Learner and R-Learner
This repository provides the Python \& R implementation of "Permutation Test via Causal Inference Objective Functions". In this project, we formulate the distribution shift detection problem as a causal inference problem(regarding the existing batch of data as control group/batch and the newly coming batch of data as treatment group/batch) via various forms of the objective functions followed by the permute-then-refit procedure. This repo include permutation-based distribution-shift testing procedures built on doubly robust pseudo-outcome learners(PO-risk) and R-learners(R-risk). This implementation allows flexible specification of nuisance estimation models, enabling users to flexibly choose among different versions of the propensity score model(logistic regression, tree-based learners - Random Forest, XGBoost, CatBoost and MLP) and the outcome model(linear regression, tree-based learners - Random Forest, XGBoost, CatBoost  and MLP).

### Motivation: Meta-Learner and Distribution Shift
We leverage the meta-learner(causal forest, doubly-robust pseudo-outcome learner and the R-learner) to quantify the distance between the two batches of datasets, then the variable importance for this version of meta-learner is a proxy for the distribution shift - from another aspect, so called OOD Variable Importance. The illustration of the feature selection as the validity of this proposed framework in both Covariate Shift(P(X) shift) and Concept Drift(P(Y|X) shift).

- For Covariate Shift(the feature selection consistency in terms of kendall's tau correlation is leveraged for efficiency of the ranking of the feature importance in the covariate shift).
<img width="900" height="970" alt="vecshift_lambda_cor06_polished" src="https://github.com/user-attachments/assets/0275705c-7a57-4edb-9832-6356cc2c541b" />
- For Concept Drift(the feature selection consistency in terms of kendall's tau correlation is leveraged for efficiency of the ranking of the feature importance in the concept drift).
<img width="900" height="970" alt="uq_vimpood_cd_on_cs_allmethods_polished" src="https://github.com/user-attachments/assets/65925cae-d308-4651-8ba1-979d122e5dae" />

- This framework is highly flexible that can be extended into multiple versions of the variable importances as well as different notions of the causal learners.
<img width="1500" height="950" alt="cfperm__diagram" src="https://github.com/user-attachments/assets/43332ee7-3877-421f-86ba-b66f11a9fbb2" />

- 

### Motivation: Causal Objective Function and Hypothesis Testing
The causal learner + permute-then-refit procedure can be leveraged for the hypothesis testing procedure with very solid theoretical guarantee.

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





## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
