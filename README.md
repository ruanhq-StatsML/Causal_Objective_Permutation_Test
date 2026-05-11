# Causal Objective Permutation Test
This repository provides the Python \& R implementation of "Permutation Test via Causal Inference Objective Functions". In this project, we formulate the distribution shift detection problem as a causal inference problem(regarding the existing batch of data as control group/batch and the newly coming batch of data as treatment group/batch) via various forms of the objective functions followed by the permute-then-refit procedure. This reppo includes permutation-based distribution-shift testing procedures built on doubly robust pseudo-outcome learners(PO-risk) and R-learners(R-risk). This implementation allows flexible specification of nuisance estimation models, enabling users to flexibly choose among different versions of the propensity score model and the outcome model.


## Python version:
```python
!pip install causal_objective_perm
from causal_objective_perm import RRPerm, DRPerm

```

## R version:
#### Installation (local)

```r
install.packages(c("devtools", "roxygen2", "testthat", "grf", "MASS", ""))
devtools::install_local("path/to/CFPerm")
```

## R example:
```R
n <- 400
p <- 16
X <- matrix(rnorm(n*p, 0, 1), n, p)
Y <- 10.0 * X[, 1] + 2.0 * X[, 2] + 1.0 * X[, 3] + rnorm(n, 0, 1)
W <- rbinom(n, 1, 0.5)
output <- RRPerm(X, Y, W, n_splits = 5, m_model = 'rf_regression', e_model = 'rf_classification')
```

## Python Example:

```python
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



## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
