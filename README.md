# CFPerm

Implementation for the "Permutation Test via Causal Inference Objective Function": It provides implementation for leveraging DR(doubly-robust) pseudo-outcome learner and R-learner for permutation test of testing the distribution shift. The permute-then-refit procedure is employed.


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

## Quick example

```python
from RRperm import *
rng = np.random.default_rng(2023)
n = 400
p = 16
X = rng.normal(size = (n, p))
Y = 10.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(scale = 1.0, size = n)
W = rng.binomial(1, 0.5, size = n)
output = RRPerm(X, Y, W, n_splits = 5, model_m = 'rf_regressor', model_e = 'rf_classifier')
output['rejected'], output['p_value'], output['statistic']

```

## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
