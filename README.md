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

```r
library(CFPerm)

set.seed(1)
sim <- LM_generation(
  n = 200,
  beta_hat = c(1, -1, 0.5),
  mean_shift = 0,
  var_shift = 1,
  cor = 0.3,
  n_nuisance = 5,
  eps = 1
)

df <- sim$df_return
df_train <- df[1:100, c(paste0("X", 1:3), paste0("X_nuis", 1:5), "Y")]
df_test  <- df[101:200, c(paste0("X", 1:3), paste0("X_nuis", 1:5), "Y")]

res <- cfperm(df_train, df_test, n_perm = 50, num.trees = 150, seed = 123)
res
```

## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
