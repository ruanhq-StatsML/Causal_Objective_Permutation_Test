#unittest for the RRPerm:
library(MASS)
library(glmnet)
library(ANN2)
library(CVST)
library(kernlab)
library(grf)
library(RandomProjectionTest)
library(caret)
library(testthat)
source('model_registry.R')
source('utils.R')

#Testing for the H_{0} null case:
test_that('RRPerm returns TRUE/FALSE and validates inputs',
  {
  skip_if_not_installed(c('randomForest'))
  set.seed(123)
  n0 <- 300
  n1 <- 300
  p <- 20
  n <- n0 + n1
  n_perm = 10
  X <- matrix(rnorm(n * p, 0, 1), nrow = n, ncol = p)
  colnames(X) <- paste0("X", seq_len(p))
  W <- c(rep(0, n0), rep(1, n1))
  Y <- 1 + 0.8 * X[, 1] - 0.5 * X[, 2] + rnorm(n, sd = 1) 
  res_rrperm <- RRPerm(X, Y, W, 
    m_model = 'rf_regression', e_model = 'logistic_classification',
    ridge_lambda = 0.25, clip_e = 0.01, n_perm = n_perm)
  expect_true((res_rrperm$p_value >= 0) & (res_rrperm$p_value <= 1))
  expect_true(res_rrperm$rejected %in% c(TRUE, FALSE, 0, 1))
  expect_gte(res_rrperm$r_risk_original, 0)
  expect_true(is.finite(res_rrperm$r_risk_original))
  expect_true(is.finite(max(res_rrperm$r_risk_perm_list)))
  expect_true(length(res_rrperm$r_risk_perm_list) == n_perm)
})

