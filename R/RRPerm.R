#RRPerm R:
library(MASS)
library(glmnet)
library(ANN2)
library(CVST)
library(kernlab)
library(grf)
library(RandomProjectionTest)
library(caret)
source('model_registry.R')
source('utils.R')

####################################################################
#Permutation Test for Distribution Shift via the R-risk
#In estimating the R-risk, the cross-fitting procedure is required(within the cross_nuisance_fit function here)
#'@param X             :Covariate Matrix, (n_exist + n_new, p)
#'@param Y             :Response Vectors, (n_exist + n_new, )
#'@param W             :Treatment(batch) Assignment Vectors, (n_exist + n_new, )
#'@param seed          :Random Seed, integer
#'@param n_splits      :Number of Splits.
#'@param clip_e        :Clipping Value for the propensity score
#'@param n_perm        :Number of permutations
#'@param alpha         :Significance Level alpha
#'@param m_model       :Outcome Model
#'@param e_model       :Propensity Score Model
####################################################################
RRPerm <- function(X, Y, W, 
    seed = 2026, n_folds = 5,
    clip_e = 0.01, n_perm = 150, alpha = 0.05,
    m_model, e_model){
  n = nrow(X)
  p = ncol(X)
  mu_hat = rep(0, n)
  e_hat = rep(0, n)
  shuffled_indices <- sample(1:n)
  fold_assignment = cut(seq(1, n), breaks = n_folds, labels = FALSE)
  folds <- split(shuffled_indices, fold_assignment)
  whole_idx <- c(1:n)
  binary_outcome = (length(unique(Y)) == 2)
  nuisance_fit <- cross_nuisance_fit(X, Y, W,
      seed = seed, n_folds = n_folds, binary_outcome = binary_outcome,
      m_model = m_model, e_model = e_model)
  Y_tilde = Y - nuisance_fit$m_hat
  W_tilde = W - nuisance_fit$e_hat
  tau_full_model <- fit_tau_rlearner_weighted(X_tilde, Y_tilde, W_tilde,
      ridge_lambda = lambda, 
      seed = seed, clip_wtilde = clip_wtilde)
  tau_full <- tau_full_model$predict(X)
  r_risk_original <- r_risk(Y_tilde, W_tilde, tau_full)
  r_risk_perm_list <- numeric(n_perm)
  for(b in 1:n_perm){
    W_perm <- sample(W)
    #Re-estimate the R-risk following the cross-fitted procedure.
    nuisance_fit_b <- cross_nuisance_fit(X, Y, W_perm,
        seed = (seed + b^2), n_folds = n_folds, binary_outcome = binary_outcome,
        m_model = m_model, e_model = e_model)
    Y_tilde_b <- Y - nuisance_fit_b$m_hat
    W_tilde_b <- W - nuisance_fit_b$e_hat
    tau_b <- tau_model$predict(X_tilde)
    r_risk_perm_list[b] <- r_risk(Y_tilde_b, W_tilde_b, tau_b)
  }
  p_value <- (1 + sum(r_risk_original > r_risk_perm_list))/(1 + n_perm)
  rejected <- (p_value < alpha)
  return(list(
    r_risk_original = r_risk_original,
    r_risk_perm_list = r_risk_perm_list,
    p_value = p_value,
    rejected = rejected
  ))
}










