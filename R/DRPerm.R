#DRPerm R:
library(MASS)
library(glmnet)
library(ANN2)
library(CVST)
library(kernlab)
library(grf)
library(RandomProjectionTest)
library(caret)

############################################################
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
#'@param tau_model     :Pseudo-Outcome Model
#
############################################################
DRPerm <- function(X, Y, W,
    model_registry, 
    m_model, e_model, tau_model,
    seed: int = 2026, n_folds: int = 5,
    clip_e = 0.01, n_perm = 150, alpha = 0.05){
  n = nrow(X)
  p = ncol(X)
  nuisance_fit <- cross_nuisance_fit(X, Y, W,
    seed = seed, n_folds = n_folds,
    m_model = m_model, e_model = e_model)
  Y_tilde = Y - nuisance_fit$m_hat
  W_tilde = W - nuisance_fit$e_hat
  pseudo_outcome <- Y_tilde * W_tilde
  tau_po_model <- model_registry[tau_model]
  tau_fit <- fit_predict_oof(
    X = X, y = pseudo_outcome, folds = n_folds,
    model_spec = tau_po_model, seed = seed
    )$pred
  observed_po_risk <- mean((pseudo_outcome - tau_fit)**2)
  permuted_po_risk <- numeric(n_perm)
  for(b in 1:n_perm){
    W_perm = sample(W)
    nuisance_fit_b <- cross_nuisance_fit(X, Y, W,
      seed = (seed + b^2), n_folds = n_folds,
      m_model = m_model, e_model = e_model)
    Y_tilde_b <- Y - nuisance_fit_b$m_hat
    W_tilde_b <- W_perm - nuisance_fit_b$e_hat
    pseudo_outcome_b <- Y_tilde_b * W_tilde_b
    tau_po_model_b <- model_registry[tau_model]
    tau_fit_b <- fit_predict_oof(X = X, y = pseudo_outcome_b,
      folds = n_folds, model_spec = tau_po_model, seed = (seed + b^3))
    permuted_po_risk[b] <- mean((pseudo_outcome_b - tau_fit_b) ** 2)
  }
  p_value <- (1 + sum(permuted_po_risk > r_risk_original))/(1 + n_perm)
  return(
    list(
      observed_po_risk = observed_po_risk,
      permuted_po_risk = permuted_po_risk,
      p_value = p_value,
      rejected = (p_value < alpha)
        )
    )
}






df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
DRPerm(X, Y, W, n_perm = 25)$p_value















