#DRPerm R:
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

####################################################################################
#'Permutation Test for Distribution Shift 
#'via Doubly Robust Pseudo-Outcome Learner
#
#'Implements a permute-then-refit permutation test based on the 
#'doubly robust pseudo-outcome risk. The procedure starts by
#'fitting the outcome model(on both treatment group and control group)
#'and propensity score model, then regress the pseudo-outcome
#'on the covariates. 
#
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
#'
#'@return A list containing the observed po-risk, permutation p-values
#'        as well as the rejection decisions.
####################################################################################
DRPerm <- function(X, Y, W,
    model_registry = model_registry, 
    m_model = 'rf_regression',
    e_model = 'logistic_classification',
    tau_model = 'rf_regression',
    seed = 2026, n_folds = 5,
    clip_e = 0.01, n_perm = 150, alpha = 0.05){
  n = nrow(X)
  p = ncol(X)
  nuisance_fit <- cross_nuisance_fit_po(X, Y, W,
    seed = seed, n_folds = n_folds,
    m_model = m_model, e_model = e_model)
  m0_hat = nuisance_fit$m0_hat
  m1_hat = nuisance_fit$m1_hat
  e_hat = nuisance_fit$e_hat
  pseudo_outcome <- (Y - ifelse(W == 1, m1_hat, m0_hat)) * (W - e_hat)/(e_hat * (1 - e_hat)) + (m1_hat - m0_hat)
  tau_po_model <- model_registry[[tau_model]]
  tau_fit <- fit_predict_oof(
    X = X, y = pseudo_outcome, folds = n_folds,
    model_spec = tau_po_model, seed = seed
    )$pred
  observed_po_risk <- mean((pseudo_outcome - tau_fit)**2)
  permuted_po_risk <- numeric(n_perm)
  for(b in 1:n_perm){
    W_perm = sample(W)
    nuisance_fit_b <- cross_nuisance_fit_po(X, Y, W_perm,
      seed = (seed + b^2), n_folds = n_folds,
      m_model = m_model, e_model = e_model)
    m0_hat_b = nuisance_fit_b$m0_hat
    m1_hat_b = nuisance_fit_b$m1_hat
    e_hat_b = nuisance_fit_b$e_hat
    pseudo_outcome_b <- (Y - ifelse(W_perm == 1, m1_hat_b, m0_hat_b)) * (W_perm - e_hat_b)/(e_hat_b * (1 - e_hat_b)) + (m1_hat_b - m0_hat_b)
    tau_po_model_b <- model_registry[[tau_model]]
    tau_fit_b <- fit_predict_oof(X = X, y = pseudo_outcome_b,
      folds = n_folds, model_spec = tau_po_model, seed = (seed + b^3))$pred
    permuted_po_risk[b] <- mean((pseudo_outcome_b - tau_fit_b) ** 2)
    print(b)
  }
  p_value <- (1 + sum(permuted_po_risk > observed_po_risk))/(1 + n_perm)
  return(
    list(
      observed_po_risk = observed_po_risk,
      permuted_po_risk = permuted_po_risk,
      p_value = p_value,
      rejected = (p_value < alpha)
        )
    )
}



model_registry <- default_model_registry()
#No Distribution Shift:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
DRPerm(X, Y, W,
 model_registry = model_registry,
 n_perm = 25)$p_value


#Concept Drift:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,0,0,0,0,0,0), cor = 0.3, eps = 3)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
DRPerm(X, Y, W,
 model_registry = model_registry,
 n_perm = 100)$p_value


#Covariate Shift(Mean Shift):
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3, mean_X = 10)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
DRPerm(X, Y, W, 
  model_registry = model_registry,
  n_perm = 25)$p_value


#Covariate Shift(Variance Shift):
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3, var_X = 5)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
DRPerm(X, Y, W, n_perm = 25)$p_value











