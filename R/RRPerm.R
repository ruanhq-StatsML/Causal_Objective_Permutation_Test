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
    m_model = 'rf_regression', e_model = 'rf_classification',
    ridge_lambda = 0.25,
    clip_e = 0.01, n_perm = 150, alpha = 0.05, clip_wtilde = 0.01){
  n = nrow(X)
  p = ncol(X)
  nuisance_fit <- cross_nuisance_fit(X, Y, W,
      seed = seed, n_folds = n_folds,
      m_model = m_model, e_model = e_model)
  Y_tilde = Y - nuisance_fit$m_hat
  W_tilde = W - nuisance_fit$e_hat
  tau_full_model <- fit_tau_rlearner_weighted(X, Y_tilde, W_tilde,
      ridge_lambda = ridge_lambda, 
      seed = seed, clip_wtilde = clip_wtilde)
  #tau_full <- pred_tau(tau_full_model, X)
  tau_full <- tau_full_model$predict(X)
  r_risk_original <- r_risk(Y_tilde, W_tilde, tau_full)
  r_risk_perm_list <- numeric(n_perm)
  for(b in 1:n_perm){
    W_perm <- sample(W)
    #Re-estimate the R-risk following the cross-fitted procedure.
    nuisance_fit_b <- cross_nuisance_fit(X, Y, W_perm,
        seed = (seed + b^2), n_folds = n_folds, 
        m_model = m_model, e_model = e_model)
    Y_tilde_b <- Y - nuisance_fit_b$m_hat
    W_tilde_b <- W - nuisance_fit_b$e_hat
    tau_b_model <- fit_tau_rlearner_weighted(X, Y_tilde_b, W_tilde_b,
        ridge_lambda = ridge_lambda, seed = seed, clip_wtilde = clip_wtilde)
    #Putting the prediction wrapper within:
    tau_b <- tau_b_model$predict(X)
    #tau_b <- pred_tau(tau_b_model, X)
    r_risk_perm_list[b] <- r_risk(Y_tilde_b, W_tilde_b, tau_b)
    print(b)
  }
  p_value <- (1 + sum(r_risk_perm_list > r_risk_original))/(1 + n_perm)
  rejected <- (p_value < alpha)
  return(list(
    r_risk_original = r_risk_original,
    r_risk_perm_list = r_risk_perm_list,
    p_value = p_value,
    rejected = rejected
  ))
}

#Testing cases:
LM_generation <- function(n, beta_hat, cor, n_nuisance = 12, eps, mean_X = 0, var_X = 1){
  p <- length(beta_hat)
  corr_matrix <- matrix(0, p, p)
  diag(corr_matrix) <- 1
  for(i in 1:p){
    for(j in 1:p){
      corr_matrix[i, j] <- cor ^ (abs(i - j)) * sqrt(var_X)
    }
  }
  X_design <- mvrnorm(n, mu = rep(mean_X, p), corr_matrix)
  Y1 <- as.matrix(X_design) %*% as.matrix(beta_hat, nrow = p)
  random_error <- rnorm(n, 0, eps)
  #adding the nuisance random noise features:
  X_nuiss <- matrix(0, n, n_nuisance)
  for(i in 1:n_nuisance){
    X_nuiss[,i] <- rnorm(n, 0, 1)
  }
  Y <- Y1 + random_error
  df_return <- data.frame(cbind(Y1, X_design, X_nuiss, Y))
  ncol_df <- ncol(df_return)
  colnames(df_return) <- c("Y1", paste("X", c(1:p), sep = ""), paste("X_nuis", c(1:n_nuisance), sep = ""), "Y")
  X_return <- df_return[,2:(ncol_df - 1)]
  return(list(df_return = df_return, X_return = X_return))
}

####################################################################
#Type-I error:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
RRPerm(X, Y, W, n_perm = 25)$p_value


#Concept Drift:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,0,0,0,0,0,0), cor = 0.3, eps = 3)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
RRPerm(X, Y, W, n_perm = 25)$p_value


#Mean Shift:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3, mean_X = 5)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
RRPerm(X, Y, W,
 e_model = 'logistic_classification',
 n_perm = 25)$p_value


#Variance Shift:
df_exist <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3)$df_return
df_new <- LM_generation(n = 1000, beta_hat = c(1,1,1,1,1,1,1,1), cor = 0.3, eps = 3, var_X = 5)$df_return
W <- c(rep(0, nrow(df_exist)), rep(1, nrow(df_new)))
X <- rbind(df_exist[,2:(ncol(df_exist)-1)], df_new[,2:(ncol(df_new)-1)])
Y <- c(df_exist$Y, df_new$Y)
RRPerm(X, Y, W,
 seed = 2026,
 m_model = 'rf_regression',
 e_model = 'logistic_classification',
 n_perm = 30)$p_value









