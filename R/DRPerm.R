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
#'@param model_m       :Outcome Model
#'@param model_e       :Propensity Score Model
#
#
#
#
#
#
#
#
############################################################
DRPerm <- function(X, Y, W,
    seed: int = 2026, n_splits: int = 5,
    clip_e = 0.01, n_perm = 150, alpha = 0.05,
    return_detail = TRUE, 
    model_m, model_e){
  n = nrow(X)
  p = ncol(X)
  mu_hat = rep(0, n)
  e_hat = rep(0, n)
  shuffled_indices <- sample(1:n)
  fold_assignment = cut(seq(1, n), breaks = k, labels = FALSE)
  folds <- split(shuffled_indices, fold_assignment)
  for(k in 1:n_fold){
    
  }
}





