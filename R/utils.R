#Util functions:
#Model Factory Registry:

#Inferring the type for the response:
infer_type <- function(X){
  p <- ncol(X)
  feature_type = rep('', p)
  for(j in 1:p){
    if(len(unique(X[, j])) <= 2){
      feature_type[j] = 'binary'
    }
  }
}

##############################################
#Calculating the R-risk:
#r_risk:
#'@param Y_tilde       :Y-m(X)
#'@param W_tilde       :(T-e(X))
#'@param tau           :tau(X)
r_risk <- function(Y_tilde, W_tilde, tau){
  mean((Y_tilde - W_tilde * tau)^2)
}



##############################################
#Calculating the PO(pseudo-outcome) risk:
po_risk <- function(tau_pred, Y, T, mu0, mu1, e, clip_eps = 1e-3){
  pi_clip <- pmax(pmin(e, 1 - clip_eps), clip_eps)
  tau_po <- (Y - ifelse(T == 1, mu1, mu0)) * (T - pi_clip)/(pi_clip * (1 - pi_clip))
  return(mean((tau_po - tau_pred)^2))
}


#Fitting the R-learner via the weighted regression:
fit_tau_rlearner_weighted <- function(X, Y_tilde, W_tilde, ridge_lambda,
    clip_wtilde = 1e-4, seed = 2026){
  mask <- (abs(W_tilde) > clip_wtilde)
  if(!is.null(seed)){
    set.seed(as.integer(seed))
  }
  X0 = X[mask, , drop = FALSE]
  z <- Y_tilde[mask]/W_tilde[mask]
  w <- (W_tilde[mask]^2)
  mu <- colMeans(X0)
  sd <- apply(X0, 2, sd)
  #Scaling the data here:
  X_scaled <- sweep(sweep(X0, 2, mu, '-'), 2, sd, '/')
  X_design <- cbind(rep(1, nrow(X_scaled)), X_scaled)
  #Solving a Weighted Ridge Regression:
  W_weights <- sqrt(w)
  Xw <- X_design %*% W_weights
  yw <- z * W_weights
  XtX <- t(Xw) %*% Xw
  Xty <- t(Xw) %*% yw
  #Solving for the beta_hat here:
  beta <- solve(XtX + ridge_lambad * diag(c(0, rep(1, ncol(X_design)))), Xty)
  pred_function <- function(Xnew){
    Xnew <- as.matrix(Xnew)
    Xs_new <- sweep(sweep(Xnew, 2, mu, '-'), 2, sd, '/')
    as.numeric(cbind(1, Xs_new) %*% beta)
  }
  return(list(coef = beta, mu = mu, sd = sd, predict = pred_function))
}

#











