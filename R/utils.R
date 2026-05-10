source('model_registry.R')
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
##############################################
r_risk <- function(Y_tilde, W_tilde, tau){
  mean((Y_tilde - W_tilde * tau)^2)
}


##########################################
#Calculating the PO(pseudo-outcome) risk:
##########################################
po_risk <- function(tau_pred, Y, T, mu0, mu1, e, clip_eps = 1e-3){
  pi_clip <- pmax(pmin(e, 1 - clip_eps), clip_eps)
  tau_po <- (Y - ifelse(T == 1, mu1, mu0)) * (T - pi_clip)/(pi_clip * (1 - pi_clip))
  return(mean((tau_po - tau_pred)^2))
}

####################################################
#Fitting the R-learner via the weighted regression:
####################################################
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
  Xw <- as.matrix(X_design * W_weights)
  yw <- Y_tilde * W_weights
  XtX <- t(Xw) %*% Xw
  Xty <- t(Xw) %*% yw
  #Solving for the beta_hat here:
  beta <- solve(XtX + ridge_lambda * diag(c(0, rep(1, ncol(X_scaled)))), Xty)
  pred_function <- function(Xnew){
    Xnew <- as.matrix(Xnew)
    Xs_new <- sweep(sweep(Xnew, 2, mu, '-'), 2, sd, '/')
    as.numeric(cbind(1, Xs_new) %*% beta)
  }
  return(list(coef = beta, mu = mu, sd = sd, predict = pred_function))
}
pred_tau <- function(rlearn_obj, Xnew){
  Xnew <- as.matrix(Xnew)
  Xs_new <- sweep(sweep(Xnew, 2, rlearner_obj$mu, '-'), 2, rlearner_obj$sd, '/')
  return(as.numeric(cbind(rep(1, nrow(Xnew)), Xnew) %*% rlearner_obj$beta))
}


###########################################################################################
#' Do the prediction in the out-of-fold:
#' @param X,y:        Covariates and the Corresponding Responses
#' @param folds:      Number of folds
#' @param model_spec: Model Specification
#' @param seed:       Random Seed
###########################################################################################
fit_predict_oof <- function(X, y, folds, model_spec, seed){
  X <- as.data.frame(X)
  y <- as.vector(y)
  n <- nrow(X)
  pred <- numeric(n)
  fits <- vector('list', length(folds))
  for(k in 1:length(folds)){
    te <- folds[[k]]
    tr <- setdiff(c(1:n), te)
    X_tr <- X[tr, , drop = FALSE]#ensure that the dimension is not reduced here.
    X_te <- X[te, , drop = FALSE]
    y_tr <- y[tr]
    fold_seed <- if (!is.null(seed)) seed + k else NULL
    model_fit <- model_spec$fit(
       X = X_tr, y = y_tr, seed = fold_seed
    )
    pred_te <- model_spec$predict(fit = model_fit, X_new = X_te)
    pred[te] <- as.numeric(pred_te)
    fits[[k]] <- model_fit
  }
  return(
    list(
        pred = pred,
        fits = fits,
        model_name = model_spec$name
      )
    )
}

########################################################################################
#' Fit the outcome model and propensity score model for the pseudo-outcome learner and X-learner with two separate outcome models.
#' Under these circumstances, you need an outcome model for both treatment group and control group.
#' Given a matrix/data.frame with two batches where for the existing batch T = 0, new batch T = 1'
#' Performs the following operations:
#' Options for the propensity score model: Logistic Regression, Random Forest, XGBoost and MLP
#' Options for the outcome model: Random Forest, XGBoost and the MLP,
#'@param X:                   Numeric matrix/data.frame of dimension (n_exist + n_new) * (p + 2), p is the number of features, with the response column(the second last column) and the batch assignment column(the last column)
#'                            For the batch assignment column T(p+2-th column), the first n_exist value represent batch 0 and the next n_new value represent batch 1        
#'@param Y: 
#'@param T: 
#'@param outcome_model:       Type of the outcome model, on T == 1 and T == 0 respectively.
#'@param n_folds:             Number of folds of cross-fitting
#'@param propensity_model:    Type of the propensity score model
#'@param n_estimator':        Number of the Trees for the outcome model and propensity score model.
#'@param binary_outcome:      Whether the response(Y) is binary or not
#'@return vimp_result:        Numeric vector-list of (p) for each of the feature
#' Cross-Fitting on the Nuisance Parameters:
#
#' Given a matrix/data.frame with two batches where for the existing batch T = 0, new batch T = 1'
#' Performs the following operations:
#' Fit the propensity score model and the outcome model with the clip here.
########################################################################################
cross_nuisance_fit_po <- function(
  X, Y, T, n_folds = 5,
  n_estimator = 100, 
  m_model = 'xgb_regression', e_model = 'rf_classification',
  clip_e = 1e-3, seed = 2026, size = 4, maxit = 20, lambda = 0.05){
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  T <- as.numeric(T)
  p <- ncol(X)
  if(p <= 0){
    stop(sprintf('Must include at least one covariate in X!'))
  }
  if(n <= 2){
    stop(sprintf('Must include at least 1 observation per batch'))
  }
  #create folds:
  folds <- caret::createFolds(
    y = factor(T), k = n_folds, list = TRUE,
    returnTrain = FALSE
    )
  m_hat <- rep(0, n)
  e_hat <- rep(0, n)
  scores <- rep(0, n)
  fits_m1 <- vector('list', length(folds))
  fits_m0 <- vector('list', length(folds))
  fits_e <- vector('list', length(folds))
  model_regis <- default_model_registry(
    ntree = n_estimator,
    size = size,
    maxit = maxit,
    lambda = lambda
    )
  if(!(m_model %in% names(model_regis))){
    stop(
      'm_model is not included in the default_model_registry.  Available models are: ',
      paste(names(model_regis), collapse = ','),
      call. = FALSE
    )
  }
  if(!(e_model %in% names(model_regis))){
    stop(
      'e_model is not included in the default_model_registry. Available models are: ',
      paste(names(model_regis), collapse = ','),
      call. = FALSE
    )
  }
  model_m1 <- model_regis[[m_model]]
  model_m0 <- model_regis[[m_model]]
  model_e <- model_regis[[e_model]]
  Y_T0 = Y[W==0]
  Y_T1 = Y[W==1]
  X_T0 = X[W==0, ]
  X_T1 = X[W==1, ]
  m0_fit <- fit_predict_oof(
    X = X_T0, y = Y_T0, folds = folds,
    model_spec = model_m0, seed = seed
    )
  m1_fit <- fit_predict_oof(
    X = X_T1, y = Y_T1, folds = folds,
    model_spec = model_m1, seed = seed
  )
  #Fit the Propensity Score Model:
  e_fit <- fit_predict_oof(
    X = X, y = T, folds = folds,
    model_spec = model_e, seed = seed
  )
  m0_hat = m0_fit$pred
  m1_hat = m1_fit$pred
  e_hat = e_fit$pred
  e_hat = pmin(pmax(e_hat, clip_e), 1 - clip_e)
  return(list(
      m0_hat = m0_hat,
      m1_hat = m1_hat,
      e_hat = e_hat,
      m_model_name = m_fit$model_name,
      e_model_name = e_fit$model_name,  
      m0_fits = m0_fit$fits,
      m1_fits = m1_fit$fits,
      e_fits = e_fit$fits
  ))
}

########################################################################################
#' Fit the outcome model and propensity score model for R-learner.
#' Given a matrix/data.frame with two batches where for the existing batch T = 0, new batch T = 1'
#' Performs the following operations:
#' 
#' Options for the propensity score model: Logistic Regression, Random Forest, XGBoost and MLP
#' Options for the outcome model: Random Forest, XGBoost and the MLP,
#'@param df:                  Numeric matrix/data.frame of dimension (n_exist + n_new) * (p + 2), p is the number of features, with the response column(the second last column) and the batch assignment column(the last column)
#'                            For the batch assignment column T(p+2-th column), the first n_exist value represent batch 0 and the next n_new value represent batch 1        
#'@param outcome_model:       Type of the outcome model
#'@param n_folds:             Number of folds of cross-fitting
#'@param propensity_model:    Type of the propensity score model
#'@param n_estimator':        Number of the Trees for the outcome model and propensity score model.
#'@param binary_outcome:      Whether the response(Y) is binary or not
#'@return vimp_result:        Numeric vector-list of (p) for each of the feature
#' Cross-Fitting on the Nuisance Parameters:
#
#' Given a matrix/data.frame with two batches where for the existing batch T = 0, new batch T = 1'
#' Performs the following operations:
#' Fit the propensity score model and the outcome model with the clip here.
########################################################################################
cross_nuisance_fit <- function(
    X, Y, T, 
    n_folds = 5, 
    n_estimator = 100,
    m_model = 'xgb_regression',
    e_model = 'rf_classification',
    clip_e = 1e-3, 
    seed = 2026){
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  T <- as.numeric(T)
  p <- ncol(X)
  if(p <= 0){
    stop(sprintf('Must include at least one covariate in X!'))
  }
  if(n <= 2){
    stop(sprintf('Must include at least 1 observation per batch'))
  }
  #Cross-Fitting:
  folds <- caret::createFolds(
    y = factor(T),
    k = n_folds,
    list = TRUE,
    returnTrain = FALSE
    )
  m_hat <- rep(0, n)
  e_hat <- rep(0, n)
  scores <- rep(0, n)
  fits_m <- vector('list', length(folds))
  fits_e <- vector('list', length(folds))
  #Fit the Outcome Model:
  model_regis <- default_model_registry(
    ntree = n_estimator,
    size = 4,
    maxit = 20,
    lambda = 0.01
    )
  if(!(m_model %in% names(model_regis))){
    stop(
      'm_model is not included in the default_model_registry.  Available models are: ',
      paste(names(model_regis), collapse = ','),
      call. = FALSE
    )
  }
  if(!(e_model %in% names(model_regis))){
    stop(
      'e_model is not included in the default_model_registry. Available models are: ',
      paste(names(model_regis), collapse = ','),
      call. = FALSE
    )
  }
  model_m <- model_regis[[m_model]]
  model_e <- model_regis[[e_model]]
  m_fit <- fit_predict_oof(
    X = X, y = Y, folds = folds,
    model_spec = model_m, seed = seed
  )
  #Fit the Propensity Score Model:
  e_fit <- fit_predict_oof(
    X = X, y = T, folds = folds,
    model_spec = model_e, seed = seed
  )
  #results for the outcome model here:
  m_hat <- m_fit$pred
  #propensity score clipping:
  e_hat <- pmin(pmax(e_fit$pred, clip_e), 1 - clip_e)
  return(list(
      m_hat = m_hat,
      e_hat = e_hat,
      m_model_name = m_fit$model_name,
      e_model_name = e_fit$model_name,  
      m_fits = m_fit$fits,
      e_fits = e_fit$fits
  ))
}










