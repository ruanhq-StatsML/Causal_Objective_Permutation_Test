#Model Registry:
library(nnet)
library(glmnet)
library(xgboost)
library(randomForest)
library(nnet)

##########################################################################################
#Model Registry: Incorporate all the possible models here for the model configurations.
##########################################################################################
#Objective:
#    Construct a named registry of individual model adapters used by the nuisance estimation layer.
#
#Inputs:
#'@param -- ntree:          Number of trees in the Random Forest or XGBoost Model
#'@param -- positive_class: Class label whose predicted probability is returned by classification adapters
#'@param -- lambda:         L-2 penalty in the ridge regression
#'@param -- nthread:        number of threads used by the xgboost package
#'@param -- size:           The number of units in the hidden dimension
#'@param -- maxit:          Maximum number of iterations for the NNet here.
#'@param -- MaxNWts:        Maximum allowable amount of weights
#
#Output Contract:
#    Returns a named list of model adapters, each adapter must contain:
#      -- name:     character identifier
#      -- fit:      function(X, y, seed = NULL) return the fitted model object
#      -- predict:  function(fit, X_new) returning a numeric vector of length nrow(X_new)
#
#Design:            The higher level model class with the model configurations
#Call:              models <- default_model_registry()
#                   m_model <- models$rf_regression
#                   e_model <- models$logistic_classification
##########################################################################################
default_model_registry <- function(
    ntree = 200, positive_class = 1,
    lambda = 0.01, nthread = 1, maxit = 200,
    xgb_max_depth = 4, xgb_gamma = 0.5, xgb_eta = 0.1,
    size = 5){
  ####
  list(
    rf_regression = make_rf_regression(
      ntree = ntree
    ),
    rf_classification = make_rf_classification(
      ntree = ntree,
      positive_class = positive_class
      ),
    lm_regression = make_lm_regression(),
    logistic_classification = make_logistic_classification(),
    xgb_regression = make_xgboost_regressor(
      ntree = ntree,
      max_depth = xgb_max_depth,
      gamma = xgb_gamma,
      eta = xgb_eta,
      nthread = nthread
      ),
    xgb_classification = make_xgb_classifier(
      ntree = ntree,
      max_depth = xgb_max_depth,
      gamma = xgb_gamma,
      eta = xgb_eta,
      nthread = nthread
      ),
    ridge_regression = make_ridge_regression(
      lambda = lambda
    ),
    ridge_classification = make_ridge_classification(
      lambda = lambda
    ),
    mlp_regression = make_mlp_regression(
      size = size, decay = 1e-5, maxit = maxit,
      linout = TRUE, trace = TRUE, MaxNWts = 10000
    ),
    mlp_classification = make_mlp_classifier(
      size = size, decay = 1e-5, maxit = maxit,
      trace = TRUE, MaxNWts = 8000
    ),
    multinomial_classifier = make_multinomial_classification(),
    mlp_multinomial = make_mlp_multinomial(
      size = size, decay = 1e-5, maxit = maxit,
      linout = TRUE, trace = TRUE, MaxNWts = 4000
    )
  )
}


make_mlp_regression <- function(
  size = 5,
  decay = 1e-5,
  maxit = 250, linout = TRUE,
  trace = FALSE, MaxNWts = 8000){
  list(
    name = 'mlp_regression',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      X <- as.matrix(X)
      y <- as.numeric(y)
      x_center <- colMeans(X)
      x_scale <- apply(X, 2, sd)
      X_scaled <- scale(X, center = x_center, scale = x_scale)
      fit = nnet::nnet(
        x = X_scaled,
        y = y,
        size = size,
        decay = decay,
        maxit = maxit,
        linout = linout,
        trace = trace,
        MaxNWts = MaxNWts
      )
      list(
        model = fit,
        x_center = x_center,
        x_scale = x_scale
      )
    },
    predict = function(fit, X_new){
      X_new <- as.matrix(X_new)
      X_new_scaled <- scale(
        X_new, center = fit$x_center,
        scale = fit$x_scale
      )
      as.numeric(predict(fit$model, X_new_scaled, type = 'raw'))
    }
  )
}

make_mlp_multinomial <- function(
  size = 5, decay = 1e-5,
  maxit = 500, linout = TRUE,
  trace = FALSE,
  MaxNWts = 80000
){
  list(
    name = 'mlp_multinomial',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      y <- as.factor(y)
      if(length(levels(y)) <= 2){
        stop('Please refer back to make_mlp_classifier for binary classification', call. = FALSE)
      }
      X <- as.matrix(X)
      X_center <- colMeans(X)
      X_scale <- apply(X, 2, sd)
      X_scaled = scale(X, center = X_center, scale = X_scale)
      fit <- nnet::nnet(
        x = X_scaled,
        y = as.factor(y),
        size = size, decay = decay,
        maxit = maxit, linout = linout,
        trace = trace, MaxNWts = MaxNWts
      )
      list(
        model = fit,
        X_center = X_center,
        X_scaled = X_scaled
      )
    },
    predict = function(fit, X_new){
      X_new = as.matrix(X_new)
      X_new_scaled <- scale(
        X_new, center = fit$X_center, scale = fit$X_scaled
        )
      as.numeric(predict(fit$model, X_new_scaled, type = 'class'))
    }
  )
}#list: name, fit_obj, predict.

#making the mlp classifier - binary: 
make_mlp_classifier <- function(
  size = 5,
  decay = 1e-5,
  maxit = 250,
  trace = FALSE, MaxNWts = 80000,
  positive_class = 1
){
  list(
    name = 'mlp_classifier',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      y <- as.factor(y)
      if(length(levels(y)) != 2){
        stop('MLP classification only focus on binary classification only now.', call. = FALSE)
      }
      X <- as.matrix(X)
      x_center <- colMeans(X)
      x_scale <- apply(X, 2, sd)
      X_scaled <- scale(X, center = x_center, scale = x_scale)
      y_num <- as.numeric(y == positive_class)
      fit = nnet::nnet(
        x = X_scaled,
        y = y_num,
        size = size,
        decay = decay,
        maxit = maxit,
        linout = FALSE,
        trace = trace,
        MaxNWts = MaxNWts
      )
      list(
        model = fit,
        x_center = x_center,
        x_scale = x_scale
      )
    },
    predict = function(fit, X_new){
      X_new <- as.matrix(X_new)
      X_new_scaled <- scale(
        X_new, center = fit$x_center, scale = fit$x_scale
      )
      as.numeric(predict(fit$model, X_new_scaled, type = 'raw'))
    }
  )
}

make_rf_regression <- function(
    ntree = 200,
    nodesize = NULL,
    mtry = NULL
){
  list(
    name = 'rf_regression',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      X <- as.data.frame(X)
      y <- as.numeric(y)
      p <- ncol(X)
      randomForest::randomForest(
        x = X,
        y = as.matrix(y),
        ntree = ntree, 
        nodesize = if (is.null(nodesize)){
          max(1, round(sqrt(nrow(X))))
        }
        else{
          nodesize
        },
        mtry = if(is.null(mtry)){
          max(1, round(p/3))
        }
        else{
          mtry
        }
      )
    },
    predict = function(fit, X_new){
      as.numeric(predict(fit, newdata = as.data.frame(X_new)))
    }
  )
}

make_rf_classification <- function(ntree = 200,
  nodesize = NULL, mtry = NULL, 
  positive_class = 1){
  list(
    name = 'rf_classification',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      X <- as.data.frame(X)
      p <- ncol(X)
      randomForest::randomForest(
        x = X,
        y = factor(y),
        ntree = ntree, 
        nodesize = if (is.null(nodesize)){
          max(1, round(sqrt(nrow(X))))
        }
        else{
          nodesize
        },
        mtry = if(is.null(mtry)){
          max(1, round(sqrt(p)))
        }
        else{
          mtry
        }
      )
    },
    predict = function(fit, X_new){
      prob <- predict(fit, newdata = as.data.frame(X_new),
        type = 'prob')
      if(!(positive_class %in% colnames(prob))){
        stop('positive class not found in predicted probability matrix', call. = FALSE)
      }
      as.numeric(prob[, positive_class])
    }
  )
}

make_multinomial_classification <- function(){
  list(
    name = 'multinomial_regression',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed))
      df = data.frame(y = factor(y), as.data.frame(X))
      nnet::multinom(y~., data = df, trace = FALSE) 
    },
    predict = function(fit, X_new){
      probs = predict(fit, newdata = as.data.frame(X_new),
        type = 'class'
         )
      as.numeric(as.character(probs))
    }
  )
}


make_lm_regression <- function(){
  list(
    name = 'linear_regression',
    fit = function(X, y, seed = NULL){
      df <- data.frame(y = as.numeric(y), as.data.frame(X))
      lm(y~., data = df)
    },
    predict = function(fit, X_new){
      as.numeric(predict(fit, newdata = as.data.frame(X_new)))
    }
  )
}


make_logistic_classification <- function(){
  list(
    name = 'logistic_classification',
    fit = function(X, y, seed = NULL){
      df <- data.frame(y = factor(y), as.data.frame(X))
      glm(y~., data = df, family = binomial())
    },
    predict = function(fit, X_new){
      probs = predict(fit, newdata = as.data.frame(X_new),
        type = 'response')
      as.numeric(probs)
    }
  )
}

make_xgboost_regressor <- function(
  ntree = 150, max_depth = 4, gamma = 0.5,
  eta = 0.1, nthread = 1
){
  list(
    name = 'xgb_regression',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      dtrain <- xgboost::xgb.DMatrix(
        data = as.matrix(X),
        label = as.numeric(y)
      )
      xgboost::xgboost(data = dtrain,
        max_depth = max_depth,
        gamma = gamma,
        eta = eta,
        nthread = nthread,
        nrounds = ntree, 
        objective = 'reg:squarederror',
        verbose = 0)
    },
    predict = function(fit, X_new){
      as.numeric(predict(fit, newdata = as.matrix(X_new)))#automatically output the response.
    }
  )
}

make_xgb_classifier <- function(
  ntree = 150, max_depth = 4,
  gamma = 0.5, eta = 0.1, nthread = 1
){
  list(
    name = 'xgb_classification',
    fit = function(X, y, seed = NULL){
      if(!is.null(seed)) set.seed(seed)
      dtrain <- xgboost::xgb.DMatrix(
        data = as.matrix(X),
        label = as.numeric(y)
      )
      xgboost::xgboost(
        data = dtrain,
        max_depth = max_depth,
        gamma = gamma,
        eta = eta,
        nthread = nthread,
        nrounds = ntree,
        objective = 'binary:logistic',
        verbose = 0
      )
    },
    predict = function(fit, X_new){
      as.numeric(predict(fit, newdata = as.matrix(X_new)))#automatically output the class probability here.
    }
  )
}

make_ridge_regression <- function(lambda = 0.01){
  list(
    name = 'ridge_regression',
    fit = function(X, y, seed = NULL){
        glmnet::glmnet(
            x = as.matrix(X),
            y = as.numeric(y),
            alpha = 0.0,
            lambda = lambda,
        )
    },
    predict = function(fit, X_new){
      as.numeric(predict(fit, newx = as.matrix(X_new),
        s = lambda))
    }
  )
}

make_ridge_classification <- function(lambda = 0.01){
  list(
    name = 'ridge_classification',
    fit = function(X, y, seed = NULL){
      glmnet::glmnet(
        x = as.matrix(X),
        y = factor(y),
        alpha = 0.0,
        lambda = lambda,
        family = 'binomial'
        )
    },
    predict = function(fit, X_new){
      as.numeric(predict(
        fit, newx = as.matrix(X_new),
        s = lambda, type = 'response'
      ))
    }
  )
}


validate_mdoel_spec <- function(model_spec, name = 'model_spec'){
  if(!is.list(model_spec)){
    stop(name, ' must be a list.', call. = FALSE)
  }
  if(is.null(model_spec$name) || 
    !is.character(model_spec$name) || 
    length(model_spec$name) != 1){
    stop(name, ' must contain a character scalar `name`.', call. = FALSE)
  }
  if(is.null(model_spec$fit) || !is.function(model_spec$fit)){
    stop(name, " must contain a function `fit`.", call. = FALSE)
  }
  if(is.null(model_spec$predict) || !is.function(model_spec$predict)){
    stop(name, " must contain a function `predict`.", call. = FALSE)
  }
  fit_args <- names(formals(model_spec$fit))
  predict_args <- names(formals(model_spec$predict))
  required_fit_args <- c("X", 'y')
  required_predict_args <- c("fit", "X_new")
  missing_fit_args <- setdiff(required_fit_args, fit_args)
  missing_predict_args <- setdiff(required_predict_args, predict_args)
  if(length(missing_fit_args) > 0){
    stop(
      name, "$fit is missing required arguments(s): ",
      paste(missing_fit_args, collapse = ', '),
      call. = FALSE
    )
  }
  if(length(missing_predict_args) > 0){
    stop(
      name, '$predict is missing required arguments(s): ',
      paste(missing_predict_args, collapse = ', '),
      call. = FALSE
    )
  }
  invisible(TRUE)
}















