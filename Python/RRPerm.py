from utils import *


'''
Permutation Test for Distribution Shift via R-risk followed by the permute-then-refit procedure:
Hyperparameters:
- X:               Covariates array - (n_exist + n_new, p)
- Y:               Responses - (n_exist + n_new, )
- W:               Treatment(Batch) Assignment - (n_exist + n_new, )
- seed:            Random State Seeds
- n_split:         Number of splits
- clip_e:          Clipping value for the propensity score
- n_perm:          Number of permutations for the test
- alpha:           Significance Level for the test
- model_m:         Outcome Model
- model_e:         Propensity Score Model
Returns the cross-fitted value (m_hat, e_hat) evaluated on the whole data
as well as the fitted outcome model and propensity score models.
'''
#Specify the model registry factory so that we can fetch the model with flexibility afterwards:
model_registry = default_model_registry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)
def RRPerm(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, *,
    model_registry = model_registry,
    model_m = 'rf_regressor', model_e = 'rf_classifier',
    seed: int = 0, n_folds: int = 5, binary_outcome = False,
    clip_e = 0.01, n_perm = 150, alpha = 0.05, return_detail = True):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    #---------------------------------------------------------------------
    # Cross-fitted nuisance estimation, via the pseudo-outcome regression
    #---------------------------------------------------------------------
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat  = np.zeros(n, dtype = float)
    mu_hat, e_hat = cross_nuisance_fit(
      X, Y, W, n_folds = n_folds, model_registry = model_registry,
      model_m = model_m, model_e = model_e
    )
    #---------------------------------------------------------------------
    # Calculate the R-risk followed by the weighted regression
    #---------------------------------------------------------------------
    Y_tilde = Y - mu_hat
    W_tilde = W - e_hat
    tau_model = _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed = seed)
    tau_hat = tau_model.predict(X)
    observed_r_risk = np.mean((Y_tilde - tau_hat * W_tilde) ** 2)
    #---------------------------------------------------------------------
    # Permute-then-refit for re-estimation of the R-risk
    #---------------------------------------------------------------------
    permuted_r_risk = np.zeros(n_perm, dtype = float)
    for b in range(n_perm):
        W_perm = rng.permutation(W)
        #Refit the model:
        mu_hat_b, e_hat_b = cross_nuisance_fit(
          X, Y, W_perm, n_folds = n_folds, model_registry = model_registry,
          model_m = model_m, model_e = model_e
          )
        Y_tilde_b = Y - mu_hat_b
        W_tilde_b = W - e_hat_b
        tau_model_b = _fit_tau_rlearner_weighted(X, Y_tilde_b, W_tilde_b, seed = seed)
        tau_hat_b = tau_model_b.predict(X)
        permuted_r_risk[b] = np.mean((Y_tilde_b - tau_hat_b * W_tilde_b) ** 2)
    p_value = (1.0 + np.sum(permuted_r_risk >= observed_r_risk))/(1.0 + n_perm)
    out = {
      'statistic': float(observed_r_risk),
      'p_value': float(p_value),
      'reject': np.bool_(p_value < alpha),
      'alpha': float(alpha),
      'model_outcome': model_outcome['name'],
      'model_propensity_score': model_propensity_score['name']
    }
    if return_detail:
        out.update({
          'mu_hat': mu_hat, 'e_hat': e_hat,
          'residual_y': residual_y, 'residual_t': residual_t,
          'pseudo_outcome': pseudo_outcome, 'tau_hat': tau_hat,
          'permuted_r_risk': permuted_r_risk
        })
    return out



#Test cases: H_{0}
rng = np.random.default_rng(2023)
n = 400
p = 16
X = rng.normal(size = (n, p))
Y = 10.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(scale = 1.0, size = n)
W = rng.binomial(1, 0.5, size = n)
output = RRPerm(X, Y, W, n_splits = 5,
 model_registry = model_registry, model_m = 'rf_regressor', model_e = 'rf_classifier')
output



