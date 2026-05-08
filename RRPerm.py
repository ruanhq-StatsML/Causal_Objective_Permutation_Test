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
    seed: int = 0, n_splits: int = 5, binary_outcome = False,
    clip_e = 0.01, n_perm = 150, alpha = 0.05, return_detail = True,
    model_m = 'rf_regressor', model_e = 'rf_classifier'):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    # ----------------------------------------------------
    # Cross-fitted nuisance estimation:
    # ----------------------------------------------------
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat  = np.zeros(n, dtype = float)
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        T_train = t[train_idx]
        #specify the outcome model and return it for each of the fold:
        model_propensity_score = model_registry[model_e]
        model_outcome = model_registry[model_m]
        fit_mu = model_outcome['fit'](
          X_train, Y_train, seed = seed + k
          )
        mu_hat[test_idx] = model_outcome['predict'](
          fit_mu, X_test
          )
        fit_e = model_propensity_score['fit'](
          X_train, T_train, seed = seed + 100 + k
          )
        e_hat[test_idx] = model_propensity_score['predict'](
          fit_e, X_test
          )
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e) 
    # --------------------------------------------------------------
    # Calculate the R-risk followed by the weighted average tau for 
    # --------------------------------------------------------------
    Y_tilde = Y - mu_hat
    T_tilde = T - e_hat
    denom = np.where(np.abs(T_tilde) < clip_e, np.sign(T_tilde) * clip_e, T_tilde)
    denom = np.where(denom == 0, clip_e, denom)
    pseudo_tau = Y_tilde/denom
    fit_tau = model_tau['fit'](
      X, pseudo_tau, seed = seed + 200 + k
    )
    tau_hat = model_tau['predict'](
      fit_obj, X
    )
    observed_r_risk = np.mean((Y_tilde - tau_hat * T_tilde) ** 2)
    # ----------------------------------------------------
    # Permuted version of the PO-risk
    # ----------------------------------------------------
    permuted_r_risk = np.zeros(n_perm, dtype = float)
    for b in range(n_perm):
        t_perm = rng.permutation(t)
        fit_e_perm = model_propensity_score['fit'](
          X, t_perm, seed = seed + 300 + b
        )
        e_perm = model_propensity_score['predict'](
          fit_e_perm, X
        )
        e_perm = np.clip(e_perm, clip_e, 1 - clip_e)
        T_tilde_perm = t_perm - e_perm
        denom_perm = np.where(np.abs(T_tilde_perm) < clip_e, np.sign(T_tilde_perm) * clip_e, T_tilde_perm)
        denom_perm = np.where(denom_perm == 0, clip_e, denom_perm)
        pseudo_tau_perm = Y_tilde/denom_perm
        fit_tau_perm = model_outcome['fit'](
          X, pseudo_tau_perm, seed = seed + 400 + b
        )
        tau_perm = model_outcome['predict'](
          fit_tau_perm, X
        )
        permuted_r_risk[b] = np.mean((Y_tilde - tau_perm * T_tilde_perm) ** 2)
    p_value = (1.0 + np.sum(permuted_r_risk >= observed_r_risk))/(n_perm + 1)
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



#Test cases:
rng = np.random.default_rng(2023)
n = 400
p = 16
X = rng.normal(size = (n, p))
Y = 10.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(scale = 1.0, size = n)
W = rng.binomial(1, 0.5, size = n)
output = RRPerm(X, Y, W, n_splits = 5, model_m = 'rf_regressor', model_e = 'rf_classifier')
output
















