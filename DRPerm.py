'''
Permutation Test for Distribution Shift via PO-risk(Pseudo-Outcome Risk) followed by the permute-then-refit procedure:

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
model_registry = ModelRegistry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)
def DRPerm(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, *,
    seed: int = 0, n_splits: int = 5,
    clip_e = 0.01, n_perm = 150, alpha = 0.05, return_detail = True,
    model_m = 'rf_regression', model_e = 'rf_classification'):
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
    # ----------------------------------------------------
    # Calculate the Pseudo-Outcome Scores/PO-risk
    # ----------------------------------------------------
    residual_y = Y - mu_hat
    residual_t = T - e_hat
    pseudo_outcome = residual_y * residual_t
    # ----------------------------------------------------
    # Estimate the unpermuted PO-risk via outcome model
    # ----------------------------------------------------
    fit_tau = model_outcome['fit'](X, pseudo_outcome, seed = seed + 200 + k)
    tau_score = model_outcome['predict'](fit_tau, X)
    observed_po_risk = np.mean(tau_score ** 2)
    # ----------------------------------------------------
    # Permuted version of the PO-risk
    # ----------------------------------------------------
    permuted_po_risk = np.zeros(n_perm, dtype = float)
    for b in range(n_perm):
        t_perm = rng.permutation(t)
        fit_e_perm = model_propensity_score['fit'](
          X, t_perm, seed = seed + 300 + b
        )
        e_perm = model_propensity_score['predict'](
          fit_e_perm, X
        )
        e_perm = np.clip(e_perm, clip_e, 1 - clip_e)
        residual_t_perm = t_perm - e_perm
        pseudo_perm = residual_y * residual_t_perm
        fit_tau_perm = model_outcome['fit'](
          X, pseudo_perm, seed = seed + 400 + b
        )
        tau_perm = model_outcome['predict'](
          fit_tau_perm, X
        )
        permuted_po_risk[b] = np.mean(tau_perm ** 2)
    p_value = (1.0 + np.sum(permuted_po_risk >= observed_po_risk))/(n_perm + 1)
    out = {
      'statistic': float(observed_po_risk),
      'p_value': float(p_value),
      'reject': bool(p_value < alpha),
      'alpha': float(alpha),
      'model_outcome': model_outcome['name'],
      'model_propensity_score': model_propensity_score['name']
    }
    if return_detail:
        out.update({
          'mu_hat': mu_hat, 'e_hat': e_hat,
          'residual_y': residual_y, 'residual_t': residual_t,
          'pseudo_outcome': pseudo_outcome, 'tau_score': tau_score,
          'permuted_po_risk': permuted_po_risk
        })
    return out