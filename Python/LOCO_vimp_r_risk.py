#LOCO r-risk variable importance:
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from model_registry_class import ModelRegistry
from utils import _as_1d


model_factory = ModelRegistry(
ntree = 150, ridge_alpha = 0.25,
nthread = 1, maxit = 500,
max_depth = 5, gamma = 0.25,
eta = 0.15, mlp_hidden_size = 4,
mlp_decay = 1e-4, mlp_max_iter = 500
)
MODEL_REGISTRY = model_factory.as_r_style_dict()


def _crossfit_nuisance(X, Y, W,
 model_registry = MODEL_REGISTRY,
 model_e='rf_classifier', model_m='rf_regressor', seed=0, n_splits=5, clip_e=0.01):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n, p = X.shape
    cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)
    model_propensity_score = model_registry[model_e]
    model_outcome = model_registry[model_m]
    m_hat = np.zeros(n, dtype = float)
    e_hat = np.zeros(n, dtype = float)
    for tr, te in cv.split(X, W):
        Xm_tr, Xm_eval = X[tr], X[te]
        Y_tr, W_tr = Y[tr], W[tr]
        model_ps_fit = model_propensity_score['fit'](
            Xm_tr, W_tr, seed = seed
        )
        e_hat[te] = model_propensity_score['predict'](
            model_ps_fit, Xm_eval
        )
        model_outcome_fit = model_outcome['fit'](
            Xm_tr, Y_tr, seed = seed
        )
        m_hat[te] = model_outcome['predict'](
            model_outcome_fit, Xm_eval
        )
    e_hat = np.clip(e_hat, clip_e, 1.0 - clip_e)
    return m_hat, e_hat


#Return the np.ndarray here:
def _fit_rlearner_tau(
    X: np.ndarray, Y_tilde: np.ndarray,
    W_tilde: np.ndarray, *,
    seed: int = 0, clip_wtilde: float = 1e-3):
    X = np.asarray(X)
    Y_tilde = _as_1d(Y_tilde).astype(float)
    W_tilde = _as_1d(W_tilde).astype(float)
    W_tilde = np.clip(W_tilde, clip_wtilde, 1 - clip_wtilde)
    #mask = np.abs(W_tilde) > clip_wtilde
    denom = np.where(
        np.abs(W_tilde) < clip_wtilde,
        clip_wtilde * np.where(W_tilde >= 0, 1.0, -1.0),
        W_tilde
    )
    z = Y_tilde/denom
    weights = W_tilde ** 2
    tau_model = Pipeline(
        steps = [
        ('scaler', StandardScaler(with_mean = True, with_std = True)),
        ('ridge', Ridge(alpha = 1.0, random_state = seed))]
    )
    tau_model.fit(X, z, ridge__sample_weight = weights)
    return tau_model

def r_risk(Y_tilde: np.ndarray,
           W_tilde: np.ndarray,
           tau_hat: np.ndarray) -> float:
    """
    The R-risk: R_{n} = \frac{1}{n}sum_{i=1}^{n}(Y_i - m(X_i) - \tau_hat*(T_i - e(X_i)))^2)
    """
    Y_tilde = _as_1d(Y_tilde)
    W_tilde = _as_1d(W_tilde)
    tau_hat = _as_1d(tau_hat)
    return float(np.mean((Y_tilde - tau_hat * W_tilde) ** 2))
    

def vimp_loco_r_risk(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray,
    model_m='rf_regressor', model_e='logistic_classifier',
    *, model_registry=MODEL_REGISTRY,
    seed: int = 0, n_splits=5, clip_e: float = 0.01,
    clip_wtilde: float = 1e-3, normalize: bool = False):
    '''
    LOCO variable importance for the causal learner via the R-risk:
    Steps:
    (1) Cross-fitting procedure on the nuisance parameters: m_hat(X) and e_hat(X)
    (2) Calculate the W_tilde and W_tilde: Y_tilde = Y - m_hat(X), W_tilde = W - e_hat(X)
    '''
    #We'd better fitting this from scratch here!
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    p = X.shape[1]
    m_hat, e_hat = _crossfit_nuisance(
        X, Y, W, model_registry=model_registry,
        model_e=model_e, model_m=model_m, seed=seed,
        n_splits=n_splits, clip_e=clip_e,
    )
    Y_tilde = Y - m_hat
    W_tilde = W - e_hat
    tau_unpermuted_model = _fit_rlearner_tau(X, Y_tilde, W_tilde)
    tau_hat_unpermuted = tau_unpermuted_model.predict(X)
    imps = np.zeros(p, dtype = float)
    r_risk_unpermuted = r_risk(Y_tilde, W_tilde, tau_hat_unpermuted)
    #calculate the R-risk here:
    for j in range(p):
        X_minus_j = np.delete(X, j, axis = 1)
        tau_permuted_model = _fit_rlearner_tau(X_minus_j, Y_tilde, W_tilde, seed = 2026 + j)
        tau_hat_permuted = tau_permuted_model.predict(X_minus_j)
        r_risk_j = r_risk(Y_tilde, W_tilde, tau_hat_permuted)
        imps[j] = r_risk_j - r_risk_unpermuted
    return imps/np.sum(imps) if normalize else imps




    

#Test case:
#X = np.random.random((100, 20))
#Y = np.random.random((100, ))
#W = np.random.choice((0,1), 100)
#vimp_loco_r_risk(X, Y, W, model_m = 'rf_regressor', model_e = 'rf_classifier')
















