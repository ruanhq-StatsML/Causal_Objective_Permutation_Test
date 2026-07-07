import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from econml.dml import CausalForestDML


def cf_variable_importance(
    X,
    Y,
    W,
    model_psm="logistic",
    model_outcome="rf",
    n_estimators=150,
    seed=0,
):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=float).ravel()

    model_t = LogisticRegression(max_iter=1000, random_state=seed)
    model_y = RandomForestRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=1
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        random_state=seed,
        discrete_treatment=True,
    )
    cf.fit(Y, W, X=X)
    return np.asarray(cf.feature_importances_, dtype=float)