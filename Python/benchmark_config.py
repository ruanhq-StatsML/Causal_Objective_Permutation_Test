import numpy as np

from model_registry_class import ModelRegistry

# Covariate-shift gamma grid: 0, 0.06, ..., 0.30
COVARIATE_SHIFT_GAMMA_GRID = np.round(np.arange(0, 0.3 + 1e-9, 0.06), 2)

DEFAULT_MODEL_FACTORY = ModelRegistry(
    ntree=150,
    ridge_alpha=0.25,
    nthread=1,
    maxit=200,
    max_depth=5,
    gamma=0.25,
    eta=0.15,
    mlp_hidden_size=4,
    mlp_decay=1e-5,
    mlp_max_iter=500,
    mlp_trace=False,
    mlp_max_coef_reg=10000,
    mlp_max_coef_clf=10000,
    warn_xgb_labels=True,
    positive_class=1,
)

MODEL_REGISTRY = DEFAULT_MODEL_FACTORY.as_r_style_dict()
