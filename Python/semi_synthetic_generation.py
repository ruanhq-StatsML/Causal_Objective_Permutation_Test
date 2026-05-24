#Real-Data Application for semi-synthetic demonstration:
'''
Incrementally inject concept drift on the response Y
via adding the information from those covariates.

'''
import os
import numpy as np
import pandas as pd
from RRPerm import *
from DRPerm import *
import numpy as np
import pandas as pd
import shap
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from model_registry import ModelRegistry


model_registry = ModelRegistry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)

def inject_cd(df_new, feature_ind, beta_shift, noise_sd, seed = seed):
	rng = np.random.default_rng(seed = seed)
	if check_continuous(Y_new):
        Y_new = np.array(df_new[:, -1])
        beta_shift = beta_shift * np.random.choice([-1, 1], size = len(beta_shift))
        Y_new = beta_shift @ df_new[:, feature_ind] + rng.normal(size = Y_new.shape[0], loc = 0, scale = noise_sd)
        df_new[:, -1] = Y_new
    else:
        Y_new = np.array(df_new[:, -1])
        beta_shift = beta_shift * np.random.choice([-1, 1], size = len(beta_shift))
        Y_odd = np.expit(Y_new) + beta_shift @ df_new[:, feature_ind] + rng.normal(size = Y_new.shape[0], loc = 0, scale = noise_sd)
        Y_new = np.exp(Y_odd)/(1.0 + np.exp(Y_odd))
        df_new[:, -1] = Y_new
    return df_new

def test_wrapper(input_csv, col_split,
 feature_ind = np.array([0, 2, 5]),
 beta_shift = np.array([1, 2, 4]),
 noise_sd = 1, model_m = 'rf_regression', model_e = 'logistic_classification',
 B = 200, mode = 'median', seed = 2000):
    '''
    Shifts the active features here:
    y_shifted = logit(f(X) + X_{active}^{T}\beta) - classifier
    y_shifted = f(X) + X_{active}^{T}\beta
    '''
	df = pd.read_csv(input_csv)
	if mode == 'median':
        df_exist = df[df[col_split] <= df[col_split].median()]
        df_new = df[df[col_split] > df[col_split].median()]
    else:
        df_exist = df[df[col_split] <= df[col_split].mean()]
        df_new = df[df[col_split] > df[col_split].mean()]
    #R-risk permtuation test:
    sig_drperm = numeric(B)
    sig_rrperm = numeric(B)
    n1 = df_exist.shape[0]
    n2 = df_new.shape[0]
    for i in range(B):
        df1_B = df1.sample(n = n1, replace = True, random_state = seed + i)
        df2_B = df2.sample(n = n2, replace = True, random_state = seed + i)
        df2_B = inject_cd(df2_B, feature_ind, beta_shift, noise_sd, seed = seed + i)
        X = np.vstack([df1_B[:, :(df1_B.shape[1]-1)],
                       df2_B[:, :(df2_B.shape[1]-1)]])
        Y = np.concatenate([df1_B[:, df1_B.shape[1]],
                            df2_B[:, df2_B.shape[1]]])
        W = np.concatenate([np.zeros(n1), np.ones(n2)])
        result_drperm = DRPerm(X, Y, W, model_m = model_m,
            model_e = model_e)
        result_rrperm = RRPerm(X, Y, W, model_m = model_m,
            model_e = model_e)
        sig_drperm[i] = result_drperm['reject']
        sig_rrperm[i] = result_rrperm['reject']
    return {
        'power_drperm': np.mean(sig_drperm).astype(float),
        'power_rrperm': np.mean(sig_rrperm).astype(float)
    }


'''
Semi-Synthetic Generation for Boston Data
'''
feature_ind = np.array([0, 2, 4])
beta_shift = np.array([0.1, 0.1, 0.05])
power_df = []
for j in range(np.linspace(0, 1, 6)):
    power_drperm, power_rrperm = test_wrapper(
      input_csv = 'dataset/Boston.csv',
      col_split = 'V6',
      feature_ind = feature_ind,
      beta_shift = beta_shift)
    power_df.append([power_drperm, power_rrperm])

power_df = pd.DataFrame(power_df)
power_df.columns = ["power_drperm", "power_rrperm"]
pd.DataFrame(power_df).to_csv('power_boston_semi.csv')
    
'''
Semi-Synthetic Generation for Aquatic Toxicity Data
'''
power_df = []
feature_ind = np.array([0, 2, 4])
beta_shift = np.array([0.1, 0.1, 0.05])
for j in range(np.linspace(0, 1, 6)):
    power_drperm, power_rrperm = test_wrapper(
      input_csv = 'dataset/aquatic_toxicity.csv',
      col_split = 'V6',
      feature_ind = feature_ind,
      beta_shift = beta_shift)
    power_df.append([power_drperm, power_rrperm])

power_df = pd.DataFrame(power_df)
power_df.columns = ["power_drperm", "power_rrperm"]
pd.DataFrame(power_df).to_csv('power_aquatic_semi.csv')
    










