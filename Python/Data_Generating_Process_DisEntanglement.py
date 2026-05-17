from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
import numpy as np
import pandas as pd
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import Literal, Dict, Any, Callable, Optional, List
import matplotlib.pyplot as plt

'''
#####################################################################################################################
--- Large Scale Simulation for Different Types of Distribution Shift: Baseline DGP here is the Linear Model and 
    we incrementally impose the 
#We incorporate the following methodologies for benchmark comparisons:
- X_shift(Covariate Shift):     ['none', 'mean', 'eigen', 'covariance', 'mixture', 'add_variable']
- C_shift(Concept Drift):       ['none', 'linear', 'nonlinear', 'interaction', 'threshold']

~ Various of types of shifts are incorporated in the new batch of data D_{new}.
Apart from the conventional Mean Shift/Quadratic Shift/Cubic Shifts, Variance Shift and Noise Shift:
#### Concept Drift:
- Threshold shift               - Truncation/Cap the response between a range in the response of D_{new}.
- Interaction Shift             - Interaction Term between X and T is imposed in D_{new}.
- Nonlinear Shift               - The additional nonlinear term is imposed in D_{new}.
- Add variable                  - The noise with arbitrary distribution is imposed in D_{new}.
#### Covariate Shift:
- Support Shift                 - The support of the covariates supp(X) is capped within a range for each direction.
- Eigen Shift                   - The eigen shift is imposed in the SVD decomposition U@(Sigma_shift)@V.T
- Mixture Shift                 - Both mean shift, variance shift are imposed.
#####################################################################################################################
'''

#Collect these datasets:
def Kennedy2022(n_ref = 1000, n_new = 1000, p_signal = 8, p_nuisance = 12, mean_shift = 0.25, epsilon = 1, seed = 2026):
    def mu(x):
        if -1 <= x <= -0.5:
            return ((x + 2)**2)/2.0
        elif -0.5 <= x < 0:
            return x/2.0 + 0.875
        elif 0 < x <= 0.5:
            return -5 * ((x - 0.2)**2) + 1.075
        else:
            return x + 0.125
    X_ref_sig = np.random.uniform(-1, 1, n_ref + n_new)
    pi_X = 0.5 + 0.4 * np.sign(X_ref_sig)
    bernoulli_list = np.array([np.random.binomial(n = 1, p = p, size = 1).item() for p in pi_X])
    X_ref_1 = np.array(X_ref_sig[bernoulli_list == 0]).reshape(-1, 1)
    X_new_1 = np.array(X_ref_sig[bernoulli_list == 1]).reshape(-1, 1)
    n_exist = X_ref_1.shape[0]
    n_new = X_new_1.shape[0]
    rng = np.random.default_rng(seed = seed)
    X_ref_sig_part2 = rng.multivariate_normal(mean = mean_ref, cov = cov_ref, size = n_exist)
    X_new_sig_part2 = rng.multivariate_normal(mean = [x + y for x, y in zip(mean_ref, [mean_shift] * len(mean_ref))], cov = cov_ref, size = n_new)
    beta = np.array(rng.normal(loc = 1, scale = 0.5, size = p_signal)).reshape(-1,1)
    Y_ref = X_ref_sig_part2 @ beta + rng.normal(loc = 0, scale = 1, size = n_exist).reshape(-1,1)
    Y_new = X_new_sig_part2 @ beta + rng.normal(loc = 0, scale = 1, size = n_new).reshape(-1,1)
    cov_ref_nuisance = np.eye(p_nuisance)
    X_nuisance_ref = rng.multivariate_normal(mean = [0] * p_nuisance, cov = cov_ref_nuisance, size = n_exist)
    X_nuisance_new = rng.multivariate_normal(mean = [0] * p_nuisance, cov = cov_ref_nuisance, size = n_new)
    X_ref = np.hstack([X_ref_1, X_ref_sig_part2, X_nuisance_ref])
    X_new = np.hstack([X_new_1, X_new_sig_part2, X_nuisance_new])
    return X_ref, Y_ref, X_new, Y_new


@dataclass
class DGPOutput:
    X_ref: np.ndarray
    Y_ref: np.ndarray
    X_new: np.ndarray
    Y_new: np.ndarray
    config: DGPConfig
    metadata: Dict[str, Any]
    def to_metadata_row(self):
        row = asdict(self.config)
        row.update({
            'label': self.config.label,
            'S_X': self.config.S_X,
            'S_C': self.config.S_C,
            'S_X_shift': self.config.x_shift_type,
            'S_C_shift': self.config.concept_drift_type
        })
        row.update(self.metadata)#The metadata include these columns
        return row


@dataclass
class DGPConfig:
    dgp_id: int = 1
    n_ref: int = 1000
    n_new: int = 1000
    d: int = 10
    X_dist = 'Normal'
    t_df: int = 3
    x_shift_type = "none"
    concept_drift_type = "none"
    n_nuisance = 10
    beta = [1,1,1,1,1]
    mean_shift_count = 4
    eigen_shift_count = 4
    delta_x: float = 0.0
    delta_c: float = 0.0
    y_type = "regression"
    noise_sd: float = 1.0
    seed: int = 123
    mixture_degree = 2.0
    @property
    def S_X(self):
        return int(self.x_shift_type != 'none' and self.delta_x > 0)
    @property 
    def S_C(self):
        return int(self.concept_drift_type != 'none' and self.delta_c > 0)
    @property
    def label(self):
        if self.S_X == 0 and self.S_C == 0:
            return 'no_shift'
        if self.S_X == 1 and self.S_C == 0:
            return 'covariate_only'
        if self.S_X == 0 and self.S_C == 1:
            return 'concept_drift_only'
        return 'mixed'


class ShiftDGP:
    def __init__(self, config: DGPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
    def generate_X_ref(self):
        #Fix a t-reference data and implement the mean shift on the T-distribution here.
        cfg = self.config
        if cfg.X_dist == 'Normal':
            return self.rng.normal(loc = 0.0, scale = 1.0, size = (cfg.n_ref, cfg.d))
        else:
            #with the univariate distribution here:
            loc = np.full(cfg.n_ref, 0)
            shape = np.eye(cfg.d) * 1.5
            return self.rng.multivariate_t(loc = loc, shape = shape, df = cfg.t_df).rvs(cfg.n_ref)
        raiseValueError("The distribution is not Supported Currently")
    def generate_X_new(self):
        #Fix the reference T-distribution with imposing the mean shift
        cfg = self.config
        if cfg.x_shift_type == 'none' or cfg.delta_x == 0:
            if cfg.X_dist == 'Normal':
                return self.rng.normal(loc = 0.0, scale = 1.0, size = (cfg.n_new, cfg.d))
            else:
                loc = np.full(cfg.n_ref, 0)
                #loc[:cfg.mean_shift_count] = cfg.delta_x
                shape = np.eye(cfg.d) * 1.5
                return self.rng.multivariate_t(loc = loc, shape = shape, df = cfg.t_df).rvs(cfg.n_ref)
        if cfg.x_shift_type == 'mean':
            if cfg.X_dist == 'normal':
                mean = np.full(cfg.d, 0)
                mean[:cfg.mean_shift_count] = cfg.delta_x
                return self.rng.nomral(loc = mean, scale = 1.0)
            else:
                #loc, 
                loc = np.full(cfg.d)
                loc[:cfg.mean_shift_count] = cfg.delta_x
                shape = np.eye(cfg.d) * 1.5
                return self.rng.multivariate_t(loc = loc, shape = shape, df = cfg.t_df).rvs(cfg.n_ref)
        if cfg.x_shift_type == 'covariance':
            if cfg.X_dist == 'normal':
                cov = np.eye(cfg.d)
                cov[0, 1] = cov[1, 0] = min(0.8, cfg.delta_x)
                cov[2, 3] = cov[3, 2] = min(0.8, cfg.delta_x/2.5)
                return self.rng.multivariate_normal(
                    mean = np.zeros(cfg.d),
                    cov = cov,
                    size = cfg.n_new
                    )
            else:
                loc = np.full(cfg_d)
                shape = np.eye(cfg.d) * 1.5
                shape[0, 1] = shape[1, 0] = min(0.8, cfg.delta_x)
                shape[2, 3] = shape[3, 2] = min(0.8, cfg.delta_x)
                return multivariate_t.rvs(loc = loc, shape = shape, df = cfg.t_df).rvs(cfg.n_ref)
        if cfg.x_shift_type == 'mixture':
            if cfg.X_dist == 'normal':
                z = self.rng.binomial(1, p = min(0.9, cfg.delta_x), size = cfg.n_new)
                X = self.rng.normal(0, 1, size = (cfg.n_new, cfg.d))
                X[z == 1, : 3] += cfg.mixture_degree
            else:
                z = self.rng.binomial(1, p = min(0.9, cfg.delta_x), size = cfg.n_new)
                loc = np.full(cfg.d)
                shape = np.eye(cfg.d) * 1.5
                X = multivariate_t(loc = loc, shape = shape, df = cfg.t_df).rvs(cfg.n_ref)
                X[z == 1, :3] = cfg.mixture_degree
                return X
        if cfg.x_shift_type == 'support':
            #truncate the support between [-half, half]
            X_ref = self.generate_X_ref()#That's where you put the config outside here.
            half = 1.0 + float(cfg.delta_x)
            min_mask = self.rng.uniform(-half - 0.25, -half + 0.25, size = (cfg.n_new, cfg.d))
            max_mask = self.rng.uniform(half - 0.25, half + 0.25, size = (cfg.n_new, cfg.d))
            return np.clip(X_ref, min_mask, max_mask)
        if cfg.x_shift_type == 'eigenshift':
            X_ref = self.generate_X_ref()
            U, S, Vh = np.linalg.svd(X_ref)
            eigen_values = np.diag(np.diag(S))
            shift_mag_vec = [shift_mag ** (1 + 2 * delta/cfg_eigen_shift_count) for delta in range(k)]
            for j in range(k):
                eigen_values[j] += shift_mag_vec[j]
            X_ref_reshaped = U @ np.diag(eigen_values) @ Vh.T
            return X
        raise ValueError(f"Unknown x-shift-type: {cfg.x_shift_type}")
    def m0(self, X: np.ndarray):
        n_beta = len(cfg.beta)
        output = (X[:,:n_beta] @ np.array(cfg.beta).reshape(-1, 1)).reshape(-1)
        #output = 0.5 * X[:, 0] + 0.75 * X[:, 1] - 0.25 * X[:, 2] + 0.25 * np.sin(X[:, 3]) + 0.75 * X[:, 4] * X[:, 5]
        return output
    def m1(self, X: np.ndarray):
        output = self.m0(X) + self.concept_drift(X)
    def concept_drift(self, X):
        cfg = self.config
        if cfg.concept_drift_type == 'none' or cfg.delta_c == 0:
            return np.zeros(X.shape[0])
        if cfg.concept_drift_type == 'linear':
            return cfg.delta_c * (X[:, 0] + X[:, 1])
        if cfg.concept_drift_type == 'nonlinear':
            return cfg.delta_c * np.sin(X[:, 0] * X[:, 1])
        if cfg.concept_drift_type == 'cubic':
            return cfg.delta_c * ((X[:, 0] ** 2) + X[:, 2] ** 3 + X[:, 1] ** 3)
        if cfg.concept_drift_type == 'interaction':
            return cfg.delta_c * X[:, 0] * X[:, 2] * (5.0/X[:, 3]) 
        if cfg.concept_drift_type == 'threshold':
            return cfg.delta_c * (X[:, 1] > 0).astype(float)
        raise ValueError(f"Unknown Concept Drift Type: {cfg.concept_drift_type}")
    def generate_y(self, X, batch):
        cfg = self.config
        mu = self.m0(X) if batch == 'ref' else self.m1(X)
        if cfg.y_type == 'regression':
            return mu + self.rng.normal(0, cfg.noise_sd, size = X.shape[0])
        if cfg.y_type == 'classification':
            probs = 1.0/(1.0 + np.exp(-mu))
            return self.rng.binomial(1, probs, size = X.shape[0])
    def generate_data(self):
        X_ref = self.generate_X_ref()
        X_new = self.generate_X_new()
        Y_ref = self.generate_y(X_ref, batch = 'ref')
        Y_new = self.generate_y(X_new, batch = 'new')
        cd_new = self.concept_drift(X_new)
        metadata = {
        'oracle_mean_shift': float(np.linalg.norm(X_new.mean(axis = 0) - 
        	X_ref.mean(axis = 0))),
        'oracle_m0_mean_ref': float(np.mean(self.m0(X_ref))),
        'oracle_m1_mean_ref': float(np.mean(self.m1(X_new))),
        'oracle_concept_signal_raw': float(np.mean(np.abs(cd_new)))
        }
        #return the config and all of the metadata:
        return DGPOutput(
            X_ref = X_ref,
            Y_ref = Y_ref,
            X_new = X_new,
            Y_new = Y_new,
            config = self.config,
            metadata = metadata
        )


@dataclass
class DGPConfig:
    dgp_id: int = 1
    n_ref: int = 1000
    n_new: int = 1000
    d: int = 12
    X_dist = 'Normal'
    t_df: int = 3
    x_shift_type = "none"
    concept_drift_type = "none"
    n_nuisance = 10
    beta = [1,1,1,1,1,1,1,1]
    mean_shift_count = 4
    eigen_shift_count = 4
    delta_x: float = 0.0
    delta_c: float = 0.0
    y_type = "regression"
    noise_sd: float = 1.0
    seed: int = 123
    mixture_degree = 2.0
    @property
    def S_X(self):
        return int(self.x_shift_type != 'none' and self.delta_x > 0)
    @property 
    def S_C(self):
        return int(self.concept_drift_type != 'none' and self.delta_c > 0)
    @property
    def label(self):
        if self.S_X == 0 and self.S_C == 0:
            return 'no_shift'
        if self.S_X == 1 and self.S_C == 0:
            return 'covariate_only'
        if self.S_X == 0 and self.S_C == 1:
            return 'concept_drift_only'
        return 'mixed'


########
#Making all of the Data-Generating Process with the Kennedy
def make_all_DGP(
    n_ref = 1000,
    n_new = 1000,
    d = 10,
    y_type = 'regression',
    base_seed = 2026):
    X_DIST = ['normal', 't_dist']
    x_shift_type = ['mean', 'covariance', 'mixture', 'support', 'eigenshift', 'none']
    concept_drift_type = ['none', 'linear', 'nonlinear', 'cubic', 'interaction', 'threshold']
    delta_x_values = {
    'none': [0.0],
    'mean': [0.25, 0.75],
    'covariance': [0.25, 0.75],
    'mixture': [0.25, 0.75],
    'support': [0.25, 0.75]
    }
    delta_c_values = {
    'none': [0.0],
    'linear': [0.25, 0.75],
    'nonlinear': [0.25, 0.75],
    'interaction': [0.25, 0.5, 0.75],
    'threshold': [0.25, 0.5, 0.75],
    'cubic': [0.25, 0.5, 0.75]
    }
    configs = []
    dgp_id = 0
    for x_dists in X_DIST:
        for cs in concept_drift_type:
            for xs in x_shift_type:
                for dc in delta_c_values[cs]:
                    for dx in delta_x_values[xs]:
                        cfg = DGPConfig(
                            dgp_id = dgp_id,
                            n_ref = n_ref,
                            n_new = n_new,
                            d = d,
                            X_dist = x_dists,
                            x_shift_type = xs,
                            concept_drift_type = cs,
                            delta_x = dx,
                            delta_c = dc,
                            y_type = y_type,
                            seed = base_seed + dgp_id * 2
                        )
                        configs.append(cfg)
                        dgp_id += 1
    return configs



def generate_all_dgp(configs):
    outputs = []
    rows = []
    for cfg in configs:
        dgp = ShiftDGP(cfg)
        output = dgp.generate()
        outputs.append(output)
        row.append(output.to_metadata_row())
    metadata_df = pd.DataFrame(rows)
    outputs.append(Kennedy2022())
    return outputs, metadata_df

    



def main():
    configs = make_all_DGP()
    outputs, metadata_df = generate_all_dgp(configs)
    np.save('results/DGP_whole_outputs_comprehensive_whole.npy',
    np.array(outputs, dtype = object),
    allow_pickle = True)
    metadata_df.to_csv('results/DGP_whole_meta_comprehensive_whole.csv', index = False)










for row_idx, dgp_output in enumerate(benchmark_data[start_idx:], start=start_idx):
    X_ref, Y_ref, X_new, Y_new = (
        dgp_output.X_ref,
        dgp_output.Y_ref,
        dgp_output.X_new,
        dgp_output.Y_new,
    )
    col_df1 = [f"X_{k}" for k in range(X_ref.shape[1] - 1)]
    col_df2 = [f"X_{k}" for k in range(X_new.shape[1] - 1)]
    col_Y = "Y"
    df1 = pd.DataFrame(
        np.concatenate([X_ref, Y_ref.reshape(-1, 1)], axis=1),
        columns=[f"X_{k}" for k in range(X_ref.shape[1])] + ["Y"],
    )
    df2 = pd.DataFrame(
        np.concatenate([X_new, Y_new.reshape(-1, 1)], axis=1),
        columns=[f"X_{k}" for k in range(X_new.shape[1])] + ["Y"],
    )
    output = benchmark_method(
        df1,
        col_df1,
        df2,
        col_df2,
        col_Y,
        B=B_REPS,
        n_resamples_rrperm=N_RESAMPLES_RRPERM,
        c2st_perm_b=C2ST_B,
        mmd_iterations=MMD_ITER,
        xu_n_perm=XU_N_PERM,
    )
    metadata.loc[row_idx, "power_rrperm"] = output["power_rrperm"][0]
    metadata.loc[row_idx, "power_miles_joint"] = output["power_miles_joint"][0]
    metadata.loc[row_idx, "power_miles_covar"] = output["power_miles_covar"][0]
    metadata.loc[row_idx, "power_xu_joint"] = output["power_xu_joint"][0]
    metadata.loc[row_idx, "power_xu_covariate"] = output["power_xu_covariate"][0]
    metadata.loc[row_idx, "power_chen14_joint"] = output["power_chen14_joint"][0]
    metadata.loc[row_idx, "power_chen14_covariate"] = output["power_chen14_covariate"][0]
    metadata.loc[row_idx, "power_c2st_joint"] = output["power_c2st_joint"][0]
    metadata.loc[row_idx, "power_c2st_covariate"] = output["power_c2st_covariate"][0]
    metadata.loc[row_idx, "power_mmd_joint"] = output["power_mmd_joint"][0]
    metadata.loc[row_idx, "power_mmd_covariate"] = output["power_mmd_covariate"][0]
    metadata.to_csv(out_csv, index=False)
    print(
        f"[progress] {row_idx + 1}/{n_total} | power_rrperm={output['power_rrperm'][0]:.4f} "
        f"| power_c2st_joint={output['power_c2st_joint'][0]:.4f}",
        flush=True,
    )





