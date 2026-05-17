#Sensitivity Analysis for R-learner based permutation test:
from itertools import product

def linear_model_sensitivity(
    n: int = 1000,
    p: int = 20,
    beta: list = [1,1,1,1,1,1,1,1],
    variance: float = 1.0,
    cor: float = 0.3,
    eps_noi: float = 1.0,
    mean: float = 0.0,
    r2_u_on_y: float = 0.25,
    r2_u_on_d: float = 0.3
):
    n_signal = len(beta)
    cov_mat = np.zeros((n_signal, n_signal))
    for ii in range(n_signal):
        for jj in range(n_signal):
            cov_mat[ii, jj] = cor ** abs(ii - jj) * variances
    cov = cov_mat @ cov_mat.T
    X_design = np.random.multivariate_normal([mean] * n_signal, cov, size = n)
    U = np.random.normal(0, 1, size = n)
    var_total_d = 1.0
    #proportion of U on the treatment
    k_d = np.sqrt(r2_u_on_d * var_total_d)
    d_latent = X_design[:, 1]
    var_noise_d = (1 - r2_u_on_d) * var_total_d
    epsilon_d = np.random.normal(0, np.sqrt(max(var_noise_d, 1e-12)), n)
    d_logits = d_latent + k_d * U + epsilon_d
    Y_signal = (X_design @ np.asarray(beta, dtype = float)).ravel()
    Y_noise = np.random.normal(0, eps_noi, n)
    var_residual_Y = float(np.var(Y_noise))
    #Generate the Y with Y = Y_signal + k_y * U(proportion of u on Y) 
    k_y = np.sqrt((r2_u_on_y * var_residual_Y)/max(1.0 - r2_u_on_y, 1e-12))
    Y = Y_signal + k_y * U + np.random.normal(
        0, np.sqrt(max(var_residual_Y - k_y ** 2, 0.0)), n
    )
    X_noise = np.random.normal(0, 1, (n, p - n_signal))
    X_all = np.concatenate((X_design, X_noise), axis = 1)
    df = pd.DataFrame(pd.concatenate((X_all, Y.reshape(-1, 1)), axis = 1))
    df.columns = [f"X{i}" for i in range(p)] + ["Y"]
    return df


#Evaluate the sensitivity for that positions:
def sensitivity_analysis_wrapper(n_t, ss_ratio, uy_r2_grid, ud_r2_grid, p,
    beta = [1,1,1,1,1,1,1,1], beta_new = [1,1,1,1,1,1,2,3], out_csv = 'sensitivity_wrapper'):
	power_df = pd.DataFrame(product(ss_ratio, uy_r2_grid, ud_r2_grid))
	power_df.columns = ['sample_size_ratio', 'uy_r2_grid', 'ud_r2_grid']
	power_df['power_rr'] = 0
	for i in range(power_df.shape[1]):
		ss = power_df['sample_size_ratio'][i]
		uy_r2 = power_df['uy_r2_grid'][i]
		ud_r2 = power_df['ud_r2_grid'][i]
		n1 = round(n_t * 1/(1+ss))
		n2 = n_t - n1
		p_val_list1 = [0] * B 
		risk_list = [0] * B
		for j in range(B):
	        df1 = linear_model_sensitivity(
	            n = n1,
		    p = p,
		    beta = beta,
		    variances = 2,
		    cor = 0.3,
		    eps_noi = 3,
		    mean = 0,
		    r2_u_on_y = float(uy_r2),
		    r2_u_on_d = float(ud_r2)
		    )
			df2 = linear_model_sensitivity(
			    n = n2,
			    p = p,
			    beta = beta_new,
			    variances = 2,
			    cor = 0.3,
			    eps_noi = 3,
			    mean = 0,
			    r2_u_on_y = float(uy_r2),
			    r2_u_on_d = float(ud_r2)
			)
		    X = np.concatenate(
	            [df1.iloc[:, :P_FEAT].values, df2.iloc[:, :P_FEAT].values]
	        )
	        Y = np.concatenate(
	            [df1.iloc[:, P_FEAT].values, df2.iloc[:, P_FEAT].values]
	        )
	        n1 = df1.shape[0]
	        n2 = df2.shape[0]
	        W = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
	        sim_s = _sim_seed(i, j, k)
	        cf_obs = (int(sim_s) * 1103515245 + 12345) % (2**32)
	        risk, _bias_arm, _, _, _ = _r_risk_on_W(
	            X,
	            Y,
	            W,
	            cf_seed=cf_obs,
	            tau_seed=sim_s + 17,
	            n_splits=N_SPLITS,
	            nuisance=NUISANCE,
	            clip_wtilde=1e-3,
	            rf_n_estimators=RF_TREES,
	        )
	        pval, perm_bias_mean, perm_bias_std, bias_orig = _r_risk_perm_w_pvalue(
	            X,
	            Y,
	            W,
	            n_resamples=N_RES,
	            sim_seed=sim_s,
	            n_splits=N_SPLITS,
	            nuisance=NUISANCE,
	            rf_n_estimators=RF_TREES,
	        )
	        p_val_list1[k] = float(pval)
	        risk_list[k] = float(risk)
	    power_df['power'][i] = np.mean(np.array(p_val_list1) < 0.05)
	    power_df.to_csv(out_csv)

   



























