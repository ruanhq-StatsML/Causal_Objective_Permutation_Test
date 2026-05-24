#python packages wrapper for the autoML two-sample test:
py <- "/Users/heqiaoruan/Desktop/Desktop - heqiao’s MacBook Air/PhD/Research/permutationTestingCovariateShift/DRPerm_MMD/.venv_autotst/bin/python"
Sys.unsetenv("RETICULATE_PYTHON_FALLBACK")  
Sys.setenv(RETICULATE_PYTHON = py)
library(reticulate)

reticulate::use_python(py, required = TRUE)
run_autotst <- function(df_exist, df_new, B, alpha = 0.05, seed = 2026){
  autotst <- import('autotst', convert = FALSE)
  np <- import('numpy', convert = FALSE)
  n1 <- nrow(df_exist)
  n2 <- nrow(df_new)
  pval_list <- numeric(B)
  for(i in 1:B){
  	set.seed(seed + i)
    df_exist_B <- df_exist[sample(c(1:n1), n1, replace = TRUE, seed = seed - i), ]
    set.seed(seed - i)
    df_new_B <- df_new[sample(c(1:n2), n2, replace = TRUE, seed = seed - i), ]
    X0 <- as.matrix(df_exist_B)
    X1 <- as.matrix(df_new_B)
    storage.mode(X0) <- 'double'
    storage.mode(X1) <- 'double'
    X0_py <- np$array(X0, dtype = 'float64')
    X1_py <- np$array(X1, dtype = 'float64')
    tst <- autotst$AutoTST(X0_py, X1_py)
    pval_list[i] <- as.numeric(py_to_r(tst$p_value()))
  }
  #Transform the p-value back to R Version:
  return(mean(pval_list < alpha))
}
df1 <- matrix(rnorm((100 * 50), 0, 1), 1000, 5)
df2 <- matrix(rnorm((100 * 50), 0, 1), 1000, 5)
run_autotst(df1, df2, B = 500)
#Other benchmark functions:

#df.pivot_table(index, column, values)
#df.melt(id_vars, var_names, values)
#df %>% pivot_longer
#df %>% pivot_wider are conventional procedures.






