#python packages wrapper:

run_autotst <- function(df_exist, df_new, B, alpha = 0.05, seed = 2026){
  library(reticulate)
  autotst <- import('autotst', convert = FALSE)
  np <- import('numpy', convert = FALSE)
  n1 <- nrow(df_exist)
  n2 <- nrow(df_new)
  pval_list <- numeric(B)
  for(i in 1:B){
  	df_exist_B <- df_exist[sample(c(1:n1), n1, replace = TRUE, seed = seed - i), ]
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


