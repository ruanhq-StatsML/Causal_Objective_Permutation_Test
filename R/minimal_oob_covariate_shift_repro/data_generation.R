# LM + nuisance design from your simulation (kept local so the example is self-describing).

LM_generation <- function(n, beta_hat, cor, n_nuisance, eps, mean_X = 0, var_X = 1) {
  p <- length(beta_hat)
  corr_matrix <- matrix(NA_real_, p, p)
  for (i in 1:p) {
    for (j in 1:p) {
      corr_matrix[i, j] <- (cor^(abs(i - j))) * var_X
    }
  }
  X_design <- MASS::mvrnorm(n, mu = rep(mean_X, p), Sigma = corr_matrix)
  Y1 <- as.matrix(X_design) %*% as.matrix(beta_hat, nrow = p)
  random_error <- rnorm(n, 0, eps)
  X_nuiss <- matrix(rnorm(n * n_nuisance, 0, 1), nrow = n, ncol = n_nuisance)
  Y <- as.vector(Y1 + random_error)
  df_return <- data.frame(cbind(Y1, X_design, X_nuiss, Y))
  ncol_df <- ncol(df_return)
  colnames(df_return) <- c(
    "Y1",
    paste0("X", seq_len(p)),
    paste0("X_nuis", seq_len(n_nuisance)),
    "Y"
  )
  X_return <- df_return[, 2:(ncol_df - 1), drop = FALSE]
  list(df_return = df_return, X_return = X_return)
}

prepare_two_sample_frames <- function(df1, df2) {
  n_col <- ncol(df1)
  df1[, 2:(n_col - 1)] <- as.data.frame(scale(df1[, 2:(n_col - 1), drop = FALSE], scale = TRUE))
  df2[, 2:(n_col - 1)] <- as.data.frame(scale(df2[, 2:(n_col - 1), drop = FALSE], scale = TRUE))
  df1 <- df1[, 2:n_col, drop = FALSE]
  df2 <- df2[, 2:n_col, drop = FALSE]
  list(df1 = df1, df2 = df2)
}
