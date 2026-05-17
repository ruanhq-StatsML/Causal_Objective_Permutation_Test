# =============================================================================
# Minimal power / Type-I sanity grid for OOB permutation + Lancaster combine
# -----------------------------------------------------------------------------
# 1) Put YOUR project's sources in `deps/` (or set COVAR_SHIFT_DEP_DIR):
#      deps/permutation_covariateShift_utils.R
#      deps/permOOBCovarShift.R
#    (Add any further files those scripts source.)
# 2) From this directory:
#      Rscript run_minimal_power.R
# =============================================================================

options(warn = 1)

get_this_dir <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", cmd[grep("^--file=", cmd)])
  if (length(file_arg) == 1L && nzchar(file_arg)) {
    return(dirname(normalizePath(file_arg)))
  }
  of <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(of)) {
    return(dirname(normalizePath(of)))
  }
  getwd()
}

this_dir <- get_this_dir()

deps_dir <- Sys.getenv(
  "COVAR_SHIFT_DEP_DIR",
  unset = file.path(this_dir, "deps")
)

source(file.path(this_dir, "config.R"))
source(file.path(this_dir, "data_generation.R"))

need_pkg <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop(
      "Missing packages: ", paste(missing, collapse = ", "),
      "\nInstall with install.packages(c(\"", paste(missing, collapse = "\", \""), "\"))"
    )
  }
  invisible(pkgs)
}

need_pkg(c("MASS", "aggregation"))

utils_r <- file.path(deps_dir, "permutation_covariateShift_utils.R")
perm_r <- file.path(deps_dir, "permOOBCovarShift.R")

if (!file.exists(utils_r) || !file.exists(perm_r)) {
  stop(
    "Expected dependency scripts not found.\n",
    "  ", utils_r, "\n",
    "  ", perm_r, "\n",
    "Copy your project's R files into deps/ or set COVAR_SHIFT_DEP_DIR."
  )
}

source(utils_r)
source(perm_r)
if (!exists("perm_oobtest", mode = "function")) {
  stop("After sourcing deps, `perm_oobtest` should be defined — check permOOBCovarShift.R.")
}

extract_perm_p <- function(x) {
  if (is.numeric(x) && length(x) == 1L) {
    return(as.numeric(x))
  }
  if (is.list(x) && !is.null(x$p_val)) {
    return(as.numeric(x$p_val))
  }
  stop(
    "Could not read a scalar p-value from perm_oobtest() result (",
    paste(class(x), collapse = ", "), "). ",
    "Adjust extract_perm_p() in run_minimal_power.R if your API differs."
  )
}

apply_repro_seed()

power_df <- expand.grid(
  sample_size_ratio = REPRO$sample_size_ratio,
  noise_shift = REPRO$noise_shift,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
power_df$power_lancaster <- NA_real_
power_df$power_p1 <- NA_real_
power_df$power_p2 <- NA_real_
power_df$mc_rep <- REPRO$mc_rep

for (j in seq_len(nrow(power_df))) {
  pval_l <- numeric(REPRO$mc_rep)
  pval_1 <- numeric(REPRO$mc_rep)
  pval_2 <- numeric(REPRO$mc_rep)
  noise <- power_df$noise_shift[j]
  r_j <- power_df$sample_size_ratio[j]

  for (i in seq_len(REPRO$mc_rep)) {
    n1 <- round(REPRO$base_n_total * r_j / (1 + r_j))
    n2 <- round(REPRO$base_n_total * 1 / (1 + r_j))

    df1 <- LM_generation(
      n = n1,
      beta_hat = REPRO$beta_train,
      cor = REPRO$cor,
      n_nuisance = REPRO$n_nuisance,
      eps = REPRO$eps_train,
      mean_X = 0,
      var_X = 1
    )$df_return

    df2 <- LM_generation(
      n = n2,
      beta_hat = REPRO$beta_train,
      cor = REPRO$cor,
      n_nuisance = REPRO$n_nuisance,
      eps = REPRO$eps_train + noise,
      mean_X = 0,
      var_X = 1
    )$df_return

    xy <- prepare_two_sample_frames(df1, df2)
    p1 <- extract_perm_p(perm_oobtest(xy$df1, xy$df2, B = REPRO$perm_replicates))
    p2 <- extract_perm_p(perm_oobtest(xy$df2, xy$df1, B = REPRO$perm_replicates))
    pval_l[i] <- aggregation::lancaster(c(p1, p2), c(n2, n1))
    pval_1[i] <- p1
    pval_2[i] <- p2
  }

  power_df$power_lancaster[j] <- mean(pval_l < REPRO$alpha, na.rm = TRUE)
  power_df$power_p1[j] <- mean(pval_1 < REPRO$alpha, na.rm = TRUE)
  power_df$power_p2[j] <- mean(pval_2 < REPRO$alpha, na.rm = TRUE)
}

out_path <- file.path(this_dir, REPRO$out_csv)
write.csv(power_df, out_path, row.names = FALSE)
message("Wrote ", normalizePath(out_path, winslash = "/", mustWork = FALSE))

sink(file.path(this_dir, REPRO$out_session))
cat("MINIMAL OOB COVARIATE-SHIFT REPRO — session snapshot\n\n")
print(sessionInfo())
sink()

message("Session info: ", file.path(this_dir, REPRO$out_session))
