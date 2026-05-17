# Reproducibility knobs — edit here only for quick scaling tests.
REPRO <- list(
  seed = 42L,
  # Match common modern defaults; exact cross-version reproducibility still needs same R + BLAS.
  rng_kind = c("Mersenne-Twister", "Inversion", "Rejection"),
  mc_rep = 20L,
  perm_replicates = 200L,
  base_n_total = 1800L,
  sample_size_ratio = c(0.3, 0.5),
  noise_shift = c(0.5, 0.7),
  beta_train = c(1, 1, 1, 1, 1, 1, 1, 1),
  cor = 0.3,
  n_nuisance = 12L,
  eps_train = 2,
  alpha = 0.05,
  out_csv = "minimal_power_oob_grid.csv",
  out_session = "minimal_session_info.txt"
)

apply_repro_seed <- function() {
  k <- REPRO$rng_kind
  ok <- tryCatch(
    {
      do.call(RNGkind, as.list(k))
      TRUE
    },
    warning = function(w) TRUE,
    error = function(e) FALSE
  )
  if (!ok) {
    RNGkind(kind = k[[1]], normal.kind = k[[2]])
  }
  set.seed(REPRO$seed)
}
