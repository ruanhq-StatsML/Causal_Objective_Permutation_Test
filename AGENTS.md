# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **research/statistics library** (no servers, DBs, or web services). It ships two
parallel implementations of the "Causal Objective Permutation Test": a Python package under
`Python/` and an R package ("CFPerm") under `R/`. "Running the app" means running the library
code / model adapters on synthetic data; there is nothing to serve.

### Python (primary; the only thing CI covers)
- Deps (`numpy pandas scikit-learn scipy xgboost flake8 pytest`) are installed by the startup
  update script. They land in `~/.local/bin`, which is **not on `PATH`** — invoke tools as
  `python3 -m flake8 ...` and `python3 -m pytest` instead of the bare commands.
- CI (`.github/workflows/python-package.yml`) runs, from the repo root:
  - Lint: `python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - Test: `python3 -m pytest`
- Known pre-existing state (NOT an environment problem, do not "fix" as setup):
  - `pytest` collects **0 tests** (there are no `test_*.py` files) and exits with code 5.
  - The blocking `flake8` lint **fails**: the committed code has many undefined-name (`F821`)
    and syntax (`E999`) errors (e.g. `MODEL_REGISTRY`, `_as_1d`, `_as_2d_float` are referenced
    but never defined).
- Because of those undefined names, the headline entry points `RRPerm`/`DRPerm` in
  `Python/RRPerm.py` and `Python/DRPerm.py` cannot be imported/run as committed. The genuinely
  runnable core is the model registry: `from model_registry_class import ModelRegistry` (run
  from inside `Python/`), whose `rf_regressor` adapter fits/predicts end-to-end.

### R (secondary; not in CI)
- R 4.3.3 plus all CRAN packages the sources import (`MASS glmnet ANN2 CVST kernlab grf`
  `RandomProjectionTest caret testthat nnet xgboost randomForest`) are **pre-installed in the VM
  snapshot** at `/usr/local/lib/R/site-library`. They are intentionally **not** reinstalled by
  the update script (heavy compiled packages). If a package is ever missing, reinstall system-wide
  with `sudo Rscript -e 'install.packages(...)'` using the Posit binary repo
  (`https://packagemanager.posit.co/cran/__linux__/noble/latest`) and set `HTTPUserAgent` so
  precompiled binaries are served (source builds of `grf`/`caret`/`xgboost` are very slow).
- Run R scripts from the `R/` directory — the sources use relative `source('model_registry.R')`
  / `source('utils.R')`. The runnable core is `default_model_registry()` from `model_registry.R`.
- Pre-existing state: `unittest_RRPerm.R`/`unittest_DRPerm.R` do not `source('RRPerm.R')`, and
  `utils.R::cross_nuisance_fit` references an undefined variable `n`, so the R permutation-test
  entry points do not run end-to-end as committed. Do not treat this as an environment issue.
