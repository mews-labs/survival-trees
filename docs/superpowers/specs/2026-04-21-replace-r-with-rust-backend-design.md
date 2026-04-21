# Replace R backend with a Rust native extension

**Date:** 2026-04-21
**Author:** Vincent Laurent
**Status:** Proposed

## Context

`survival-trees` relies on R through `rpy2` for its core algorithms
(`LTRCART` from R package `LTRCtrees`, `randomForestSRC`, Kaplan-Meier
via `survival::survfit`). This forces users to install an R toolchain
plus ~12 R packages (see `setup.py` / `build.sh`), which is a heavy
operational tax and blocks distribution via a standard wheel.

The goal is to remove R entirely and replace it with a Rust native
extension compiled into the Python package via `maturin` / `PyO3`.

## Goals

- Eliminate every R dependency (`rpy2`, `.R` scripts, R packages) from
  the library at runtime and install time.
- Keep the public Python API strictly backward compatible for
  `LTRCTrees`, `RandomForestLTRC`, `RandomForestLTRCFitter`,
  `LTRCTreesFitter`. Soft-deprecate `RandomForestSRC`.
- Deliver a Rust backend embedded in the distributed wheel so
  `pip install` works without any extra toolchain on the user side
  (for the supported target).
- Maintain or improve runtime performance compared to the current R
  backend on the reference benchmark.

## Non-goals

- Bit-for-bit reproduction of R/LTRCART's tree structure. We accept
  free functional equivalence (same problem, possibly different
  internal splitting heuristic).
- Supporting macOS or Windows binary wheels in the first release.
  Only Linux `x86_64` wheels are shipped.
- Rewriting `scikit-survival`-style RSF from scratch. `RandomForestSRC`
  is deprecated and delegates to `RandomForestLTRC` during the
  deprecation window.

## Constraints (agreed during brainstorming)

1. **Scope** — replace all R code including `RandomForestSRC`
   (soft-deprecated, not reimplemented).
2. **Target language** — Rust, exposed via `PyO3` + `maturin`.
3. **Numerical fidelity** — free functional equivalence. Split
   algorithm may differ from `rpart` LTRCART.
4. **Public API** — strictly backward compatible (except deprecation of
   `RandomForestSRC`).
5. **Distribution** — Linux `x86_64` wheels only.
6. **Validation** — existing tests pass + regression test on AUC +
   performance benchmark ≥ R on reference datasets.
7. **Rollout** — parallel branch then swap PR. No runtime backend
   selection.

## Architecture

### Repository layout

```
survival-trees/
├── Cargo.toml                        # Rust workspace (1 crate)
├── pyproject.toml                    # replaces setup.py, maturin backend
├── rust/
│   └── survival_trees_rs/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs                # PyO3 entry point, exposed classes
│           ├── tree.rs               # LTRC tree + fit recursion
│           ├── split.rs              # LTRC log-rank + best split search
│           ├── forest.rs             # bagging + feature subsampling
│           ├── km.rs                 # leaf Kaplan-Meier with LTRC correction
│           └── types.rs              # shared structs (Sample, Node, …)
├── survival_trees/                   # externally unchanged public package
│   ├── __init__.py
│   ├── _base.py                      # rewritten, delegates to Rust
│   ├── _fitters.py                   # unchanged
│   ├── metric.py, plotting.py, …     # unchanged
│   └── tools/
├── tests/
└── docs/superpowers/specs/
```

The compiled native module is imported as `survival_trees._rust`.
It is an implementation detail — only `_base.py` imports it.

### Toolchain

- Rust stable, MSRV 1.74+.
- Crates: `pyo3`, `numpy`, `ndarray`, `rayon`.
- `pyproject.toml` declares `requires-python = ">=3.9"`, runtime
  dependencies: `numpy`, `pandas`, `scikit-learn`, `lifelines`.
- Build system: `maturin`. `setup.py`, `build.sh`, `requirements.txt`
  are removed in the swap PR.

### Rust components

`types.rs`
```rust
pub struct Sample { pub entry: f64, pub time: f64, pub event: bool }
pub struct Dataset { pub x: Array2<f64>, pub samples: Vec<Sample> }
pub struct Node {
    pub feature: Option<usize>,
    pub threshold: Option<f64>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
    pub leaf_id: Option<usize>,
    pub importance_contrib: f64,
}
pub struct KmCurve { pub times: Vec<f64>, pub surv: Vec<f64> }
pub struct Control {
    pub max_depth: Option<usize>,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub min_impurity_decrease: f64,
}
```

`split.rs` — algorithmic core
- `fn log_rank_ltrc(left, right) -> f64` — log-rank statistic on
  left-truncated right-censored data, with risk set corrected for
  truncation (a subject is at risk only after its entry time).
  **This is an explicit simplification under constraint 3 (free
  functional equivalence):** the original LTRCART uses rpart's
  deviance-based splitting (`method = "exp"`), we use the standard
  log-rank statistic instead. Both are valid LTRC-aware split rules.
- `fn find_best_split(ds, indices, control) -> Option<Split>` —
  iterates features × candidate thresholds (one sort per feature,
  linear scan on unique thresholds), returns the best split by
  log-rank.
- Enforces `min_samples_leaf`, `min_samples_split`,
  `min_impurity_decrease` (pruning threshold analogous to rpart's
  `cp`: a split is kept only if it improves the parent log-rank
  statistic by at least this fraction), `max_depth`.

`tree.rs`
- `fn fit_tree(ds, control) -> Tree` — depth-first recursion,
  computes a leaf KM curve via `km.rs` at each leaf.
- `fn predict_leaf_ids(tree, x) -> Vec<usize>` — deterministic
  descent.
- `fn feature_importances(tree, n_features) -> Vec<f64>` — normalized
  sum over nodes of the split's log-rank improvement weighted by the
  number of samples reaching the node (rpart-like aggregation rule,
  adapted to our log-rank splitter).

`km.rs`
- `fn kaplan_meier_ltrc(samples) -> KmCurve` — Kaplan-Meier with
  left-truncation correction (risk set = subjects already entered and
  not yet exited).

`forest.rs`
- `fn fit_forest(ds, control, n_trees, max_samples, max_features, seed, n_jobs) -> Forest` —
  bootstrap + feature subsampling per tree, parallel fit via
  `rayon::par_iter`.
- `fn predict_forest(forest, x) -> PredictionBundle` — returns
  `(leaf_ids_per_tree, km_curves_per_tree)`; aggregation into
  pandas structures happens on the Python side.

`lib.rs` — PyO3 surface, exposes two classes:
- `_RustLtrcTree`: `.fit(x, entry, time, event, control_dict)`,
  `.predict_curves(x) -> (curves_ndarray, node_ids_ndarray, times_ndarray)`,
  `.feature_importances() -> ndarray`.
- `_RustLtrcForest`: same + `n_trees`, `max_samples`, `max_features`,
  `seed`, `n_jobs`.

All inputs and outputs flow through `numpy::PyReadonlyArray` /
`IntoPyArray` for zero-copy interop.

### Threading

- Forest fit: `rayon`, bounded by `n_jobs` (default `num_cpus::get()`).
- Single-tree fit: sequential.

## Python API mapping (unchanged surface)

| Public Python class | Backed by | Public attributes kept |
|---|---|---|
| `LTRCTrees` | `_RustLtrcTree` | `feature_importances_`, `results_` (opaque wrapper for backward compat) |
| `RandomForestLTRC` | `_RustLtrcForest` | `feature_importances_`, `n_features_in_`, `feature_names_in_`, `results_`, `base_estimator_` |
| `RandomForestLTRCFitter` | same | same |
| `LTRCTreesFitter` | same | same |
| `RandomForestSRC` | deprecated shim over `RandomForestLTRC` | raises `DeprecationWarning` in `__init__` |

Method signatures preserved:
- `.fit(X: pd.DataFrame, y: pd.DataFrame) -> None`
- `.predict(X) -> pd.DataFrame` (dense, indexed like `X`, columns = times)
- `.predict_curves(X) -> tuple[pd.DataFrame, pd.Series]` (on `LTRCTrees`)
- `.fast_predict_(X) -> None` plus `nodes_`, `km_estimates_` (on `RandomForestLTRC`)

Undocumented R-specific internal attributes (`_r_data_frame`,
`_id_run`, `__hashes`, `__select_feature`) are removed silently.

## Data flow

### Fit — `LTRCTrees`
1. `X`, `y` pandas inputs → `_validate_y` (unchanged).
2. Extract `entry`, `time`, `event` arrays + feature matrix as
   `numpy.ndarray[float64]`.
3. Call `_RustLtrcTree.fit(x, entry, time, event, control_dict)`.
4. Rust builds the tree and stores one KM curve per leaf internally.
5. Python retrieves `feature_importances_`.

### Predict — `LTRCTrees`
1. `_RustLtrcTree.predict_curves(x)` returns
   `(curves[n_leaves, n_times], leaf_ids[n_samples], times[n_times])`.
2. Python reshapes to the current public `DataFrame` layout
   (`km_mat` indexed like `X`, columns = times).
3. Existing `get_dense_prediction` / `interpolate_prediction` logic
   applies unchanged — it is pure Python post-processing over the
   output of the new backend.

### Fit — `RandomForestLTRC`
1. Validation → `_RustLtrcForest.fit(..., n_jobs)`.
2. Rust handles bootstrap and per-tree feature subsampling, fits in
   parallel via `rayon`.
3. Python retrieves aggregated `feature_importances_`.

### Predict — `RandomForestLTRC.fast_predict_`
1. Rust returns per-tree `(leaf_id_per_sample, km_curve_per_leaf)`.
2. Python runs the current `__post_processing` (duplicate-node
   deduplication, averaging of KM curves) unchanged — this code
   produces the public `nodes_` / `km_estimates_` attributes and has
   no R dependency.

## Error handling

- Python-level validation (`_validate_y`) unchanged; same error
  messages.
- Rust errors surface as typed Python exceptions via `PyResult`:
  - `n_samples < min_samples_split` after filtering → `ValueError`.
  - feature with zero variance / constant dataset → split skipped,
    no error.
  - zero events observed → `ValueError("no event observed")`.
- No `panic!` in Rust code paths reachable from Python; everything
  converts through `PyResult<T>`.
- The current `ro.r("sink('/dev/null')")` workaround is removed.

## Testing strategy

### Existing tests
- `survival_trees/test/test_restimator.py` is adapted.
  `test_random_forest_src` is updated to assert the
  `DeprecationWarning` and that output remains semantically
  consistent (survival between 0 and 1, monotonically non-increasing).
- All other Python tests pass unchanged, with numerical tolerances
  adjusted where needed.

### Regression test (new)
- Datasets: `lifelines.datasets.load_larynx()` and
  `load_flchain()` (real LTRC data).
- Reference values generated once against the current R backend,
  stored in `tests/fixtures/reference_auc.json` (AUC + concordance +
  inference time).
- Acceptance criterion: `auc_rust >= auc_r - 0.02` on each dataset.

### Performance benchmark (new)
- `tests/bench/bench_fit_predict.py` re-creates the workload behind
  `public/benchmark.png`.
- Measures `fit` + `predict` on synthetic datasets of size 1k / 10k /
  50k.
- CI criterion: `time_rust <= 1.1 * time_r_snapshot`, where
  `time_r_snapshot` is measured once on the same hardware and frozen
  in the fixtures.

### Rust unit tests
- Under `rust/survival_trees_rs/src/` with `#[cfg(test)]`:
  - Log-rank statistic on small known cases (golden values).
  - KM with left-truncation vs. `lifelines` fixtures stored as JSON.
  - Deterministic tree construction on a seeded toy dataset.

## Transition plan

### Phase 1 — PR `rust-backend-parallel` (feature branch)
1. Add `rust/`, `Cargo.toml`, and `pyproject.toml` (maturin backend).
   `pyproject.toml` becomes the authoritative build file and replaces
   `setup.py` **even in Phase 1** — mixing both build backends is
   brittle. `rpy2` is still declared in runtime dependencies.
2. The R packages post-install step (current `PostInstall` class in
   `setup.py`) is dropped. Phase 1 is a feature branch, not a release,
   so end users are unaffected. Developers working on the branch run
   `bash build.sh` manually to install the R packages.
3. Add the `survival_trees._rust` compiled module.
4. Add internal classes `_RustBackedLTRCTrees` / `_RustBackedForest`
   in a new `survival_trees/_rust_base.py`. They are not wired into
   the public API yet.
5. Rust unit tests + regression tests + benchmark pass in CI.
6. Existing R code remains functional at runtime; no user-visible
   change to the public Python API.

### Phase 2 — PR `rust-backend-swap`
1. Rewrite `_base.py` so public classes delegate to `_rust_base`.
2. Remove `rpy2` from `requirements.txt`, delete `REstimator`,
   the `.R` scripts, and the `PostInstall` block from `setup.py`
   (replaced by `pyproject.toml` / maturin).
3. Update `build.sh` to install the Rust toolchain and run maturin.
4. Update `README.md` — remove the "Requirements: R" section.
5. Bump version to `0.1.0` (minor, API backward compatible except
   for the `RandomForestSRC` deprecation warning).

### Phase 3 — later release
- Remove `RandomForestSRC` entirely after one deprecation cycle.

## `RandomForestSRC` deprecation

- Release N (swap PR): `RandomForestSRC.__init__` raises
  `DeprecationWarning("RandomForestSRC is deprecated; use "
  "RandomForestLTRC instead. Will be removed in v0.2.0.")`, and
  delegates internally to `RandomForestLTRC` with parameter
  mapping (`n_estimator` → `n_estimators`).
- Target-variable adaptation: the legacy `RandomForestSRC.fit(X, y)`
  accepts `y` with two columns (time, event); `RandomForestLTRC`
  requires three columns (entry, time, event). The shim injects an
  `entry` column of zeros when the input has two columns, preserving
  the historical non-LTRC semantics of `RandomForestSRC`.
- Release N+1: class removed from `__init__.py`.
- Test `test_random_forest_src` verifies the warning, the two-column
  `y` input still works, and that output remains semantically
  coherent (survival ∈ [0, 1], monotone non-increasing per row).

## Open items

None. All decisions from the brainstorming session are captured above.

## References

- Current R scripts: `survival_trees/base_script_n.R`,
  `survival_trees/predict_n.R`.
- Current Python backend: `survival_trees/_base.py`.
- LTRCtrees paper: https://academic.oup.com/biostatistics/article/18/2/352/2739324
- PyO3 guide: https://pyo3.rs
- maturin: https://www.maturin.rs
