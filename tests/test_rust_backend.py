"""Smoke tests for the Rust backend — shape and survival-curve sanity."""

import numpy as np
import pandas as pd
import pytest

lifelines = pytest.importorskip(
    "lifelines",
    reason="lifelines not installed; the Rust smoke tests need lifelines.datasets",
)

from survival_trees import ExtraSurvivalTrees  # noqa: E402
from survival_trees._rust_base import (  # noqa: E402
    _RustBackedForest,
    _RustBackedLTRCTrees,
)


@pytest.fixture(scope="module")
def ltrc_xy():
    data = lifelines.datasets.load_larynx().dropna().copy()
    data["entry_date"] = data["age"].astype(float)
    data["time"] = data["time"].astype(float) + data["entry_date"]
    data["death"] = data["death"].astype(bool)
    y = data[["entry_date", "time", "death"]]
    x = data.drop(columns=y.columns.tolist())
    return x, y


def _assert_survival_sanity(df: pd.DataFrame):
    arr = df.to_numpy(dtype=float)
    mask = np.isfinite(arr)
    assert mask.any(), "all survival values are NaN"
    vals = arr[mask]
    assert (vals >= -1e-6).all() and (vals <= 1.0 + 1e-6).all(), \
        f"survival out of [0, 1]; min={vals.min()}, max={vals.max()}"
    for i in range(df.shape[0]):
        row = arr[i, :]
        row = row[np.isfinite(row)]
        diffs = np.diff(row)
        assert (diffs <= 1e-6).all(), f"non-monotone survival on row {i}"


def test_tree_fit_predict(ltrc_xy):
    x, y = ltrc_xy
    tree = _RustBackedLTRCTrees(
        max_depth=3,
        min_samples_leaf=3,
        min_samples_split=4,
        min_impurity_decrease=0.0,
    )
    tree.fit(x, y)
    imp = tree.feature_importances_
    assert len(imp) == x.shape[1]
    assert all(v >= 0.0 for v in imp)

    pred = tree.predict(x)
    assert isinstance(pred, pd.DataFrame)
    assert pred.shape[0] == x.shape[0]
    assert pred.shape[1] >= 1
    _assert_survival_sanity(pred)


def test_tree_predict_curves_shape(ltrc_xy):
    x, y = ltrc_xy
    tree = _RustBackedLTRCTrees(max_depth=2, min_samples_leaf=3)
    tree.fit(x, y)
    curves, indexes = tree.predict_curves(x)
    assert isinstance(curves, pd.DataFrame)
    assert isinstance(indexes, pd.Series)
    assert indexes.index.equals(x.index)
    n_leaves = int(indexes.max()) + 1
    assert curves.shape[0] == n_leaves


def test_forest_fit_predict(ltrc_xy):
    x, y = ltrc_xy
    forest = _RustBackedForest(
        n_estimators=5,
        max_samples=0.8,
        min_samples_leaf=3,
        min_impurity_decrease=0.0,
        random_state=42,
        n_jobs=1,
    )
    forest.fit(x, y)
    assert forest.n_features_in_ == x.shape[1]
    assert forest.feature_names_in_ == list(x.columns)
    assert len(forest.feature_importances_) == x.shape[1]

    pred = forest.predict(x)
    assert isinstance(pred, pd.DataFrame)
    assert pred.shape[0] == x.shape[0]
    assert hasattr(forest, "nodes_")
    assert hasattr(forest, "km_estimates_")
    _assert_survival_sanity(pred)


def test_forest_reproducible(ltrc_xy):
    x, y = ltrc_xy
    kwargs = {
        "n_estimators": 4,
        "max_samples": 0.8,
        "min_samples_leaf": 3,
        "min_impurity_decrease": 0.0,
        "random_state": 123,
        "n_jobs": 1,
    }
    f1 = _RustBackedForest(**kwargs).fit(x, y)
    f2 = _RustBackedForest(**kwargs).fit(x, y)
    assert f1.feature_importances_ == f2.feature_importances_


def test_forest_time_grid_int(ltrc_xy):
    x, y = ltrc_xy
    forest = _RustBackedForest(
        n_estimators=5,
        min_samples_leaf=3,
        min_impurity_decrease=0.0,
        random_state=0,
        n_jobs=1,
        time_grid=5,
    )
    forest.fit(x, y)
    pred = forest.predict(x)
    assert pred.shape[0] == x.shape[0]
    assert pred.shape[1] <= 5
    cols = np.asarray(pred.columns, dtype=float)
    assert (np.diff(cols) >= 0).all()
    _assert_survival_sanity(pred)


def test_forest_time_grid_array(ltrc_xy):
    x, y = ltrc_xy
    grid = np.linspace(float(y.iloc[:, 0].min()) + 0.1,
                       float(y.iloc[:, 1].max()), 7)
    forest = _RustBackedForest(
        n_estimators=5, min_samples_leaf=3, min_impurity_decrease=0.0,
        random_state=0, n_jobs=1, time_grid=grid,
    )
    forest.fit(x, y)
    pred = forest.predict(x)
    assert pred.shape[1] == grid.size
    np.testing.assert_allclose(np.asarray(pred.columns, dtype=float), grid)
    _assert_survival_sanity(pred)


def test_forest_lazy_matches_dense(ltrc_xy):
    x, y = ltrc_xy
    kwargs = {"n_estimators": 5, "min_samples_leaf": 3,
              "min_impurity_decrease": 0.0, "random_state": 0, "n_jobs": 1}
    forest = _RustBackedForest(**kwargs).fit(x, y)
    lazy = forest.predict(x, lazy=True)
    assert lazy.shape[0] == x.shape[0]
    assert lazy.n_samples == x.shape[0]
    dense_from_lazy = lazy.to_dense()
    dense_direct = _RustBackedForest(**kwargs).fit(x, y).predict(x)
    # Lazy sums trees in a different order than the eager Rust path;
    # f32 summation order yields differences ~1e-7.
    np.testing.assert_allclose(
        dense_from_lazy.to_numpy(dtype=np.float32),
        dense_direct.to_numpy(dtype=np.float32),
        rtol=1e-5, atol=1e-6,
    )


def test_forest_fit_predict_with_nan_features(ltrc_xy):
    x, y = ltrc_xy
    rng = np.random.default_rng(0)
    x_nan = x.astype(float).copy()
    mask = rng.random(x_nan.shape) < 0.2
    x_nan.values[mask] = np.nan
    forest = _RustBackedForest(
        n_estimators=5, min_samples_leaf=3, min_impurity_decrease=0.0,
        random_state=0, n_jobs=1,
    )
    forest.fit(x_nan, y)
    pred = forest.predict(x_nan)
    _assert_survival_sanity(pred)


def test_forest_predict_nan_on_clean_fit(ltrc_xy):
    """Fit on clean data, predict on X with NaN → follows learned defaults."""
    x, y = ltrc_xy
    forest = _RustBackedForest(
        n_estimators=5, min_samples_leaf=3, min_impurity_decrease=0.0,
        random_state=0, n_jobs=1,
    ).fit(x, y)
    x_query = x.astype(float).copy()
    x_query.iloc[:5] = np.nan
    pred = forest.predict(x_query)
    assert pred.shape[0] == x.shape[0]
    _assert_survival_sanity(pred)


def test_forest_rejects_nan_in_time(ltrc_xy):
    x, y = ltrc_xy
    y_bad = y.copy()
    y_bad.iloc[0, 1] = np.nan
    forest = _RustBackedForest(n_estimators=2, n_jobs=1)
    with pytest.raises(Exception):
        forest.fit(x, y_bad)


def test_extra_survival_trees_fit_predict(ltrc_xy):
    x, y = ltrc_xy
    forest = ExtraSurvivalTrees(
        n_estimators=10, min_samples_leaf=3,
        min_impurity_decrease=0.0, random_state=0, n_jobs=1,
    ).fit(x, y)
    assert forest.splitter == "random"
    pred = forest.predict(x)
    _assert_survival_sanity(pred)


def test_extra_survival_trees_reproducible(ltrc_xy):
    x, y = ltrc_xy
    kw = {"n_estimators": 10, "min_samples_leaf": 3,
          "min_impurity_decrease": 0.0, "random_state": 42, "n_jobs": 1}
    f1 = ExtraSurvivalTrees(**kw).fit(x, y)
    f2 = ExtraSurvivalTrees(**kw).fit(x, y)
    np.testing.assert_allclose(
        f1.predict(x).to_numpy(dtype=np.float32),
        f2.predict(x).to_numpy(dtype=np.float32),
    )


def test_extra_survival_trees_differs_from_rf(ltrc_xy):
    """Same seed, different splitters → different predictions."""
    x, y = ltrc_xy
    kw = {"n_estimators": 10, "min_samples_leaf": 3,
          "min_impurity_decrease": 0.0, "random_state": 0, "n_jobs": 1}
    rf = _RustBackedForest(**kw).fit(x, y)
    ext = ExtraSurvivalTrees(**kw).fit(x, y)
    p_rf = rf.predict(x).to_numpy(dtype=float)
    p_ext = ext.predict(x).to_numpy(dtype=float)
    assert not np.allclose(p_rf, p_ext, atol=1e-4), \
        "random and best splitter must not give identical predictions"


def test_forest_pipelines_differ(ltrc_xy):
    """Aalen and Km pipelines produce different but valid forest predictions."""
    x, y = ltrc_xy
    kwargs = {"n_estimators": 8, "min_samples_leaf": 3,
              "min_impurity_decrease": 0.0, "random_state": 0, "n_jobs": 1}
    f_aalen = _RustBackedForest(pipeline="aalen", **kwargs).fit(x, y)
    f_km = _RustBackedForest(pipeline="km", **kwargs).fit(x, y)
    p_aalen = f_aalen.predict(x).to_numpy(dtype=float)
    p_km = f_km.predict(x).to_numpy(dtype=float)
    _assert_survival_sanity(f_aalen.predict(x))
    _assert_survival_sanity(f_km.predict(x))
    assert not np.allclose(p_aalen, p_km, atol=1e-4), \
        "aalen and km pipelines must not give identical predictions"


def test_forest_lazy_at_time(ltrc_xy):
    x, y = ltrc_xy
    forest = _RustBackedForest(
        n_estimators=5, min_samples_leaf=3, min_impurity_decrease=0.0,
        random_state=0, n_jobs=1,
    ).fit(x, y)
    lazy = forest.predict(x, lazy=True)
    dense = lazy.to_dense()
    times = lazy.times
    assert times.size > 0
    probes = [times[0] - 1.0, times[0], times[times.size // 2],
              times[-1], times[-1] + 1000.0]
    for t in probes:
        got = lazy.at_time(t).to_numpy(dtype=np.float32)
        if t < times[0]:
            expected = np.ones(x.shape[0], dtype=np.float32)
        else:
            col_idx = int(np.searchsorted(times, t, side="right") - 1)
            expected = dense.iloc[:, col_idx].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(got, expected)
