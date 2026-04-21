"""Phase 1 smoke tests for the Rust backend of survival-trees.

These hit the Rust compiled module via ``survival_trees._rust_base`` and
do not touch the current R-backed public API. The spec accepts free
functional equivalence (not numerical parity with LTRCART), so we check
shape + basic survival-curve sanity only.
"""

import numpy as np
import pandas as pd
import pytest

lifelines = pytest.importorskip(
    "lifelines",
    reason="lifelines not installed; the Rust smoke tests need lifelines.datasets",
)

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
    kwargs = dict(
        n_estimators=4,
        max_samples=0.8,
        min_samples_leaf=3,
        min_impurity_decrease=0.0,
        random_state=123,
        n_jobs=1,
    )
    f1 = _RustBackedForest(**kwargs).fit(x, y)
    f2 = _RustBackedForest(**kwargs).fit(x, y)
    assert f1.feature_importances_ == f2.feature_importances_
