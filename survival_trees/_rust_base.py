"""Rust-backed implementations of LTRCTrees and RandomForestLTRC.

This module is internal and not re-exported via ``__init__.py`` in Phase 1.
It mirrors the public signatures of ``_base.LTRCTrees`` and
``_base.RandomForestLTRC`` so that Phase 2 can simply swap the public
classes to these implementations.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from . import _rust
from ._common import forest_post_processing, validate_y


def _extract_xy(X: pd.DataFrame, y: pd.DataFrame):
    """Return arrays (x, entry, time, event) with the dtypes expected by the
    Rust backend, given the public ``_validate_y`` contract:
    y columns = [truncation, age_of_death, death]."""
    entry = np.ascontiguousarray(y.iloc[:, 0].to_numpy(dtype=np.float64))
    time = np.ascontiguousarray(y.iloc[:, 1].to_numpy(dtype=np.float64))
    event_raw = y.iloc[:, 2]
    if event_raw.dtype != bool:
        values = np.unique(event_raw)
        if set(np.asarray(values).tolist()) <= {0, 1, True, False}:
            event = np.ascontiguousarray(event_raw.astype(bool).to_numpy())
        else:
            raise ValueError("event column must be boolean or 0/1")
    else:
        event = np.ascontiguousarray(event_raw.to_numpy(dtype=bool))
    x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    return x, entry, time, event


def _build_control(tree: "_RustBackedLTRCTrees") -> dict:
    return {
        "max_depth": tree.max_depth,
        "min_samples_leaf": (1 if tree.min_samples_leaf is None
                             else int(tree.min_samples_leaf)),
        "min_samples_split": (2 if tree.min_samples_split is None
                              else int(tree.min_samples_split)),
        "min_impurity_decrease": (0.0 if tree.min_impurity_decrease is None
                                  else float(tree.min_impurity_decrease)),
    }


def _assemble_dense_curves(curves_mat: np.ndarray,
                           leaf_ids: np.ndarray,
                           times: np.ndarray,
                           index: pd.Index) -> pd.DataFrame:
    """curves_mat: (n_leaves, n_times); leaf_ids: (n_samples,).
    Returns DataFrame of shape (n_samples, n_times) indexed like `index`."""
    if times.size == 0:
        return pd.DataFrame(index=index, dtype="float32")
    rows = curves_mat[leaf_ids, :]
    return pd.DataFrame(rows, index=index, columns=times, dtype="float32")


class _RustBackedLTRCTrees(BaseEstimator, ClassifierMixin):
    """Drop-in Rust-backed replacement for ``_base.LTRCTrees``."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        get_dense_prediction: bool = True,
        interpolate_prediction: bool = True,
        min_impurity_decrease: Optional[float] = None,
        min_samples_split: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.get_dense_prediction = get_dense_prediction
        self.interpolate_prediction = interpolate_prediction
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        validate_y(y)
        x, entry, time, event = _extract_xy(X, y)
        self._inner = _rust._RustLtrcTree()
        self._inner.fit(x, entry, time, event, _build_control(self))
        self._feature_names_ = list(X.columns)
        importances = np.asarray(self._inner.feature_importances(), dtype=float)
        self.feature_importances_ = importances.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        curves_mat, leaf_ids, times = self._predict_raw(X)
        km_mat = _assemble_dense_curves(curves_mat, leaf_ids, times, X.index)
        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float32", np.nan))
        if self.interpolate_prediction:
            if km_mat.shape[1] == 0 or 0 not in km_mat.columns:
                km_mat[0] = 1.0
            km_mat = km_mat[np.sort(km_mat.columns)]
            km_mat = km_mat.astype("float32").T.ffill().T
        return km_mat

    def predict_curves(self, X: pd.DataFrame):
        curves_mat, leaf_ids, times = self._predict_raw(X)
        curves_df = pd.DataFrame(curves_mat, columns=times, dtype="float32")
        indexes = pd.Series(leaf_ids, index=X.index, dtype="int64")
        return curves_df, indexes

    def _predict_raw(self, X: pd.DataFrame):
        x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
        curves_mat, leaf_ids, times = self._inner.predict_curves(x)
        return (np.asarray(curves_mat, dtype=np.float32),
                np.asarray(leaf_ids, dtype=np.int64),
                np.asarray(times, dtype=np.float64))


class _RustBackedForest(ClassifierMixin):
    """Drop-in Rust-backed replacement for ``_base.RandomForestLTRC``."""

    def __init__(
        self,
        n_estimators: int = 3,
        max_features: Union[float, int, None] = None,
        max_depth: Optional[float] = None,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        min_samples_leaf: Optional[int] = None,
        min_impurity_decrease: float = 0.01,
        min_samples_split: int = 2,
        base_estimator: Optional[_RustBackedLTRCTrees] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_estimator = n_estimators
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        if base_estimator is None:
            self.base_estimator_ = _RustBackedLTRCTrees(
                interpolate_prediction=False,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_samples_split=self.min_samples_split,
            )
        else:
            self.base_estimator_ = base_estimator

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, float):
            return max(2, min(n_features, int(round(self.max_features * n_features))))
        if self.max_features == "auto":
            return max(2, n_features // 3)
        return max(2, min(n_features, int(self.max_features)))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        validate_y(y)
        x, entry, time, event = _extract_xy(X, y)
        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.feature_names_in_ = X.columns.to_list()

        inner = _rust._RustLtrcForest()
        seed = 0 if self.random_state is None else int(self.random_state)
        n_jobs = int(self.n_jobs) if self.n_jobs else 0
        if n_jobs <= 0:
            try:
                import os
                n_jobs = max(1, os.cpu_count() or 1)
            except Exception:
                n_jobs = 1
        inner.fit(
            x, entry, time, event,
            _build_control(self.base_estimator_),
            int(self.n_estimators),
            float(self.max_samples) if self.bootstrap else 1.0,
            int(self._resolve_max_features(n_features)),
            int(seed),
            int(n_jobs),
        )
        self._inner = inner
        importances = np.asarray(inner.feature_importances(), dtype=float)
        self.feature_importances_ = importances.tolist()
        self._feature_subsets = [list(subset) for subset in inner.feature_subsets()]
        return self

    def fast_predict_(self, X: pd.DataFrame) -> None:
        x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
        per_tree = self._inner.predict_forest(x)
        result = {}
        for e, (curves_mat, leaf_ids, times) in enumerate(per_tree):
            curves_mat = np.asarray(curves_mat, dtype=np.float32)
            leaf_ids = np.asarray(leaf_ids, dtype=np.int64)
            times = np.asarray(times, dtype=np.float64)
            curves_df = pd.DataFrame(curves_mat, columns=times, dtype="float32")
            indexes = pd.Series(leaf_ids, index=X.index, dtype="int64")
            result[e] = (curves_df, indexes)
        self.km_estimates_, self.nodes_ = forest_post_processing(result, X)

    def predict(self, X: pd.DataFrame, return_type: str = "dense") -> pd.DataFrame:
        self.fast_predict_(X)
        if return_type == "dense":
            return pd.merge(
                self.nodes_, self.km_estimates_,
                left_on="curve_index", right_index=True,
            ).set_index("x_index").drop(columns=["curve_index"]).loc[X.index]
        raise ValueError(f"return_type : {return_type} is not implemented yet")
