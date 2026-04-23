from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from . import _rust
from ._common import validate_y


def _extract_xy(X: pd.DataFrame, y: pd.DataFrame):
    entry = np.ascontiguousarray(y.iloc[:, 0].to_numpy(dtype=np.float64))
    time = np.ascontiguousarray(y.iloc[:, 1].to_numpy(dtype=np.float64))
    event_raw = y.iloc[:, 2]
    if event_raw.dtype != bool:
        if not set(np.unique(event_raw).tolist()) <= {0, 1}:
            raise ValueError("event column must be boolean or 0/1")
    event = np.ascontiguousarray(event_raw.to_numpy(dtype=bool))
    x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    return x, entry, time, event


def _build_control(tree: _RustBackedLTRCTrees) -> dict:
    return {
        "max_depth": tree.max_depth,
        "min_samples_leaf": tree.min_samples_leaf or 1,
        "min_samples_split": tree.min_samples_split or 2,
        "min_impurity_decrease": tree.min_impurity_decrease or 0.0,
        "criterion": tree.criterion,
        "pipeline": tree.pipeline,
        "splitter": tree.splitter,
    }


def _assemble_dense_curves(curves_mat: np.ndarray, leaf_ids: np.ndarray,
                           times: np.ndarray, index: pd.Index) -> pd.DataFrame:
    if times.size == 0:
        return pd.DataFrame(index=index, dtype="float32")
    return pd.DataFrame(curves_mat[leaf_ids, :], index=index,
                        columns=times, dtype="float32")


class _RustBackedLTRCTrees(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        get_dense_prediction: bool = True,
        interpolate_prediction: bool = True,
        min_impurity_decrease: Optional[float] = None,
        min_samples_split: Optional[int] = None,
        criterion: str = "log-rank",
        pipeline: str = "aalen",
        splitter: str = "best",
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.get_dense_prediction = get_dense_prediction
        self.interpolate_prediction = interpolate_prediction
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.pipeline = pipeline
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        validate_y(y)
        x, entry, time, event = _extract_xy(X, y)
        self._inner = _rust._RustLtrcTree()
        seed = 0 if self.random_state is None else int(self.random_state)
        self._inner.fit(x, entry, time, event, _build_control(self), seed)
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
        time_grid: Optional[Union[int, np.ndarray]] = None,
        criterion: str = "log-rank",
        pipeline: str = "aalen",
        splitter: str = "best",
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
        self.time_grid = time_grid
        self.criterion = criterion
        self.pipeline = pipeline
        self.splitter = splitter
        if base_estimator is None:
            self.base_estimator_ = _RustBackedLTRCTrees(
                interpolate_prediction=False,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                pipeline=self.pipeline,
                splitter=self.splitter,
            )
        else:
            self.base_estimator_ = base_estimator

    def _resolve_time_grid(self, y: pd.DataFrame) -> Optional[np.ndarray]:
        if self.time_grid is None:
            return None
        if isinstance(self.time_grid, (int, np.integer)):
            n_bins = int(self.time_grid)
            if n_bins <= 0:
                return None
            event_mask = y.iloc[:, 2].astype(bool).to_numpy()
            event_times = y.iloc[:, 1].to_numpy(dtype=np.float64)[event_mask]
            if event_times.size == 0:
                return None
            qs = np.linspace(0.0, 1.0, n_bins + 1)[1:]
            return np.unique(np.quantile(event_times, qs)).astype(np.float64)
        grid = np.asarray(self.time_grid, dtype=np.float64)
        if grid.ndim != 1:
            raise ValueError("time_grid array must be 1-dimensional")
        if grid.size and np.any(np.diff(grid) < 0):
            raise ValueError("time_grid must be sorted ascending")
        return grid

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, float):
            return max(2, min(n_features, round(self.max_features * n_features)))
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
        self._time_grid_ = self._resolve_time_grid(y)
        return self

    def fast_predict_(self, X: pd.DataFrame) -> None:
        x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
        grid = getattr(self, "_time_grid_", None)
        if grid is None:
            curves, node_index, times = self._inner.predict_aggregated(x, None)
        else:
            curves, node_index, times = self._inner.predict_aggregated(x, grid)
        curves = np.asarray(curves, dtype=np.float32)
        node_index = np.asarray(node_index, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        self.km_estimates_ = pd.DataFrame(
            curves, columns=times, index=pd.RangeIndex(curves.shape[0]), dtype="float32"
        )
        self.nodes_ = pd.DataFrame(
            {"x_index": X.index.to_numpy(), "curve_index": node_index}
        )

    def predict(self, X: pd.DataFrame, return_type: str = "dense",
                lazy: bool = False):
        if lazy:
            from ._lazy import LazyForestSurvival
            x = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
            inner = self._inner.predict_lazy(x)
            return LazyForestSurvival(
                inner=inner,
                sample_index=X.index,
                default_grid=getattr(self, "_time_grid_", None),
            )
        self.fast_predict_(X)
        if return_type == "dense":
            node_index = self.nodes_["curve_index"].to_numpy()
            result = self.km_estimates_.iloc[node_index].copy()
            result.index = X.index
            return result
        raise ValueError(f"return_type : {return_type} is not implemented yet")
