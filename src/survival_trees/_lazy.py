"""Lazy sparse forest prediction.

Thin Python wrapper over the Rust `_LazyForest` handle. Stores only a
sample index and an optional default grid; all aggregation
(arithmetic mean of per-tree survivals) is done in Rust with rayon.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class LazyForestSurvival:
    """Deferred forest-level survival predictions.

    Holds a reference to an opaque Rust handle that owns the per-sample
    leaf ids and per-tree per-leaf sparse KM curves. Aggregation is
    computed on demand in `at_time`, `at_times`, or `to_dense`.
    """

    __slots__ = ("_inner", "_sample_index", "_default_grid")

    def __init__(
        self,
        inner,
        sample_index: pd.Index,
        default_grid: Optional[np.ndarray] = None,
    ):
        self._inner = inner
        self._sample_index = pd.Index(sample_index)
        self._default_grid = (
            np.asarray(default_grid, dtype=np.float64)
            if default_grid is not None
            else None
        )

    @property
    def n_samples(self) -> int:
        return int(self._inner.n_samples())

    @property
    def n_trees(self) -> int:
        return int(self._inner.n_trees())

    @property
    def index(self) -> pd.Index:
        return self._sample_index

    @property
    def times(self) -> np.ndarray:
        """Default time grid: the constructor-provided grid if any,
        otherwise the sorted union of every leaf's event times across
        all trees."""
        if self._default_grid is not None:
            return self._default_grid
        return np.asarray(self._inner.union_times(), dtype=np.float64)

    @property
    def n_times(self) -> int:
        return self.times.size

    @property
    def shape(self):
        return (self.n_samples, self.n_times)

    def at_time(self, t: float) -> pd.Series:
        """Forest-averaged survival S(t | X_i) for every sample."""
        t = float(t)
        arr = np.asarray(self._inner.at_time(t), dtype=np.float32)
        return pd.Series(arr, index=self._sample_index, name=t)

    def at_times(self, ts) -> pd.DataFrame:
        """Forest-averaged survival at every time in `ts`,
        DataFrame of shape `(n_samples, len(ts))`."""
        ts_arr = np.asarray(ts, dtype=np.float64)
        if ts_arr.ndim != 1:
            raise ValueError("ts must be 1-dimensional")
        mat = np.asarray(self._inner.at_times(ts_arr), dtype=np.float32)
        return pd.DataFrame(mat, index=self._sample_index, columns=ts_arr)

    def to_dense(self, time_grid=None) -> pd.DataFrame:
        """Materialize the full dense `(n_samples, n_times)` DataFrame
        on the requested grid (or the default one)."""
        grid = self.times if time_grid is None else np.asarray(time_grid, dtype=np.float64)
        return self.at_times(grid)

    def to_numpy(self, dtype=None):
        return self.to_dense().to_numpy(dtype=dtype)

    def iloc_samples(self, ix) -> pd.DataFrame:
        """Return a dense DataFrame for a subset of samples."""
        ix_arr = np.asarray(ix, dtype=np.int64)
        return self.to_dense().iloc[ix_arr]

    def __repr__(self) -> str:
        return (
            f"<LazyForestSurvival n_samples={self.n_samples} "
            f"n_trees={self.n_trees}>"
        )
