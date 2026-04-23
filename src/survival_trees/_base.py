"""Public estimator classes backed by the Rust extension module.

Phase 2: the R/rpy2 backend has been removed. All estimators now delegate
to :mod:`survival_trees._rust_base`. The public API (class names, method
signatures, attribute names) is preserved.
"""

import warnings

import pandas as pd

from ._common import validate_y
from ._rust_base import _RustBackedForest, _RustBackedLTRCTrees


class LTRCTrees(_RustBackedLTRCTrees):
    """A left-truncated right-censored survival tree regressor.

    Backed by the Rust extension since v0.1.0. The public surface
    (``.fit``, ``.predict``, ``.predict_curves``, ``feature_importances_``)
    is unchanged from the previous R-backed version. See the spec at
    ``docs/superpowers/specs/2026-04-21-replace-r-with-rust-backend-design.md``.
    """


class RandomForestLTRC(_RustBackedForest):
    """A left-truncated right-censored survival random forest.

    Backed by the Rust extension since v0.1.0."""


class ExtraSurvivalTrees(_RustBackedForest):
    """Extra-Survival-Trees — LTRC survival forest with randomised splits.

    At each node, one threshold per feature is drawn uniformly between
    the observed min and max (Geurts et al. 2006), scored by log-rank,
    and the best feature wins. Trees are less accurate individually
    but much less correlated → smoother `Λ_F` curves, better suited
    to the Cox-like decomposition API."""

    def __init__(self, **kwargs):
        kwargs.setdefault("splitter", "random")
        super().__init__(**kwargs)


class RandomForestSRC(RandomForestLTRC):
    """Deprecated: use :class:`RandomForestLTRC` instead.

    Kept as a thin shim that accepts the legacy two-column ``y``
    (``time``, ``event``) by injecting an entry column of zeros, so
    callers written against the old R-backed ``randomForestSRC`` wrapper
    keep working during the deprecation window. Will be removed in
    v0.2.0.
    """

    def __init__(self, n_estimator: int = 100, **kwargs):
        warnings.warn(
            "RandomForestSRC is deprecated; use RandomForestLTRC instead. "
            "Will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("n_estimators", n_estimator)
        super().__init__(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if y.shape[1] == 2:
            y = y.copy()
            y.insert(0, "__entry__", 0.0)
        return super().fit(X, y)


def _validate_y(y: pd.DataFrame):
    """Backwards-compatible alias for :func:`._common.validate_y`."""
    return validate_y(y)
