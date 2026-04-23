import warnings

import pandas as pd

from ._common import validate_y
from ._rust_base import _RustBackedForest, _RustBackedLTRCTrees


class LTRCTrees(_RustBackedLTRCTrees):
    """Left-truncated right-censored survival tree regressor."""


class RandomForestLTRC(_RustBackedForest):
    """Left-truncated right-censored survival random forest."""


class ExtraSurvivalTrees(_RustBackedForest):
    """LTRC survival forest with randomised splits (Geurts et al. 2006):
    one threshold per feature drawn uniformly between observed min and
    max, scored by log-rank."""

    def __init__(self, **kwargs):
        kwargs.setdefault("splitter", "random")
        super().__init__(**kwargs)


class RandomForestSRC(RandomForestLTRC):
    """Deprecated: use :class:`RandomForestLTRC`. Accepts two-column ``y``
    ``(time, event)`` by injecting an entry column of zeros."""

    def __init__(self, n_estimator: int = 100, **kwargs):
        warnings.warn(
            "RandomForestSRC is deprecated; use RandomForestLTRC instead.",
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
    return validate_y(y)
