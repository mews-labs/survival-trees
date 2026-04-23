from __future__ import annotations

import numpy as np
import pandas as pd

from survival_trees import LTRCTrees as LTRCT
from survival_trees import RandomForestLTRC as RF


class RandomForestLTRC(RF):
    """``data``-oriented fitter around :class:`survival_trees.RandomForestLTRC`.

    Accepts a single DataFrame with explicit ``duration_col``, ``event_col``
    and ``entry_col`` instead of the ``(X, y)`` split of the base class."""

    def __init__(self,
                 n_estimators: int = 3,
                 max_features: float | int | None = None,
                 max_depth: float | None = None,
                 bootstrap: bool = True,
                 max_samples: float = 1,
                 min_samples_leaf: int | None = None,
                 min_impurity_decrease: float | None = None,
                 min_samples_split: int = 2,
                 base_estimator: LTRCTrees | None = None,
                 ):
        super().__init__(n_estimators=n_estimators,
                         max_features=max_features,
                         max_depth=max_depth,
                         bootstrap=bootstrap,
                         max_samples=max_samples,
                         min_samples_leaf=min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split,
                         base_estimator=base_estimator)

    def fit(self, data: pd.DataFrame, duration_col: str,
            event_col: str, entry_col: str):
        X = data.drop(columns=[entry_col, duration_col, event_col])
        y = data[[entry_col, duration_col, event_col]]
        return super().fit(X, y)

    def predict_survival(self, X: pd.DataFrame, return_type="dense"
                         ) -> pd.DataFrame | None:
        return self.predict(X, return_type).T

    def predict_cumulative_hazard(self, X: pd.DataFrame, return_type="dense"):
        data = self.predict(X, return_type).T
        return pd.DataFrame(-np.log(data), index=data.index, columns=data.columns)


class LTRCTrees(LTRCT):
    def __init__(self,
                 max_depth: int | None = None,
                 min_samples_leaf: int | None = None,
                 min_impurity_decrease: float | None = None,
                 min_samples_split: float | None = None,
                 ):
        super().__init__(max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split
                         )

    def fit(self, data: pd.DataFrame, duration_col: str,
            event_col: str, entry_col: str):
        X = data.drop(columns=[entry_col, duration_col, event_col])
        y = data[[entry_col, duration_col, event_col]]
        return super().fit(X, y)

    def predict_survival(self, X: pd.DataFrame) -> pd.DataFrame | None:
        return self.predict(X).T

    def predict_cumulative_hazard(self, X: pd.DataFrame):
        data = self.predict(X).T
        return pd.DataFrame(-np.log(data), index=data.index, columns=data.columns)
