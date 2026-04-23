from typing import Union

import numpy as np
import pandas as pd

from survival_trees import LTRCTrees as LTRCT
from survival_trees import RandomForestLTRC as RF


class RandomForestLTRC(RF):
    """
        A left truncated right censored survival random forest regressor.
        LeTRiCS-RF

        This is a meta estimator that fits a number of LTRC trees
        in order to improve the prediction by averging several surival function

        The sub-sample size is controlled with the `max_samples` parameter if
        `bootstrap=True` (default), otherwise the whole dataset is used to build
        each tree.

        Parameters
        ----------
        n_estimators : int, default=3
            The number of trees in the forest.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.


        max_features : int or float, default="auto"
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `round(max_features * n_features)` features are considered at each
              split.
            - If None, then `max_features=n_features//3`.

        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees. If False,
            the  whole dataset is used to build each tree.

        max_samples : float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.

            - If None (default), then draw `X.shape[0]` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
              `max_samples` should be in the interval `(0.0, 1.0]`.

        min_impurity_decrease: float, default=0.01
            complexity parameter. Any split that does not decrease the overall
            lack of fit by a factor of cp is not attempted

        Attributes
        ----------
        base_estimator_ : LTRCTrees
            The child estimator template used to create the collection of fitted
            sub-estimators.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

        feature_importances_ : ndarray of shape (n_features,)
            The impurity-based feature importances.
            The higher, the more important the feature.
            The importance of a feature is computed as the (normalized)
            total reduction of the criterion brought by that feature.  It is also
            known as the Gini importance.

            Warning: impurity-based feature importances can be misleading for
            high cardinality features (many unique values). See
            :func:`sklearn.inspection.permutation_importance` as an alternative.



        Notes
        -----
        The default values for the parameters controlling the size of the trees
        (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
        unpruned trees which can potentially be very large on some data sets. To
        reduce memory consumption, the complexity and size of the trees should be
        controlled by setting those parameter values.

        The features are always randomly permuted at each split. Therefore,
        the best found split may vary, even with the same training data,
        ``max_features=n_features`` and ``bootstrap=False``, if the improvement
        of the criterion is identical for several splits enumerated during the
        search of the best split. To obtain a deterministic behaviour during
        fitting, ``random_state`` has to be fixed.

        References
        ----------

        Examples
        --------
        """

    def __init__(self,
                 n_estimators: int = 3,
                 max_features: Union[float, int] = None,
                 max_depth: float = None,
                 bootstrap: bool = True,
                 max_samples: float = 1,
                 min_samples_leaf: int = None,
                 min_impurity_decrease: float = None,
                 min_samples_split: int = 2,
                 base_estimator: "LTRCTrees" = None,
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
        """

        Parameters
        ----------
        data: pandas DataFrame
            Input data
        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.
        event_col: string
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.
        entry_col: string
            the  name of the column in DataFrame that contains the subjects'
            entry time.
        Returns
        -------

        """
        X = data.drop(columns=[entry_col, duration_col, event_col])
        y = data[[entry_col, duration_col, event_col]]
        return super().fit(X, y)

    def predict_survival(self, X: pd.DataFrame, return_type="dense"
                         ) -> Union[pd.DataFrame, None]:
        return self.predict(X, return_type).T

    def predict_cumulative_hazard(self, X: pd.DataFrame, return_type="dense"):
        data = self.predict(X, return_type).T
        return pd.DataFrame(-np.log(data), index=data.index, columns=data.columns)


class LTRCTrees(LTRCT):
    def __init__(self,
                 max_depth: int = None,
                 min_samples_leaf: int = None,
                 min_impurity_decrease: float = None,
                 min_samples_split: float = None
                 ):
        super().__init__(max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split
                         )

    def fit(self, data: pd.DataFrame, duration_col: str,
            event_col: str, entry_col: str):
        """

        Parameters
        ----------
        data: pandas DataFrame
            Input data
        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.
        event_col: string
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.
        entry_col: string
            the  name of the column in DataFrame that contains the subjects'
            entry time.
        Returns
        -------

        """
        X = data.drop(columns=[entry_col, duration_col, event_col])
        y = data[[entry_col, duration_col, event_col]]
        return super().fit(X, y)

    def predict_survival(self, X: pd.DataFrame) -> Union[pd.DataFrame, None]:
        return self.predict(X).T

    def predict_cumulative_hazard(self, X: pd.DataFrame):
        data = self.predict(X).T
        return pd.DataFrame(-np.log(data), index=data.index, columns=data.columns)
