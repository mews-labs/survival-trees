import os
from typing import List
from typing import Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, ClassifierMixin

from survival_trees.tools import execution

path = os.path.abspath(__file__).replace(os.path.basename(__file__), "")

r_session = ro.r


class REstimator(BaseEstimator):

    def __init__(self):
        from rpy2.rinterface_lib.embedded import RRuntimeError
        self.__utils = rpackages.importr('utils')
        ro.r("sink('/dev/null')")
        try:
            self.__utils.chooseCRANmirror(ind=1)
        except RRuntimeError:
            pass
        self.__learn_name = "data.X"
        self.__test_name = "data.X"
        self._r_data_frame = pd.DataFrame()

    def _send_to_r_space(self, X, y=None):
        if y is not None:
            name = self.__learn_name
            data = pd.concat((X, y), axis=1)
        else:
            name = self.__test_name
            data = X
        with localconverter(ro.default_converter + pandas2ri.converter):
            self._r_data_frame = ro.conversion.py2rpy(data)
        ro.globalenv[name] = self._r_data_frame

    @staticmethod
    def _import_packages(list_package: List[str]):
        for package in list_package:
            with execution.silence_stdout():
                rpackages.importr(package)

    @staticmethod
    def _get_from_r_space(list_object: List[str]):
        dict_result = {}
        for o in list_object:
            with localconverter(ro.default_converter + pandas2ri.converter):
                dict_result[o] = ro.conversion.rpy2py(ro.globalenv[o])
        return dict_result


class RandomForestSRC(REstimator, ClassifierMixin):
    def __init__(self, n_estimator=100):
        super().__init__()
        self.n_estimator = n_estimator
        self.name = "randomForestSRC"
        install_if_needed([self.name])
        self._import_packages(["randomForestSRC"])

    def fit(self, X, y: pd.DataFrame):
        """
        :param X: data frame
        :param y: 2D data set (time and status)
        :return:
        """
        if y.shape[1] != 2:
            raise ValueError("Target data should "
                             "be a dataframe with two columns")
        duration = y.columns[0]
        event = y.columns[1]
        self._send_to_r_space(X, y)
        r_session(f"""
        forest.obj <- randomForestSRC::rfsrc(
            Surv({duration}, {event}) ~ .,
            data = data.X, 
            ntree = {self.n_estimator}, 
            tree.err=TRUE)""")
        r_session("imp <- vimp(forest.obj)$importance")
        self.feature_importances_ = self._get_from_r_space(["imp"])["imp"]

    def predict(self, X) -> pd.DataFrame:
        self._send_to_r_space(X, y=None)
        r_session("predict.obj <- predict(forest.obj, data.X)")
        r_session("res <- predict.obj$survival")
        r_session("times <- predict.obj$time.interest")
        getter = self._get_from_r_space(["res", "times"])
        prediction = getter["res"]
        times = getter["times"]
        return pd.DataFrame(prediction, columns=times, index=X.index)


class LTRCTrees(REstimator, ClassifierMixin):
    """
    A left truncated right censored trees regressor.

    Based on rpart algorithm from R, and LTRC R package.
    This algorithm evaluates Kaplan Meier estimate.

    Parameters
    ----------

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure (no death) or until all leaves contain less than
        min_samples_leaf samples.

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

    min_impurity_decrease: float, default=0.01
        complexity parameter. Any split that does not decrease the overall
        lack of fit by a factor of min_impurity_decrease is not attempted

    min_samples_split: float, default=2

    Attributes
    ----------

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.


    See Also
    --------

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] LTRC Trees `manual`_
    .. [2] Rpart algorithm

    Examples
    --------

    .. _manual: https://cran.r-project.org/web/packages/LTRCtrees/LTRCtrees.pdf
    """

    def __init__(
            self, max_depth=None,
            min_samples_leaf=None,
            get_dense_prediction=True,
            interpolate_prediction=True,
            min_impurity_decrease: float = None,
            min_samples_split=None,
    ):
        super().__init__()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._import_packages(["data.table", "LTRCtrees", "survival", 'hash'])
        self.get_dense_prediction = get_dense_prediction
        self.interpolate_prediction = interpolate_prediction
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.__hash = "id.run"
        self.min_samples_split = min_samples_split

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        _validate_y(y)
        y_copy = y.copy()
        y_copy.columns = ["troncature", "age_mort", "mort"]
        self._send_to_r_space(X, y_copy)
        r_cmd = open(path + "/base_script_n.R").read()
        r_cmd = r_cmd % self.__param_r_setter()
        r_session(r_cmd)
        self._id_run = str(self._get_from_r_space(["id.run"])["id.run"][0])
        self.results_ = r_session("result.ltrc.tree")
        self.__get_feature_importances(X.columns.to_list())
        ro.r("gc()")

    def __get_feature_importances(self, features: iter):
        try:
            importance = pd.Series(
                list(ro.r("var.importance")),
                index=list(ro.r("names(var.importance)"))
            )
            importance = importance.reindex(features).fillna(0)
            self.feature_importances_ = [importance.loc[i] for i in
                                         importance.index]
            self.feature_importances_ = [c / sum(self.feature_importances_)
                                         for c in self.feature_importances_]
        except TypeError:
            self.feature_importances_ = [np.nan for _ in features]

    def __param_r_setter(self):
        param = "xval=2, "
        if self.min_samples_leaf is not None:
            param += "minbucket=%s, " % self.min_samples_leaf
        if self.min_impurity_decrease is not None:
            param += "cp=%s, " % self.min_impurity_decrease
        if self.max_depth is not None:
            param += "maxdepth=%s, " % self.max_depth
        if self.min_samples_split is not None:
            param += "minsplit=%s, " % self.min_samples_split
        if param == "":
            return ""
        else:
            return "control = rpart::rpart.control({param})".format(
                param=param[:-2])

    def __get_prediction_data(self, X: pd.DataFrame):
        self._send_to_r_space(X)
        ro.globalenv["result.ltrc.tree"] = self.results_
        r_cmd_str = open(path + "/predict_n.R").read().replace(
            "id.run", "'%s'" % self._id_run
        )
        r_session(r_cmd_str)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.__get_prediction_data(X)
        km_mat = pd.DataFrame(
            columns=list(np.array(r_session("time.stamp"))),
            index=X.index, dtype="float16")

        for k in list(r_session("Keys")):
            subset = np.where(np.array(r_session("data.X$ID")) == k)[0]

            curves = list(r_session(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = list(r_session(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            km_mat.loc[km_mat.index[subset], np.array(time)] = curves
        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float16", np.nan))
        if self.interpolate_prediction:
            km_mat[0] = 1
            km_mat = km_mat[np.sort(km_mat.columns)]
            km_mat = km_mat.astype("float32").T.fillna(method="pad").T

        ro.r("gc()")
        return km_mat

    def predict_curves(self, X: pd.DataFrame) -> tuple:
        self.__get_prediction_data(X)
        all_times = list(np.array(r_session("time.stamp")))
        curves = pd.DataFrame(columns=all_times, index=range(len(
            list(r_session("Keys")))), dtype="float32")
        indexes = pd.Series(index=X.index, dtype="int64")
        for i, k in enumerate(list(r_session("Keys"))):
            subset = np.where(np.array(r_session("data.X$ID")) == k)[0]
            curve = np.array(r_session(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = np.array(r_session(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            curve = pd.Series(curve, index=time, dtype="float32").reindex(
                all_times)
            curves.loc[i] = curve.fillna(method="pad")
            indexes.iloc[subset] = i
        return curves, indexes


class RandomForestLTRC(ClassifierMixin):
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

        min_samples_split: float, default=2

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
                 max_samples: float = 1.,
                 min_samples_leaf: int = None,
                 min_impurity_decrease: float = 0.01,
                 min_samples_split: int = 2,
                 base_estimator: LTRCTrees = None,
                 ):
        self.__select_feature = {}
        self.bootstrap = bootstrap
        self.n_estimator = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        if base_estimator is None:
            self.base_estimator_ = LTRCTrees(
                interpolate_prediction=False,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_samples_split=self.min_samples_split
            )
        else:
            self.base_estimator_ = base_estimator

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """

        Parameters
        ----------
        X: pandas DataFrame
            Input data
        y: pandas DataFrame
            Target. For survival analysis in presence of right censored and left
            truncated data, the target must have three columns

            - first column : the truncation or entry points. The date at which
              data start to be seen (numeric)
            - second column : the age of death or truncation (numeric)
            - third column : a boolean stating id the event occurred (boolean)

        Returns
        -------

        """
        _validate_y(y)
        self.__hashes = {}
        self.results_ = {}
        self.__var_imp = pd.DataFrame(index=range(self.n_estimator),
                                      columns=X.columns)
        self.__max_feature(X.columns.to_list())
        for e in range(self.n_estimator):
            x_train, y_train = self.__bootstrap(X, y)
            self.__select_feature[e] = np.random.choice(
                X.columns, size=int(self.__m_features),
                replace=False)
            x_train = x_train.loc[:, self.__select_feature[e]]
            self.base_estimator_.fit(x_train, y_train)
            self.results_[e] = self.base_estimator_.results_
            self.__hashes[e] = self.base_estimator_._get_from_r_space([
                "id.run"])["id.run"][0]

            self.__var_imp.loc[
                e, self.__select_feature[e]
            ] = self.base_estimator_.feature_importances_
        self.feature_importances_ = list(self.__var_imp.mean(axis=0).values)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_list()

    def __max_feature(self, features: iter):

        if isinstance(self.max_features, float):
            self.__m_features = int(
                np.round(self.max_features * len(features), 0))
        elif self.max_features is None:
            self.__m_features = len(features)

        elif self.max_features == "auto":
            self.__m_features = len(features) // 3
        else:
            self.__m_features = self.max_features

        self.__m_features = max(2, self.__m_features)
        self.__m_features = min(len(features), self.__m_features)

    def __bootstrap(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.bootstrap:
            select_index = np.random.choice(X.index, size=int(
                self.max_samples * X.shape[0]))
            x_train, y_train = X.loc[select_index], y.loc[
                select_index]
            return x_train, y_train
        return X, y

    def predict(self, X: pd.DataFrame, return_type="dense"
                ) -> Union[pd.DataFrame, None]:
        self.fast_predict_(X)
        if return_type == "dense":
            return pd.merge(self.nodes_, self.km_estimates_,
                            left_on="curve_index",
                            right_index=True).set_index("x_index").drop(
                columns=["curve_index"]).loc[X.index]
        raise ValueError(f"return_type : {return_type} is not "
                         f"implemented yet")

    def fast_predict_(self, X: pd.DataFrame) -> None:
        result = {}
        for e in range(self.n_estimator):
            x_predict = X.loc[:, self.__select_feature[e]]
            self.base_estimator_.results_ = self.results_[e]
            self.base_estimator_._id_run = self.__hashes[e]
            result[e] = self.base_estimator_.predict_curves(x_predict)
        self.km_estimates_, self.nodes_ = self.__post_processing(result, X)

    @staticmethod
    def __post_processing(result, X):
        from ._common import forest_post_processing
        return forest_post_processing(result, X)


def _validate_y(y: pd.DataFrame):
    from ._common import validate_y
    return validate_y(y)
