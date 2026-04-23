"""
https://medium.com/analytics-vidhya/concordance-index-72298c11eac7
https://anaqol.org/ea4275/js2012/3-Combescure.pdf?PHPSESSID=9fe65030a58af33d1370d7ccff343fa4
"""
from functools import wraps

import numpy as np
import pandas as pd
from lifelines import utils
from sklearn import metrics as sk_metrics


# life lines signature
# event_times, predicted_scores, event_observed


def __validate(matrix: pd.DataFrame, vector: iter):
    shape_m = matrix.shape
    shape_v = len(vector)
    if shape_m[0] != shape_v:
        raise ValueError(f"Shape of input matrix must be n_observation ({shape_m[0]}) x "
                         f"number of timestamp ({shape_m[1]}) \n"
                         f"Number of observation in vector ({shape_v})")


def __validate_decorator(function):
    wraps(function)

    def wrapper(temporal_score: pd.DataFrame, event_observed: iter,
                censoring_time: iter, *args, **kwargs):
        __validate(temporal_score, event_observed)
        __validate(temporal_score, censoring_time)
        result = function(temporal_score, event_observed,
                          censoring_time, *args, **kwargs)
        return result

    return wrapper


@__validate_decorator
def concordance_index(temporal_score: pd.DataFrame, event_observed: iter,
                      censoring_time: iter):
    c_index = pd.Series(index=temporal_score.columns)
    for date in c_index.index:
        try:
            c_index.loc[date] = utils.concordance_index(
                censoring_time,
                temporal_score[date], event_observed & (date >= censoring_time))
        except ZeroDivisionError:
            pass

    return c_index


@__validate_decorator
def mean_concordance_index(temporal_score: pd.DataFrame, death: iter,
                           censoring_time: iter):
    return concordance_index(temporal_score, death,
                             censoring_time).mean()


@__validate_decorator
def time_dependent_helper(
        temporal_score: pd.DataFrame,
        event_observed: iter,
        censoring_time: iter,
        function: callable,
        method: str = "harrell",

):
    result = {}

    if method == "harrell":
        def outcome(_):
            return event_observed

        def marker(t_):
            return temporal_score[t_]

    elif method == "roc-cd":
        def outcome(t_):
            return (censoring_time <= t_) & event_observed

        def marker(t_):
            return temporal_score[t_]

    elif method == "roc-id":
        def outcome(t_):
            out = np.where(censoring_time < t_, np.nan, censoring_time)
            return (t_ == out) & event_observed

        def marker(t_):
            return temporal_score[t_]
    else:
        raise ValueError(f"method : {method} is not known")

    for t in temporal_score.columns:
        try:
            y_true = np.array(outcome(t))
            y_score = np.array(marker(t))

            nan_ = np.isnan(y_true)
            nan_ |= np.isnan(y_score)

            result[t] = function(
                y_true=y_true[~nan_],
                y_score=y_score[~nan_]
            )
        except ValueError:
            pass
    return result


@__validate_decorator
def time_dependent_auc(
        temporal_score: pd.DataFrame,
        event_observed: iter,
        censoring_time: iter,
        method="harrell"
) -> pd.Series:
    """
    method
        - roc-cd : Cumulative sensitivity and dynamic specificity (C/D)
        - roc-id : Incident sensitivity and dynamic specificity (I/D)
    """
    result = time_dependent_helper(
        temporal_score,
        event_observed,
        censoring_time,
        sk_metrics.roc_auc_score,
        method)
    return pd.Series(result, name=method)
