import numpy as np
import pandas as pd

_DEFAULT_DATE_INITIAL = pd.to_datetime("2000")
_DEFAULT_DATE_FINAL = pd.to_datetime("2050")


def shift_origin(
        data: pd.DataFrame, starting_dates: pd.Series, period,
        date_final=_DEFAULT_DATE_FINAL,
        date_initial=_DEFAULT_DATE_INITIAL, dtypes: str = "float16",
):
    """Shift the time origin of a temporal DataFrame whose columns are
    datetime, so each row is re-aligned on its own ``starting_dates``."""
    range_ = np.arange(
        starting_dates.min(),
        date_final + period,
        step=period)
    data = data[data.columns.sort_values()]
    s_origin = pd.DataFrame(
        index=data.index, columns=range_, dtype=dtypes)
    columns = s_origin.columns
    for d in starting_dates.dropna().unique():
        loc = starting_dates.index[starting_dates == d]
        i_d = np.argmin(np.abs((s_origin.columns - d).days))
        if i_d > -1:
            i_f = i_d + len(data.columns) - 1
            iter_columns = columns[i_d:i_f]
            ncols = len(iter_columns)
            s_origin.loc[loc, iter_columns] = data.loc[loc].values[:, :ncols]
    i_f, i_i = [range_[np.searchsorted(s_origin.columns, d)]
                for d in (date_final, date_initial)]
    res = s_origin.loc[:, i_i:i_f]
    del s_origin
    return res
