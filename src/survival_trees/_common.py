import numpy as np
import pandas as pd


def validate_y(y: pd.DataFrame) -> None:
    from pandas.api.types import is_numeric_dtype
    assert y.shape[1] == 3, \
        "y must have 3 columns: truncation, age of death, death"
    assert is_numeric_dtype(y[y.columns[0]]), "truncation column must be numeric"
    assert is_numeric_dtype(y[y.columns[1]]), "age-of-death column must be numeric"
    assert len(np.unique(y[y.columns[2]])) == 2, "death column must be boolean"


def forest_post_processing(result, X):
    """Aggregate per-tree ``(curves, node-index)`` predictions into a
    ``(unique_curves, nodes)`` pair."""
    nodes = pd.DataFrame(False, index=X.index, columns=result.keys())
    all_times = [result[e][0].columns for e in result]
    all_times = np.unique(np.sort(np.concatenate(all_times)))

    for e in result:
        data = result[e][1]
        nodes.loc[data.index, e] = data.values

    unique_nodes = nodes.drop_duplicates()
    unique_curves = pd.DataFrame(
        0, columns=all_times, index=unique_nodes.index,
        dtype="float32")
    unique_curves = unique_curves.loc[
                    :, ~unique_curves.columns.duplicated()]
    unique_curves_mask = pd.DataFrame(
        0, columns=unique_curves.columns,
        index=unique_nodes.index, dtype="Int8")

    for c in unique_nodes.columns:
        data = result[c][0].loc[unique_nodes[c]]
        data = data.T.reindex(unique_curves.columns).ffill().bfill().T
        unique_curves += data.to_numpy()
        unique_curves_mask[data.columns] += (~data.isna()).astype(int).to_numpy()

    unique_curves /= unique_curves_mask
    nodes = nodes.reset_index()
    unique_nodes = unique_nodes.reset_index()
    on = list(result.keys())
    nodes = pd.merge(nodes, unique_nodes,
                     on=on).drop(columns=on)
    nodes.columns = ["x_index", "curve_index"]
    return unique_curves, nodes
