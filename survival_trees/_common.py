"""Shared helpers between the legacy R backend (`_base.py`) and the new Rust
backend (`_rust_base.py`). Logic is lifted verbatim from `_base.py` so any
consumer — R or Rust — produces the same post-processed output."""

import numpy as np
import pandas as pd


def validate_y(y: pd.DataFrame) -> None:
    from pandas.api.types import is_numeric_dtype
    assert y.shape[1] == 3, "Target data must be a data frame" \
                            "with 3 columns : truncation, age of death " \
                            "and death"
    assert is_numeric_dtype(y[y.columns[0]]), "The first column of target" \
                                              " (truncation) should be " \
                                              " numeric"
    assert is_numeric_dtype(y[y.columns[1]]), "The second column of" \
                                              " target (age of death) " \
                                              "should be numeric"
    assert len(np.unique(y[y.columns[2]])) == 2, "The third column of " \
                                                 "target (death) " \
                                                 "should be boolean"


def forest_post_processing(result, X):
    """Aggregate per-tree (curves, node-index) predictions into a
    (unique_curves, nodes) pair, unchanged from the original R-based
    `RandomForestLTRC.__post_processing`."""
    nodes = pd.DataFrame(False, index=X.index, columns=result.keys())
    all_times = [result[e][0].columns for e in result.keys()]
    all_times = np.unique(np.sort(np.concatenate(all_times)))

    for e in result.keys():
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
        data = data.T.reindex(
            unique_curves.columns).fillna(
            method="ffill").fillna(
            method="bfill").T
        unique_curves += data.values
        unique_curves_mask[data.columns] += (~data.isna()
                                             ).astype(int).values

    unique_curves /= unique_curves_mask
    nodes = nodes.reset_index()
    unique_nodes = unique_nodes.reset_index()
    on = list(result.keys())
    nodes = pd.merge(nodes, unique_nodes,
                     on=on).drop(columns=on)
    nodes.columns = ["x_index", "curve_index"]
    return unique_curves, nodes
