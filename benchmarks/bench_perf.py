"""Benchmark the Rust LTRC forest on real datasets — fit, predict,
`predict(lazy=True)`, `.at_time`, `.at_times`, `.to_dense` — and AUC."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from survival_trees import RandomForestLTRC
from survival_trees.metric import concordance_index


def _split_xy(data: pd.DataFrame, y_cols: list[str],
              event_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[y_cols].copy()
    y[event_col] = y[event_col].astype(bool)
    y[[c for c in y_cols if c != event_col]] = (
        y[[c for c in y_cols if c != event_col]].astype(float))
    X = (data.drop(columns=y_cols)
             .select_dtypes(include=np.number).astype(float))
    return X, y


def load_datasets_no_r():
    from lifelines import datasets as ld

    out = {}

    data = ld.load_larynx()
    data["entry_date"] = data["age"]
    data["time"] += data["entry_date"]
    out["larynx"] = _split_xy(data, ["entry_date", "time", "death"], "death")

    data = ld.load_lung().dropna()
    data["entry_date"] = data["age"] * 365.25
    data["time"] += data["entry_date"]
    out["lung"] = _split_xy(data, ["entry_date", "time", "status"], "status")

    data = ld.load_gbsg2().dropna()
    data["death"] = 1 - data["cens"]
    data = data.drop(columns="cens", axis=1)
    data["entry_date"] = data["age"]
    data["time"] /= 365.25
    data["time"] += data["entry_date"]
    data["horTh"] = data["horTh"] == "yes"
    data["menostat"] = data["menostat"] == "Post"
    data["tgrade"] = data["tgrade"] == "III"
    out["gbsg2"] = _split_xy(data, ["entry_date", "time", "death"], "death")

    data = ld.load_dd()
    data["entry_date"] = 0.0
    out["dd"] = _split_xy(data, ["entry_date", "duration", "observed"], "observed")

    data = ld.load_rossi()
    data["entry_date"] = 0.0
    out["rossi"] = _split_xy(data, ["entry_date", "week", "arrest"], "arrest")

    import synthetic as syn
    data = pd.concat(
        (syn.X.astype(float), syn.Y.astype(bool), syn.L, syn.R), axis=1
    )
    out["synthetic"] = _split_xy(
        data, ["left_truncation", "right_censoring", "target"], "target")

    return out


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def _compute_auc(pred: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """Mean c-index over the time grid of ``pred``."""
    pred = pred.astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if pred.shape[1] == 0:
        return float("nan")
    event = y_test.iloc[:, 2].astype(bool).loc[pred.index]
    censoring = y_test.iloc[:, 1].astype(float).loc[pred.index]
    c = concordance_index(pred, event_observed=event, censoring_time=censoring)
    return float(np.nanmean(c.to_numpy(dtype=float)))


def bench_dataset(name, X, y, n_estimators=300, min_samples_leaf=10, seed=0):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=seed
    )

    forest = RandomForestLTRC(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.8,
        min_impurity_decrease=1e-7,
        random_state=seed,
        n_jobs=0,
    )

    _, t_fit = _time(forest.fit, x_train, y_train)
    pred, t_predict = _time(forest.predict, x_test)
    lazy, t_predict_lazy = _time(forest.predict, x_test, lazy=True)

    all_times = lazy._inner.union_times()
    all_times = np.asarray(all_times)
    t_median = float(all_times[all_times.size // 2]) if all_times.size else 1.0
    _, t_at = _time(lazy.at_time, t_median)

    ts10 = np.linspace(all_times.min(), all_times.max(), 10)
    _, t_at10 = _time(lazy.at_times, ts10)

    grid200 = np.linspace(all_times.min(), all_times.max(), 200)
    _, t_dense200 = _time(lazy.to_dense, grid200)

    auc = _compute_auc(pred, y_test)

    return {
        "dataset": name,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_feat": int(X.shape[1]),
        "fit": t_fit,
        "predict": t_predict,
        "predict_lazy": t_predict_lazy,
        "at_time": t_at,
        "at_times(10)": t_at10,
        "to_dense(200)": t_dense200,
        "auc": auc,
        "fit>lazy": t_fit > t_predict_lazy,
    }


def main(n_seeds: int = 5):
    warnings.simplefilter("ignore")
    datasets = load_datasets_no_r()

    all_rows = []
    for name, (X, y) in datasets.items():
        print(f"running {name} (x{n_seeds} seeds)...", flush=True)
        for seed in range(n_seeds):
            row = bench_dataset(name, X, y, seed=seed)
            row["seed"] = seed
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    timing_cols = ["fit", "predict", "predict_lazy",
                   "at_time", "at_times(10)", "to_dense(200)"]
    shape_cols = ["n_train", "n_test", "n_feat"]
    agg_time = df.groupby("dataset")[timing_cols].mean()
    agg_shape = df.groupby("dataset")[shape_cols].first()
    auc_mean = df.groupby("dataset")["auc"].mean().rename("auc_mean")
    auc_std = df.groupby("dataset")["auc"].std().rename("auc_std")

    summary = pd.concat([agg_shape, agg_time, auc_mean, auc_std], axis=1)
    order = list(datasets.keys())
    summary = summary.loc[order]

    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 160):
        print()
        print(f"--- Timings (ms, mean over {n_seeds} seeds) "
              "and AUC (c-index, mean ± std) ---")
        print(summary.to_string())
        print()
        print("--- AUC alone ---")
        print(pd.DataFrame({"auc_mean": auc_mean, "auc_std": auc_std}).to_string())


if __name__ == "__main__":
    main()
