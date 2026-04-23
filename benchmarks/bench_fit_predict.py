"""Measures fit+predict wall time for the Rust forest on synthetic LTRC
datasets of increasing size. Not run under pytest — invoke with
``python benchmarks/bench_fit_predict.py`` after ``maturin develop --release``."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def make_synth_ltrc(
    n: int, p: int = 10, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    entry = np.maximum(0.0, rng.exponential(scale=5.0, size=n))
    baseline = np.exp(0.5 * x[:, 0])
    lifespan = rng.exponential(scale=1.0 / baseline)
    time_col = entry + lifespan
    censor = entry + rng.exponential(scale=50.0, size=n)
    observed_time = np.minimum(time_col, censor)
    event = time_col <= censor
    df_x = pd.DataFrame(x, columns=[f"f{i}" for i in range(p)])
    df_y = pd.DataFrame(
        {"entry": entry, "time": observed_time, "event": event.astype(bool)}
    )
    return df_x, df_y


def bench_one(n: int) -> dict:
    from survival_trees._rust_base import _RustBackedForest
    x, y = make_synth_ltrc(n, p=10, seed=0)
    forest = _RustBackedForest(
        n_estimators=20,
        max_samples=0.8,
        min_samples_leaf=5,
        min_impurity_decrease=0.0,
        random_state=0,
        n_jobs=0,
    )
    t0 = time.perf_counter()
    forest.fit(x, y)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = forest.predict(x)
    t_predict = time.perf_counter() - t0

    return {"n": n, "fit_s": t_fit, "predict_s": t_predict}


def main():
    for n in (1_000, 10_000, 50_000):
        row = bench_one(n)
        print(
            f"n={row['n']:>6}  fit={row['fit_s']:.3f}s  "
            f"predict={row['predict_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
