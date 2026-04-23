import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import synthetic
from lifelines import datasets
from lifelines.fitters import coxph_fitter, log_logistic_aft_fitter
from lifelines.plotting import plot_lifetimes
from sklearn.model_selection import train_test_split

from survival_trees import (
    LTRCTrees,
    LTRCTreesFitter,
    RandomForestLTRC,
    RandomForestLTRCFitter,
    plotting,
)
from survival_trees.metric import concordance_index as harrell_concordance
from survival_trees.metric import time_dependent_auc

plot.rc('font', family='ubuntu')


def load_datasets():
    datasets_dict = {}
    # ==========================================================================
    data = datasets.load_larynx()
    data["entry_date"] = data["age"]
    data["time"] += data["entry_date"]
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())
    datasets_dict["Larynx Cancer"] = X, y
    # ==========================================================================
    data = datasets.load_lung()
    data["entry_date"] = data["age"] * 365.25
    data["time"] += data["entry_date"]
    y = data[["entry_date", "time", "status"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Lung Cancer"] = X, y
    # ==========================================================================
    data = datasets.load_gbsg2().dropna()
    data["death"] = 1 - data["cens"]
    data = data.drop(columns='cens', axis=1)
    data["entry_date"] = data["age"]
    data["time"] /= 365.25
    data["time"] += data["entry_date"]

    data["horTh"] = data["horTh"] == "yes"
    data["menostat"] = data["menostat"] == "Post"
    data["tgrade"] = data["tgrade"] == "III"
    y = data[["entry_date", "time", "death"]].copy()
    X = data.drop(columns=y.columns.tolist())
    X = X.astype(float).select_dtypes(include=np.number)
    datasets_dict["Breast Cancer"] = X, y
    # ==========================================================================
    data = datasets.load_dd()
    data["entry_date"] = 0
    y = data[["entry_date", "duration", "observed"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict[r"Dictatorship \& Democracy"] = X, y
    # ==========================================================================
    data = datasets.load_rossi()
    data["entry_date"] = 0
    y = data[["entry_date", "week", "arrest"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Convicts"] = X, y
    # ==========================================================================
    data = pd.concat((synthetic.X.astype(int), synthetic.Y, synthetic.L,
                      synthetic.R), axis=1)
    y = data[["left_truncation", "right_censoring", "target"]]
    X = data.drop(columns=y.columns.tolist())
    X = X.select_dtypes(include=np.number)
    datasets_dict["Synthetic data"] = X, y
    return datasets_dict


def benchmark(n_exp=2):
    all_datasets = load_datasets()
    models = {
        "ltrc-forest": RandomForestLTRCFitter(
            n_estimators=300,
            min_impurity_decrease=0.01,
            min_samples_leaf=5,
            max_samples=0.8),
        "ltrc-trees": LTRCTreesFitter(min_samples_leaf=5,
                                      min_impurity_decrease=0.01),
        "cox-semi-parametric": coxph_fitter.SemiParametricPHFitter(penalizer=0.1),
        "aft-log-logistic": log_logistic_aft_fitter.LogLogisticAFTFitter(penalizer=0.1),
    }
    auc_res = {}
    cidx_res = {}
    for j in range(n_exp):
        auc_res[j] = pd.DataFrame(index=all_datasets.keys(), columns=models.keys())
        cidx_res[j] = pd.DataFrame(index=all_datasets.keys(), columns=models.keys())
        for k, (X, y) in all_datasets.items():
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, random_state=j)
            for key in models:
                try:
                    models[key].fit(
                        pd.concat((x_train, y_train), axis=1).dropna(),
                        entry_col=y_train.columns[0],
                        duration_col=y_train.columns[1],
                        event_col=y_train.columns[2]
                    )
                    # Risk marker = 1 - exp(-Λ) = 1 - S ∈ [0, 1]. Monotone
                    # in Λ → invariant for AUC, finite everywhere.
                    ch = models[key].predict_cumulative_hazard(x_test).astype(float).T
                    risk = (1.0 - np.exp(-ch)).dropna()
                    event_col = y_test.loc[risk.index].iloc[:, 2]
                    time_col = y_test.loc[risk.index].iloc[:, 1]

                    auc = time_dependent_auc(risk, event_observed=event_col,
                                             censoring_time=time_col)
                    auc_res[j].loc[k, key] = np.nanmean(auc)

                    # Harrell c-index uses S(t|X) = exp(-Λ) as survival score.
                    surv = np.exp(-ch).loc[risk.index]
                    ci = harrell_concordance(surv, event_observed=event_col,
                                             censoring_time=time_col)
                    cidx_res[j].loc[k, key] = np.nanmean(ci)
                except Exception:
                    pass

    def _concat(d):
        out = pd.DataFrame()
        for run in d:
            d[run]["num_expe"] = run
            out = pd.concat((d[run].astype(float), out), axis=0)
        out.index.name = "dataset"
        return out

    auc_all = _concat(auc_res)
    cidx_all = _concat(cidx_res)
    auc_all.to_csv("benchmarks/benchmark_data_v2.csv")
    cidx_all.to_csv("benchmarks/benchmark_cindex_v2.csv")

    mean_ = auc_all.reset_index().groupby("dataset").mean().drop(columns=["num_expe"])
    std_ = auc_all.reset_index().groupby("dataset").std().drop(columns=["num_expe"])
    return mean_, std_

def test_larynx():
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())

    models = (RandomForestLTRC(max_features=2, n_estimators=30,
                               min_samples_leaf=4),
              LTRCTrees(min_samples_leaf=4))
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)

    for i, model in enumerate(models):
        model.fit(x_train, y_train)
        test = model.predict(x_test).astype(float)
        c_index = harrell_concordance(
            test, event_observed=y_test["death"],
            censoring_time=y_test["time"])
        c_index.plot()


def test_metrics():
    from importlib import reload
    reload(plotting)
    data = datasets.load_larynx()
    data["entry_date"] = 0
    y = data[["entry_date", "time", "death"]]
    X = data.drop(columns=y.columns.tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8)
    model = RandomForestLTRC(max_features=2, n_estimators=30,
                             min_samples_leaf=1,
                             cp=0.000000001)
    model.fit(x_train, y_train)
    plot.figure()
    for method in ["harrell", "roc-cd", "roc-id"]:
        test = model.predict(x_test).astype(float)
        tdr = time_dependent_auc(1 - test, event_observed=y_test["death"],
                                 censoring_time=y_test["time"],
                                 method=method)
        tdr.dropna().plot(marker=".", label=method)
    plot.legend()
    plot.savefig("benchmark/metric.png")

    plot.figure()
    plot_lifetimes(y["time"], entry=y["entry_date"], event_observed=y["death"])
    plot.savefig("benchmark/lifelines.png")

    plot.figure()

    plotting.tagged_curves(temporal_curves=test, label=y_test["death"],
                           time_event=y_test["time"],
                           add_marker=False)
    # plot.savefig("benchmark/curves.png", dpi=200)


def properties():
    data = load_datasets()
    for k, (X, y) in data.items():
        print(X.shape)




if __name__ == '__main__':
    datasets_dict = load_datasets()
    data_names = list(datasets_dict.keys())
    # mean_, _ = benchmark(n_exp=20)
    # mean_.to_csv("benchmark/benchmark.csv")


    def _as_markdown(csv_path, md_path, caption):
        df = pd.read_csv(csv_path, index_col="dataset")
        grouped = df.reset_index().groupby("dataset")
        m = grouped.mean().drop(columns=["num_expe"])
        s = grouped.std().drop(columns=["num_expe"])
        m = m.loc[data_names]
        s = s.loc[data_names]
        m.index = [e.replace(r"\&", "&") for e in m.index]
        s.index = m.index
        m.columns = [e.replace("trees", "cart") for e in m.columns]
        s.columns = m.columns
        drop = [c for c in m.columns if "cart" in c]
        m = m.drop(columns=drop, errors="ignore")
        s = s.drop(columns=drop, errors="ignore")

        winners = m.astype(float).idxmax(axis=1, skipna=True)
        cols = list(m.columns)

        lines = [f"### {caption}", ""]
        lines.append("| dataset | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for r in m.index:
            cells = [r]
            for c in cols:
                mv, sv = m.loc[r, c], s.loc[r, c]
                if pd.isna(mv):
                    cells.append("")
                    continue
                bold = (winners.get(r) == c)
                mean_s = f"**{mv:.2f}**" if bold else f"{mv:.2f}"
                cells.append(mean_s if pd.isna(sv) else f"{mean_s} ± {sv:.2f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        with open(md_path, "w") as fh:
            fh.write("\n".join(lines))

    def _splice_into_readme(readme_path, md_paths,
                            start="<!-- BENCH:START -->",
                            end="<!-- BENCH:END -->"):
        with open(readme_path) as fh:
            src = fh.read()
        i, j = src.find(start), src.find(end)
        if i < 0 or j < 0 or j < i:
            return
        body = ["", start,
                "<!-- auto-generated from " +
                " + ".join(md_paths) +
                " by benchmark.py -->", ""]
        for p in md_paths:
            with open(p) as fh:
                body.append(fh.read().rstrip() + "\n")
        body.append(end)
        new = src[:i] + "\n".join(body) + src[j + len(end):]
        with open(readme_path, "w") as fh:
            fh.write(new)

    _as_markdown("benchmarks/benchmark_data_v2.csv",
                 "./public/benchmark.md",
                 "Time-dependent AUC (Harrell, mean ± std over 5 seeds)")
    _as_markdown("benchmarks/benchmark_cindex_v2.csv",
                 "./public/benchmark_cindex.md",
                 "Harrell c-index (mean ± std over 5 seeds)")

    _splice_into_readme("./README.md",
                        ["./public/benchmark_cindex.md",
                         "./public/benchmark.md"])


    for k, data_ in datasets_dict.items():
        X, y = data_
        y.columns = ["entry_date", "time", "death"]
        pd.concat((X, y), axis=1).iloc[:600].to_csv(
            f"benchmarks/data/{k}.txt", index=False)
