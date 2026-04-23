# LTRC Survival Forest

<img src="https://img.shields.io/github/languages/code-size/eurobios-scb/survival-trees" alt="Alternative text" />

### Install notice

To install the package you can run

```shell
python -m pip install git+https://eurobios-mews-labs/survival-trees.git
```


### Usage

```python
import numpy as np
from survival_trees import RandomForestLTRCFitter
from survival_trees.metric import time_dependent_auc
from lifelines import datasets
from sklearn.model_selection import train_test_split

# load dataset
data = datasets.load_larynx().dropna()
data["entry_date"] = data["age"]
data["time"] += data["entry_date"]
y = data[["entry_date", "time", "death"]]
X = data.drop(columns=y.columns.tolist())

# split dataset    
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7)

# initialise and fit model    
model = RandomForestLTRCFitter(
    n_estimators=30,
    min_impurity_decrease=0.0000001,
    min_samples_leaf=3,
    max_samples=0.89)
model.fit(
    data.loc[x_train.index],
    entry_col="entry_date",
    duration_col="time",
    event_col='death'
)


survival_function = - np.log(model.predict_cumulative_hazard(
                    x_test).astype(float)).T

auc_cd = time_dependent_auc(
    - survival_function, 
    event_observed=y_test.loc[survival_function.index].iloc[:, 2],
    censoring_time=y_test.loc[survival_function.index].iloc[:, 1])

```


## Benchmark



<!-- BENCH:START -->
<!-- auto-generated from ./public/benchmark_cindex.md + ./public/benchmark.md by benchmark.py -->

### Harrell c-index (mean ± std over 5 seeds)

| dataset | ltrc-forest | cox-semi-parametric | aft-log-logistic |
|---|---|---|---|
| Larynx Cancer | **0.82** ± 0.03 | 0.65 ± 0.05 | 0.11 ± 0.03 |
| Lung Cancer | **0.88** ± 0.03 | 0.57 ± 0.06 | 0.50 ± 0.00 |
| Breast Cancer | 0.86 ± 0.01 | **0.96** ± 0.01 | 0.08 ± 0.01 |
| Dictatorship & Democracy | **0.60** ± 0.01 | 0.54 ± 0.02 | 0.53 ± 0.02 |
| Convicts | 0.60 ± 0.05 | 0.60 ± 0.09 | **0.60** ± 0.10 |
| Synthetic data | **0.64** ± 0.03 | 0.54 ± 0.03 | 0.53 ± 0.03 |

### Time-dependent AUC (Harrell, mean ± std over 5 seeds)

| dataset | ltrc-forest | cox-semi-parametric | aft-log-logistic |
|---|---|---|---|
| Larynx Cancer | 0.44 ± 0.04 | **0.60** ± 0.08 | 0.50 ± 0.00 |
| Lung Cancer | 0.43 ± 0.07 | **0.61** ± 0.04 | 0.50 ± 0.00 |
| Breast Cancer | **0.55** ± 0.03 | 0.51 ± 0.03 | 0.50 ± 0.00 |
| Dictatorship & Democracy | **0.75** ± 0.02 | 0.54 ± 0.06 | 0.59 ± 0.04 |
| Convicts | 0.62 ± 0.05 | **0.65** ± 0.02 | 0.65 ± 0.02 |
| Synthetic data | **0.63** ± 0.02 | 0.51 ± 0.02 | 0.51 ± 0.04 |

<!-- BENCH:END -->

## References

* https://academic.oup.com/biostatistics/article/18/2/352/2739324

## Requirements

Since v0.1.0 the package ships a native Rust backend (via
[maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs)). No R
toolchain is required. Prebuilt Linux x86_64 wheels target Python 3.9+
via the stable `abi3` ABI.

Installing from source requires a Rust stable toolchain; see `build.sh`.

## Project

This implementation come from an SNCF DTIPG project and is developped and maintained by Eurobios Scientific Computation
Branch and SNCF IR

<img src="https://www.sncf.com/themes/contrib/sncf_theme/images/logo-sncf.svg?v=3102549095" alt="drawing" width="100"/>

<img src="https://www.mews-partners.com/wp-content/uploads/2021/09/Eurobios-Mews-Labs-logo-768x274.png.webp" alt="drawing" width="175"/>

## Authors

- Vincent LAURENT 
