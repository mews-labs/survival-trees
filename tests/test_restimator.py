import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

from survival_trees import LTRCTrees, RandomForestLTRC, RandomForestSRC


@pytest.fixture(scope="module")
def get_data():
    from sklearn.model_selection import train_test_split
    n = 300
    y_cols = ["start_year", "age", "observed"]
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"x1": rng.uniform(size=n), "x2": rng.uniform(size=n)},
        index=range(n),
    )
    y = pd.DataFrame(columns=y_cols, index=X.index)
    y["age"] = X["x1"] * 10 + rng.uniform(size=n)
    y["start_year"] = 0
    y["observed"] = (y["age"] + rng.uniform(size=n) + X["x2"]) > 6
    y["observed"] = y["observed"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return x_train, x_test, y_train, y_test


def test_random_forest_src_deprecated(get_data):
    x_train, x_test, _, y_test = get_data
    _, _, y_train, _ = get_data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rf_src = RandomForestSRC(n_estimator=10)
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), "RandomForestSRC must raise DeprecationWarning"
    rf_src.fit(x_train, y_train.drop(columns=["start_year"]))
    pred = rf_src.predict(x_test)
    assert sum(pred.index != x_test.index) == 0
    assert len(rf_src.feature_importances_) == x_train.shape[1]
    values = pred.to_numpy(dtype=float)
    mask = np.isfinite(values)
    assert mask.any()
    assert (values[mask] >= -1e-6).all() and (values[mask] <= 1.0 + 1e-6).all()


def test_ltrc_trees(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = LTRCTrees(min_samples_leaf=1, min_impurity_decrease=0.001)
    y_save = y_train.__deepcopy__()
    est.fit(x_train, y_train)
    assert sum(np.array(y_train.columns == y_save.columns)) == y_train.shape[1]


def test_ltrc_trees_n(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = LTRCTrees(
        get_dense_prediction=False,
        interpolate_prediction=True,
    )
    est.fit(x_train, y_train)
    test = est.predict(x_test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.ffill()
    c_index = pd.Series(index=test.columns, dtype=float)
    for date in c_index.index:
        try:
            c_index.loc[date] = concordance_index(
                date * np.ones(len(test)),
                test[date], y_test["observed"])
        except ValueError:
            pass
    imp = est.feature_importances_
    assert len(imp) == x_train.shape[1]
    assert c_index.mean() > 0.5


def test_ltrc_trees_predict_curves(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = LTRCTrees(min_impurity_decrease=0.000001, min_samples_leaf=1)
    est.fit(x_train, y_train)
    curves, indexes = est.predict_curves(x_test)
    assert curves.shape[0] >= 1
    assert indexes.index.equals(x_test.index)


def _assert_survival_matrix(df):
    arr = df.to_numpy(dtype=float)
    assert np.isfinite(arr).all(), "non-finite survival value"
    assert (arr >= -1e-6).all() and (arr <= 1.0 + 1e-6).all()
    diffs = np.diff(arr, axis=1)
    assert (diffs <= 1e-6).all(), "non-monotone survival curve"


def _mean_c_index(test: pd.DataFrame, y_observed: pd.Series) -> float:
    c_index = pd.Series(index=test.columns, dtype=float)
    for date in c_index.index:
        try:
            c_index.loc[date] = concordance_index(
                date * np.ones(len(test)),
                test[date], y_observed)
        except Exception:
            pass
    return c_index.mean()


def test_rf_ltrc(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = RandomForestLTRC(n_estimators=3, min_impurity_decrease=0.001, random_state=0)
    est.fit(x_train, y_train)
    test = est.predict(x_test)
    _assert_survival_matrix(test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.ffill()
    c_mean = _mean_c_index(test, y_test["observed"])
    assert abs(c_mean - 0.5) > 0.1, f"forest predictions not informative; c={c_mean}"


def test_rf_ltrc_fast(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = RandomForestLTRC(
        n_estimators=20,
        max_features=2,
        bootstrap=True,
        max_samples=0.5,
        min_samples_split=3,
        random_state=0,
    )
    est.fit(x_train, y_train)
    test = est.predict(x_test)
    _assert_survival_matrix(test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.ffill()
    c_mean = _mean_c_index(test, y_test["observed"])
    assert abs(c_mean - 0.5) > 0.1, f"forest predictions not informative; c={c_mean}"


def _eval_survival_at(pred: pd.DataFrame, t: float) -> np.ndarray:
    cols = np.asarray(pred.columns, dtype=float)
    idx = np.searchsorted(cols, t, side="right") - 1
    idx = max(idx, 0)
    return pred.iloc[:, idx].to_numpy(dtype=float)


def _make_two_group_data(n: int, seed: int, lam_a: float, lam_b: float):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    lam = np.where(x1 < 0.5, lam_a, lam_b)
    t_event = rng.exponential(scale=1.0 / lam)
    # light independent censoring → KM unbiased, satisfies 2-unique-values validator
    t_cens = rng.exponential(scale=10.0, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.DataFrame({"start": np.zeros(n), "age": time, "observed": event})
    return X, y


def test_two_group_exponential_recovery():
    """Two groups with known exponential S(t); KM leaves must recover it."""
    lam_a, lam_b = 0.5, 2.0
    X, y = _make_two_group_data(n=4000, seed=42, lam_a=lam_a, lam_b=lam_b)

    est = LTRCTrees(max_depth=1, min_samples_leaf=100, min_impurity_decrease=0.0)
    est.fit(X, y)

    X_test = pd.DataFrame({"x1": [0.25, 0.75], "x2": [0.5, 0.5]})
    pred = est.predict(X_test)

    for t in (0.5, 1.0, 1.5, 2.0):
        s_hat = _eval_survival_at(pred, t)
        expected = np.array([np.exp(-lam_a * t), np.exp(-lam_b * t)])
        assert np.allclose(s_hat, expected, atol=0.02), (
            f"t={t}: got {s_hat}, expected {expected}")


def test_two_group_exponential_recovery_forest():
    """Same ground truth, recovered by a small forest via aggregated predict."""
    lam_a, lam_b = 0.5, 2.0
    X, y = _make_two_group_data(n=4000, seed=7, lam_a=lam_a, lam_b=lam_b)

    est = RandomForestLTRC(
        n_estimators=20,
        max_features=2,
        min_samples_leaf=100,
        min_impurity_decrease=0.0,
        random_state=0,
    )
    est.fit(X, y)

    X_test = pd.DataFrame({"x1": [0.25, 0.75], "x2": [0.5, 0.5]})
    pred = est.predict(X_test)

    for t in (0.5, 1.0, 1.5, 2.0):
        s_hat = _eval_survival_at(pred, t)
        expected = np.array([np.exp(-lam_a * t), np.exp(-lam_b * t)])
        assert np.allclose(s_hat, expected, atol=0.05), (
            f"t={t}: got {s_hat}, expected {expected}")


def test_rf_ltrc_fast_max_features(get_data):
    x_train, x_test, y_train, y_test = get_data
    est = RandomForestLTRC(
        n_estimators=5,
        max_features=0.5,
        bootstrap=True,
        max_samples=0.5,
        min_impurity_decrease=0.001,
        random_state=0,
    )
    est.fit(x_train, y_train)
    test = est.predict(x_test)
    _assert_survival_matrix(test)
    test[0] = 1
    test = test[np.sort(test.columns)]
    test = test.ffill()
    c_mean = _mean_c_index(test, y_test["observed"])
    assert abs(c_mean - 0.5) > 0.1, f"forest predictions not informative; c={c_mean}"
