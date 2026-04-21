mod forest;
mod km;
mod split;
mod tree;
mod types;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::forest::{
    aggregated_feature_importances, fit_forest, predict_forest, ForestFitParams,
};
use crate::tree::{feature_importances, fit_tree, leaf_km_curves, predict_leaf_ids};
use crate::types::{Control, Dataset, Forest, Sample, Tree};

fn control_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Control> {
    let mut ctrl = Control::default();
    if let Some(v) = dict.get_item("max_depth")? {
        if !v.is_none() {
            ctrl.max_depth = Some(v.extract::<usize>()?);
        }
    }
    if let Some(v) = dict.get_item("min_samples_leaf")? {
        if !v.is_none() {
            ctrl.min_samples_leaf = v.extract::<usize>()?.max(1);
        }
    }
    if let Some(v) = dict.get_item("min_samples_split")? {
        if !v.is_none() {
            ctrl.min_samples_split = v.extract::<usize>()?.max(2);
        }
    }
    if let Some(v) = dict.get_item("min_impurity_decrease")? {
        if !v.is_none() {
            ctrl.min_impurity_decrease = v.extract::<f64>()?;
        }
    }
    Ok(ctrl)
}

fn build_dataset(
    x: &PyReadonlyArray2<f64>,
    entry: &PyReadonlyArray1<f64>,
    time: &PyReadonlyArray1<f64>,
    event: &PyReadonlyArray1<bool>,
) -> PyResult<Dataset> {
    let x_view = x.as_array();
    let entry_view = entry.as_array();
    let time_view = time.as_array();
    let event_view = event.as_array();
    let n = x_view.nrows();
    if entry_view.len() != n || time_view.len() != n || event_view.len() != n {
        return Err(PyValueError::new_err(
            "x, entry, time, event must have matching row counts",
        ));
    }
    if n == 0 {
        return Err(PyValueError::new_err("empty dataset"));
    }
    if !event_view.iter().any(|&b| b) {
        return Err(PyValueError::new_err("no event observed"));
    }
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        samples.push(Sample {
            entry: entry_view[i],
            time: time_view[i],
            event: event_view[i],
        });
    }
    Ok(Dataset {
        x: x_view.to_owned(),
        samples,
    })
}

/// Pack KM curves into dense (n_leaves, n_unique_times) matrix + time vector.
/// Missing values after the last observed time are forward-filled with the
/// last observed survival; times before the first event have survival 1.
fn curves_to_dense(curves: &[crate::types::KmCurve]) -> (Array2<f64>, Vec<f64>) {
    let mut unique_times: Vec<f64> = curves.iter().flat_map(|c| c.times.iter().copied()).collect();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup_by(|a, b| a == b);

    let n_leaves = curves.len();
    let n_times = unique_times.len();
    let mut out = Array2::<f64>::ones((n_leaves, n_times));
    for (leaf_i, curve) in curves.iter().enumerate() {
        let mut last = 1.0_f64;
        let mut src = 0usize;
        for (j, &t) in unique_times.iter().enumerate() {
            while src < curve.times.len() && curve.times[src] <= t {
                last = curve.surv[src];
                src += 1;
            }
            out[[leaf_i, j]] = last;
        }
    }
    (out, unique_times)
}

#[pyclass]
struct _RustLtrcTree {
    tree: Option<Tree>,
    n_features: usize,
}

#[pymethods]
impl _RustLtrcTree {
    #[new]
    fn new() -> Self {
        Self { tree: None, n_features: 0 }
    }

    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        entry: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<bool>,
        control: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let ctrl = control_from_dict(control)?;
        let ds = build_dataset(&x, &entry, &time, &event)?;
        self.n_features = ds.n_features();
        self.tree = Some(fit_tree(&ds, &ctrl));
        Ok(())
    }

    fn predict_curves<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
    {
        let tree = self.tree.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        let x_view = x.as_array();
        if x_view.ncols() != self.n_features {
            return Err(PyValueError::new_err(format!(
                "expected {} features, got {}",
                self.n_features,
                x_view.ncols()
            )));
        }
        let leaf_ids = predict_leaf_ids(tree, &x_view.to_owned());
        let curves = leaf_km_curves(tree);
        let (dense, times) = curves_to_dense(&curves);
        let leaf_ids_i64: Vec<i64> = leaf_ids.iter().map(|&v| v as i64).collect();
        Ok((
            dense.into_pyarray_bound(py),
            leaf_ids_i64.into_pyarray_bound(py),
            times.into_pyarray_bound(py),
        ))
    }

    fn feature_importances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let tree = self.tree.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        Ok(feature_importances(tree).into_pyarray_bound(py))
    }
}

#[pyclass]
struct _RustLtrcForest {
    forest: Option<Forest>,
    n_features: usize,
}

#[pymethods]
impl _RustLtrcForest {
    #[new]
    fn new() -> Self {
        Self { forest: None, n_features: 0 }
    }

    #[pyo3(signature = (
        x, entry, time, event, control,
        n_trees, max_samples, max_features, seed, n_jobs,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        entry: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<bool>,
        control: &Bound<'_, PyDict>,
        n_trees: usize,
        max_samples: f64,
        max_features: usize,
        seed: u64,
        n_jobs: usize,
    ) -> PyResult<()> {
        let ctrl = control_from_dict(control)?;
        let ds = build_dataset(&x, &entry, &time, &event)?;
        self.n_features = ds.n_features();
        let params = ForestFitParams {
            control: &ctrl,
            n_trees,
            max_samples,
            max_features,
            seed,
            n_jobs,
        };
        self.forest = Some(fit_forest(&ds, &params));
        Ok(())
    }

    /// Returns a list of (dense_curves [n_leaves, n_times], leaf_ids [n_samples],
    /// times [n_times]) tuples, one per tree.
    fn predict_forest<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>>
    {
        let forest = self.forest.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        let x_owned = x.as_array().to_owned();
        let preds = predict_forest(forest, &x_owned);
        let mut out = Vec::with_capacity(preds.len());
        for pred in preds {
            let (dense, times) = curves_to_dense(&pred.curves);
            let leaf_ids_i64: Vec<i64> = pred.leaf_ids.iter().map(|&v| v as i64).collect();
            out.push((
                dense.into_pyarray_bound(py),
                leaf_ids_i64.into_pyarray_bound(py),
                times.into_pyarray_bound(py),
            ));
        }
        Ok(out)
    }

    fn feature_importances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let forest = self.forest.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        Ok(aggregated_feature_importances(forest).into_pyarray_bound(py))
    }

    fn feature_subsets(&self) -> PyResult<Vec<Vec<usize>>> {
        let forest = self.forest.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        Ok(forest.feature_subsets.clone())
    }
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_RustLtrcTree>()?;
    m.add_class::<_RustLtrcForest>()?;
    Ok(())
}
