mod forest;
mod km;
mod split;
mod tree;
mod types;

use std::cmp::Ordering;

use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::forest::{
    aggregated_feature_importances, fit_forest, predict_forest,
    predict_forest_aggregated, project_features, ForestFitParams,
};
use crate::tree::{feature_importances, fit_tree, leaf_km_curves, predict_leaf_ids};
use crate::types::{Control, Dataset, Forest, KmCurve, Pipeline, Sample, SplitCriterion,
                    SplitterMode, Tree};

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
    if let Some(v) = dict.get_item("criterion")? {
        if !v.is_none() {
            let name: String = v.extract()?;
            ctrl.criterion = match name.as_str() {
                "log-rank" | "logrank" => SplitCriterion::LogRank,
                "poisson-exp" | "poisson" | "exp" => SplitCriterion::PoissonExp,
                other => return Err(PyValueError::new_err(format!(
                    "unknown criterion '{other}'; expected 'log-rank' or 'poisson-exp'"
                ))),
            };
        }
    }
    if let Some(v) = dict.get_item("pipeline")? {
        if !v.is_none() {
            let name: String = v.extract()?;
            ctrl.pipeline = match name.as_str() {
                "aalen" | "nelson-aalen" | "na" => Pipeline::Aalen,
                "km" | "kaplan-meier" => Pipeline::Km,
                other => return Err(PyValueError::new_err(format!(
                    "unknown pipeline '{other}'; expected 'aalen' or 'km'"
                ))),
            };
        }
    }
    if let Some(v) = dict.get_item("splitter")? {
        if !v.is_none() {
            let name: String = v.extract()?;
            ctrl.splitter = match name.as_str() {
                "best" => SplitterMode::Best,
                "random" | "extratrees" | "extra" => SplitterMode::Random,
                other => return Err(PyValueError::new_err(format!(
                    "unknown splitter '{other}'; expected 'best' or 'random'"
                ))),
            };
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
    for i in 0..n {
        if !entry_view[i].is_finite() {
            return Err(PyValueError::new_err(format!(
                "entry[{i}] is not finite ({}); entry times must be finite floats",
                entry_view[i]
            )));
        }
        if !time_view[i].is_finite() {
            return Err(PyValueError::new_err(format!(
                "time[{i}] is not finite ({}); event times must be finite floats",
                time_view[i]
            )));
        }
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

/// Pack KM curves into a dense `(n_leaves, n_unique_times)` matrix + time
/// vector. Before the first event: S=1. After the last observed time:
/// forward-fill with the last S.
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

    #[pyo3(signature = (x, entry, time, event, control, seed=0))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        entry: PyReadonlyArray1<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<bool>,
        control: &Bound<'_, PyDict>,
        seed: u64,
    ) -> PyResult<()> {
        let ctrl = control_from_dict(control)?;
        let ds = build_dataset(&x, &entry, &time, &event)?;
        self.n_features = ds.n_features();
        self.tree = Some(fit_tree(&ds, &ctrl, seed));
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

    fn predict_lazy(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<_LazyForest>> {
        let forest = self
            .forest
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit first"))?;
        let x_owned = x.as_array().to_owned();
        let n_samples = x_owned.nrows();
        let n_trees = forest.trees.len();

        let per_tree: Vec<(Vec<usize>, Vec<KmCurve>)> = py.allow_threads(|| {
            forest
                .trees
                .par_iter()
                .zip(forest.feature_subsets.par_iter())
                .map(|(tree, feats)| {
                    let sub_x = project_features(&x_owned, feats);
                    let leaves = predict_leaf_ids(tree, &sub_x);
                    let curves = leaf_km_curves(tree);
                    (leaves, curves)
                })
                .collect()
        });

        let mut leaf_ids = Array2::<i64>::zeros((n_samples, n_trees));
        let mut tree_curves: Vec<Vec<KmCurve>> = Vec::with_capacity(n_trees);
        for (e, (leaves, curves)) in per_tree.into_iter().enumerate() {
            for (i, l) in leaves.iter().enumerate() {
                leaf_ids[[i, e]] = *l as i64;
            }
            tree_curves.push(curves);
        }

        Py::new(py, _LazyForest { leaf_ids, tree_curves, pipeline: forest.pipeline })
    }

    /// Returns `(unique_curves, node_index_per_sample, times)`.
    /// Samples that share the same per-tree leaf tuple share a curve row.
    #[pyo3(signature = (x, time_grid=None))]
    fn predict_aggregated<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time_grid: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let forest = self.forest.as_ref().ok_or_else(|| PyValueError::new_err("fit first"))?;
        let x_owned = x.as_array().to_owned();
        let grid_vec: Option<Vec<f64>> = time_grid
            .as_ref()
            .map(|g| g.as_array().iter().copied().collect());
        let grid_slice: Option<&[f64]> = grid_vec.as_deref();
        let (curves, node_index, times) =
            predict_forest_aggregated(forest, &x_owned, grid_slice);
        let node_index_i64: Vec<i64> = node_index.iter().map(|&v| v as i64).collect();
        Ok((
            curves.into_pyarray_bound(py),
            node_index_i64.into_pyarray_bound(py),
            times.into_pyarray_bound(py),
        ))
    }
}

#[pyclass]
struct _LazyForest {
    leaf_ids: Array2<i64>,
    tree_curves: Vec<Vec<KmCurve>>,
    pipeline: Pipeline,
}

const S_MIN: f32 = 1e-12;

fn leaf_surv_at(curve: &KmCurve, t: f64) -> f32 {
    let idx = curve.times.partition_point(|&tk| tk <= t);
    if idx == 0 {
        1.0
    } else {
        curve.surv[idx - 1] as f32
    }
}

#[inline]
fn leaf_chf_at(curve: &KmCurve, t: f64) -> f32 {
    -leaf_surv_at(curve, t).max(S_MIN).ln()
}

#[pymethods]
impl _LazyForest {
    fn n_samples(&self) -> usize {
        self.leaf_ids.nrows()
    }

    fn n_trees(&self) -> usize {
        self.leaf_ids.ncols()
    }

    /// Aalen: S_F(t) = exp(− mean_b Λ_b(t)).
    /// Km:    S_F(t) = mean_b S_b(t).
    fn at_time<'py>(&self, py: Python<'py>, t: f64) -> Bound<'py, PyArray1<f32>> {
        let n_samples = self.leaf_ids.nrows();
        let n_trees = self.leaf_ids.ncols();
        let inv = if n_trees > 0 { 1.0_f32 / n_trees as f32 } else { 0.0 };
        let leaf_ids_view = self.leaf_ids.view();
        let tree_curves = &self.tree_curves;
        let aalen = self.pipeline == Pipeline::Aalen;

        let acc: Vec<f32> = py.allow_threads(|| {
            let per_tree: Vec<Vec<f32>> = (0..n_trees)
                .into_par_iter()
                .map(|e| {
                    tree_curves[e]
                        .iter()
                        .map(|c| if aalen { leaf_chf_at(c, t) } else { leaf_surv_at(c, t) })
                        .collect()
                })
                .collect();

            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut acc = 0.0_f32;
                    for e in 0..n_trees {
                        let l = leaf_ids_view[[i, e]] as usize;
                        acc += per_tree[e][l];
                    }
                    if aalen { (-acc * inv).exp() } else { acc * inv }
                })
                .collect()
        });

        acc.into_pyarray_bound(py)
    }

    fn at_times<'py>(
        &self,
        py: Python<'py>,
        ts: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray2<f32>> {
        let ts_vec: Vec<f64> = ts.as_array().iter().copied().collect();
        let k = ts_vec.len();
        let n_samples = self.leaf_ids.nrows();
        let n_trees = self.leaf_ids.ncols();
        let inv = if n_trees > 0 { 1.0_f32 / n_trees as f32 } else { 0.0 };
        let leaf_ids_view = self.leaf_ids.view();
        let tree_curves = &self.tree_curves;

        let aalen = self.pipeline == Pipeline::Aalen;
        let result: Array2<f32> = py.allow_threads(|| {
            // Per-tree leaf values: Λ_b(t) in Aalen, S_b(t) in Km.
            let per_tree: Vec<Array2<f32>> = (0..n_trees)
                .into_par_iter()
                .map(|e| {
                    let curves = &tree_curves[e];
                    let n_leaves = curves.len();
                    let init: f32 = if aalen { 0.0 } else { 1.0 };
                    let mut mat = Array2::<f32>::from_elem((n_leaves, k), init);
                    for (l, curve) in curves.iter().enumerate() {
                        for (j, &t) in ts_vec.iter().enumerate() {
                            let idx = curve.times.partition_point(|&tk| tk <= t);
                            let s = if idx > 0 { curve.surv[idx - 1] as f32 } else { 1.0 };
                            mat[[l, j]] = if aalen { -s.max(S_MIN).ln() } else { s };
                        }
                    }
                    mat
                })
                .collect();

            let mut result = Array2::<f32>::zeros((n_samples, k));
            result
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for e in 0..n_trees {
                        let l = leaf_ids_view[[i, e]] as usize;
                        let leaf_row = per_tree[e].row(l);
                        for j in 0..k {
                            row[j] += leaf_row[j];
                        }
                    }
                    if aalen {
                        for j in 0..k { row[j] = (-row[j] * inv).exp(); }
                    } else {
                        for j in 0..k { row[j] *= inv; }
                    }
                });
            result
        });

        result.into_pyarray_bound(py)
    }

    fn union_times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let mut all_times: Vec<f64> = self
            .tree_curves
            .iter()
            .flat_map(|tc| tc.iter().flat_map(|c| c.times.iter().copied()))
            .collect();
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        all_times.dedup_by(|a, b| a == b);
        all_times.into_pyarray_bound(py)
    }
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_RustLtrcTree>()?;
    m.add_class::<_RustLtrcForest>()?;
    m.add_class::<_LazyForest>()?;
    Ok(())
}
