use std::collections::HashMap;

use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::tree::{feature_importances, fit_tree, leaf_km_curves, predict_leaf_ids};
use crate::types::{Control, Dataset, Forest, KmCurve, Pipeline, Sample, Tree};

pub struct ForestFitParams<'a> {
    pub control: &'a Control,
    pub n_trees: usize,
    pub max_samples: f64,
    pub max_features: usize,
    pub seed: u64,
    pub n_jobs: usize,
}

fn bootstrap_indices(n: usize, max_samples: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let size = ((max_samples * n as f64).round() as usize).max(1);
    (0..size).map(|_| rng.gen_range(0..n)).collect()
}

fn choose_features(total: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let k = k.min(total).max(1);
    let mut pool: Vec<usize> = (0..total).collect();
    pool.shuffle(rng);
    pool.truncate(k);
    pool.sort();
    pool
}

pub fn fit_forest(ds: &Dataset, params: &ForestFitParams) -> Forest {
    let n_features = ds.n_features();

    // Per-tree seeds/features/indices drawn serially so parallel fitting
    // stays reproducible regardless of `n_jobs`.
    let mut master_rng = ChaCha8Rng::seed_from_u64(params.seed);
    let mut per_tree_specs: Vec<(u64, Vec<usize>, Vec<usize>)> =
        Vec::with_capacity(params.n_trees);
    for _ in 0..params.n_trees {
        let tree_seed: u64 = master_rng.gen();
        let mut tree_rng = ChaCha8Rng::seed_from_u64(tree_seed);
        let features = choose_features(n_features, params.max_features, &mut tree_rng);
        let indices = bootstrap_indices(ds.n_samples(), params.max_samples, &mut tree_rng);
        per_tree_specs.push((tree_seed, features, indices));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.n_jobs.max(1))
        .build()
        .expect("rayon pool");

    let fit_one = |spec: &(u64, Vec<usize>, Vec<usize>)| {
        let (seed, feat_subset, indices) = spec;
        let sub_x = build_sub_x(&ds.x, indices, feat_subset);
        let sub_samples: Vec<Sample> = indices.iter().map(|&i| ds.samples[i]).collect();
        let sub_ds = Dataset { x: sub_x, samples: sub_samples };
        fit_tree(&sub_ds, params.control, *seed)
    };

    let trees: Vec<Tree> = pool.install(|| per_tree_specs.par_iter().map(fit_one).collect());
    let feature_subsets: Vec<Vec<usize>> =
        per_tree_specs.into_iter().map(|(_, f, _)| f).collect();

    Forest {
        trees,
        feature_subsets,
        n_features,
        pipeline: params.control.pipeline,
    }
}

fn build_sub_x(x: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), cols.len()));
    for (out_i, &src_i) in rows.iter().enumerate() {
        for (out_j, &src_j) in cols.iter().enumerate() {
            out[[out_i, out_j]] = x[[src_i, src_j]];
        }
    }
    out
}

pub struct TreePrediction {
    pub leaf_ids: Vec<usize>,
    pub curves: Vec<KmCurve>,
}

pub fn predict_forest(forest: &Forest, x: &Array2<f64>) -> Vec<TreePrediction> {
    forest
        .trees
        .iter()
        .zip(forest.feature_subsets.iter())
        .map(|(tree, feats)| {
            let sub_x = project_features(x, feats);
            TreePrediction {
                leaf_ids: predict_leaf_ids(tree, &sub_x),
                curves: leaf_km_curves(tree),
            }
        })
        .collect()
}

pub(crate) fn project_features(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
    for (out_j, &src_j) in cols.iter().enumerate() {
        let src_col = x.index_axis(Axis(1), src_j);
        let mut dst_col = out.index_axis_mut(Axis(1), out_j);
        dst_col.assign(&src_col);
    }
    out
}

/// Returns `(unique_curves[n_unique, n_times], node_index[n_samples],
/// times[n_times])`. One curve per unique per-tree-leaf-id tuple;
/// `node_index[i]` is the row of sample `i` in `unique_curves`.
pub fn predict_forest_aggregated(
    forest: &Forest,
    x: &Array2<f64>,
    time_grid: Option<&[f64]>,
) -> (Array2<f32>, Vec<usize>, Vec<f64>) {
    let n_samples = x.nrows();
    let n_trees = forest.trees.len();

    let mut tree_leaf_ids: Vec<Vec<usize>> = Vec::with_capacity(n_trees);
    let mut tree_curves: Vec<Vec<KmCurve>> = Vec::with_capacity(n_trees);
    for (tree, feats) in forest.trees.iter().zip(forest.feature_subsets.iter()) {
        let sub_x = project_features(x, feats);
        tree_leaf_ids.push(predict_leaf_ids(tree, &sub_x));
        tree_curves.push(leaf_km_curves(tree));
    }

    let all_times: Vec<f64> = match time_grid {
        Some(g) => g.to_vec(),
        None => {
            let mut v: Vec<f64> = tree_curves
                .iter()
                .flat_map(|tc| tc.iter().flat_map(|c| c.times.iter().copied()))
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v.dedup_by(|a, b| a == b);
            v
        }
    };
    let n_times = all_times.len();

    let mut unique: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut node_index: Vec<usize> = Vec::with_capacity(n_samples);
    let mut tuple_list: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_samples {
        let mut key = Vec::with_capacity(n_trees);
        for e in 0..n_trees {
            key.push(tree_leaf_ids[e][i]);
        }
        let len = unique.len();
        let idx = *unique.entry(key.clone()).or_insert(len);
        if idx == len {
            tuple_list.push(key);
        }
        node_index.push(idx);
    }

    // Per-tree inverted index `leaf_id → tuples using that leaf`:
    // densify each leaf curve at most once into `val_buf`.
    let mut leaf_to_tuples: Vec<HashMap<usize, Vec<usize>>> =
        (0..n_trees).map(|_| HashMap::new()).collect();
    for (u, tuple) in tuple_list.iter().enumerate() {
        for (e, &leaf) in tuple.iter().enumerate() {
            leaf_to_tuples[e].entry(leaf).or_default().push(u);
        }
    }

    // Aalen: Λ_F = (1/B) Σ_b Λ_b,  S_F = exp(−Λ_F),  Λ_b = −ln S_b
    // Km:    S_F = (1/B) Σ_b S_b                                    (Breiman)
    let n_unique = tuple_list.len();
    let mut unique_curves = Array2::<f32>::zeros((n_unique, n_times));
    let inv_trees = if n_trees > 0 { 1.0_f32 / (n_trees as f32) } else { 0.0_f32 };
    let mut val_buf = vec![0.0_f32; n_times];
    let aalen = forest.pipeline == Pipeline::Aalen;

    for e in 0..n_trees {
        let tc = &tree_curves[e];
        for (leaf_id, tuples) in leaf_to_tuples[e].iter() {
            if tuples.is_empty() {
                continue;
            }
            let curve = &tc[*leaf_id];
            let mut last_s = 1.0_f32;
            let mut src = 0usize;
            for (j, &t) in all_times.iter().enumerate() {
                while src < curve.times.len() && curve.times[src] <= t {
                    last_s = curve.surv[src] as f32;
                    src += 1;
                }
                val_buf[j] = if aalen { -last_s.max(S_MIN).ln() } else { last_s };
            }
            for &u in tuples.iter() {
                let mut row = unique_curves.row_mut(u);
                for j in 0..n_times {
                    row[j] += val_buf[j] * inv_trees;
                }
            }
        }
    }
    if aalen {
        for u in 0..n_unique {
            let mut row = unique_curves.row_mut(u);
            for j in 0..n_times {
                row[j] = (-row[j]).exp();
            }
        }
    }

    (unique_curves, node_index, all_times)
}

const S_MIN: f32 = 1e-12;

pub fn aggregated_feature_importances(forest: &Forest) -> Vec<f64> {
    let mut out = vec![0.0_f64; forest.n_features];
    let mut counts = vec![0u64; forest.n_features];
    for (tree, feats) in forest.trees.iter().zip(forest.feature_subsets.iter()) {
        let imp = feature_importances(tree);
        for (idx_local, &global_idx) in feats.iter().enumerate() {
            out[global_idx] += imp[idx_local];
            counts[global_idx] += 1;
        }
    }
    for i in 0..out.len() {
        if counts[i] > 0 {
            out[i] /= counts[i] as f64;
        }
    }
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        out.iter().map(|v| v / sum).collect()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Sample;
    use ndarray::array;

    fn s(entry: f64, time: f64, event: bool) -> Sample {
        Sample { entry, time, event }
    }

    /// Forest built twice with the same seed must give identical importances.
    #[test]
    fn forest_reproducible_with_same_seed() {
        let x = array![
            [0.0, 0.3],
            [0.1, 0.7],
            [0.2, 0.2],
            [0.3, 0.8],
            [0.8, 0.1],
            [0.9, 0.6],
            [1.0, 0.4],
            [1.1, 0.9],
        ];
        let samples = vec![
            s(0.0, 1.0, true),
            s(0.0, 1.2, true),
            s(0.0, 0.9, true),
            s(0.0, 1.1, true),
            s(0.0, 10.0, true),
            s(0.0, 11.0, true),
            s(0.0, 9.5, true),
            s(0.0, 12.0, true),
        ];
        let ds = Dataset { x: x.clone(), samples: samples.clone() };
        let ctrl = Control::default();
        let params = ForestFitParams {
            control: &ctrl,
            n_trees: 5,
            max_samples: 1.0,
            max_features: 2,
            seed: 42,
            n_jobs: 1,
        };
        let f1 = fit_forest(&ds, &params);
        let ds2 = Dataset { x, samples };
        let f2 = fit_forest(&ds2, &params);
        let imp1 = aggregated_feature_importances(&f1);
        let imp2 = aggregated_feature_importances(&f2);
        assert_eq!(imp1, imp2, "forest not reproducible under same seed");
    }
}
