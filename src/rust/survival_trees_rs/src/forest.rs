use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::tree::{feature_importances, fit_tree, leaf_km_curves, predict_leaf_ids};
use crate::types::{Control, Dataset, Forest, KmCurve, Sample, Tree};

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

    // Pre-generate per-tree RNG seeds, feature subsets, and bootstrap indices
    // in a deterministic order so parallel execution stays reproducible.
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
        let (_seed, feat_subset, indices) = spec;
        let sub_x = build_sub_x(&ds.x, indices, feat_subset);
        let sub_samples: Vec<Sample> = indices.iter().map(|&i| ds.samples[i]).collect();
        let sub_ds = Dataset { x: sub_x, samples: sub_samples };
        fit_tree(&sub_ds, params.control)
    };

    let trees: Vec<Tree> = pool.install(|| per_tree_specs.par_iter().map(fit_one).collect());
    let feature_subsets: Vec<Vec<usize>> =
        per_tree_specs.into_iter().map(|(_, f, _)| f).collect();

    Forest {
        trees,
        feature_subsets,
        n_features,
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

fn project_features(x: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.nrows(), cols.len()));
    for (out_j, &src_j) in cols.iter().enumerate() {
        let src_col = x.index_axis(Axis(1), src_j);
        let mut dst_col = out.index_axis_mut(Axis(1), out_j);
        dst_col.assign(&src_col);
    }
    out
}

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
