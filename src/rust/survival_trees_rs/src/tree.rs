use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::km::{kaplan_meier_ltrc, nelson_aalen_survival_ltrc};
use crate::split::find_best_split;
use crate::types::{Control, Dataset, Direction, Node, Pipeline, Tree};

pub fn fit_tree(ds: &Dataset, control: &Control, seed: u64) -> Tree {
    let n_features = ds.n_features();
    let indices: Vec<usize> = (0..ds.n_samples()).collect();
    let mut leaf_counter: usize = 0;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let root = build_node(ds, &indices, control, 0, &mut leaf_counter, &mut rng);
    Tree {
        root,
        n_leaves: leaf_counter,
        n_features,
        pipeline: control.pipeline,
    }
}

fn build_node(
    ds: &Dataset,
    indices: &[usize],
    control: &Control,
    depth: usize,
    leaf_counter: &mut usize,
    rng: &mut ChaCha8Rng,
) -> Node {
    let at_depth_limit = control.max_depth.map_or(false, |d| depth >= d);
    let too_few = indices.len() < control.min_samples_split.max(2);
    let split = if at_depth_limit || too_few {
        None
    } else {
        find_best_split(ds, indices, control, rng)
    };

    match split {
        None => make_leaf(ds, indices, leaf_counter, control.pipeline),
        Some(s) => {
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &i in indices {
                let v = ds.x[[i, s.feature]];
                let go_left = if v.is_nan() {
                    s.default_direction == Direction::Left
                } else {
                    v <= s.threshold
                };
                if go_left {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            if left_idx.len() < control.min_samples_leaf
                || right_idx.len() < control.min_samples_leaf
            {
                return make_leaf(ds, indices, leaf_counter, control.pipeline);
            }
            let left = Box::new(build_node(ds, &left_idx, control, depth + 1, leaf_counter, rng));
            let right = Box::new(build_node(ds, &right_idx, control, depth + 1, leaf_counter, rng));
            Node::Internal {
                feature: s.feature,
                threshold: s.threshold,
                default_direction: s.default_direction,
                left,
                right,
                n_samples: indices.len(),
                improvement: s.score,
            }
        }
    }
}

fn make_leaf(ds: &Dataset, indices: &[usize], leaf_counter: &mut usize,
             pipeline: Pipeline) -> Node {
    let samples: Vec<_> = indices.iter().map(|&i| ds.samples[i]).collect();
    let km = match pipeline {
        Pipeline::Km => kaplan_meier_ltrc(&samples),
        Pipeline::Aalen => nelson_aalen_survival_ltrc(&samples),
    };
    let leaf_id = *leaf_counter;
    *leaf_counter += 1;
    Node::Leaf { leaf_id, km }
}

pub fn predict_leaf_ids(tree: &Tree, x: &Array2<f64>) -> Vec<usize> {
    (0..x.nrows())
        .map(|i| {
            let mut node = &tree.root;
            loop {
                match node {
                    Node::Leaf { leaf_id, .. } => return *leaf_id,
                    Node::Internal {
                        feature,
                        threshold,
                        default_direction,
                        left,
                        right,
                        ..
                    } => {
                        let v = x[[i, *feature]];
                        let go_left = if v.is_nan() {
                            *default_direction == Direction::Left
                        } else {
                            v <= *threshold
                        };
                        node = if go_left { left } else { right };
                    }
                }
            }
        })
        .collect()
}

/// Aggregate KM curves per leaf id, in order of leaf id.
pub fn leaf_km_curves(tree: &Tree) -> Vec<crate::types::KmCurve> {
    let mut curves: Vec<Option<crate::types::KmCurve>> = vec![None; tree.n_leaves];
    collect_curves(&tree.root, &mut curves);
    curves
        .into_iter()
        .map(|c| {
            c.unwrap_or(crate::types::KmCurve {
                times: vec![],
                surv: vec![],
            })
        })
        .collect()
}

fn collect_curves(node: &Node, out: &mut [Option<crate::types::KmCurve>]) {
    match node {
        Node::Leaf { leaf_id, km } => out[*leaf_id] = Some(km.clone()),
        Node::Internal { left, right, .. } => {
            collect_curves(left, out);
            collect_curves(right, out);
        }
    }
}

pub fn feature_importances(tree: &Tree) -> Vec<f64> {
    let mut imp = vec![0.0_f64; tree.n_features];
    accumulate_importance(&tree.root, &mut imp);
    let sum: f64 = imp.iter().sum();
    if sum > 0.0 {
        imp.iter().map(|v| v / sum).collect()
    } else {
        imp
    }
}

fn accumulate_importance(node: &Node, imp: &mut [f64]) {
    if let Node::Internal {
        feature,
        left,
        right,
        n_samples,
        improvement,
        ..
    } = node
    {
        imp[*feature] += (*n_samples as f64) * improvement;
        accumulate_importance(left, imp);
        accumulate_importance(right, imp);
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

    /// Two well-separated groups on feature 0 should produce a tree with 2 leaves
    /// and the split should pick feature 0.
    #[test]
    fn tree_splits_on_discriminative_feature() {
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
        let ds = Dataset { x: x.clone(), samples };
        let ctrl = Control {
            max_depth: Some(1),
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
            criterion: crate::types::SplitCriterion::LogRank,
            pipeline: Pipeline::Aalen,
            splitter: crate::types::SplitterMode::Best,
        };
        let tree = fit_tree(&ds, &ctrl, 0);
        assert_eq!(tree.n_leaves, 2);
        let ids = predict_leaf_ids(&tree, &x);
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0], ids[2]);
        assert_ne!(ids[0], ids[4]);

        let imp = feature_importances(&tree);
        assert!(imp[0] > imp[1]);
    }
}
