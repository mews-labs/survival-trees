use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::types::{Control, Dataset, Direction, Split, SplitCriterion, SplitterMode};

pub fn find_best_split(
    ds: &Dataset,
    indices: &[usize],
    control: &Control,
    rng: &mut ChaCha8Rng,
) -> Option<Split> {
    match control.criterion {
        SplitCriterion::LogRank => find_best_split_log_rank(ds, indices, control, rng),
        SplitCriterion::PoissonExp => find_best_split_poisson_exp(ds, indices, control),
    }
}

// ---------------------------------------------------------------------------
// Poisson-exponential deviance (rpart method = "exp").
//
//     hazard_S  = d_S / e_S       e_S = sum_i (time_i - entry_i) over S
//     loglik_S  = d_S * ln(d_S / e_S) - d_S
//     gain      = loglik_L + loglik_R - loglik_parent
// ---------------------------------------------------------------------------

#[inline]
fn node_score_pe(d: f64, e: f64) -> f64 {
    if d <= 0.0 || e <= 0.0 { 0.0 } else { d * (d / e).ln() }
}

fn totals_pe(ds: &Dataset, indices: &[usize]) -> (f64, f64) {
    let mut d = 0.0;
    let mut e = 0.0;
    for &i in indices {
        let s = ds.samples[i];
        if s.event { d += 1.0; }
        e += (s.time - s.entry).max(0.0);
    }
    (d, e)
}

fn find_best_split_poisson_exp(
    ds: &Dataset,
    indices: &[usize],
    control: &Control,
) -> Option<Split> {
    let n = indices.len();
    if n < control.min_samples_split.max(2) { return None; }
    let (d_total, e_total) = totals_pe(ds, indices);
    if d_total <= 0.0 || e_total <= 0.0 { return None; }
    let parent_score = node_score_pe(d_total, e_total);

    let n_features = ds.n_features();
    let mut best: Option<Split> = None;
    let mut observed: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut missing: Vec<usize> = Vec::with_capacity(n);

    for f in 0..n_features {
        observed.clear();
        missing.clear();
        for &i in indices {
            let v = ds.x[[i, f]];
            if v.is_nan() { missing.push(i); } else { observed.push((v, i)); }
        }
        observed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if observed.len() < 2 { continue; }

        let (d_miss, e_miss) = totals_pe(ds, &missing);
        let n_miss = missing.len();
        let n_obs = observed.len();

        let mut cum_d = 0.0_f64;
        let mut cum_e = 0.0_f64;
        for k in 0..(n_obs - 1) {
            let sample = ds.samples[observed[k].1];
            if sample.event { cum_d += 1.0; }
            cum_e += (sample.time - sample.entry).max(0.0);

            let x_k = observed[k].0;
            let x_next = observed[k + 1].0;
            if x_k == x_next { continue; }
            let left_obs = k + 1;
            let right_obs = n_obs - left_obs;
            let threshold = 0.5 * (x_k + x_next);

            if left_obs + n_miss >= control.min_samples_leaf
                && right_obs >= control.min_samples_leaf
            {
                let d_left = cum_d + d_miss;
                let e_left = cum_e + e_miss;
                let d_right = d_total - d_left;
                let e_right = e_total - e_left;
                let improvement = node_score_pe(d_left, e_left)
                    + node_score_pe(d_right, e_right)
                    - parent_score;
                if improvement.is_finite() && improvement > 0.0 {
                    best = keep_best(best, Split {
                        feature: f, threshold, score: improvement,
                        default_direction: Direction::Left,
                    });
                }
            }
            if left_obs >= control.min_samples_leaf
                && right_obs + n_miss >= control.min_samples_leaf
            {
                let d_left = cum_d;
                let e_left = cum_e;
                let d_right = d_total - d_left;
                let e_right = e_total - e_left;
                let improvement = node_score_pe(d_left, e_left)
                    + node_score_pe(d_right, e_right)
                    - parent_score;
                if improvement.is_finite() && improvement > 0.0 {
                    best = keep_best(best, Split {
                        feature: f, threshold, score: improvement,
                        default_direction: Direction::Right,
                    });
                }
            }
        }
    }

    if let Some(b) = best {
        let floor = control.min_impurity_decrease * parent_score.abs().max(1.0);
        if b.score < floor { return None; }
    }
    best
}

// ---------------------------------------------------------------------------
// Log-rank (Cox-Mantel) — Ishwaran et al. 2008. For each distinct event time
// t_m in the node:
//     n_m   = # at risk at t_m      (entry_i < t_m <= time_i)
//     d_m   = # events at t_m
//     n_Lm  = at-risk count in left child
//     d_Lm  = events in left child
//     Z^2   = (O_L - E_L)^2 / V_L
//     O_L   = sum_m d_Lm
//     E_L   = sum_m d_m * n_Lm / n_m
//     V_L   = sum_m d_m * (n_m - d_m) * n_Lm * (n_m - n_Lm)
//             / ( n_m^2 * (n_m - 1) )
// ---------------------------------------------------------------------------

fn build_event_grid(ds: &Dataset, indices: &[usize]) -> Vec<f64> {
    let mut times: Vec<f64> = indices.iter()
        .filter_map(|&i| {
            let s = ds.samples[i];
            if s.event { Some(s.time) } else { None }
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup();
    times
}

fn find_best_split_log_rank(
    ds: &Dataset,
    indices: &[usize],
    control: &Control,
    rng: &mut ChaCha8Rng,
) -> Option<Split> {
    let n = indices.len();
    if n < control.min_samples_split.max(2) { return None; }

    let grid = build_event_grid(ds, indices);
    let t_len = grid.len();
    if t_len == 0 { return None; }

    let mut at_risk: Vec<(u32, u32)> = Vec::with_capacity(n);
    let mut event_at: Vec<i32> = Vec::with_capacity(n);
    for &i in indices {
        let s = ds.samples[i];
        let lo = grid.partition_point(|&t| t <= s.entry) as u32;
        let hi = grid.partition_point(|&t| t <= s.time) as u32;
        at_risk.push((lo, hi));
        if s.event {
            let idx = grid.binary_search_by(|t| t.partial_cmp(&s.time).unwrap())
                .map(|x| x as i32).unwrap_or(-1);
            event_at.push(idx);
        } else {
            event_at.push(-1);
        }
    }

    let mut n_tot = vec![0i64; t_len];
    let mut d_tot = vec![0i64; t_len];
    for local_i in 0..n {
        let (lo, hi) = at_risk[local_i];
        for m in (lo as usize)..(hi as usize) { n_tot[m] += 1; }
        let e = event_at[local_i];
        if e >= 0 { d_tot[e as usize] += 1; }
    }
    if d_tot.iter().all(|&d| d == 0) { return None; }

    // p[m] = d_m / n_m ;  c[m] = d_m (n_m - d_m) / (n_m^2 (n_m - 1))
    let mut p_vec = vec![0.0_f64; t_len];
    let mut c_vec = vec![0.0_f64; t_len];
    for m in 0..t_len {
        let n_m = n_tot[m] as f64;
        let d_m = d_tot[m] as f64;
        if n_m >= 2.0 && d_m > 0.0 {
            p_vec[m] = d_m / n_m;
            c_vec[m] = d_m * (n_m - d_m) / (n_m * n_m * (n_m - 1.0));
        }
    }

    let n_features = ds.n_features();
    let mut best: Option<Split> = None;

    let mut observed: Vec<(f64, u32)> = Vec::with_capacity(n);
    let mut missing: Vec<u32> = Vec::with_capacity(n);
    let mut n_miss = vec![0i64; t_len];
    let mut n_left = vec![0i64; t_len];

    for f in 0..n_features {
        observed.clear();
        missing.clear();
        for local_i in 0..n {
            let v = ds.x[[indices[local_i], f]];
            if v.is_nan() { missing.push(local_i as u32); }
            else { observed.push((v, local_i as u32)); }
        }
        observed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n_obs = observed.len();
        if n_obs < 2 { continue; }

        for b in n_miss.iter_mut() { *b = 0; }
        let mut o_miss: i64 = 0;
        let mut w_miss: f64 = 0.0;
        for &lm in &missing {
            let li = lm as usize;
            let (lo, hi) = at_risk[li];
            for m in (lo as usize)..(hi as usize) {
                n_miss[m] += 1;
                w_miss += p_vec[m];
            }
            let e = event_at[li];
            if e >= 0 { o_miss += 1; }
        }
        let n_missing_total = missing.len();

        for b in n_left.iter_mut() { *b = 0; }
        let mut o_left: i64 = 0;
        let mut w_left: f64 = 0.0;
        let mut v_left: f64 = 0.0;

        let candidates: Vec<usize> = match control.splitter {
            SplitterMode::Best => (1..n_obs)
                .filter(|&k| observed[k - 1].0 < observed[k].0)
                .collect(),
            SplitterMode::Random => {
                let min_v = observed[0].0;
                let max_v = observed[n_obs - 1].0;
                if min_v >= max_v {
                    Vec::new()
                } else {
                    let threshold = rng.gen_range(min_v..max_v);
                    let k_star = observed.partition_point(|(v, _)| *v <= threshold);
                    if k_star == 0 || k_star == n_obs { Vec::new() } else { vec![k_star] }
                }
            }
        };

        let mut next_cand = 0usize;
        let mut cum_left = 0usize;
        for k in 1..n_obs {
            let local_i = observed[k - 1].1 as usize;
            let (lo, hi) = at_risk[local_i];
            for m in (lo as usize)..(hi as usize) {
                let nlm = n_left[m] as f64;
                let nm = n_tot[m] as f64;
                // Δ v_left = c * [(nlm+1)(nm-nlm-1) - nlm(nm-nlm)] = c * (nm - 2 nlm - 1)
                v_left += c_vec[m] * (nm - 2.0 * nlm - 1.0);
                w_left += p_vec[m];
                n_left[m] += 1;
            }
            if event_at[local_i] >= 0 { o_left += 1; }
            cum_left += 1;

            while next_cand < candidates.len() && candidates[next_cand] < k {
                next_cand += 1;
            }
            if next_cand >= candidates.len() || candidates[next_cand] != k { continue; }

            let right_obs = n_obs - cum_left;
            let threshold = 0.5 * (observed[k - 1].0 + observed[k].0);

            if cum_left + n_missing_total >= control.min_samples_leaf
                && right_obs >= control.min_samples_leaf
            {
                let v_with_miss = v_with_missing(&n_left, &n_tot, &n_miss, &c_vec);
                let score = log_rank_score(
                    (o_left + o_miss) as f64,
                    w_left + w_miss,
                    v_with_miss,
                );
                if score.is_finite() && score > 0.0 {
                    best = keep_best(best, Split {
                        feature: f, threshold, score,
                        default_direction: Direction::Left,
                    });
                }
            }
            if cum_left >= control.min_samples_leaf
                && right_obs + n_missing_total >= control.min_samples_leaf
            {
                let score = log_rank_score(o_left as f64, w_left, v_left);
                if score.is_finite() && score > 0.0 {
                    best = keep_best(best, Split {
                        feature: f, threshold, score,
                        default_direction: Direction::Right,
                    });
                }
            }

            if control.splitter == SplitterMode::Random { break; }
        }
    }

    if let Some(b) = best {
        // χ² scales with n → reference by n so cp matches rpart's Poisson-exp cp.
        let floor = control.min_impurity_decrease * (n as f64).max(1.0);
        if b.score < floor { return None; }
    }
    best
}

#[inline]
fn v_with_missing(n_left: &[i64], n_tot: &[i64], n_miss: &[i64], c_vec: &[f64]) -> f64 {
    let mut s = 0.0;
    for m in 0..n_left.len() {
        let nm = n_tot[m] as f64;
        let nlm = (n_left[m] + n_miss[m]) as f64;
        if nlm <= 0.0 || nlm >= nm { continue; }
        s += c_vec[m] * nlm * (nm - nlm);
    }
    s
}

#[inline]
fn log_rank_score(o_left: f64, w_left: f64, v_left: f64) -> f64 {
    if v_left <= 0.0 { return 0.0; }
    let num = o_left - w_left;
    (num * num) / v_left
}

#[inline]
fn keep_best(current: Option<Split>, candidate: Split) -> Option<Split> {
    match current {
        None => Some(candidate),
        Some(b) if candidate.score > b.score => Some(candidate),
        other => other,
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

    fn test_rng() -> ChaCha8Rng {
        use rand::SeedableRng;
        ChaCha8Rng::seed_from_u64(0)
    }

    fn ctrl(criterion: SplitCriterion) -> Control {
        Control {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
            criterion,
            pipeline: crate::types::Pipeline::Aalen,
            splitter: crate::types::SplitterMode::Best,
        }
    }

    #[test]
    fn best_split_picks_discriminative_feature_lr() {
        let x = array![
            [0.0, 0.3], [0.1, 0.7], [0.2, 0.2], [0.3, 0.8],
            [0.8, 0.1], [0.9, 0.6], [1.0, 0.4], [1.1, 0.9],
        ];
        let samples = vec![
            s(0.0, 1.0, true), s(0.0, 1.2, true), s(0.0, 0.9, true), s(0.0, 1.1, true),
            s(0.0, 10.0, true), s(0.0, 11.0, true), s(0.0, 9.5, true), s(0.0, 12.0, true),
        ];
        let ds = Dataset { x, samples };
        let indices: Vec<usize> = (0..8).collect();
        let split = find_best_split(&ds, &indices, &ctrl(SplitCriterion::LogRank), &mut test_rng())
            .expect("should find a split");
        assert_eq!(split.feature, 0);
        assert!(split.threshold > 0.3 && split.threshold < 0.8);
        assert!(split.score > 0.0);
    }

    #[test]
    fn best_split_picks_discriminative_feature_pe() {
        let x = array![
            [0.0, 0.3], [0.1, 0.7], [0.2, 0.2], [0.3, 0.8],
            [0.8, 0.1], [0.9, 0.6], [1.0, 0.4], [1.1, 0.9],
        ];
        let samples = vec![
            s(0.0, 1.0, true), s(0.0, 1.2, true), s(0.0, 0.9, true), s(0.0, 1.1, true),
            s(0.0, 10.0, true), s(0.0, 11.0, true), s(0.0, 9.5, true), s(0.0, 12.0, true),
        ];
        let ds = Dataset { x, samples };
        let indices: Vec<usize> = (0..8).collect();
        let split = find_best_split(&ds, &indices, &ctrl(SplitCriterion::PoissonExp), &mut test_rng())
            .expect("should find a split");
        assert_eq!(split.feature, 0);
        assert!(split.threshold > 0.3 && split.threshold < 0.8);
        assert!(split.score > 0.0);
    }

    #[test]
    fn mia_learns_direction_matching_events_lr() {
        let nan = f64::NAN;
        let x = array![
            [0.0], [0.1], [0.2], [0.3],
            [0.8], [0.9], [1.0], [1.1],
            [nan], [nan], [nan], [nan],
        ];
        let samples = vec![
            s(0.0, 1.0, true), s(0.0, 1.2, true), s(0.0, 0.9, true), s(0.0, 1.1, true),
            s(0.0, 10.0, true), s(0.0, 11.0, true), s(0.0, 9.5, true), s(0.0, 12.0, true),
            s(0.0, 10.5, true), s(0.0, 11.5, true), s(0.0, 9.8, true), s(0.0, 11.2, true),
        ];
        let ds = Dataset { x, samples };
        let indices: Vec<usize> = (0..12).collect();
        let split = find_best_split(&ds, &indices, &ctrl(SplitCriterion::LogRank), &mut test_rng())
            .expect("should find a split");
        assert_eq!(split.feature, 0);
        assert_eq!(split.default_direction, Direction::Right);
    }

    #[test]
    fn no_split_when_homogeneous_lr() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let samples = vec![
            s(0.0, 5.0, true), s(0.0, 5.0, true), s(0.0, 5.0, true), s(0.0, 5.0, true),
        ];
        let ds = Dataset { x, samples };
        let indices: Vec<usize> = (0..4).collect();
        let split = find_best_split(&ds, &indices, &ctrl(SplitCriterion::LogRank), &mut test_rng());
        assert!(split.is_none());
    }
}
