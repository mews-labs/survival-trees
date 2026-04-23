use crate::types::{Control, Dataset, Sample, Split};

/// Two-sample log-rank statistic on left-truncated right-censored data.
///
/// At each distinct event time `t`:
///   O1_t = events in group 1
///   N1_t = size of risk set of group 1
///   O_t  = events overall (both groups)
///   N_t  = size of overall risk set
///   E1_t = O_t * N1_t / N_t
///   V_t  = O_t * (N_t - O_t) * N1_t * (N_t - N1_t) / (N_t^2 * (N_t - 1))
/// Returns (sum O1_t - E1_t)^2 / sum V_t. Zero if no events or undefined.
pub fn log_rank_ltrc(left: &[Sample], right: &[Sample]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    // Collect and sort unique event times across both groups.
    let mut event_times: Vec<f64> = left
        .iter()
        .chain(right.iter())
        .filter(|s| s.event)
        .map(|s| s.time)
        .collect();
    if event_times.is_empty() {
        return 0.0;
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    event_times.dedup_by(|a, b| a == b);

    let mut o_minus_e: f64 = 0.0;
    let mut var_sum: f64 = 0.0;

    for t in event_times {
        let n1 = left.iter().filter(|s| s.entry < t && s.time >= t).count() as f64;
        let n2 = right.iter().filter(|s| s.entry < t && s.time >= t).count() as f64;
        let n = n1 + n2;
        if n < 2.0 {
            continue;
        }
        let o1 = left.iter().filter(|s| s.event && s.time == t).count() as f64;
        let o2 = right.iter().filter(|s| s.event && s.time == t).count() as f64;
        let o = o1 + o2;
        if o == 0.0 {
            continue;
        }
        let e1 = o * n1 / n;
        o_minus_e += o1 - e1;
        if n > 1.0 {
            let v = o * (n - o) * n1 * (n - n1) / (n * n * (n - 1.0));
            var_sum += v;
        }
    }

    if var_sum <= 0.0 {
        0.0
    } else {
        (o_minus_e * o_minus_e) / var_sum
    }
}

/// Find the best (feature, threshold) split on the subset `indices` of `ds`.
/// Returns `None` if no valid split exists under `control`.
///
/// Candidate thresholds per feature are midpoints between consecutive unique
/// sorted values. For each candidate, computes log-rank between left (x<=t)
/// and right (x>t).
pub fn find_best_split(
    ds: &Dataset,
    indices: &[usize],
    control: &Control,
) -> Option<Split> {
    let n = indices.len();
    if n < control.min_samples_split.max(2) {
        return None;
    }

    // Score of the parent node — used for min_impurity_decrease.
    // We interpret min_impurity_decrease relative to the best candidate score:
    // split kept only if best_score >= min_impurity_decrease * n_total_events
    // (comparable to rpart's cp which normalizes by deviance).
    let n_events: usize = indices.iter().filter(|&&i| ds.samples[i].event).count();
    if n_events == 0 {
        return None;
    }

    let n_features = ds.n_features();
    let mut best: Option<Split> = None;

    for f in 0..n_features {
        // Collect (x_value, idx) for this feature restricted to indices, sort by value.
        let mut vals: Vec<(f64, usize)> = indices.iter().map(|&i| (ds.x[[i, f]], i)).collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Unique x values; candidate thresholds = midpoints.
        for i in 0..(vals.len() - 1) {
            let x1 = vals[i].0;
            let x2 = vals[i + 1].0;
            if x1 == x2 {
                continue;
            }
            let threshold = 0.5 * (x1 + x2);
            let left_n = i + 1;
            let right_n = vals.len() - left_n;
            if left_n < control.min_samples_leaf || right_n < control.min_samples_leaf {
                continue;
            }
            let left_samples: Vec<Sample> =
                vals[..=i].iter().map(|(_, idx)| ds.samples[*idx]).collect();
            let right_samples: Vec<Sample> =
                vals[(i + 1)..].iter().map(|(_, idx)| ds.samples[*idx]).collect();
            let score = log_rank_ltrc(&left_samples, &right_samples);
            if !score.is_finite() || score <= 0.0 {
                continue;
            }
            match best {
                None => best = Some(Split { feature: f, threshold, score }),
                Some(b) if score > b.score => {
                    best = Some(Split { feature: f, threshold, score })
                }
                _ => {}
            }
        }
    }

    // Minimum improvement threshold: scaled by number of events to keep it
    // roughly scale-invariant. If the best log-rank chi-square per event is
    // below min_impurity_decrease * n_events, reject.
    if let Some(b) = best {
        if b.score < control.min_impurity_decrease * (n_events as f64) {
            return None;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(entry: f64, time: f64, event: bool) -> Sample {
        Sample { entry, time, event }
    }

    /// Two perfectly-separated groups should produce a strictly positive log-rank.
    #[test]
    fn logrank_clear_separation() {
        let left = vec![s(0.0, 1.0, true), s(0.0, 1.0, true), s(0.0, 2.0, true)];
        let right = vec![s(0.0, 10.0, true), s(0.0, 11.0, true), s(0.0, 12.0, true)];
        let lr = log_rank_ltrc(&left, &right);
        assert!(lr > 3.0, "log-rank should be large for clear separation; got {}", lr);
    }

    /// Identical survival distributions should give ~0.
    #[test]
    fn logrank_identical() {
        let left = vec![s(0.0, 1.0, true), s(0.0, 2.0, true), s(0.0, 3.0, true)];
        let right = vec![s(0.0, 1.0, true), s(0.0, 2.0, true), s(0.0, 3.0, true)];
        let lr = log_rank_ltrc(&left, &right);
        assert!(lr.abs() < 1e-9, "log-rank should be ~0 for identical; got {}", lr);
    }

    /// find_best_split picks the feature that separates groups, on a 2D toy.
    #[test]
    fn best_split_picks_discriminative_feature() {
        use ndarray::array;
        // feature 0 splits clearly (<0.5 vs >=0.5), feature 1 is noise.
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
        let ds = Dataset { x, samples };
        let indices: Vec<usize> = (0..8).collect();
        let ctrl = Control {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
        };
        let split = find_best_split(&ds, &indices, &ctrl).expect("should find a split");
        assert_eq!(split.feature, 0, "expected feature 0 to be chosen; got {}", split.feature);
        assert!(split.threshold > 0.3 && split.threshold < 0.8);
    }
}
