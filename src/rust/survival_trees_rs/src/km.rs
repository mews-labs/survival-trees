use crate::types::{KmCurve, Sample};

/// Kaplan-Meier with left-truncation:
/// risk set at `t` = { i : entry_i < t <= time_i },
/// S(t) = ∏_{t_k <= t} (1 - d_k / n_k).
pub fn kaplan_meier_ltrc(samples: &[Sample]) -> KmCurve {
    if samples.is_empty() {
        return KmCurve { times: vec![], surv: vec![] };
    }

    let mut order: Vec<u32> = (0..samples.len() as u32).collect();
    order.sort_by(|&a, &b| samples[a as usize]
        .time
        .partial_cmp(&samples[b as usize].time)
        .unwrap_or(std::cmp::Ordering::Equal));

    let mut entries: Vec<f64> = samples.iter().map(|s| s.entry).collect();
    entries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut times = Vec::new();
    let mut surv = Vec::new();
    let mut s: f64 = 1.0;

    let mut exit_cursor: usize = 0;
    let n = samples.len();
    let mut i = 0;
    while i < n {
        let t = samples[order[i] as usize].time;
        while exit_cursor < n && samples[order[exit_cursor] as usize].time < t {
            exit_cursor += 1;
        }
        let mut events_at_t = 0usize;
        let mut j = i;
        while j < n && samples[order[j] as usize].time == t {
            if samples[order[j] as usize].event {
                events_at_t += 1;
            }
            j += 1;
        }

        if events_at_t > 0 {
            // n_at_risk = #{entry_i < t} - #{time_i < t}
            let entered = lower_bound(&entries, t);
            let n_at_risk = entered.saturating_sub(exit_cursor);
            if n_at_risk > 0 {
                s *= 1.0 - (events_at_t as f64) / (n_at_risk as f64);
                times.push(t);
                surv.push(s);
            }
        }
        i = j;
    }

    KmCurve { times, surv }
}

/// Nelson-Aalen estimator with left-truncation correction, exported as
/// S_NA(t) = exp(-Λ̂_NA(t)), with Λ̂_NA(t) = Σ_{t_k ≤ t} d_k / n_k.
/// Never reaches 0 by construction.
pub fn nelson_aalen_survival_ltrc(samples: &[Sample]) -> KmCurve {
    if samples.is_empty() {
        return KmCurve { times: vec![], surv: vec![] };
    }
    let mut order: Vec<u32> = (0..samples.len() as u32).collect();
    order.sort_by(|&a, &b| samples[a as usize]
        .time
        .partial_cmp(&samples[b as usize].time)
        .unwrap_or(std::cmp::Ordering::Equal));

    let mut entries: Vec<f64> = samples.iter().map(|s| s.entry).collect();
    entries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut times = Vec::new();
    let mut surv = Vec::new();
    let mut chf: f64 = 0.0;

    let mut exit_cursor: usize = 0;
    let n = samples.len();
    let mut i = 0;
    while i < n {
        let t = samples[order[i] as usize].time;
        while exit_cursor < n && samples[order[exit_cursor] as usize].time < t {
            exit_cursor += 1;
        }
        let mut events_at_t = 0usize;
        let mut j = i;
        while j < n && samples[order[j] as usize].time == t {
            if samples[order[j] as usize].event {
                events_at_t += 1;
            }
            j += 1;
        }
        if events_at_t > 0 {
            let entered = lower_bound(&entries, t);
            let n_at_risk = entered.saturating_sub(exit_cursor);
            if n_at_risk > 0 {
                chf += (events_at_t as f64) / (n_at_risk as f64);
                times.push(t);
                surv.push((-chf).exp());
            }
        }
        i = j;
    }
    KmCurve { times, surv }
}

/// `#{i : sorted[i] < v}` via binary search.
fn lower_bound(sorted: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(entry: f64, time: f64, event: bool) -> Sample {
        Sample { entry, time, event }
    }

    #[test]
    fn km_no_truncation_no_censoring() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(0.0, 2.0, true),
            s(0.0, 3.0, true),
            s(0.0, 4.0, true),
        ];
        let km = kaplan_meier_ltrc(&samples);
        assert_eq!(km.times, vec![1.0, 2.0, 3.0, 4.0]);
        for (got, want) in km.surv.iter().zip(&[0.75, 0.5, 0.25, 0.0]) {
            assert!((got - want).abs() < 1e-9, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn km_right_censoring() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(0.0, 2.5, false),
            s(0.0, 3.0, true),
            s(0.0, 4.0, true),
        ];
        let km = kaplan_meier_ltrc(&samples);
        assert_eq!(km.times, vec![1.0, 3.0, 4.0]);
        for (got, want) in km.surv.iter().zip(&[0.75, 0.375, 0.0]) {
            assert!((got - want).abs() < 1e-9, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn km_left_truncation() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(2.0, 3.0, true),
            s(0.0, 4.0, true),
        ];
        let km = kaplan_meier_ltrc(&samples);
        assert_eq!(km.times, vec![1.0, 3.0, 4.0]);
        for (got, want) in km.surv.iter().zip(&[0.5, 0.25, 0.0]) {
            assert!((got - want).abs() < 1e-9, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn km_empty_input() {
        let km = kaplan_meier_ltrc(&[]);
        assert!(km.times.is_empty());
        assert!(km.surv.is_empty());
    }

    /// NA with no censoring: Λ(t) = Σ 1/(n-k+1) → well-known values.
    /// n=4 samples: Λ(t1)=1/4, Λ(t2)=1/4+1/3, etc.
    #[test]
    fn na_no_truncation_no_censoring() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(0.0, 2.0, true),
            s(0.0, 3.0, true),
            s(0.0, 4.0, true),
        ];
        let na = nelson_aalen_survival_ltrc(&samples);
        let lambdas: [f64; 4] = [1.0/4.0,
                                 1.0/4.0 + 1.0/3.0,
                                 1.0/4.0 + 1.0/3.0 + 1.0/2.0,
                                 1.0/4.0 + 1.0/3.0 + 1.0/2.0 + 1.0];
        for (got, lam) in na.surv.iter().zip(lambdas.iter()) {
            let expected = (-*lam).exp();
            assert!((got - expected).abs() < 1e-9, "got {got}, lambda {lam}");
        }
    }

    /// NA never returns 0, even with all-uncensored deaths.
    #[test]
    fn na_never_zero() {
        let samples = vec![
            s(0.0, 1.0, true), s(0.0, 2.0, true), s(0.0, 3.0, true), s(0.0, 4.0, true),
        ];
        let na = nelson_aalen_survival_ltrc(&samples);
        for s in &na.surv {
            assert!(*s > 0.0, "NA survival must be strictly positive, got {s}");
        }
    }
}
