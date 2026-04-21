use crate::types::{KmCurve, Sample};

/// Kaplan-Meier estimator with left-truncation correction.
///
/// Risk set at time `t` = subjects `i` such that `entry_i < t <= time_i`.
/// At each distinct observed event time, survival drops by `1 - d/n`
/// where `d` is the count of events at `t` and `n` is the size of the
/// risk set just before `t`.
pub fn kaplan_meier_ltrc(samples: &[Sample]) -> KmCurve {
    if samples.is_empty() {
        return KmCurve {
            times: vec![],
            surv: vec![],
        };
    }

    let mut event_times: Vec<f64> = samples
        .iter()
        .filter(|s| s.event)
        .map(|s| s.time)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    event_times.dedup_by(|a, b| a == b);

    let mut times = Vec::with_capacity(event_times.len());
    let mut surv = Vec::with_capacity(event_times.len());
    let mut s: f64 = 1.0;

    for t in event_times {
        let n_at_risk = samples
            .iter()
            .filter(|sp| sp.entry < t && sp.time >= t)
            .count();
        let d = samples
            .iter()
            .filter(|sp| sp.event && sp.time == t)
            .count();
        if n_at_risk == 0 {
            continue;
        }
        s *= 1.0 - (d as f64) / (n_at_risk as f64);
        times.push(t);
        surv.push(s);
    }

    KmCurve { times, surv }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(entry: f64, time: f64, event: bool) -> Sample {
        Sample { entry, time, event }
    }

    /// No truncation, no censoring: 4 subjects dying at times 1,2,3,4.
    /// S(t) = (n-k)/n at each step: 0.75, 0.5, 0.25, 0.0
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

    /// Right-censoring: subject censored at time 2.5 does not drop S.
    #[test]
    fn km_right_censoring() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(0.0, 2.5, false), // censored
            s(0.0, 3.0, true),
            s(0.0, 4.0, true),
        ];
        let km = kaplan_meier_ltrc(&samples);
        // event times: 1, 3, 4
        // at t=1: n=4, d=1 -> S=0.75
        // at t=3: n=2 (censored at 2.5 gone, subject at 4 still there), d=1 -> S=0.375
        // at t=4: n=1, d=1 -> S=0.0
        assert_eq!(km.times, vec![1.0, 3.0, 4.0]);
        for (got, want) in km.surv.iter().zip(&[0.75, 0.375, 0.0]) {
            assert!((got - want).abs() < 1e-9, "got {}, want {}", got, want);
        }
    }

    /// Left truncation: subject with entry=2 is NOT at risk at t=1.
    #[test]
    fn km_left_truncation() {
        let samples = vec![
            s(0.0, 1.0, true),
            s(2.0, 3.0, true), // enters at 2, not at risk at t=1
            s(0.0, 4.0, true),
        ];
        let km = kaplan_meier_ltrc(&samples);
        // event times: 1, 3, 4
        // at t=1: risk set = {subj1 (entry<1, time>=1), subj3 (entry<1, time>=1)}; subj2 has entry=2 so not at risk. n=2, d=1 -> S=0.5
        // at t=3: risk set = {subj2, subj3}; n=2, d=1 -> S=0.25
        // at t=4: risk set = {subj3}; n=1, d=1 -> S=0.0
        assert_eq!(km.times, vec![1.0, 3.0, 4.0]);
        for (got, want) in km.surv.iter().zip(&[0.5, 0.25, 0.0]) {
            assert!((got - want).abs() < 1e-9, "got {}, want {}", got, want);
        }
    }

    /// Edge case: empty risk set at some time (should not panic).
    #[test]
    fn km_empty_input() {
        let km = kaplan_meier_ltrc(&[]);
        assert!(km.times.is_empty());
        assert!(km.surv.is_empty());
    }
}
