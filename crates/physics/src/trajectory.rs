//! Trajectory analysis and post-processing utilities.
//!
//! Functions for sampling, interpolating, and computing derived statistics
//! from a solved [`Trajectory`].

use simucad_core::types::{Trajectory, TrajectoryPoint, Vec2};

// ---------------------------------------------------------------------------
// TrajectoryStats
// ---------------------------------------------------------------------------

/// Derived statistics computed from a solved trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryStats {
    /// Maximum altitude reached (m).
    pub max_height: f64,
    /// Horizontal distance from launch to impact (m).
    pub range: f64,
    /// Total time of flight (s).
    pub flight_time: f64,
    /// Peak speed at any point along the trajectory (m/s).
    pub max_speed: f64,
    /// Speed at the moment of ground impact (m/s).
    pub impact_speed: f64,
    /// Angle of the velocity vector at impact, measured below horizontal
    /// (radians, positive downward). Returns 0.0 for empty trajectories.
    pub impact_angle: f64,
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Evenly sample `num_samples` points along a trajectory by interpolating
/// in the time domain.
///
/// If `num_samples` is 0 or the trajectory is empty, an empty vector is
/// returned. If `num_samples` is 1, the midpoint is returned. For larger
/// values the samples are distributed uniformly from the first to the last
/// point in time.
///
/// Interpolation is linear between consecutive trajectory points.
pub fn sample_trajectory(trajectory: &Trajectory, num_samples: usize) -> Vec<TrajectoryPoint> {
    if num_samples == 0 || trajectory.points.is_empty() {
        return Vec::new();
    }

    let pts = &trajectory.points;

    if pts.len() == 1 || num_samples == 1 {
        // Return the midpoint (by time) or just the single point
        let mid = &pts[pts.len() / 2];
        return vec![mid.clone()];
    }

    let t_start = pts.first().unwrap().time;
    let t_end = pts.last().unwrap().time;
    let t_span = t_end - t_start;

    if t_span.abs() < f64::EPSILON {
        // All points at the same time -- just return copies of the first
        return vec![pts[0].clone(); num_samples];
    }

    let mut result = Vec::with_capacity(num_samples);
    let mut seg_idx = 0_usize; // current segment lower index

    for i in 0..num_samples {
        let t_target = t_start + t_span * (i as f64) / ((num_samples - 1) as f64);

        // Advance segment index so that pts[seg_idx].time <= t_target
        while seg_idx + 1 < pts.len() - 1 && pts[seg_idx + 1].time <= t_target {
            seg_idx += 1;
        }

        let a = &pts[seg_idx];
        let b = &pts[(seg_idx + 1).min(pts.len() - 1)];

        let seg_dt = b.time - a.time;
        if seg_dt.abs() < f64::EPSILON {
            result.push(a.clone());
            continue;
        }

        let frac = ((t_target - a.time) / seg_dt).clamp(0.0, 1.0);

        let position = Vec2::new(
            a.position.x + frac * (b.position.x - a.position.x),
            a.position.y + frac * (b.position.y - a.position.y),
        );
        let velocity = Vec2::new(
            a.velocity.x + frac * (b.velocity.x - a.velocity.x),
            a.velocity.y + frac * (b.velocity.y - a.velocity.y),
        );
        let speed = velocity.magnitude();

        result.push(TrajectoryPoint {
            time: t_target,
            position,
            velocity,
            speed,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Derived statistics
// ---------------------------------------------------------------------------

/// Compute summary statistics from a solved trajectory.
///
/// For empty trajectories all fields are 0.0.
pub fn compute_derived_stats(trajectory: &Trajectory) -> TrajectoryStats {
    if trajectory.points.is_empty() {
        return TrajectoryStats {
            max_height: 0.0,
            range: 0.0,
            flight_time: 0.0,
            max_speed: 0.0,
            impact_speed: 0.0,
            impact_angle: 0.0,
        };
    }

    let pts = &trajectory.points;

    let max_height = pts
        .iter()
        .map(|p| p.position.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let max_speed = pts
        .iter()
        .map(|p| p.speed)
        .fold(f64::NEG_INFINITY, f64::max);

    let last = pts.last().unwrap();
    let first = pts.first().unwrap();

    let impact_speed = last.speed;
    let range = (last.position.x - first.position.x).abs();
    let flight_time = last.time - first.time;

    // Impact angle: angle of velocity vector below horizontal at the last
    // point. We use atan2(-vy, vx) so that a downward velocity gives a
    // positive angle.
    let impact_angle = (-last.velocity.y).atan2(last.velocity.x).abs();

    TrajectoryStats {
        max_height,
        range,
        flight_time,
        max_speed,
        impact_speed,
        impact_angle,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use simucad_core::types::Trajectory;

    /// Build a simple parabolic trajectory for testing.
    fn make_test_trajectory() -> Trajectory {
        let g = 9.806_65;
        let v0 = 100.0;
        let angle = std::f64::consts::FRAC_PI_4;
        let vx = v0 * angle.cos();
        let vy = v0 * angle.sin();
        let flight_time = 2.0 * vy / g;

        let n = 200;
        let mut points = Vec::with_capacity(n);
        let mut max_height: f64 = 0.0;

        for i in 0..n {
            let t = flight_time * (i as f64) / ((n - 1) as f64);
            let x = vx * t;
            let y = vy * t - 0.5 * g * t * t;
            let cur_vy = vy - g * t;
            let speed = (vx * vx + cur_vy * cur_vy).sqrt();
            max_height = max_height.max(y);

            points.push(TrajectoryPoint {
                time: t,
                position: Vec2::new(x, y),
                velocity: Vec2::new(vx, cur_vy),
                speed,
            });
        }

        Trajectory {
            points,
            max_height,
            range: vx * flight_time,
            flight_time,
        }
    }

    // ----- sample_trajectory tests -----

    #[test]
    fn sample_empty_trajectory() {
        let traj = Trajectory::empty();
        let samples = sample_trajectory(&traj, 10);
        assert!(samples.is_empty());
    }

    #[test]
    fn sample_zero_samples() {
        let traj = make_test_trajectory();
        let samples = sample_trajectory(&traj, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn sample_one_sample() {
        let traj = make_test_trajectory();
        let samples = sample_trajectory(&traj, 1);
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn sample_correct_count() {
        let traj = make_test_trajectory();
        for n in [2, 5, 10, 50, 100, 500] {
            let samples = sample_trajectory(&traj, n);
            assert_eq!(samples.len(), n, "expected {n} samples");
        }
    }

    #[test]
    fn sample_endpoints_match() {
        let traj = make_test_trajectory();
        let samples = sample_trajectory(&traj, 50);

        let first_orig = traj.points.first().unwrap();
        let last_orig = traj.points.last().unwrap();

        let first_sample = samples.first().unwrap();
        let last_sample = samples.last().unwrap();

        assert!((first_sample.time - first_orig.time).abs() < 1e-10);
        assert!((last_sample.time - last_orig.time).abs() < 1e-10);
        assert!((first_sample.position.x - first_orig.position.x).abs() < 1e-8);
        assert!((last_sample.position.x - last_orig.position.x).abs() < 1e-8);
    }

    #[test]
    fn sample_monotonic_time() {
        let traj = make_test_trajectory();
        let samples = sample_trajectory(&traj, 100);
        for window in samples.windows(2) {
            assert!(
                window[1].time >= window[0].time,
                "time should be monotonically increasing"
            );
        }
    }

    #[test]
    fn sample_speed_matches_velocity() {
        let traj = make_test_trajectory();
        let samples = sample_trajectory(&traj, 30);
        for pt in &samples {
            let computed_speed = pt.velocity.magnitude();
            assert!(
                (pt.speed - computed_speed).abs() < 1e-10,
                "speed {} != |velocity| {}",
                pt.speed,
                computed_speed
            );
        }
    }

    // ----- compute_derived_stats tests -----

    #[test]
    fn stats_empty_trajectory() {
        let traj = Trajectory::empty();
        let stats = compute_derived_stats(&traj);
        assert!((stats.max_height).abs() < 1e-12);
        assert!((stats.range).abs() < 1e-12);
        assert!((stats.flight_time).abs() < 1e-12);
        assert!((stats.max_speed).abs() < 1e-12);
        assert!((stats.impact_speed).abs() < 1e-12);
        assert!((stats.impact_angle).abs() < 1e-12);
    }

    #[test]
    fn stats_max_height() {
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);

        // Analytical max height for 45 deg, 100 m/s
        let expected = 100.0_f64.powi(2) * 0.5 / (2.0 * 9.806_65);
        assert!(
            (stats.max_height - expected).abs() < 1.0,
            "max_height {} vs expected {}",
            stats.max_height,
            expected
        );
    }

    #[test]
    fn stats_range() {
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);

        let expected = 100.0_f64.powi(2) / 9.806_65;
        assert!(
            (stats.range - expected).abs() < 1.0,
            "range {} vs expected {}",
            stats.range,
            expected
        );
    }

    #[test]
    fn stats_max_speed_at_least_v0() {
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);
        // For a trajectory starting at v0=100, max speed should be >= 100
        assert!(stats.max_speed >= 99.9);
    }

    #[test]
    fn stats_impact_speed_equals_launch_speed_in_vacuum() {
        // In vacuum from ground level, impact speed == launch speed
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);
        let launch_speed = traj.points.first().unwrap().speed;

        assert!(
            (stats.impact_speed - launch_speed).abs() < 1.0,
            "impact speed {} vs launch speed {}",
            stats.impact_speed,
            launch_speed
        );
    }

    #[test]
    fn stats_impact_angle_positive() {
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);
        assert!(
            stats.impact_angle > 0.0,
            "impact angle should be positive (below horizontal)"
        );
    }

    #[test]
    fn stats_impact_angle_symmetric_for_45_degrees() {
        // For a 45-degree launch from ground level, the impact angle should
        // also be ~45 degrees.
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);
        let expected = std::f64::consts::FRAC_PI_4;
        assert!(
            (stats.impact_angle - expected).abs() < 0.05,
            "impact angle {:.3} vs expected {:.3}",
            stats.impact_angle,
            expected
        );
    }

    #[test]
    fn stats_flight_time() {
        let traj = make_test_trajectory();
        let stats = compute_derived_stats(&traj);

        let v0 = 100.0;
        let vy = v0 * std::f64::consts::FRAC_PI_4.sin();
        let expected = 2.0 * vy / 9.806_65;
        assert!(
            (stats.flight_time - expected).abs() < 0.01,
            "flight_time {} vs expected {}",
            stats.flight_time,
            expected
        );
    }
}
