//! Parameter sweep engine for batch trajectory computation.
//!
//! Provides [`ParameterSweep`] for defining ranges of launch angle, velocity,
//! and mass, and [`execute_sweep`] for computing trajectories across the full
//! cartesian product of those ranges in parallel using [`rayon`].

use rayon::prelude::*;
use simucad_core::types::{SimulationConfig, Trajectory};

use crate::drag::DragModel;
use crate::integrator::{EulerIntegrator, Integrator, RK4Integrator};

// ---------------------------------------------------------------------------
// Sweep configuration types
// ---------------------------------------------------------------------------

/// Defines a parameter space to sweep over. Each range is optional; if
/// `None`, the corresponding parameter is taken from [`SweepConfig`]
/// defaults.
///
/// Ranges are specified as `(min, max, steps)` where `steps` is the number
/// of evenly-spaced samples including both endpoints.
#[derive(Debug, Clone)]
pub struct ParameterSweep {
    /// Range of launch angles in radians: (min, max, steps).
    pub angle_range: Option<(f64, f64, usize)>,
    /// Range of initial velocities in m/s: (min, max, steps).
    pub velocity_range: Option<(f64, f64, usize)>,
    /// Range of projectile masses in kg: (min, max, steps).
    pub mass_range: Option<(f64, f64, usize)>,
}

/// Summary parameters and results for a single trajectory in a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepParameters {
    /// Launch angle (radians).
    pub angle: f64,
    /// Initial speed (m/s).
    pub velocity: f64,
    /// Projectile mass (kg).
    pub mass: f64,
    /// Peak altitude reached (m).
    pub max_height: f64,
    /// Horizontal distance from launch to impact (m).
    pub range: f64,
    /// Total time of flight (s).
    pub flight_time: f64,
}

/// Aggregate result of a parameter sweep.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// Per-trajectory summary parameters.
    pub parameters: Vec<SweepParameters>,
    /// Full trajectory data for each parameter combination.
    pub trajectories: Vec<Trajectory>,
}

/// Which numerical integrator to use during the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorChoice {
    /// First-order symplectic Euler.
    Euler,
    /// Classic 4th-order Runge-Kutta.
    RK4,
}

/// Configuration for a parameter sweep that is constant across all
/// trajectory evaluations.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Gravitational acceleration magnitude (m/s^2).
    pub gravity: f64,
    /// Aerodynamic drag model.
    pub drag_model: DragModel,
    /// Initial launch height above ground (m).
    pub initial_height: f64,
    /// Simulation stepping configuration (timestep, max_steps).
    pub simulation_config: SimulationConfig,
    /// Integrator to use for each trajectory.
    pub integrator: IntegratorChoice,
    /// Default launch angle if not swept (radians).
    pub default_angle: f64,
    /// Default initial speed if not swept (m/s).
    pub default_velocity: f64,
    /// Default mass if not swept (kg).
    pub default_mass: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate `steps` evenly-spaced values from `min` to `max` inclusive.
/// Returns a single-element vec of `min` when `steps <= 1`.
fn linspace(min: f64, max: f64, steps: usize) -> Vec<f64> {
    if steps <= 1 {
        return vec![min];
    }
    (0..steps)
        .map(|i| min + (max - min) * (i as f64) / ((steps - 1) as f64))
        .collect()
}

/// Run a single trajectory with the given parameters, returning the
/// trajectory and summary statistics.
fn run_single(
    angle: f64,
    velocity: f64,
    mass: f64,
    config: &SweepConfig,
    integrator: &dyn Integrator,
) -> Option<(SweepParameters, Trajectory)> {
    use simucad_core::types::{KinematicState, TrajectoryPoint, Vec2};

    let vx = velocity * angle.cos();
    let vy = velocity * angle.sin();

    let mut state = KinematicState {
        time: 0.0,
        position: Vec2::new(0.0, config.initial_height),
        velocity: Vec2::new(vx, vy),
    };

    let gravity = config.gravity;
    let drag_model = &config.drag_model;

    let forces = |s: &KinematicState| -> Vec2 {
        let gravity_force = Vec2::new(0.0, -gravity);
        let drag_accel = drag_model.drag_force(s.velocity) * (1.0 / mass);
        gravity_force + drag_accel
    };

    let dt = config.simulation_config.timestep;
    let max_steps = config.simulation_config.max_steps;

    let mut points = Vec::new();
    let mut max_height: f64 = config.initial_height;

    let initial_speed = state.velocity.magnitude();
    points.push(TrajectoryPoint {
        time: state.time,
        position: state.position,
        velocity: state.velocity,
        speed: initial_speed,
    });

    for _step in 1..=max_steps {
        state = integrator.step(&state, &forces, dt);

        let speed = state.velocity.magnitude();
        max_height = max_height.max(state.position.y);

        points.push(TrajectoryPoint {
            time: state.time,
            position: state.position,
            velocity: state.velocity,
            speed,
        });

        if !state.position.x.is_finite() || !state.position.y.is_finite() {
            return None; // diverged -- skip this combination
        }

        if state.position.y < 0.0 {
            // Linear interpolation to ground impact
            let prev = &points[points.len() - 2];
            let curr = &points[points.len() - 1];
            let y_prev = prev.position.y;
            let y_curr = curr.position.y;

            if (y_curr - y_prev).abs() > f64::EPSILON {
                let frac = y_prev / (y_prev - y_curr);
                let t_impact = prev.time + frac * (curr.time - prev.time);
                let x_impact = prev.position.x + frac * (curr.position.x - prev.position.x);
                let vx_impact = prev.velocity.x + frac * (curr.velocity.x - prev.velocity.x);
                let vy_impact = prev.velocity.y + frac * (curr.velocity.y - prev.velocity.y);
                let impact_vel = Vec2::new(vx_impact, vy_impact);

                let last = points.last_mut().unwrap();
                last.time = t_impact;
                last.position = Vec2::new(x_impact, 0.0);
                last.velocity = impact_vel;
                last.speed = impact_vel.magnitude();
            }

            let range = points.last().unwrap().position.x;
            let flight_time = points.last().unwrap().time;

            let traj = Trajectory {
                points,
                max_height,
                range,
                flight_time,
            };

            let params = SweepParameters {
                angle,
                velocity,
                mass,
                max_height,
                range,
                flight_time,
            };

            return Some((params, traj));
        }
    }

    None // exceeded max steps
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Execute a parameter sweep across the cartesian product of all specified
/// ranges.
///
/// Trajectories are computed in parallel using rayon. Combinations that
/// diverge or exceed the maximum step count are silently omitted from the
/// result.
///
/// # Panics
///
/// Panics if any range specifies 0 steps.
pub fn execute_sweep(sweep: &ParameterSweep, base_config: &SweepConfig) -> SweepResult {
    let angles = match sweep.angle_range {
        Some((min, max, steps)) => {
            assert!(steps > 0, "angle_range steps must be > 0");
            linspace(min, max, steps)
        }
        None => vec![base_config.default_angle],
    };

    let velocities = match sweep.velocity_range {
        Some((min, max, steps)) => {
            assert!(steps > 0, "velocity_range steps must be > 0");
            linspace(min, max, steps)
        }
        None => vec![base_config.default_velocity],
    };

    let masses = match sweep.mass_range {
        Some((min, max, steps)) => {
            assert!(steps > 0, "mass_range steps must be > 0");
            linspace(min, max, steps)
        }
        None => vec![base_config.default_mass],
    };

    // Build the cartesian product of all parameter combinations
    let mut combos: Vec<(f64, f64, f64)> =
        Vec::with_capacity(angles.len() * velocities.len() * masses.len());
    for &a in &angles {
        for &v in &velocities {
            for &m in &masses {
                combos.push((a, v, m));
            }
        }
    }

    // Select the integrator
    let integrator: Box<dyn Integrator> = match base_config.integrator {
        IntegratorChoice::Euler => Box::new(EulerIntegrator::new()),
        IntegratorChoice::RK4 => Box::new(RK4Integrator::new()),
    };

    // Run all trajectories in parallel
    let results: Vec<(SweepParameters, Trajectory)> = combos
        .par_iter()
        .filter_map(|&(angle, velocity, mass)| {
            run_single(angle, velocity, mass, base_config, integrator.as_ref())
        })
        .collect();

    let (parameters, trajectories) = results.into_iter().unzip();

    SweepResult {
        parameters,
        trajectories,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drag::{DragModel, DragShape};
    use simucad_core::constants::STANDARD_GRAVITY;
    use std::f64::consts::FRAC_PI_4;

    fn default_sweep_config() -> SweepConfig {
        SweepConfig {
            gravity: STANDARD_GRAVITY,
            drag_model: DragModel::at_sea_level(DragShape::Sphere, 0.01),
            initial_height: 0.0,
            simulation_config: SimulationConfig {
                timestep: 0.01,
                max_steps: 100_000,
                ..SimulationConfig::default()
            },
            integrator: IntegratorChoice::Euler,
            default_angle: FRAC_PI_4,
            default_velocity: 100.0,
            default_mass: 1.0,
        }
    }

    #[test]
    fn sweep_single_point() {
        let sweep = ParameterSweep {
            angle_range: None,
            velocity_range: None,
            mass_range: None,
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        assert_eq!(result.parameters.len(), 1);
        assert_eq!(result.trajectories.len(), 1);
        assert!((result.parameters[0].angle - FRAC_PI_4).abs() < 1e-12);
        assert!((result.parameters[0].velocity - 100.0).abs() < 1e-12);
        assert!((result.parameters[0].mass - 1.0).abs() < 1e-12);
        assert!(result.parameters[0].range > 0.0);
        assert!(result.parameters[0].flight_time > 0.0);
    }

    #[test]
    fn sweep_angle_range_correct_count() {
        let sweep = ParameterSweep {
            angle_range: Some((0.2, 1.2, 11)),
            velocity_range: None,
            mass_range: None,
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        assert_eq!(result.parameters.len(), 11);
        assert_eq!(result.trajectories.len(), 11);
    }

    #[test]
    fn sweep_cartesian_product_count() {
        let sweep = ParameterSweep {
            angle_range: Some((0.3, 1.0, 3)),
            velocity_range: Some((50.0, 150.0, 4)),
            mass_range: Some((0.5, 2.0, 2)),
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        // 3 * 4 * 2 = 24 combinations
        assert_eq!(result.parameters.len(), 24);
        assert_eq!(result.trajectories.len(), 24);
    }

    #[test]
    fn sweep_higher_velocity_goes_further() {
        let sweep = ParameterSweep {
            angle_range: None,
            velocity_range: Some((50.0, 200.0, 5)),
            mass_range: None,
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        // Range should monotonically increase with velocity (all at 45 deg)
        for window in result.parameters.windows(2) {
            assert!(
                window[1].range > window[0].range,
                "range should increase with velocity: v={} range={} vs v={} range={}",
                window[0].velocity,
                window[0].range,
                window[1].velocity,
                window[1].range,
            );
        }
    }

    #[test]
    fn sweep_heavier_mass_goes_further_with_drag() {
        let sweep = ParameterSweep {
            angle_range: None,
            velocity_range: None,
            mass_range: Some((0.1, 10.0, 5)),
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        // Heavier projectiles are less affected by drag => go further
        for window in result.parameters.windows(2) {
            assert!(
                window[1].range > window[0].range,
                "range should increase with mass: m={} range={} vs m={} range={}",
                window[0].mass,
                window[0].range,
                window[1].mass,
                window[1].range,
            );
        }
    }

    #[test]
    fn sweep_parallel_matches_sequential() {
        let sweep = ParameterSweep {
            angle_range: Some((0.3, 1.2, 5)),
            velocity_range: Some((50.0, 150.0, 3)),
            mass_range: None,
        };
        let config = default_sweep_config();

        // Run the parallel sweep
        let parallel_result = execute_sweep(&sweep, &config);

        // Run sequential using run_single directly
        let angles = linspace(0.3, 1.2, 5);
        let velocities = linspace(50.0, 150.0, 3);
        let integrator = EulerIntegrator::new();

        let mut sequential_params = Vec::new();
        for &a in &angles {
            for &v in &velocities {
                if let Some((params, _traj)) =
                    run_single(a, v, config.default_mass, &config, &integrator)
                {
                    sequential_params.push(params);
                }
            }
        }

        assert_eq!(
            parallel_result.parameters.len(),
            sequential_params.len(),
            "count mismatch"
        );

        // Sort both by (angle, velocity) for deterministic comparison
        let mut par_sorted = parallel_result.parameters.clone();
        par_sorted.sort_by(|a, b| {
            a.angle
                .partial_cmp(&b.angle)
                .unwrap()
                .then(a.velocity.partial_cmp(&b.velocity).unwrap())
        });
        let mut seq_sorted = sequential_params.clone();
        seq_sorted.sort_by(|a, b| {
            a.angle
                .partial_cmp(&b.angle)
                .unwrap()
                .then(a.velocity.partial_cmp(&b.velocity).unwrap())
        });

        for (p, s) in par_sorted.iter().zip(seq_sorted.iter()) {
            assert!(
                (p.range - s.range).abs() < 1e-6,
                "range mismatch: parallel {} vs sequential {} for angle={} vel={}",
                p.range,
                s.range,
                p.angle,
                p.velocity,
            );
            assert!(
                (p.flight_time - s.flight_time).abs() < 1e-6,
                "flight_time mismatch"
            );
            assert!(
                (p.max_height - s.max_height).abs() < 1e-6,
                "max_height mismatch"
            );
        }
    }

    #[test]
    fn sweep_rk4_integrator() {
        let sweep = ParameterSweep {
            angle_range: Some((0.5, 1.0, 3)),
            velocity_range: None,
            mass_range: None,
        };
        let mut config = default_sweep_config();
        config.integrator = IntegratorChoice::RK4;
        let result = execute_sweep(&sweep, &config);

        assert_eq!(result.parameters.len(), 3);
        // All trajectories should have landed
        for traj in &result.trajectories {
            let last = traj.points.last().unwrap();
            assert!(
                last.position.y.abs() < 0.1,
                "trajectory should end near ground, got y={}",
                last.position.y
            );
        }
    }

    #[test]
    fn sweep_vacuum_matches_analytical_range() {
        // With no drag, the sweep results should closely match the
        // analytical range formula: R = v0^2 * sin(2*theta) / g
        let sweep = ParameterSweep {
            angle_range: Some((0.3, 1.2, 5)),
            velocity_range: None,
            mass_range: None,
        };
        let mut config = default_sweep_config();
        config.drag_model = DragModel::vacuum();
        config.simulation_config.timestep = 0.001;
        config.simulation_config.max_steps = 500_000;

        let result = execute_sweep(&sweep, &config);

        for p in &result.parameters {
            let analytical_range =
                p.velocity * p.velocity * (2.0 * p.angle).sin() / config.gravity;
            let rel_err = (p.range - analytical_range).abs() / analytical_range;
            assert!(
                rel_err < 0.02,
                "range {} vs analytical {} (rel err {}) for angle={}",
                p.range,
                analytical_range,
                rel_err,
                p.angle,
            );
        }
    }

    #[test]
    fn sweep_trajectories_have_points() {
        let sweep = ParameterSweep {
            angle_range: Some((0.5, 1.0, 3)),
            velocity_range: Some((80.0, 120.0, 2)),
            mass_range: None,
        };
        let config = default_sweep_config();
        let result = execute_sweep(&sweep, &config);

        for traj in &result.trajectories {
            assert!(
                traj.points.len() >= 2,
                "trajectory should have at least 2 points"
            );
            assert!(traj.flight_time > 0.0);
            assert!(traj.max_height > 0.0);
        }
    }

    #[test]
    fn linspace_single_step() {
        let vals = linspace(1.0, 5.0, 1);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn linspace_two_steps() {
        let vals = linspace(0.0, 10.0, 2);
        assert_eq!(vals.len(), 2);
        assert!((vals[0]).abs() < 1e-12);
        assert!((vals[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn linspace_five_steps() {
        let vals = linspace(0.0, 1.0, 5);
        assert_eq!(vals.len(), 5);
        for (i, &v) in vals.iter().enumerate() {
            let expected = i as f64 * 0.25;
            assert!(
                (v - expected).abs() < 1e-12,
                "linspace[{i}] = {v}, expected {expected}"
            );
        }
    }
}
