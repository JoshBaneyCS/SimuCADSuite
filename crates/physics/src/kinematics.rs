//! Trajectory solvers for projectile motion.
//!
//! Two solvers are provided:
//!
//! - [`vacuum_trajectory`]: closed-form analytical solution for projectile
//!   motion without air resistance.
//! - [`drag_trajectory`]: numerical (Euler integration) solution that
//!   includes aerodynamic drag via a [`DragModel`].

use simucad_core::error::PhysicsError;
use simucad_core::types::{KinematicState, SimulationConfig, Trajectory, TrajectoryPoint, Vec2};

use crate::drag::DragModel;
use crate::integrator::Integrator;

// ---------------------------------------------------------------------------
// Vacuum trajectory (analytical)
// ---------------------------------------------------------------------------

/// Compute the trajectory of a projectile in a vacuum using the closed-form
/// kinematic equations.
///
/// # Parameters
///
/// - `v0` -- initial speed (m/s), must be positive.
/// - `angle_rad` -- launch angle above horizontal (radians).
/// - `gravity` -- gravitational acceleration magnitude (m/s^2), must be positive.
/// - `initial_height` -- launch height above ground (m), must be non-negative.
/// - `num_points` -- number of trajectory sample points to generate.
///
/// # Returns
///
/// A [`Trajectory`] containing evenly-spaced time samples from launch to
/// impact, along with computed max height, range, and flight time.
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if any parameter is out of
/// its valid domain.
pub fn vacuum_trajectory(
    v0: f64,
    angle_rad: f64,
    gravity: f64,
    initial_height: f64,
    num_points: usize,
) -> Result<Trajectory, PhysicsError> {
    // --- Input validation ---
    if v0 <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "v0",
            value: v0,
            reason: "initial speed must be positive",
        });
    }
    if gravity <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "gravity",
            value: gravity,
            reason: "gravitational acceleration must be positive",
        });
    }
    if initial_height < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "initial_height",
            value: initial_height,
            reason: "initial height must be non-negative",
        });
    }
    if num_points < 2 {
        return Err(PhysicsError::InvalidParameter {
            field: "num_points",
            value: num_points as f64,
            reason: "need at least 2 points to define a trajectory",
        });
    }

    let vx = v0 * angle_rad.cos();
    let vy = v0 * angle_rad.sin();

    // Flight time from quadratic: y(t) = initial_height + vy*t - 0.5*g*t^2 = 0
    // => 0.5*g*t^2 - vy*t - initial_height = 0
    // => t = (vy + sqrt(vy^2 + 2*g*h0)) / g
    let discriminant = vy * vy + 2.0 * gravity * initial_height;
    let flight_time = (vy + discriminant.sqrt()) / gravity;

    // Max height: occurs at t_peak = vy / g (only if vy > 0)
    let t_peak = if vy > 0.0 { vy / gravity } else { 0.0 };
    let max_height = initial_height + vy * t_peak - 0.5 * gravity * t_peak * t_peak;

    // Range = vx * flight_time
    let range = vx * flight_time;

    // Generate evenly spaced time samples
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let t = flight_time * (i as f64) / ((num_points - 1) as f64);
        let x = vx * t;
        let y = initial_height + vy * t - 0.5 * gravity * t * t;
        let cur_vx = vx;
        let cur_vy = vy - gravity * t;
        let speed = (cur_vx * cur_vx + cur_vy * cur_vy).sqrt();

        points.push(TrajectoryPoint {
            time: t,
            position: Vec2::new(x, y),
            velocity: Vec2::new(cur_vx, cur_vy),
            speed,
        });
    }

    Ok(Trajectory {
        points,
        max_height,
        range,
        flight_time,
    })
}

// ---------------------------------------------------------------------------
// Drag trajectory (numerical)
// ---------------------------------------------------------------------------

/// Compute the trajectory of a projectile subject to aerodynamic drag using
/// numerical integration.
///
/// # Parameters
///
/// - `v0` -- initial speed (m/s), must be positive.
/// - `angle_rad` -- launch angle above horizontal (radians).
/// - `gravity` -- gravitational acceleration magnitude (m/s^2), must be positive.
/// - `drag_model` -- aerodynamic drag parameters.
/// - `mass` -- projectile mass (kg), must be positive.
/// - `initial_height` -- launch height above ground (m), must be non-negative.
/// - `config` -- simulation configuration (timestep, max steps).
/// - `integrator` -- the numerical integrator to use (Euler, RK4, etc.).
///
/// # Returns
///
/// A [`Trajectory`] populated with one point per integration step,
/// terminated when the projectile returns to ground level (y < 0).
///
/// # Errors
///
/// Returns [`PhysicsError`] variants for invalid parameters or if the
/// simulation exceeds `config.max_steps` without the projectile landing.
pub fn drag_trajectory(
    v0: f64,
    angle_rad: f64,
    gravity: f64,
    drag_model: &DragModel,
    mass: f64,
    initial_height: f64,
    config: &SimulationConfig,
    integrator: &dyn Integrator,
) -> Result<Trajectory, PhysicsError> {
    // --- Input validation ---
    if v0 <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "v0",
            value: v0,
            reason: "initial speed must be positive",
        });
    }
    if gravity <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "gravity",
            value: gravity,
            reason: "gravitational acceleration must be positive",
        });
    }
    if mass <= 0.0 {
        return Err(PhysicsError::NegativeMass(mass));
    }
    if drag_model.cross_section_area < 0.0 {
        return Err(PhysicsError::NegativeArea(drag_model.cross_section_area));
    }
    if initial_height < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "initial_height",
            value: initial_height,
            reason: "initial height must be non-negative",
        });
    }
    if config.timestep <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            field: "timestep",
            value: config.timestep,
            reason: "timestep must be positive",
        });
    }

    let vx = v0 * angle_rad.cos();
    let vy = v0 * angle_rad.sin();

    let mut state = KinematicState {
        time: 0.0,
        position: Vec2::new(0.0, initial_height),
        velocity: Vec2::new(vx, vy),
    };

    let mut points = Vec::new();
    let mut max_height: f64 = initial_height;

    // Record the initial point
    let initial_speed = state.velocity.magnitude();
    points.push(TrajectoryPoint {
        time: state.time,
        position: state.position,
        velocity: state.velocity,
        speed: initial_speed,
    });

    // Force function: gravity + drag / mass
    // We capture drag_model and mass by reference, gravity by value.
    let forces = |s: &KinematicState| -> Vec2 {
        let gravity_force = Vec2::new(0.0, -gravity);
        let drag_accel = drag_model.drag_force(s.velocity) * (1.0 / mass);
        gravity_force + drag_accel
    };

    let dt = config.timestep;

    for step in 1..=config.max_steps {
        state = integrator.step(&state, &forces, dt);

        let speed = state.velocity.magnitude();
        max_height = max_height.max(state.position.y);

        points.push(TrajectoryPoint {
            time: state.time,
            position: state.position,
            velocity: state.velocity,
            speed,
        });

        // Check for divergence (NaN or Inf)
        if !state.position.x.is_finite() || !state.position.y.is_finite() {
            return Err(PhysicsError::Divergence {
                step,
                detail: format!(
                    "position became non-finite: ({}, {})",
                    state.position.x, state.position.y
                ),
            });
        }

        // Terminate when projectile hits ground
        if state.position.y < 0.0 {
            // Linearly interpolate to find the exact ground-impact point
            let prev = &points[points.len() - 2];
            let curr = &points[points.len() - 1];
            let y_prev = prev.position.y;
            let y_curr = curr.position.y;

            if (y_curr - y_prev).abs() > f64::EPSILON {
                let frac = y_prev / (y_prev - y_curr);
                let t_impact = prev.time + frac * (curr.time - prev.time);
                let x_impact = prev.position.x + frac * (curr.position.x - prev.position.x);
                let vx_impact =
                    prev.velocity.x + frac * (curr.velocity.x - prev.velocity.x);
                let vy_impact =
                    prev.velocity.y + frac * (curr.velocity.y - prev.velocity.y);
                let impact_vel = Vec2::new(vx_impact, vy_impact);

                // Replace the last point with the interpolated impact point
                let last = points.last_mut().unwrap();
                last.time = t_impact;
                last.position = Vec2::new(x_impact, 0.0);
                last.velocity = impact_vel;
                last.speed = impact_vel.magnitude();
            }

            let range = points.last().unwrap().position.x;
            let flight_time = points.last().unwrap().time;
            return Ok(Trajectory {
                points,
                max_height,
                range,
                flight_time,
            });
        }
    }

    Err(PhysicsError::MaxStepsExceeded {
        max_steps: config.max_steps,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drag::{DragModel, DragShape};
    use crate::integrator::{EulerIntegrator, RK4Integrator};
    use simucad_core::constants::STANDARD_GRAVITY;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    const G: f64 = STANDARD_GRAVITY;
    const TOL: f64 = 1e-6;

    // ----- vacuum_trajectory tests -----

    #[test]
    fn vacuum_45_degrees_from_ground() {
        let v0 = 100.0;
        let traj = vacuum_trajectory(v0, FRAC_PI_4, G, 0.0, 500).unwrap();

        // Analytical range = v0^2 * sin(2*theta) / g = v0^2 / g for 45 deg
        let expected_range = v0 * v0 / G;
        assert!(
            (traj.range - expected_range).abs() < 0.01,
            "range: {} vs expected: {}",
            traj.range,
            expected_range
        );

        // Analytical max height = v0^2 * sin^2(45) / (2g)
        let expected_height = v0 * v0 * 0.5 / (2.0 * G);
        assert!(
            (traj.max_height - expected_height).abs() < 0.01,
            "max_height: {} vs expected: {}",
            traj.max_height,
            expected_height
        );

        // Flight time = 2 * v0 * sin(45) / g
        let expected_time = 2.0 * v0 * FRAC_PI_4.sin() / G;
        assert!((traj.flight_time - expected_time).abs() < 0.001);

        // First point at origin, last point at ground level
        assert!((traj.points.first().unwrap().position.x).abs() < TOL);
        assert!((traj.points.first().unwrap().position.y).abs() < TOL);
    }

    #[test]
    fn vacuum_vertical_launch() {
        let v0 = 50.0;
        let traj = vacuum_trajectory(v0, FRAC_PI_2, G, 0.0, 100).unwrap();

        // Range should be ~0 for vertical launch
        assert!(traj.range.abs() < 0.01);

        // Max height = v0^2 / (2g)
        let expected_height = v0 * v0 / (2.0 * G);
        assert!((traj.max_height - expected_height).abs() < 0.01);

        // Flight time = 2*v0/g
        let expected_time = 2.0 * v0 / G;
        assert!((traj.flight_time - expected_time).abs() < 0.001);
    }

    #[test]
    fn vacuum_with_initial_height() {
        let v0 = 20.0;
        let h0 = 50.0;
        let traj = vacuum_trajectory(v0, 0.0, G, h0, 200).unwrap();

        // Horizontal launch from height: flight_time = sqrt(2*h0/g)
        let expected_time = (2.0 * h0 / G).sqrt();
        assert!((traj.flight_time - expected_time).abs() < 0.001);

        // Range = v0 * flight_time
        let expected_range = v0 * expected_time;
        assert!((traj.range - expected_range).abs() < 0.01);

        // Max height = initial height (horizontal launch)
        assert!((traj.max_height - h0).abs() < TOL);
    }

    #[test]
    fn vacuum_invalid_velocity() {
        let result = vacuum_trajectory(0.0, FRAC_PI_4, G, 0.0, 100);
        assert!(result.is_err());

        let result = vacuum_trajectory(-10.0, FRAC_PI_4, G, 0.0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn vacuum_invalid_gravity() {
        let result = vacuum_trajectory(10.0, FRAC_PI_4, 0.0, 0.0, 100);
        assert!(result.is_err());

        let result = vacuum_trajectory(10.0, FRAC_PI_4, -9.8, 0.0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn vacuum_invalid_height() {
        let result = vacuum_trajectory(10.0, FRAC_PI_4, G, -5.0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn vacuum_too_few_points() {
        let result = vacuum_trajectory(10.0, FRAC_PI_4, G, 0.0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn vacuum_point_count() {
        let n = 250;
        let traj = vacuum_trajectory(50.0, FRAC_PI_4, G, 0.0, n).unwrap();
        assert_eq!(traj.points.len(), n);
    }

    #[test]
    fn vacuum_monotonic_time() {
        let traj = vacuum_trajectory(50.0, FRAC_PI_4, G, 10.0, 500).unwrap();
        for window in traj.points.windows(2) {
            assert!(window[1].time > window[0].time);
        }
    }

    // ----- drag_trajectory tests -----

    #[test]
    fn drag_reduces_range() {
        let v0 = 100.0;
        let angle = FRAC_PI_4;
        let config = SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };

        let vacuum = vacuum_trajectory(v0, angle, G, 0.0, 1000).unwrap();

        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let mass = 1.0;
        let with_drag =
            drag_trajectory(v0, angle, G, &drag_model, mass, 0.0, &config, &EulerIntegrator::new()).unwrap();

        assert!(
            with_drag.range < vacuum.range,
            "drag range ({}) should be less than vacuum range ({})",
            with_drag.range,
            vacuum.range
        );
        assert!(
            with_drag.flight_time < vacuum.flight_time,
            "drag flight time ({}) should be less than vacuum flight time ({})",
            with_drag.flight_time,
            vacuum.flight_time
        );
    }

    #[test]
    fn drag_trajectory_terminates_at_ground() {
        let config = SimulationConfig {
            timestep: 0.01,
            max_steps: 100_000,
            ..SimulationConfig::default()
        };
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.005);
        let traj = drag_trajectory(50.0, FRAC_PI_4, G, &drag_model, 0.5, 0.0, &config, &EulerIntegrator::new()).unwrap();

        // Last point should be at or very near y=0
        let last = traj.points.last().unwrap();
        assert!(
            last.position.y.abs() < 0.01,
            "last y = {}",
            last.position.y
        );
    }

    #[test]
    fn drag_with_initial_height() {
        let config = SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let traj = drag_trajectory(30.0, 0.0, G, &drag_model, 1.0, 100.0, &config, &EulerIntegrator::new()).unwrap();

        // Should have positive range
        assert!(traj.range > 0.0);
        // Max height should be >= initial height (horizontal launch)
        assert!(traj.max_height >= 99.9);
    }

    #[test]
    fn drag_invalid_mass() {
        let config = SimulationConfig::default();
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);

        let result = drag_trajectory(50.0, FRAC_PI_4, G, &drag_model, 0.0, 0.0, &config, &EulerIntegrator::new());
        assert!(result.is_err());

        let result = drag_trajectory(50.0, FRAC_PI_4, G, &drag_model, -1.0, 0.0, &config, &EulerIntegrator::new());
        assert!(result.is_err());
    }

    #[test]
    fn drag_invalid_area() {
        let config = SimulationConfig::default();
        let drag_model = DragModel::new(DragShape::Sphere, -0.01, 1.225);
        let result = drag_trajectory(50.0, FRAC_PI_4, G, &drag_model, 1.0, 0.0, &config, &EulerIntegrator::new());
        assert!(result.is_err());
    }

    #[test]
    fn drag_max_steps_exceeded() {
        let config = SimulationConfig {
            timestep: 0.0001,
            max_steps: 10, // very few steps
            ..SimulationConfig::default()
        };
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let result = drag_trajectory(100.0, FRAC_PI_4, G, &drag_model, 1.0, 0.0, &config, &EulerIntegrator::new());
        assert!(matches!(
            result,
            Err(PhysicsError::MaxStepsExceeded { .. })
        ));
    }

    #[test]
    fn drag_heavier_projectile_goes_further() {
        let config = SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);

        let euler = EulerIntegrator::new();
        let light =
            drag_trajectory(100.0, FRAC_PI_4, G, &drag_model, 0.1, 0.0, &config, &euler).unwrap();
        let heavy =
            drag_trajectory(100.0, FRAC_PI_4, G, &drag_model, 10.0, 0.0, &config, &euler).unwrap();

        assert!(
            heavy.range > light.range,
            "heavy range ({}) should exceed light range ({})",
            heavy.range,
            light.range
        );
    }

    #[test]
    fn rk4_more_accurate_than_euler_for_vacuum() {
        // RK4 with a larger timestep should still be more accurate than Euler
        let v0 = 80.0;
        let angle = FRAC_PI_4;
        let analytical = vacuum_trajectory(v0, angle, G, 0.0, 1000).unwrap();

        let vacuum_model = DragModel::vacuum();

        // Euler with dt=0.01
        let euler_config = SimulationConfig {
            timestep: 0.01,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };
        let euler_traj =
            drag_trajectory(v0, angle, G, &vacuum_model, 1.0, 0.0, &euler_config, &EulerIntegrator::new()).unwrap();

        // RK4 with same dt=0.01
        let rk4_traj =
            drag_trajectory(v0, angle, G, &vacuum_model, 1.0, 0.0, &euler_config, &RK4Integrator).unwrap();

        let euler_err = (euler_traj.range - analytical.range).abs();
        let rk4_err = (rk4_traj.range - analytical.range).abs();

        assert!(
            rk4_err < euler_err,
            "RK4 error ({:.6}) should be smaller than Euler error ({:.6})",
            rk4_err,
            euler_err
        );
    }

    #[test]
    fn rk4_drag_trajectory_terminates() {
        let config = SimulationConfig {
            timestep: 0.01,
            max_steps: 100_000,
            ..SimulationConfig::default()
        };
        let drag_model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let traj = drag_trajectory(80.0, FRAC_PI_4, G, &drag_model, 1.0, 0.0, &config, &RK4Integrator).unwrap();

        let last = traj.points.last().unwrap();
        assert!(last.position.y.abs() < 0.05, "last y = {}", last.position.y);
        assert!(traj.range > 0.0);
    }

    #[test]
    fn drag_vacuum_model_matches_analytical() {
        // Using a vacuum DragModel should give results close to analytical
        let v0 = 80.0;
        let angle = FRAC_PI_4;
        let config = SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };

        let vacuum_model = DragModel::vacuum();
        let numerical =
            drag_trajectory(v0, angle, G, &vacuum_model, 1.0, 0.0, &config, &EulerIntegrator::new()).unwrap();
        let analytical = vacuum_trajectory(v0, angle, G, 0.0, 1000).unwrap();

        // Should agree within ~1%
        assert!(
            (numerical.range - analytical.range).abs() / analytical.range < 0.01,
            "numerical range {} vs analytical range {}",
            numerical.range,
            analytical.range
        );
    }
}
