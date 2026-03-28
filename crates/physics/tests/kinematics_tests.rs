//! End-to-end integration tests for the kinematics and trajectory subsystem.
//!
//! These tests exercise cross-module workflows: computing trajectories,
//! sampling them, deriving statistics, running parameter sweeps, comparing
//! integrators, and exporting to CSV via DataTable.

use std::f64::consts::FRAC_PI_4;

use simucad_core::constants::STANDARD_GRAVITY;
use simucad_core::types::SimulationConfig;
use simucad_physics::drag::{DragModel, DragShape};
use simucad_physics::integrator::EulerIntegrator;
use simucad_physics::kinematics::{drag_trajectory, vacuum_trajectory};
use simucad_physics::sweep::{
    execute_sweep, IntegratorChoice, ParameterSweep, SweepConfig,
};
use simucad_physics::trajectory::{compute_derived_stats, sample_trajectory};

const G: f64 = STANDARD_GRAVITY;

// ---------------------------------------------------------------------------
// 1. Vacuum trajectory -> sample -> stats -> verify analytical values
// ---------------------------------------------------------------------------

#[test]
fn vacuum_trajectory_sample_and_stats() {
    let v0 = 100.0;
    let angle = FRAC_PI_4;

    // Compute the trajectory
    let traj = vacuum_trajectory(v0, angle, G, 0.0, 500).unwrap();

    // Sample it down to 50 evenly spaced points
    let samples = sample_trajectory(&traj, 50);
    assert_eq!(samples.len(), 50);

    // Verify the first sample is near origin, last is near ground
    assert!(samples.first().unwrap().position.x.abs() < 0.1);
    assert!(samples.first().unwrap().position.y.abs() < 0.1);
    assert!(samples.last().unwrap().position.y.abs() < 1.0);

    // Compute derived stats
    let stats = compute_derived_stats(&traj);

    // Analytical values for 45-degree launch from ground
    let expected_range = v0 * v0 / G; // ~1019.7 m
    let expected_max_height = v0 * v0 * 0.5 / (2.0 * G); // ~254.9 m
    let expected_flight_time = 2.0 * v0 * FRAC_PI_4.sin() / G;

    assert!(
        (stats.range - expected_range).abs() < 1.0,
        "range: {} vs expected: {}",
        stats.range,
        expected_range,
    );
    assert!(
        (stats.max_height - expected_max_height).abs() < 1.0,
        "max_height: {} vs expected: {}",
        stats.max_height,
        expected_max_height,
    );
    assert!(
        (stats.flight_time - expected_flight_time).abs() < 0.1,
        "flight_time: {} vs expected: {}",
        stats.flight_time,
        expected_flight_time,
    );

    // Impact speed should equal launch speed in vacuum (conservation of energy)
    assert!(
        (stats.impact_speed - v0).abs() < 1.0,
        "impact_speed: {} vs v0: {}",
        stats.impact_speed,
        v0,
    );

    // Impact angle should be ~45 degrees for symmetric trajectory
    assert!(
        (stats.impact_angle - FRAC_PI_4).abs() < 0.1,
        "impact_angle: {} vs expected: {}",
        stats.impact_angle,
        FRAC_PI_4,
    );
}

// ---------------------------------------------------------------------------
// 2. Drag trajectory with sphere -> verify range < vacuum range
// ---------------------------------------------------------------------------

#[test]
fn drag_trajectory_reduces_range_compared_to_vacuum() {
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

    // Drag should reduce range
    assert!(
        with_drag.range < vacuum.range,
        "drag range ({}) should be < vacuum range ({})",
        with_drag.range,
        vacuum.range,
    );

    // Drag should reduce max height
    assert!(
        with_drag.max_height < vacuum.max_height,
        "drag max_height ({}) should be < vacuum max_height ({})",
        with_drag.max_height,
        vacuum.max_height,
    );

    // Stats should also reflect the reduction
    let drag_stats = compute_derived_stats(&with_drag);
    let vacuum_stats = compute_derived_stats(&vacuum);
    assert!(drag_stats.range < vacuum_stats.range);
    assert!(drag_stats.flight_time < vacuum_stats.flight_time);
}

// ---------------------------------------------------------------------------
// 3. Parameter sweep across angles 15-75 degrees, max range at ~45 degrees
// ---------------------------------------------------------------------------

#[test]
fn parameter_sweep_max_range_near_45_degrees() {
    let sweep = ParameterSweep {
        angle_range: Some((
            15.0_f64.to_radians(),
            75.0_f64.to_radians(),
            13, // 5-degree steps: (75-15)/5 + 1 = 13
        )),
        velocity_range: None,
        mass_range: None,
    };

    // Use vacuum drag model so analytical result applies
    let config = SweepConfig {
        gravity: G,
        drag_model: DragModel::vacuum(),
        initial_height: 0.0,
        simulation_config: SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        },
        integrator: IntegratorChoice::Euler,
        default_angle: FRAC_PI_4,
        default_velocity: 100.0,
        default_mass: 1.0,
    };

    let result = execute_sweep(&sweep, &config);

    assert_eq!(result.parameters.len(), 13);
    assert_eq!(result.trajectories.len(), 13);

    // Find the angle that gives max range
    let best = result
        .parameters
        .iter()
        .max_by(|a, b| a.range.partial_cmp(&b.range).unwrap())
        .unwrap();

    // In vacuum, max range occurs at exactly 45 degrees
    let best_angle_deg = best.angle.to_degrees();
    assert!(
        (best_angle_deg - 45.0).abs() < 6.0, // within one step of 45
        "best angle: {} degrees (expected ~45)",
        best_angle_deg,
    );

    // Verify all trajectories have positive range and flight time
    for p in &result.parameters {
        assert!(p.range > 0.0);
        assert!(p.flight_time > 0.0);
        assert!(p.max_height > 0.0);
    }
}

// ---------------------------------------------------------------------------
// 4. RK4 integrator produces more accurate results than Euler for same dt
// ---------------------------------------------------------------------------

#[test]
fn rk4_more_accurate_than_euler_in_sweep() {
    // Use a coarse timestep where Euler error is noticeable
    let dt = 0.05;

    let sweep = ParameterSweep {
        angle_range: None,
        velocity_range: None,
        mass_range: None,
    };

    let base_config = SweepConfig {
        gravity: G,
        drag_model: DragModel::vacuum(),
        initial_height: 0.0,
        simulation_config: SimulationConfig {
            timestep: dt,
            max_steps: 500_000,
            ..SimulationConfig::default()
        },
        integrator: IntegratorChoice::Euler,
        default_angle: FRAC_PI_4,
        default_velocity: 100.0,
        default_mass: 1.0,
    };

    let euler_result = execute_sweep(&sweep, &base_config);

    let mut rk4_config = base_config.clone();
    rk4_config.integrator = IntegratorChoice::RK4;
    let rk4_result = execute_sweep(&sweep, &rk4_config);

    assert_eq!(euler_result.parameters.len(), 1);
    assert_eq!(rk4_result.parameters.len(), 1);

    // Analytical range for vacuum at 45 degrees
    let v0 = 100.0;
    let analytical_range = v0 * v0 * (2.0 * FRAC_PI_4).sin() / G;

    let euler_error = (euler_result.parameters[0].range - analytical_range).abs();
    let rk4_error = (rk4_result.parameters[0].range - analytical_range).abs();

    assert!(
        rk4_error < euler_error,
        "RK4 error ({}) should be less than Euler error ({})",
        rk4_error,
        euler_error,
    );

    // RK4 should be very close for this constant-acceleration problem
    let rk4_rel_error = rk4_error / analytical_range;
    assert!(
        rk4_rel_error < 0.001,
        "RK4 relative error ({}) should be < 0.1%",
        rk4_rel_error,
    );
}

// ---------------------------------------------------------------------------
// 5. Export trajectory to DataTable, verify CSV output has correct columns
// ---------------------------------------------------------------------------

#[test]
fn trajectory_to_csv_export() {
    let traj = vacuum_trajectory(50.0, FRAC_PI_4, G, 0.0, 100).unwrap();

    let table = simucad_core::export::trajectory_to_table(&traj, "Test Export");
    assert_eq!(table.title, "Test Export");
    assert_eq!(table.columns.len(), 6);
    assert_eq!(table.row_count(), 100);

    // Verify column names
    let col_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(col_names, vec!["time", "x", "y", "vx", "vy", "speed"]);

    // Verify units
    let col_units: Vec<&str> = table.columns.iter().map(|c| c.unit.as_str()).collect();
    assert_eq!(col_units, vec!["s", "m", "m", "m/s", "m/s", "m/s"]);

    // Generate CSV and verify structure
    let csv = table.to_csv().unwrap();
    let lines: Vec<&str> = csv.lines().collect();

    // Header + 100 data rows
    assert_eq!(lines.len(), 101);

    // Header should contain column names with units
    assert!(lines[0].contains("time (s)"));
    assert!(lines[0].contains("x (m)"));
    assert!(lines[0].contains("speed (m/s)"));

    // Each data row should have 6 comma-separated values
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            6,
            "expected 6 fields per row, got {}: '{}'",
            fields.len(),
            line,
        );
    }

    // First time value should be 0
    let first_data = lines[1];
    let first_time: f64 = first_data.split(',').next().unwrap().parse().unwrap();
    assert!(first_time.abs() < 1e-10);
}
