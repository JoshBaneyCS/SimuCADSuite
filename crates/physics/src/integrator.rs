//! Numerical integrators for ordinary differential equations.
//!
//! Provides the [`Integrator`] trait and concrete implementations used by the
//! kinematics and trajectory solvers throughout the physics crate.

use simucad_core::types::{KinematicState, Vec2};

// ---------------------------------------------------------------------------
// Integrator trait
// ---------------------------------------------------------------------------

/// A single-step ODE integrator that advances a [`KinematicState`] forward in
/// time by `dt` given an external force function.
///
/// The force function receives the *current* state and returns the net
/// acceleration (force per unit mass) acting on the body. Implementations
/// are required to be `Send + Sync` so that they can be shared across
/// threads when running parallel trajectory sweeps.
pub trait Integrator: Send + Sync {
    /// Advance `state` by one timestep `dt`.
    ///
    /// `forces` computes the net acceleration vector given the current state.
    fn step(
        &self,
        state: &KinematicState,
        forces: &dyn Fn(&KinematicState) -> Vec2,
        dt: f64,
    ) -> KinematicState;
}

// ---------------------------------------------------------------------------
// Forward Euler integrator
// ---------------------------------------------------------------------------

/// Simplest explicit integrator: first-order forward Euler.
///
/// Update rules:
/// ```text
/// a  = forces(state)
/// v' = v + a * dt
/// x' = x + v' * dt
/// t' = t + dt
/// ```
///
/// This uses the *updated* velocity to advance position (symplectic Euler),
/// which provides better energy behaviour for conservative systems than the
/// naive forward Euler variant.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerIntegrator;

impl EulerIntegrator {
    pub fn new() -> Self {
        Self
    }
}

impl Integrator for EulerIntegrator {
    fn step(
        &self,
        state: &KinematicState,
        forces: &dyn Fn(&KinematicState) -> Vec2,
        dt: f64,
    ) -> KinematicState {
        let acceleration = forces(state);
        let new_velocity = state.velocity + acceleration * dt;
        let new_position = state.position + new_velocity * dt;

        KinematicState {
            time: state.time + dt,
            position: new_position,
            velocity: new_velocity,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: constant acceleration of (0, -g).
    fn gravity_only(_state: &KinematicState) -> Vec2 {
        Vec2::new(0.0, -9.806_65)
    }

    #[test]
    fn euler_single_step_free_fall() {
        let integrator = EulerIntegrator::new();
        let state = KinematicState {
            time: 0.0,
            position: Vec2::new(0.0, 100.0),
            velocity: Vec2::ZERO,
        };

        let dt = 0.01;
        let next = integrator.step(&state, &gravity_only, dt);

        // v' = 0 + (-9.80665)*0.01 = -0.0980665
        let expected_vy = -9.806_65 * dt;
        assert!((next.velocity.y - expected_vy).abs() < 1e-12);

        // x' = 100 + v'*dt  (symplectic Euler uses updated velocity)
        let expected_y = 100.0 + expected_vy * dt;
        assert!((next.position.y - expected_y).abs() < 1e-12);

        assert!((next.time - dt).abs() < 1e-15);
    }

    #[test]
    fn euler_multiple_steps_constant_velocity() {
        let integrator = EulerIntegrator::new();
        let mut state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(10.0, 0.0),
        };

        let zero_force = |_: &KinematicState| Vec2::ZERO;
        let dt = 0.1;

        for _ in 0..100 {
            state = integrator.step(&state, &zero_force, dt);
        }

        // After 10 s at 10 m/s, x should be 100 m.
        assert!((state.position.x - 100.0).abs() < 1e-9);
        assert!((state.position.y).abs() < 1e-12);
        assert!((state.time - 10.0).abs() < 1e-9);
    }

    #[test]
    fn euler_parabolic_trajectory_rough_check() {
        // Launch at 45 degrees, 100 m/s, check range and height are in the
        // right ballpark for vacuum trajectory.
        let integrator = EulerIntegrator::new();
        let v0 = 100.0;
        let angle = std::f64::consts::FRAC_PI_4;
        let vx = v0 * angle.cos();
        let vy = v0 * angle.sin();

        let mut state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(vx, vy),
        };

        let dt = 0.001;
        let mut max_height: f64 = 0.0;

        loop {
            state = integrator.step(&state, &gravity_only, dt);
            max_height = max_height.max(state.position.y);
            if state.position.y < 0.0 {
                break;
            }
        }

        // Analytical range = v0^2 / g ~ 1019.7 m
        let analytical_range = v0 * v0 / 9.806_65;
        assert!((state.position.x - analytical_range).abs() / analytical_range < 0.01);

        // Analytical max height = v0^2 sin^2(45) / (2g) ~ 254.9 m
        let analytical_height = v0 * v0 * angle.sin().powi(2) / (2.0 * 9.806_65);
        assert!((max_height - analytical_height).abs() / analytical_height < 0.01);
    }

    #[test]
    fn euler_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EulerIntegrator>();
    }

    #[test]
    fn integrator_trait_object_works() {
        let integrator: Box<dyn Integrator> = Box::new(EulerIntegrator::new());
        let state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(1.0, 0.0),
        };
        let next = integrator.step(&state, &|_| Vec2::ZERO, 1.0);
        assert!((next.position.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn euler_velocity_dependent_force() {
        // Simple linear drag: F = -0.1 * v
        let integrator = EulerIntegrator::new();
        let drag_coeff = 0.1;
        let drag = move |s: &KinematicState| s.velocity * (-drag_coeff);

        let mut state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(100.0, 0.0),
        };

        let dt = 0.01;
        for _ in 0..1000 {
            state = integrator.step(&state, &drag, dt);
        }

        // Velocity should decay exponentially: v(t) = v0 * exp(-0.1*t)
        // After 10 s: v ~ 100 * exp(-1) ~ 36.79
        let analytical = 100.0 * (-0.1_f64 * 10.0).exp();
        // Euler has first-order error, allow 5% tolerance
        assert!((state.velocity.x - analytical).abs() / analytical < 0.05);
    }
}
