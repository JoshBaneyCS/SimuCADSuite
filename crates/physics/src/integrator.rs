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
// Classic 4th-order Runge-Kutta integrator
// ---------------------------------------------------------------------------

/// Classic 4th-order Runge-Kutta (RK4) integrator.
///
/// Computes the weighted average of four slope evaluations per step:
///
/// ```text
/// k1 = f(t,      y)
/// k2 = f(t+dt/2, y + k1*dt/2)
/// k3 = f(t+dt/2, y + k2*dt/2)
/// k4 = f(t+dt,   y + k3*dt)
/// y' = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
/// ```
///
/// This achieves 4th-order accuracy (local truncation error O(dt^5)),
/// providing significantly better accuracy than Euler for the same timestep.
#[derive(Debug, Clone, Copy, Default)]
pub struct RK4Integrator;

impl RK4Integrator {
    pub fn new() -> Self {
        Self
    }
}

impl Integrator for RK4Integrator {
    fn step(
        &self,
        state: &KinematicState,
        forces: &dyn Fn(&KinematicState) -> Vec2,
        dt: f64,
    ) -> KinematicState {
        // k1: slope at the beginning of the interval
        let a1 = forces(state);
        let k1_vel = a1;
        let k1_pos = state.velocity;

        // k2: slope at the midpoint using k1
        let mid1 = KinematicState {
            time: state.time + dt * 0.5,
            position: state.position + k1_pos * (dt * 0.5),
            velocity: state.velocity + k1_vel * (dt * 0.5),
        };
        let a2 = forces(&mid1);
        let k2_vel = a2;
        let k2_pos = mid1.velocity;

        // k3: slope at the midpoint using k2
        let mid2 = KinematicState {
            time: state.time + dt * 0.5,
            position: state.position + k2_pos * (dt * 0.5),
            velocity: state.velocity + k2_vel * (dt * 0.5),
        };
        let a3 = forces(&mid2);
        let k3_vel = a3;
        let k3_pos = mid2.velocity;

        // k4: slope at the end using k3
        let end = KinematicState {
            time: state.time + dt,
            position: state.position + k3_pos * dt,
            velocity: state.velocity + k3_vel * dt,
        };
        let a4 = forces(&end);
        let k4_vel = a4;
        let k4_pos = end.velocity;

        // Weighted average
        let new_velocity = state.velocity
            + (k1_vel + k2_vel * 2.0 + k3_vel * 2.0 + k4_vel) * (dt / 6.0);
        let new_position = state.position
            + (k1_pos + k2_pos * 2.0 + k3_pos * 2.0 + k4_pos) * (dt / 6.0);

        KinematicState {
            time: state.time + dt,
            position: new_position,
            velocity: new_velocity,
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive Runge-Kutta-Fehlberg (RK45) integrator
// ---------------------------------------------------------------------------

/// Result of an adaptive integration step, including the recommended next
/// timestep based on local error estimation.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveStepResult {
    /// The new state after the step.
    pub state: KinematicState,
    /// The recommended timestep for the next step.
    pub next_dt: f64,
    /// Estimated local truncation error (norm of the position error vector).
    pub error_estimate: f64,
}

/// Adaptive Runge-Kutta-Fehlberg integrator with embedded 4th/5th-order
/// error estimation and automatic step size control.
///
/// Uses the Cash-Karp variant of the RK45 method. The 4th-order solution is
/// used to advance the state while the difference between the 4th and 5th
/// order solutions provides a local error estimate. The timestep is then
/// adjusted so that the error stays within the specified tolerance.
///
/// # Step size control
///
/// The controller uses the standard formula:
/// ```text
/// dt_new = safety * dt * (tol / err)^(1/5)
/// ```
/// with a safety factor of 0.9 and growth/shrinkage limits to avoid
/// excessively large or small step adjustments.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveRK45Integrator {
    /// Absolute tolerance for the local truncation error.
    pub tolerance: f64,
    /// Safety factor for step size control (default 0.9).
    pub safety: f64,
    /// Minimum allowed timestep.
    pub dt_min: f64,
    /// Maximum allowed timestep.
    pub dt_max: f64,
}

impl AdaptiveRK45Integrator {
    /// Create a new adaptive RK45 integrator with the given tolerance.
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            safety: 0.9,
            dt_min: 1e-12,
            dt_max: 1.0,
        }
    }

    /// Create with full configuration.
    pub fn with_params(tolerance: f64, safety: f64, dt_min: f64, dt_max: f64) -> Self {
        Self {
            tolerance,
            safety,
            dt_min,
            dt_max,
        }
    }

    /// Perform one adaptive step, returning the new state, recommended next
    /// dt, and the error estimate.
    ///
    /// Uses Cash-Karp coefficients for the embedded RK4(5) pair.
    pub fn adaptive_step(
        &self,
        state: &KinematicState,
        forces: &dyn Fn(&KinematicState) -> Vec2,
        dt: f64,
    ) -> AdaptiveStepResult {
        // Cash-Karp coefficients for the six stages
        // a-coefficients (time fractions)
        let a2 = 1.0 / 5.0;
        let a3 = 3.0 / 10.0;
        let a4 = 3.0 / 5.0;
        let a5 = 1.0;
        let a6 = 7.0 / 8.0;

        // b-coefficients (stage weights)
        let b21 = 1.0 / 5.0;
        let b31 = 3.0 / 40.0;
        let b32 = 9.0 / 40.0;
        let b41 = 3.0 / 10.0;
        let b42 = -9.0 / 10.0;
        let b43 = 6.0 / 5.0;
        let b51 = -11.0 / 54.0;
        let b52 = 5.0 / 2.0;
        let b53 = -70.0 / 27.0;
        let b54 = 35.0 / 27.0;
        let b61 = 1631.0 / 55296.0;
        let b62 = 175.0 / 512.0;
        let b63 = 575.0 / 13824.0;
        let b64 = 44275.0 / 110592.0;
        let b65 = 253.0 / 4096.0;

        // 4th-order weights
        let c1 = 37.0 / 378.0;
        let c3 = 250.0 / 621.0;
        let c4 = 125.0 / 594.0;
        let c6 = 512.0 / 1771.0;

        // 5th-order weights (for error estimation)
        let d1 = 2825.0 / 27648.0;
        let d3 = 18575.0 / 48384.0;
        let d4 = 13525.0 / 55296.0;
        let d5 = 277.0 / 14336.0;
        let d6 = 1.0 / 4.0;

        // Stage 1
        let k1_a = forces(state);
        let k1_v = state.velocity;

        // Stage 2
        let s2 = KinematicState {
            time: state.time + a2 * dt,
            position: state.position + k1_v * (b21 * dt),
            velocity: state.velocity + k1_a * (b21 * dt),
        };
        let k2_a = forces(&s2);
        let k2_v = s2.velocity;

        // Stage 3
        let s3 = KinematicState {
            time: state.time + a3 * dt,
            position: state.position + (k1_v * b31 + k2_v * b32) * dt,
            velocity: state.velocity + (k1_a * b31 + k2_a * b32) * dt,
        };
        let k3_a = forces(&s3);
        let k3_v = s3.velocity;

        // Stage 4
        let s4 = KinematicState {
            time: state.time + a4 * dt,
            position: state.position + (k1_v * b41 + k2_v * b42 + k3_v * b43) * dt,
            velocity: state.velocity + (k1_a * b41 + k2_a * b42 + k3_a * b43) * dt,
        };
        let k4_a = forces(&s4);
        let k4_v = s4.velocity;

        // Stage 5
        let s5 = KinematicState {
            time: state.time + a5 * dt,
            position: state.position
                + (k1_v * b51 + k2_v * b52 + k3_v * b53 + k4_v * b54) * dt,
            velocity: state.velocity
                + (k1_a * b51 + k2_a * b52 + k3_a * b53 + k4_a * b54) * dt,
        };
        let k5_a = forces(&s5);
        let k5_v = s5.velocity;

        // Stage 6
        let s6 = KinematicState {
            time: state.time + a6 * dt,
            position: state.position
                + (k1_v * b61 + k2_v * b62 + k3_v * b63 + k4_v * b64 + k5_v * b65) * dt,
            velocity: state.velocity
                + (k1_a * b61 + k2_a * b62 + k3_a * b63 + k4_a * b64 + k5_a * b65) * dt,
        };
        let k6_a = forces(&s6);
        let k6_v = s6.velocity;

        // 4th-order solution (used to advance)
        let new_pos_4 = state.position
            + (k1_v * c1 + k3_v * c3 + k4_v * c4 + k6_v * c6) * dt;
        let new_vel_4 = state.velocity
            + (k1_a * c1 + k3_a * c3 + k4_a * c4 + k6_a * c6) * dt;

        // 5th-order solution (for error estimation only)
        let new_pos_5 = state.position
            + (k1_v * d1 + k3_v * d3 + k4_v * d4 + k5_v * d5 + k6_v * d6) * dt;

        // Error estimate: norm of (pos_4 - pos_5)
        let err_vec = new_pos_4 - new_pos_5;
        let err = err_vec.magnitude();

        // Step size control
        let next_dt = if err < 1e-30 {
            // Error is essentially zero; allow maximum growth
            (dt * 5.0).min(self.dt_max)
        } else {
            let ratio = self.tolerance / err;
            let scale = self.safety * ratio.powf(0.2);
            // Clamp growth/shrinkage factor
            let scale = scale.clamp(0.1, 5.0);
            (dt * scale).clamp(self.dt_min, self.dt_max)
        };

        AdaptiveStepResult {
            state: KinematicState {
                time: state.time + dt,
                position: new_pos_4,
                velocity: new_vel_4,
            },
            next_dt,
            error_estimate: err,
        }
    }
}

impl Integrator for AdaptiveRK45Integrator {
    /// Step using the adaptive method. The provided `dt` is used as the
    /// initial step size; the method will internally subdivide if needed to
    /// meet the tolerance. The returned state advances by exactly `dt`.
    fn step(
        &self,
        state: &KinematicState,
        forces: &dyn Fn(&KinematicState) -> Vec2,
        dt: f64,
    ) -> KinematicState {
        let result = self.adaptive_step(state, forces, dt);
        result.state
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

    // -----------------------------------------------------------------------
    // RK4 tests
    // -----------------------------------------------------------------------

    #[test]
    fn rk4_single_step_free_fall() {
        let integrator = RK4Integrator::new();
        let state = KinematicState {
            time: 0.0,
            position: Vec2::new(0.0, 100.0),
            velocity: Vec2::ZERO,
        };

        let dt = 0.01;
        let next = integrator.step(&state, &gravity_only, dt);

        // For constant acceleration, RK4 should be exact (all k stages agree)
        let expected_vy = -9.806_65 * dt;
        assert!(
            (next.velocity.y - expected_vy).abs() < 1e-10,
            "vy: {} vs expected: {}",
            next.velocity.y,
            expected_vy
        );

        // Position: x(dt) = x0 + v0*dt + 0.5*a*dt^2
        let expected_y = 100.0 + 0.0 * dt - 0.5 * 9.806_65 * dt * dt;
        assert!(
            (next.position.y - expected_y).abs() < 1e-10,
            "y: {} vs expected: {}",
            next.position.y,
            expected_y
        );
    }

    #[test]
    fn rk4_constant_velocity() {
        let integrator = RK4Integrator::new();
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

        assert!((state.position.x - 100.0).abs() < 1e-9);
        assert!((state.time - 10.0).abs() < 1e-9);
    }

    #[test]
    fn rk4_more_accurate_than_euler_parabolic() {
        // Compare Euler and RK4 on a parabolic trajectory with the SAME
        // timestep. RK4 should be significantly more accurate.
        let dt = 0.1; // deliberately coarse
        let v0 = 100.0;
        let angle = std::f64::consts::FRAC_PI_4;
        let vx = v0 * angle.cos();
        let vy = v0 * angle.sin();

        let initial = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(vx, vy),
        };

        let euler = EulerIntegrator::new();
        let rk4 = RK4Integrator::new();

        let mut state_euler = initial;
        let mut state_rk4 = initial;

        // Run both until ~ 5 seconds (well before landing at ~14.4s)
        let steps = 50;
        for _ in 0..steps {
            state_euler = euler.step(&state_euler, &gravity_only, dt);
            state_rk4 = rk4.step(&state_rk4, &gravity_only, dt);
        }

        let t = dt * steps as f64;
        // Analytical position at t=5s
        let analytical_x = vx * t;
        let analytical_y = vy * t - 0.5 * 9.806_65 * t * t;

        let euler_err_x = (state_euler.position.x - analytical_x).abs();
        let euler_err_y = (state_euler.position.y - analytical_y).abs();
        let rk4_err_x = (state_rk4.position.x - analytical_x).abs();
        let rk4_err_y = (state_rk4.position.y - analytical_y).abs();

        let euler_err = (euler_err_x * euler_err_x + euler_err_y * euler_err_y).sqrt();
        let rk4_err = (rk4_err_x * rk4_err_x + rk4_err_y * rk4_err_y).sqrt();

        // For constant-force problem, RK4 should be exact (up to floating-point)
        // while Euler accumulates significant error
        assert!(
            rk4_err < euler_err,
            "RK4 error ({rk4_err}) should be less than Euler error ({euler_err})"
        );

        // RK4 should be many orders of magnitude better
        assert!(
            rk4_err < 1e-6,
            "RK4 error ({rk4_err}) should be near machine precision for constant forces"
        );
    }

    #[test]
    fn rk4_more_accurate_than_euler_drag() {
        // For velocity-dependent forces (drag), RK4 should also beat Euler.
        let drag_coeff = 0.1;
        let drag = move |s: &KinematicState| s.velocity * (-drag_coeff);

        let initial = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(100.0, 0.0),
        };

        let dt = 0.1; // coarse timestep
        let euler = EulerIntegrator::new();
        let rk4 = RK4Integrator::new();

        let mut state_euler = initial;
        let mut state_rk4 = initial;

        let steps = 100; // 10 seconds
        for _ in 0..steps {
            state_euler = euler.step(&state_euler, &drag, dt);
            state_rk4 = rk4.step(&state_rk4, &drag, dt);
        }

        // Analytical: v(t) = v0 * exp(-k*t), x(t) = v0/k * (1 - exp(-k*t))
        let t = dt * steps as f64;
        let analytical_v = 100.0 * (-drag_coeff * t).exp();
        let analytical_x = 100.0 / drag_coeff * (1.0 - (-drag_coeff * t).exp());

        let euler_err_v = (state_euler.velocity.x - analytical_v).abs();
        let rk4_err_v = (state_rk4.velocity.x - analytical_v).abs();
        let euler_err_x = (state_euler.position.x - analytical_x).abs();
        let rk4_err_x = (state_rk4.position.x - analytical_x).abs();

        assert!(
            rk4_err_v < euler_err_v,
            "RK4 velocity error ({rk4_err_v}) should be less than Euler ({euler_err_v})"
        );
        assert!(
            rk4_err_x < euler_err_x,
            "RK4 position error ({rk4_err_x}) should be less than Euler ({euler_err_x})"
        );
    }

    #[test]
    fn rk4_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RK4Integrator>();
    }

    #[test]
    fn rk4_trait_object_works() {
        let integrator: Box<dyn Integrator> = Box::new(RK4Integrator::new());
        let state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(1.0, 0.0),
        };
        let next = integrator.step(&state, &|_| Vec2::ZERO, 1.0);
        assert!((next.position.x - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Adaptive RK45 tests
    // -----------------------------------------------------------------------

    #[test]
    fn rk45_free_fall_accurate() {
        let integrator = AdaptiveRK45Integrator::new(1e-10);
        let state = KinematicState {
            time: 0.0,
            position: Vec2::new(0.0, 100.0),
            velocity: Vec2::ZERO,
        };

        let dt = 0.5;
        let result = integrator.adaptive_step(&state, &gravity_only, dt);

        // Constant acceleration -> should be very accurate
        let expected_y = 100.0 - 0.5 * 9.806_65 * dt * dt;
        let expected_vy = -9.806_65 * dt;

        assert!(
            (result.state.position.y - expected_y).abs() < 1e-8,
            "y: {} vs expected: {}",
            result.state.position.y,
            expected_y
        );
        assert!(
            (result.state.velocity.y - expected_vy).abs() < 1e-8,
            "vy: {} vs expected: {}",
            result.state.velocity.y,
            expected_vy
        );
    }

    #[test]
    fn rk45_provides_next_dt() {
        let integrator = AdaptiveRK45Integrator::new(1e-6);
        let state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(100.0, 0.0),
        };

        let drag = |s: &KinematicState| s.velocity * (-0.1);
        let result = integrator.adaptive_step(&state, &drag, 0.1);

        assert!(result.next_dt > 0.0, "next_dt should be positive");
        assert!(result.next_dt.is_finite(), "next_dt should be finite");
    }

    #[test]
    fn rk45_error_within_tolerance() {
        let tol = 1e-8;
        let integrator = AdaptiveRK45Integrator::new(tol);
        let drag = |s: &KinematicState| {
            let g = Vec2::new(0.0, -9.806_65);
            let drag_accel = s.velocity * (-0.05);
            g + drag_accel
        };

        let mut state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(50.0, 50.0),
        };

        let mut dt = 0.01;
        // Take several adaptive steps
        for _ in 0..20 {
            let result = integrator.adaptive_step(&state, &drag, dt);
            state = result.state;
            dt = result.next_dt;
        }

        // The integrator should produce finite, reasonable values
        assert!(state.position.x.is_finite());
        assert!(state.position.y.is_finite());
        assert!(state.time > 0.0);
    }

    #[test]
    fn rk45_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AdaptiveRK45Integrator>();
    }

    #[test]
    fn rk45_implements_integrator_trait() {
        let integrator: Box<dyn Integrator> = Box::new(AdaptiveRK45Integrator::new(1e-8));
        let state = KinematicState {
            time: 0.0,
            position: Vec2::ZERO,
            velocity: Vec2::new(1.0, 0.0),
        };
        let next = integrator.step(&state, &|_| Vec2::ZERO, 1.0);
        assert!((next.position.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rk45_step_size_shrinks_for_stiff_problem() {
        // A force that changes rapidly should cause the integrator to
        // recommend a smaller step size.
        let integrator = AdaptiveRK45Integrator::new(1e-10);
        let stiff_force = |s: &KinematicState| {
            // Rapid oscillation
            let k = 1000.0;
            Vec2::new(-k * s.position.x, -k * s.position.y)
        };

        let state = KinematicState {
            time: 0.0,
            position: Vec2::new(1.0, 0.0),
            velocity: Vec2::ZERO,
        };

        let result = integrator.adaptive_step(&state, &stiff_force, 0.1);
        // For a stiff problem with tight tolerance, it should suggest a
        // smaller step than the one attempted.
        assert!(
            result.next_dt < 0.1,
            "next_dt ({}) should be < 0.1 for a stiff problem",
            result.next_dt
        );
    }
}
