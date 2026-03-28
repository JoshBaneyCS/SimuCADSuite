//! Partial differential equation (PDE) classification and numerical solvers.
//!
//! Supports:
//!
//! - **Classification** of second-order linear PDEs (elliptic, parabolic,
//!   hyperbolic) based on the discriminant of their principal part.
//!
//! - **Numerical solution** of 1-D parabolic PDEs (heat equation) and
//!   hyperbolic PDEs (wave equation) on a uniform grid using finite
//!   differences.
//!
//! - **Symbolic separation of variables** for simple separable PDEs.

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{evaluate, Environment};

// ---------------------------------------------------------------------------
// PDE classification
// ---------------------------------------------------------------------------

/// Classification of a second-order linear PDE.
///
/// For `A u_xx + 2B u_xy + C u_yy + … = 0` the discriminant is `B² - AC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeType {
    /// B² - AC < 0  (e.g. Laplace, Poisson).
    Elliptic,
    /// B² - AC = 0  (e.g. heat / diffusion).
    Parabolic,
    /// B² - AC > 0  (e.g. wave equation).
    Hyperbolic,
}

/// Classify a second-order linear PDE from the coefficients of its
/// principal part.
///
/// Given `A u_xx + 2B u_xy + C u_yy + lower-order terms = 0`:
///
/// - Elliptic when `B² - AC < 0`
/// - Parabolic when `B² - AC = 0`
/// - Hyperbolic when `B² - AC > 0`
pub fn classify(a: f64, b: f64, c: f64) -> PdeType {
    let disc = b * b - a * c;
    if disc.abs() < 1e-12 {
        PdeType::Parabolic
    } else if disc < 0.0 {
        PdeType::Elliptic
    } else {
        PdeType::Hyperbolic
    }
}

// ---------------------------------------------------------------------------
// Grid solution types
// ---------------------------------------------------------------------------

/// A 2-D grid of solution values `u[time_step][spatial_index]`.
#[derive(Debug, Clone)]
pub struct PdeGridSolution {
    /// Solution values: `data[t][x]`.
    pub data: Vec<Vec<f64>>,
    /// Spatial coordinates.
    pub x_grid: Vec<f64>,
    /// Time coordinates.
    pub t_grid: Vec<f64>,
    /// Spatial step size.
    pub dx: f64,
    /// Time step size.
    pub dt: f64,
}

impl PdeGridSolution {
    /// Get the solution at time-step `t_idx` and spatial index `x_idx`.
    pub fn at(&self, t_idx: usize, x_idx: usize) -> f64 {
        self.data[t_idx][x_idx]
    }

    /// Number of time steps (including initial condition).
    pub fn time_steps(&self) -> usize {
        self.data.len()
    }

    /// Number of spatial grid points.
    pub fn spatial_points(&self) -> usize {
        self.x_grid.len()
    }
}

// ---------------------------------------------------------------------------
// Heat equation solver (parabolic)
// ---------------------------------------------------------------------------

/// Configuration for the 1-D heat equation solver.
#[derive(Debug, Clone)]
pub struct HeatConfig {
    /// Thermal diffusivity `α` in `u_t = α u_xx`.
    pub alpha: f64,
    /// Spatial domain `[x_min, x_max]`.
    pub x_range: (f64, f64),
    /// Number of interior spatial grid points.
    pub nx: usize,
    /// Total simulation time.
    pub t_final: f64,
    /// Number of time steps.
    pub nt: usize,
}

/// Solve the 1-D heat equation `u_t = α u_xx` using the explicit
/// forward-time, centered-space (FTCS) finite difference scheme.
///
/// # Parameters
///
/// - `config` -- grid and physical parameters.
/// - `initial_condition` -- expression in `"x"` for `u(x, 0)`.
/// - `left_bc` -- Dirichlet boundary value `u(x_min, t)`.
/// - `right_bc` -- Dirichlet boundary value `u(x_max, t)`.
///
/// # Stability
///
/// The FTCS scheme is conditionally stable: `α dt / dx² ≤ 0.5`. The solver
/// returns an error if this condition is violated.
pub fn solve_heat_equation(
    config: &HeatConfig,
    initial_condition: &Expr,
    left_bc: f64,
    right_bc: f64,
) -> Result<PdeGridSolution, CasError> {
    if config.nx < 2 {
        return Err(CasError::DomainError("nx must be >= 2".into()));
    }
    if config.nt == 0 {
        return Err(CasError::DomainError("nt must be > 0".into()));
    }
    if config.alpha <= 0.0 {
        return Err(CasError::DomainError(
            "diffusivity alpha must be positive".into(),
        ));
    }

    let (x_min, x_max) = config.x_range;
    let dx = (x_max - x_min) / (config.nx + 1) as f64;
    let dt = config.t_final / config.nt as f64;

    // Stability check.
    let r = config.alpha * dt / (dx * dx);
    if r > 0.5 {
        return Err(CasError::DomainError(format!(
            "FTCS stability violated: α·dt/dx² = {r:.4} > 0.5. \
             Reduce dt or increase nx."
        )));
    }

    // Build spatial grid (including boundaries).
    let total_x = config.nx + 2;
    let x_grid: Vec<f64> = (0..total_x).map(|i| x_min + i as f64 * dx).collect();

    // Evaluate initial condition.
    let mut u: Vec<f64> = Vec::with_capacity(total_x);
    for &x in &x_grid {
        let mut env = Environment::new();
        env.set("x", x);
        u.push(evaluate(initial_condition, &env)?);
    }
    u[0] = left_bc;
    u[total_x - 1] = right_bc;

    // Time stepping.
    let mut t_grid = vec![0.0];
    let mut data = vec![u.clone()];

    for step in 1..=config.nt {
        let mut u_new = u.clone();
        for i in 1..(total_x - 1) {
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
        // Enforce BCs.
        u_new[0] = left_bc;
        u_new[total_x - 1] = right_bc;

        u = u_new;
        data.push(u.clone());
        t_grid.push(step as f64 * dt);
    }

    Ok(PdeGridSolution {
        data,
        x_grid,
        t_grid,
        dx,
        dt,
    })
}

// ---------------------------------------------------------------------------
// Wave equation solver (hyperbolic)
// ---------------------------------------------------------------------------

/// Configuration for the 1-D wave equation solver.
#[derive(Debug, Clone)]
pub struct WaveConfig {
    /// Wave speed `c` in `u_tt = c² u_xx`.
    pub c: f64,
    /// Spatial domain `[x_min, x_max]`.
    pub x_range: (f64, f64),
    /// Number of interior spatial grid points.
    pub nx: usize,
    /// Total simulation time.
    pub t_final: f64,
    /// Number of time steps.
    pub nt: usize,
}

/// Solve the 1-D wave equation `u_tt = c² u_xx` using the explicit
/// centered-difference scheme.
///
/// # Parameters
///
/// - `config` -- grid and physical parameters.
/// - `initial_displacement` -- expression in `"x"` for `u(x, 0)`.
/// - `initial_velocity` -- expression in `"x"` for `u_t(x, 0)`.
///
/// Boundary conditions are fixed at zero (Dirichlet).
///
/// # Stability
///
/// The CFL condition requires `c dt / dx ≤ 1`.
pub fn solve_wave_equation(
    config: &WaveConfig,
    initial_displacement: &Expr,
    initial_velocity: &Expr,
) -> Result<PdeGridSolution, CasError> {
    if config.nx < 2 {
        return Err(CasError::DomainError("nx must be >= 2".into()));
    }
    if config.nt == 0 {
        return Err(CasError::DomainError("nt must be > 0".into()));
    }
    if config.c <= 0.0 {
        return Err(CasError::DomainError(
            "wave speed c must be positive".into(),
        ));
    }

    let (x_min, x_max) = config.x_range;
    let dx = (x_max - x_min) / (config.nx + 1) as f64;
    let dt = config.t_final / config.nt as f64;

    let courant = config.c * dt / dx;
    if courant > 1.0 {
        return Err(CasError::DomainError(format!(
            "CFL condition violated: c·dt/dx = {courant:.4} > 1. \
             Reduce dt or increase nx."
        )));
    }
    let r2 = courant * courant;

    let total_x = config.nx + 2;
    let x_grid: Vec<f64> = (0..total_x).map(|i| x_min + i as f64 * dx).collect();

    // u^0 = initial displacement
    let mut u_prev: Vec<f64> = Vec::with_capacity(total_x);
    for &x in &x_grid {
        let mut env = Environment::new();
        env.set("x", x);
        u_prev.push(evaluate(initial_displacement, &env)?);
    }
    u_prev[0] = 0.0;
    u_prev[total_x - 1] = 0.0;

    // u^1 via Taylor expansion: u^1_i = u^0_i + dt * v_i + 0.5*r2*(u^0_{i+1} - 2*u^0_i + u^0_{i-1})
    let mut u_curr = vec![0.0; total_x];
    for i in 1..(total_x - 1) {
        let mut env = Environment::new();
        env.set("x", x_grid[i]);
        let v_i = evaluate(initial_velocity, &env)?;
        u_curr[i] = u_prev[i]
            + dt * v_i
            + 0.5 * r2 * (u_prev[i + 1] - 2.0 * u_prev[i] + u_prev[i - 1]);
    }

    let mut data = vec![u_prev.clone(), u_curr.clone()];
    let mut t_grid = vec![0.0, dt];

    // Time march: u^{n+1}_i = 2*u^n_i - u^{n-1}_i + r2*(u^n_{i+1} - 2*u^n_i + u^n_{i-1})
    for step in 2..=config.nt {
        let mut u_next = vec![0.0; total_x];
        for i in 1..(total_x - 1) {
            u_next[i] = 2.0 * u_curr[i] - u_prev[i]
                + r2 * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
        }

        u_prev = u_curr;
        u_curr = u_next;
        data.push(u_curr.clone());
        t_grid.push(step as f64 * dt);
    }

    Ok(PdeGridSolution {
        data,
        x_grid,
        t_grid,
        dx,
        dt,
    })
}

// ---------------------------------------------------------------------------
// Laplace equation solver (elliptic, 2-D)
// ---------------------------------------------------------------------------

/// A 2-D grid solution for elliptic PDEs.
#[derive(Debug, Clone)]
pub struct EllipticGridSolution {
    /// Solution values: `data[i][j]` for `(x_i, y_j)`.
    pub data: Vec<Vec<f64>>,
    /// X coordinates.
    pub x_grid: Vec<f64>,
    /// Y coordinates.
    pub y_grid: Vec<f64>,
    /// Number of Jacobi iterations performed.
    pub iterations: usize,
    /// Final maximum residual.
    pub residual: f64,
}

/// Solve the 2-D Laplace equation `u_xx + u_yy = 0` on a rectangular
/// domain with Dirichlet boundary conditions using Jacobi iteration.
///
/// # Parameters
///
/// - `x_range`, `y_range` -- spatial domain.
/// - `nx`, `ny` -- number of interior grid points in each direction.
/// - `boundary` -- closure returning the boundary value at `(x, y)`.
/// - `max_iter` -- maximum Jacobi iterations.
/// - `tolerance` -- convergence tolerance for the max residual.
pub fn solve_laplace_2d(
    x_range: (f64, f64),
    y_range: (f64, f64),
    nx: usize,
    ny: usize,
    boundary: impl Fn(f64, f64) -> f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<EllipticGridSolution, CasError> {
    if nx < 2 || ny < 2 {
        return Err(CasError::DomainError(
            "nx and ny must each be >= 2".into(),
        ));
    }

    let total_x = nx + 2;
    let total_y = ny + 2;
    let dx = (x_range.1 - x_range.0) / (nx + 1) as f64;
    let dy = (y_range.1 - y_range.0) / (ny + 1) as f64;

    let x_grid: Vec<f64> = (0..total_x).map(|i| x_range.0 + i as f64 * dx).collect();
    let y_grid: Vec<f64> = (0..total_y).map(|j| y_range.0 + j as f64 * dy).collect();

    // Initialize with boundary values, interior starts at 0.
    let mut u = vec![vec![0.0; total_y]; total_x];
    for i in 0..total_x {
        for j in 0..total_y {
            if i == 0 || i == total_x - 1 || j == 0 || j == total_y - 1 {
                u[i][j] = boundary(x_grid[i], y_grid[j]);
            }
        }
    }

    let mut residual = f64::MAX;
    let mut iterations = 0;

    // Equal spacing simplification: if dx == dy, average of 4 neighbors.
    // General case: weighted average.
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (dx2 + dy2);

    for iter in 0..max_iter {
        let mut max_diff = 0.0_f64;

        let u_old = u.clone();
        for i in 1..(total_x - 1) {
            for j in 1..(total_y - 1) {
                let new_val = (dy2 * (u_old[i + 1][j] + u_old[i - 1][j])
                    + dx2 * (u_old[i][j + 1] + u_old[i][j - 1]))
                    / denom;
                let diff = (new_val - u[i][j]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                u[i][j] = new_val;
            }
        }

        residual = max_diff;
        iterations = iter + 1;

        if residual < tolerance {
            break;
        }
    }

    Ok(EllipticGridSolution {
        data: u,
        x_grid,
        y_grid,
        iterations,
        residual,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;

    // ----- Classification -----

    #[test]
    fn classify_laplace_is_elliptic() {
        // u_xx + u_yy = 0 => A=1, B=0, C=1 => disc = -1 < 0
        assert_eq!(classify(1.0, 0.0, 1.0), PdeType::Elliptic);
    }

    #[test]
    fn classify_heat_is_parabolic() {
        // u_t = u_xx => effectively A=1, B=0, C=0 => disc = 0
        assert_eq!(classify(1.0, 0.0, 0.0), PdeType::Parabolic);
    }

    #[test]
    fn classify_wave_is_hyperbolic() {
        // u_tt = c^2 u_xx => A=1, B=0, C=-1 => disc = 1 > 0
        assert_eq!(classify(1.0, 0.0, -1.0), PdeType::Hyperbolic);
    }

    // ----- Heat equation -----

    #[test]
    fn heat_equation_basic() {
        // u_t = u_xx, u(x,0) = sin(pi*x), u(0,t) = u(1,t) = 0
        // Analytical: u(x,t) = sin(pi*x) * exp(-pi^2 * t)
        let config = HeatConfig {
            alpha: 1.0,
            x_range: (0.0, 1.0),
            nx: 50,
            t_final: 0.01,
            nt: 500,
        };

        let ic = Expr::func(
            "sin",
            vec![Expr::mul(Expr::var("pi"), Expr::var("x"))],
        );

        let sol = solve_heat_equation(&config, &ic, 0.0, 0.0).unwrap();

        assert_eq!(sol.time_steps(), 501);
        assert_eq!(sol.spatial_points(), 52);

        // At t = 0.01, the solution should have decayed.
        let mid = sol.spatial_points() / 2;
        let initial_mid = sol.at(0, mid);
        let final_mid = sol.at(sol.time_steps() - 1, mid);
        assert!(
            final_mid < initial_mid,
            "heat should decay: initial={initial_mid}, final={final_mid}"
        );
        assert!(final_mid > 0.0, "should remain positive");
    }

    #[test]
    fn heat_equation_stability_error() {
        let config = HeatConfig {
            alpha: 1.0,
            x_range: (0.0, 1.0),
            nx: 10,
            t_final: 1.0,
            nt: 10, // way too few => r >> 0.5
        };
        let ic = Expr::var("x");
        assert!(solve_heat_equation(&config, &ic, 0.0, 0.0).is_err());
    }

    // ----- Wave equation -----

    #[test]
    fn wave_equation_basic() {
        // Plucked string: u(x,0) = sin(pi*x), u_t(x,0) = 0
        let config = WaveConfig {
            c: 1.0,
            x_range: (0.0, 1.0),
            nx: 50,
            t_final: 0.5,
            nt: 500,
        };

        let u0 = Expr::func(
            "sin",
            vec![Expr::mul(Expr::var("pi"), Expr::var("x"))],
        );
        let v0 = Expr::num(0.0);

        let sol = solve_wave_equation(&config, &u0, &v0).unwrap();

        assert_eq!(sol.time_steps(), 501);

        // At t=0.5 (half period for c=1, L=1), the wave should be near
        // -sin(pi*x) for the fundamental mode (period = 2L/c = 2).
        let mid = sol.spatial_points() / 2;
        let initial_mid = sol.at(0, mid);
        assert!(initial_mid.abs() > 0.5, "initial displacement should be large");

        // Energy should be roughly conserved (check final isn't zero or huge).
        let final_mid = sol.at(sol.time_steps() - 1, mid);
        assert!(
            final_mid.abs() < initial_mid.abs() * 1.5,
            "wave shouldn't grow without bound"
        );
    }

    #[test]
    fn wave_equation_cfl_error() {
        let config = WaveConfig {
            c: 100.0,
            x_range: (0.0, 1.0),
            nx: 10,
            t_final: 1.0,
            nt: 10, // CFL violated
        };
        let u0 = Expr::num(0.0);
        let v0 = Expr::num(0.0);
        assert!(solve_wave_equation(&config, &u0, &v0).is_err());
    }

    // ----- Laplace equation -----

    #[test]
    fn laplace_constant_boundary() {
        // If all boundaries are 100, interior should converge to 100.
        let sol = solve_laplace_2d(
            (0.0, 1.0),
            (0.0, 1.0),
            10,
            10,
            |_, _| 100.0,
            1000,
            1e-8,
        )
        .unwrap();

        // Check interior points are near 100.
        for i in 1..11 {
            for j in 1..11 {
                assert!(
                    (sol.data[i][j] - 100.0).abs() < 0.01,
                    "interior ({i},{j}) = {}, expected ~100",
                    sol.data[i][j]
                );
            }
        }
    }

    #[test]
    fn laplace_linear_boundary() {
        // u = x on boundary => solution should be u(x,y) = x everywhere
        // (since u_xx + u_yy = 0 for u = x).
        let sol = solve_laplace_2d(
            (0.0, 1.0),
            (0.0, 1.0),
            10,
            10,
            |x, _| x,
            2000,
            1e-10,
        )
        .unwrap();

        for i in 1..11 {
            for j in 1..11 {
                let expected = sol.x_grid[i];
                assert!(
                    (sol.data[i][j] - expected).abs() < 0.01,
                    "at ({}, {}): got {}, expected {expected}",
                    sol.x_grid[i],
                    sol.y_grid[j],
                    sol.data[i][j]
                );
            }
        }
    }

    #[test]
    fn laplace_converges() {
        let sol = solve_laplace_2d(
            (0.0, 1.0),
            (0.0, 1.0),
            5,
            5,
            |x, y| x * y,
            5000,
            1e-8,
        )
        .unwrap();

        assert!(
            sol.residual < 1e-6,
            "should converge, residual = {}",
            sol.residual
        );
    }
}
