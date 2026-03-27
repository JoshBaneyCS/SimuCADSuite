/// Standard acceleration due to gravity at Earth's surface (m/s²).
pub const STANDARD_GRAVITY: f64 = 9.806_65;

/// Air density at sea level, 15°C, 1 atm (kg/m³).
/// Used as the default medium density for drag calculations.
pub const AIR_DENSITY_SEA_LEVEL: f64 = 1.225;

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Boltzmann constant (J/K).
pub const BOLTZMANN: f64 = 1.380_649e-23;

/// Pi — re-exported for convenience alongside physical constants.
pub const PI: f64 = std::f64::consts::PI;

/// Euler's number.
pub const E: f64 = std::f64::consts::E;

// ---------------------------------------------------------------------------
// Drag coefficients — reference values for common cross-section shapes.
// Scientific assumption: these are nominal values for steady-state,
// subsonic flow at moderate Reynolds numbers (Re ~ 10⁴–10⁶).
// Real drag coefficients vary with Reynolds number and surface roughness.
// ---------------------------------------------------------------------------

/// Drag coefficient for a flat circular disk perpendicular to flow.
pub const CD_CIRCLE: f64 = 1.17;

/// Drag coefficient for a flat square plate perpendicular to flow.
pub const CD_SQUARE: f64 = 2.1;

/// Drag coefficient for a rhombus (diamond) cross-section.
pub const CD_RHOMBUS: f64 = 1.6;

/// Drag coefficient for a streamlined airfoil cross-section.
pub const CD_AIRFOIL: f64 = 0.045;

/// Drag coefficient for a smooth sphere.
pub const CD_SPHERE: f64 = 0.47;

// ---------------------------------------------------------------------------
// Default simulation parameters
// ---------------------------------------------------------------------------

/// Default number of trajectory sample points for kinematics visualization.
pub const DEFAULT_TRAJECTORY_SAMPLES: usize = 1000;

/// Default number of sampled vector overlay points on a trajectory.
pub const DEFAULT_VECTOR_SAMPLES: usize = 20;

/// Default particle count for fluid dynamics simulations.
pub const DEFAULT_PARTICLE_COUNT: usize = 4_000_000;

/// Default number of simulation steps for fluid dynamics.
pub const DEFAULT_FLUID_STEPS: usize = 20;

/// Default integration timestep (seconds).
pub const DEFAULT_TIMESTEP: f64 = 0.01;

// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------

/// Default floating-point comparison tolerance for simulation results.
pub const DEFAULT_EPSILON: f64 = 1e-9;

/// Tolerance for GPU vs CPU result comparison (looser due to float precision).
pub const GPU_CPU_EPSILON: f64 = 1e-5;
