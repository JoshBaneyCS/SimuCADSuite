//! Aerodynamic drag models for projectile and body simulations.
//!
//! Provides [`DragShape`] for common cross-section geometries and
//! [`DragModel`] for computing the drag force vector given a velocity.

use simucad_core::constants::{
    AIR_DENSITY_SEA_LEVEL, CD_AIRFOIL, CD_CIRCLE, CD_RHOMBUS, CD_SPHERE, CD_SQUARE,
};
use simucad_core::types::Vec2;

// ---------------------------------------------------------------------------
// DragShape
// ---------------------------------------------------------------------------

/// Cross-section shape determining the drag coefficient.
///
/// Each variant maps to a reference drag coefficient for steady-state,
/// subsonic flow at moderate Reynolds numbers. The `None` variant disables
/// drag entirely (Cd = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DragShape {
    /// Flat circular disk perpendicular to flow (Cd ~ 1.17).
    Circle,
    /// Flat square plate perpendicular to flow (Cd ~ 2.1).
    Square,
    /// Rhombus (diamond) cross-section (Cd ~ 1.6).
    Rhombus,
    /// Streamlined airfoil profile (Cd ~ 0.045).
    Airfoil,
    /// Smooth sphere (Cd ~ 0.47).
    Sphere,
    /// No drag -- vacuum environment.
    None,
}

impl DragShape {
    /// Return the reference drag coefficient for this shape.
    pub fn drag_coefficient(&self) -> f64 {
        match self {
            DragShape::Circle => CD_CIRCLE,
            DragShape::Square => CD_SQUARE,
            DragShape::Rhombus => CD_RHOMBUS,
            DragShape::Airfoil => CD_AIRFOIL,
            DragShape::Sphere => CD_SPHERE,
            DragShape::None => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// DragModel
// ---------------------------------------------------------------------------

/// Parametric drag model encapsulating shape, reference area, and medium
/// density.
///
/// The drag force is computed using the standard aerodynamic drag equation:
///
/// ```text
/// F_drag = -0.5 * Cd * A * rho * |v|^2 * v_hat
/// ```
///
/// where `v_hat` is the unit vector in the direction of velocity. The
/// negative sign indicates that drag always opposes the direction of motion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragModel {
    /// Cross-section shape (determines Cd).
    pub shape: DragShape,
    /// Reference cross-sectional area perpendicular to flow (m^2).
    pub cross_section_area: f64,
    /// Density of the surrounding medium (kg/m^3).
    pub air_density: f64,
}

impl DragModel {
    /// Create a new drag model.
    pub fn new(shape: DragShape, cross_section_area: f64, air_density: f64) -> Self {
        Self {
            shape,
            cross_section_area,
            air_density,
        }
    }

    /// Create a drag model using sea-level air density.
    pub fn at_sea_level(shape: DragShape, cross_section_area: f64) -> Self {
        Self::new(shape, cross_section_area, AIR_DENSITY_SEA_LEVEL)
    }

    /// Create a vacuum (no-drag) model.
    pub fn vacuum() -> Self {
        Self::new(DragShape::None, 0.0, 0.0)
    }

    /// Compute the drag force vector for a given velocity.
    ///
    /// Returns `Vec2::ZERO` when the velocity magnitude is negligible
    /// (below `f64::EPSILON`) to avoid division by zero in the unit vector
    /// calculation.
    pub fn drag_force(&self, velocity: Vec2) -> Vec2 {
        let speed = velocity.magnitude();
        if speed < f64::EPSILON {
            return Vec2::ZERO;
        }

        let cd = self.shape.drag_coefficient();
        let magnitude = 0.5 * cd * self.cross_section_area * self.air_density * speed * speed;
        let v_hat = velocity.normalized();

        // Drag opposes velocity
        v_hat * (-magnitude)
    }

    /// Return the drag coefficient of the underlying shape.
    pub fn drag_coefficient(&self) -> f64 {
        self.shape.drag_coefficient()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn drag_coefficients_match_constants() {
        assert!((DragShape::Circle.drag_coefficient() - CD_CIRCLE).abs() < EPSILON);
        assert!((DragShape::Square.drag_coefficient() - CD_SQUARE).abs() < EPSILON);
        assert!((DragShape::Rhombus.drag_coefficient() - CD_RHOMBUS).abs() < EPSILON);
        assert!((DragShape::Airfoil.drag_coefficient() - CD_AIRFOIL).abs() < EPSILON);
        assert!((DragShape::Sphere.drag_coefficient() - CD_SPHERE).abs() < EPSILON);
        assert!((DragShape::None.drag_coefficient()).abs() < EPSILON);
    }

    #[test]
    fn drag_force_opposes_velocity() {
        let model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let velocity = Vec2::new(50.0, 0.0);
        let force = model.drag_force(velocity);

        // Force should point in the -x direction
        assert!(force.x < 0.0);
        assert!(force.y.abs() < EPSILON);
    }

    #[test]
    fn drag_force_magnitude_formula() {
        let area = 0.05;
        let rho = 1.225;
        let model = DragModel::new(DragShape::Sphere, area, rho);
        let velocity = Vec2::new(30.0, 40.0); // speed = 50 m/s

        let force = model.drag_force(velocity);
        let speed = 50.0;
        let expected_magnitude = 0.5 * CD_SPHERE * area * rho * speed * speed;

        assert!((force.magnitude() - expected_magnitude).abs() < 1e-8);

        // Force should be anti-parallel to velocity
        let cos_angle = force.dot(&velocity) / (force.magnitude() * velocity.magnitude());
        assert!((cos_angle - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn drag_force_scales_with_speed_squared() {
        let model = DragModel::at_sea_level(DragShape::Circle, 0.1);
        let v1 = Vec2::new(10.0, 0.0);
        let v2 = Vec2::new(20.0, 0.0);

        let f1 = model.drag_force(v1).magnitude();
        let f2 = model.drag_force(v2).magnitude();

        // Doubling speed should quadruple force
        assert!((f2 / f1 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn drag_force_zero_velocity() {
        let model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let force = model.drag_force(Vec2::ZERO);
        assert!((force.x).abs() < EPSILON);
        assert!((force.y).abs() < EPSILON);
    }

    #[test]
    fn vacuum_model_no_drag() {
        let model = DragModel::vacuum();
        let force = model.drag_force(Vec2::new(1000.0, 500.0));
        assert!((force.magnitude()).abs() < EPSILON);
    }

    #[test]
    fn drag_force_diagonal_direction() {
        let model = DragModel::at_sea_level(DragShape::Sphere, 0.01);
        let velocity = Vec2::new(1.0, 1.0);
        let force = model.drag_force(velocity);

        // Force components should be equal and negative (45 degree velocity)
        assert!((force.x - force.y).abs() < 1e-12);
        assert!(force.x < 0.0);
    }

    #[test]
    fn airfoil_has_lowest_drag() {
        let area = 1.0;
        let vel = Vec2::new(100.0, 0.0);

        let shapes = [
            DragShape::Circle,
            DragShape::Square,
            DragShape::Rhombus,
            DragShape::Sphere,
        ];

        let airfoil_model = DragModel::at_sea_level(DragShape::Airfoil, area);
        let airfoil_force = airfoil_model.drag_force(vel).magnitude();

        for shape in &shapes {
            let model = DragModel::at_sea_level(*shape, area);
            let force = model.drag_force(vel).magnitude();
            assert!(
                airfoil_force < force,
                "Airfoil drag ({airfoil_force}) should be less than {shape:?} drag ({force})"
            );
        }
    }

    #[test]
    fn drag_model_at_sea_level_uses_correct_density() {
        let model = DragModel::at_sea_level(DragShape::Sphere, 0.05);
        assert!((model.air_density - AIR_DENSITY_SEA_LEVEL).abs() < EPSILON);
    }
}
