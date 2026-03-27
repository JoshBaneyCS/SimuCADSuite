use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

// ---------------------------------------------------------------------------
// Vec2 — 2D vector for planar kinematics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            return Self::ZERO;
        }
        Self {
            x: self.x / mag,
            y: self.y / mag,
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Angle in radians from the positive x-axis (using `atan2`).
    pub fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Rotate the vector by `angle` radians counter-clockwise.
    pub fn rotate(&self, angle: f64) -> Vec2 {
        let (sin, cos) = angle.sin_cos();
        Vec2 {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }

    /// Linear interpolation between `self` and `other`.
    pub fn lerp(&self, other: &Vec2, t: f64) -> Vec2 {
        Vec2 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Vec2) -> f64 {
        (*self - *other).magnitude()
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3 — 3D vector for spatial simulations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            return Self::ZERO;
        }
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Linear interpolation between `self` and `other`.
    pub fn lerp(&self, other: &Vec3, t: f64) -> Vec3 {
        Vec3 {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Vec3) -> f64 {
        (*self - *other).magnitude()
    }

    /// Reflect this vector off a surface with the given normal.
    ///
    /// `normal` should be a unit vector.  The result is `self - 2 * (self . normal) * normal`.
    pub fn reflect(&self, normal: &Vec3) -> Vec3 {
        let d = self.dot(normal);
        Vec3 {
            x: self.x - 2.0 * d * normal.x,
            y: self.y - 2.0 * d * normal.y,
            z: self.z - 2.0 * d * normal.z,
        }
    }

    /// Project this vector onto `other`.
    ///
    /// Returns the zero vector if `other` has zero length.
    pub fn project_onto(&self, other: &Vec3) -> Vec3 {
        let denom = other.dot(other);
        if denom < f64::EPSILON {
            return Vec3::ZERO;
        }
        let scalar = self.dot(other) / denom;
        *other * scalar
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// ---------------------------------------------------------------------------
// Particle — used by fluid dynamics simulation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
}

impl Particle {
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self { position, velocity }
    }

    pub fn at_rest(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Trajectory types — output of kinematics solvers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub time: f64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub speed: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub points: Vec<TrajectoryPoint>,
    pub max_height: f64,
    pub range: f64,
    pub flight_time: f64,
}

impl Trajectory {
    pub fn empty() -> Self {
        Self {
            points: Vec::new(),
            max_height: 0.0,
            range: 0.0,
            flight_time: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Kinematic state — intermediate solver state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KinematicState {
    pub time: f64,
    pub position: Vec2,
    pub velocity: Vec2,
}

// ---------------------------------------------------------------------------
// Simulation configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub use_gpu: bool,
    pub thread_count: Option<usize>,
    pub timestep: f64,
    pub max_steps: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            use_gpu: false,
            thread_count: None,
            timestep: 0.01,
            max_steps: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Bounding box — used by mesh and fluid simulations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox3 {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, point: &Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// The geometric center of the bounding box.
    pub fn center(&self) -> Vec3 {
        Vec3 {
            x: (self.min.x + self.max.x) * 0.5,
            y: (self.min.y + self.max.y) * 0.5,
            z: (self.min.z + self.max.z) * 0.5,
        }
    }

    /// Return a new bounding box expanded by `margin` on every side.
    pub fn expand(&self, margin: f64) -> BoundingBox3 {
        BoundingBox3 {
            min: Vec3::new(self.min.x - margin, self.min.y - margin, self.min.z - margin),
            max: Vec3::new(self.max.x + margin, self.max.y + margin, self.max.z + margin),
        }
    }

    /// Returns `true` if this box overlaps `other` (touching counts).
    pub fn intersects(&self, other: &BoundingBox3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Return the smallest bounding box that contains both `self` and `other`.
    pub fn union(&self, other: &BoundingBox3) -> BoundingBox3 {
        BoundingBox3 {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Build a bounding box from a slice of points.
    ///
    /// Returns `None` if the slice is empty.
    pub fn from_points(points: &[Vec3]) -> Option<BoundingBox3> {
        let first = points.first()?;
        let mut min = *first;
        let mut max = *first;
        for p in &points[1..] {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        Some(BoundingBox3 { min, max })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < 1e-12);

        let b = Vec2::new(1.0, 2.0);
        let sum = a + b;
        assert_eq!(sum, Vec2::new(4.0, 6.0));

        let scaled = a * 2.0;
        assert_eq!(scaled, Vec2::new(6.0, 8.0));
    }

    #[test]
    fn vec3_cross_product() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let j = Vec3::new(0.0, 1.0, 0.0);
        let k = i.cross(&j);
        assert_eq!(k, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn bounding_box_contains() {
        let bbox = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(bbox.contains(&Vec3::ZERO));
        assert!(!bbox.contains(&Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn particle_at_rest() {
        let p = Particle::at_rest(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(p.velocity, Vec3::ZERO);
    }

    // -- Vec2 new methods ---------------------------------------------------

    #[test]
    fn vec2_angle() {
        let v = Vec2::new(1.0, 0.0);
        assert!((v.angle()).abs() < 1e-12);

        let v = Vec2::new(0.0, 1.0);
        assert!((v.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let v = Vec2::new(-1.0, 0.0);
        assert!((v.angle() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let rotated = v.rotate(std::f64::consts::FRAC_PI_2);
        assert!((rotated.x).abs() < 1e-12);
        assert!((rotated.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vec2_rotate_full_circle() {
        let v = Vec2::new(3.0, 4.0);
        let rotated = v.rotate(2.0 * std::f64::consts::PI);
        assert!((rotated.x - v.x).abs() < 1e-10);
        assert!((rotated.y - v.y).abs() < 1e-10);
    }

    #[test]
    fn vec2_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-12);
        assert!((mid.y - 10.0).abs() < 1e-12);
    }

    #[test]
    fn vec2_distance_to() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-12);
    }

    // -- Vec3 new methods ---------------------------------------------------

    #[test]
    fn vec3_lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 4.0, 6.0);
        let mid = a.lerp(&b, 0.25);
        assert!((mid.x - 0.5).abs() < 1e-12);
        assert!((mid.y - 1.0).abs() < 1e-12);
        assert!((mid.z - 1.5).abs() < 1e-12);
    }

    #[test]
    fn vec3_distance_to() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 6.0, 3.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn vec3_reflect() {
        // Reflect (1, -1, 0) off horizontal surface normal (0, 1, 0)
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(&n);
        assert!((r.x - 1.0).abs() < 1e-12);
        assert!((r.y - 1.0).abs() < 1e-12);
        assert!((r.z).abs() < 1e-12);
    }

    #[test]
    fn vec3_reflect_head_on() {
        // Straight into a wall: (-1, 0, 0) off normal (1, 0, 0) => (1, 0, 0)
        let v = Vec3::new(-1.0, 0.0, 0.0);
        let n = Vec3::new(1.0, 0.0, 0.0);
        let r = v.reflect(&n);
        assert!((r.x - 1.0).abs() < 1e-12);
        assert!((r.y).abs() < 1e-12);
        assert!((r.z).abs() < 1e-12);
    }

    #[test]
    fn vec3_project_onto() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let proj = v.project_onto(&axis);
        assert!((proj.x - 3.0).abs() < 1e-12);
        assert!((proj.y).abs() < 1e-12);
        assert!((proj.z).abs() < 1e-12);
    }

    #[test]
    fn vec3_project_onto_zero() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let zero = Vec3::ZERO;
        let proj = v.project_onto(&zero);
        assert_eq!(proj, Vec3::ZERO);
    }

    // -- BoundingBox3 new methods -------------------------------------------

    #[test]
    fn bbox_center() {
        let bb = BoundingBox3::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let c = bb.center();
        assert!((c.x).abs() < 1e-12);
        assert!((c.y).abs() < 1e-12);
        assert!((c.z).abs() < 1e-12);
    }

    #[test]
    fn bbox_expand() {
        let bb = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let expanded = bb.expand(0.5);
        assert!((expanded.min.x - (-0.5)).abs() < 1e-12);
        assert!((expanded.max.x - 1.5).abs() < 1e-12);
    }

    #[test]
    fn bbox_intersects() {
        let a = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let b = BoundingBox3::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));

        let c = BoundingBox3::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn bbox_intersects_touching() {
        let a = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = BoundingBox3::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn bbox_union() {
        let a = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = BoundingBox3::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(u.max, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn bbox_from_points() {
        let points = vec![
            Vec3::new(1.0, 5.0, -2.0),
            Vec3::new(-3.0, 0.0, 4.0),
            Vec3::new(2.0, 2.0, 2.0),
        ];
        let bb = BoundingBox3::from_points(&points).unwrap();
        assert_eq!(bb.min, Vec3::new(-3.0, 0.0, -2.0));
        assert_eq!(bb.max, Vec3::new(2.0, 5.0, 4.0));
    }

    #[test]
    fn bbox_from_points_empty() {
        assert!(BoundingBox3::from_points(&[]).is_none());
    }

    #[test]
    fn bbox_from_points_single() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let bb = BoundingBox3::from_points(&[p]).unwrap();
        assert_eq!(bb.min, p);
        assert_eq!(bb.max, p);
    }
}
