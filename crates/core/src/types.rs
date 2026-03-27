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
}
