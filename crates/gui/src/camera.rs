//! Orbit camera for the 3D viewport.
//!
//! Provides a camera that revolves around a target point with mouse-driven
//! rotation, zoom, and pan. Computes view and perspective projection matrices
//! in column-major order suitable for wgpu/WGSL.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6};

// ---------------------------------------------------------------------------
// OrbitCamera
// ---------------------------------------------------------------------------

/// An orbit camera that revolves around a target point.
pub struct OrbitCamera {
    /// Horizontal angle in radians.
    pub azimuth: f32,
    /// Vertical angle in radians (clamped to avoid gimbal lock).
    pub elevation: f32,
    /// Distance from the target.
    pub distance: f32,
    /// The point the camera orbits around.
    pub target: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            azimuth: FRAC_PI_4,
            elevation: FRAC_PI_6,
            distance: 10.0,
            target: [0.0, 0.0, 0.0],
            fov_y: FRAC_PI_4,
            near: 0.01,
            far: 1000.0,
        }
    }
}

impl OrbitCamera {
    /// Compute the camera eye position in world space.
    pub fn eye_position(&self) -> [f32; 3] {
        let cos_elev = self.elevation.cos();
        [
            self.target[0] + self.distance * cos_elev * self.azimuth.cos(),
            self.target[1] + self.distance * self.elevation.sin(),
            self.target[2] + self.distance * cos_elev * self.azimuth.sin(),
        ]
    }

    /// Rotate the camera by pixel deltas (primary drag).
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * 0.01;
        self.elevation =
            (self.elevation + dy * 0.01).clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
    }

    /// Zoom by scroll delta (positive = zoom in).
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.1)).clamp(0.1, 10_000.0);
    }

    /// Pan the target in the camera's local right/up plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let eye = self.eye_position();
        let forward = normalize(sub(self.target, eye));
        let world_up = [0.0, 1.0, 0.0];
        let right = normalize(cross(forward, world_up));
        let up = cross(right, forward);

        let scale = self.distance * 0.002;
        for i in 0..3 {
            self.target[i] -= (right[i] * dx + up[i] * dy) * scale;
        }
    }

    /// Adjust camera to frame a bounding box.
    pub fn fit_to_bounds(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.target = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        let diagonal = (dx * dx + dy * dy + dz * dz).sqrt();
        self.distance = diagonal.max(1.0) * 1.5;
    }

    /// Compute the 4x4 view matrix (column-major).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        look_at(self.eye_position(), self.target, [0.0, 1.0, 0.0])
    }

    /// Compute the 4x4 perspective projection matrix (column-major).
    pub fn projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        perspective(self.fov_y, aspect, self.near, self.far)
    }

    /// Compute the combined view-projection matrix (column-major).
    pub fn view_projection(&self, aspect: f32) -> [[f32; 4]; 4] {
        mat4_mul(self.projection_matrix(aspect), self.view_matrix())
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (f32, column-major)
// ---------------------------------------------------------------------------

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt();
    if len < 1e-10 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Build a look-at view matrix (column-major, right-handed).
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

/// Build a perspective projection matrix (column-major, clip-space z in 0..1
/// for wgpu).
fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y / 2.0).tan();
    let range_inv = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * range_inv, -1.0],
        [0.0, 0.0, near * far * range_inv, 0.0],
    ]
}

/// Multiply two 4x4 column-major matrices: result = a * b.
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_looks_at_origin() {
        let cam = OrbitCamera::default();
        assert_eq!(cam.target, [0.0, 0.0, 0.0]);
        assert!(cam.distance > 0.0);
    }

    #[test]
    fn rotate_changes_azimuth() {
        let mut cam = OrbitCamera::default();
        let original = cam.azimuth;
        cam.rotate(100.0, 0.0);
        assert_ne!(cam.azimuth, original);
    }

    #[test]
    fn zoom_changes_distance() {
        let mut cam = OrbitCamera::default();
        let original = cam.distance;
        cam.zoom(1.0);
        assert!(cam.distance < original);
    }

    #[test]
    fn elevation_is_clamped() {
        let mut cam = OrbitCamera::default();
        cam.rotate(0.0, 10_000.0);
        assert!(cam.elevation < FRAC_PI_2);
        cam.rotate(0.0, -20_000.0);
        assert!(cam.elevation > -FRAC_PI_2);
    }

    #[test]
    fn fit_to_bounds_centers_target() {
        let mut cam = OrbitCamera::default();
        cam.fit_to_bounds([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
        assert_eq!(cam.target, [0.0, 0.0, 0.0]);
        assert!(cam.distance > 10.0);
    }

    #[test]
    fn view_projection_is_finite() {
        let cam = OrbitCamera::default();
        let vp = cam.view_projection(1.5);
        for col in &vp {
            for &val in col {
                assert!(val.is_finite(), "view_projection has non-finite value");
            }
        }
    }
}
