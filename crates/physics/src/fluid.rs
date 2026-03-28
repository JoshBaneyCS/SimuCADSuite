//! Particle-based fluid simulation.
//!
//! Provides a simple Lagrangian particle system for advection and velocity
//! field reconstruction. Particles are confined within a [`BoundingBox3`]
//! and can be advected by a uniform velocity field. A velocity field can be
//! reconstructed at arbitrary node positions by averaging nearby particle
//! displacements.
//!
//! Parallelism is achieved via [`rayon`] for both advection and velocity
//! field computation.

use std::collections::HashMap;

use rayon::prelude::*;
use simucad_core::types::{BoundingBox3, Particle, Vec3};

// ---------------------------------------------------------------------------
// Pseudo-random number generator (xorshift64)
// ---------------------------------------------------------------------------

/// A minimal xorshift64 PRNG to avoid pulling in an external RNG crate.
///
/// This is adequate for distributing particles within a domain. It is NOT
/// cryptographically secure.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Seed the generator. A seed of 0 is replaced with a non-zero default.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x1234_5678_9ABC_DEF0 } else { seed },
        }
    }

    /// Generate the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a pseudo-random `f64` in the range [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for a uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// ParticleSystem
// ---------------------------------------------------------------------------

/// A system of Lagrangian particles within a 3D bounding box.
///
/// Particles are represented as position + velocity pairs and can be
/// advected, reflected at boundaries, and queried for velocity field
/// reconstruction.
pub struct ParticleSystem {
    /// The collection of particles.
    pub particles: Vec<Particle>,
    /// Spatial bounds for the simulation domain.
    pub bounds: BoundingBox3,
}

impl ParticleSystem {
    /// Create a new particle system with `count` particles randomly
    /// distributed within `bounds`. All particles start at rest.
    ///
    /// The distribution is deterministic for a given `count` (seeded by
    /// `count` itself) to make simulations reproducible.
    pub fn initialize(bounds: BoundingBox3, count: usize) -> Self {
        let size = bounds.size();
        let mut rng = Xorshift64::new(count as u64);

        let particles = (0..count)
            .map(|_| {
                let x = bounds.min.x + rng.next_f64() * size.x;
                let y = bounds.min.y + rng.next_f64() * size.y;
                let z = bounds.min.z + rng.next_f64() * size.z;
                Particle::at_rest(Vec3::new(x, y, z))
            })
            .collect();

        Self { particles, bounds }
    }

    /// Advect all particles by a uniform velocity field for a timestep `dt`.
    ///
    /// Each particle's position is updated as:
    /// ```text
    /// position += velocity_field * dt
    /// ```
    ///
    /// Particles that leave the bounding box are reflected back inside
    /// (elastic boundary reflection) so that the domain remains populated.
    ///
    /// Uses rayon for parallel iteration.
    pub fn advect(&mut self, velocity: Vec3, dt: f64) {
        let bounds = self.bounds;
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity = velocity;
            p.position = p.position + velocity * dt;

            // Reflect particles back into the domain
            reflect_into_bounds(&mut p.position, &bounds);
        });
    }

    /// Compute a velocity field at the given `node_positions`.
    ///
    /// For each node, the velocity is computed as the mean of
    /// `(particle.position - node_position)` over all particles. This gives
    /// a simple displacement-based velocity estimate.
    ///
    /// Uses rayon for parallel iteration over nodes.
    pub fn compute_velocity_field(&self, node_positions: &[Vec3]) -> Vec<Vec3> {
        if self.particles.is_empty() {
            return vec![Vec3::ZERO; node_positions.len()];
        }

        let inv_count = 1.0 / self.particles.len() as f64;

        // Pre-compute the sum of all particle positions once (O(N) instead
        // of O(N*M) where N = particles, M = nodes).
        let sum = self
            .particles
            .par_iter()
            .fold(
                || Vec3::ZERO,
                |acc, p| acc + p.position,
            )
            .reduce(|| Vec3::ZERO, |a, b| a + b);

        node_positions
            .par_iter()
            .map(|node| {
                // mean(particle_pos - node) = mean(particle_pos) - node
                let mean_pos = sum * inv_count;
                mean_pos - *node
            })
            .collect()
    }

    /// Return the number of particles currently in the system.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Return the number of particles that are inside the bounding box.
    pub fn particles_in_bounds(&self) -> usize {
        self.particles
            .par_iter()
            .filter(|p| self.bounds.contains(&p.position))
            .count()
    }
}

// ---------------------------------------------------------------------------
// Boundary reflection helper
// ---------------------------------------------------------------------------

/// Reflect a position back into the bounding box along each axis
/// independently (elastic reflection).
pub fn reflect_into_bounds(pos: &mut Vec3, bounds: &BoundingBox3) {
    reflect_axis(&mut pos.x, bounds.min.x, bounds.max.x);
    reflect_axis(&mut pos.y, bounds.min.y, bounds.max.y);
    reflect_axis(&mut pos.z, bounds.min.z, bounds.max.z);
}

/// Reflect a scalar coordinate into [lo, hi] using a fold-back strategy.
fn reflect_axis(val: &mut f64, lo: f64, hi: f64) {
    let span = hi - lo;
    if span <= 0.0 {
        *val = lo;
        return;
    }

    // Normalize to [0, span]
    let mut v = *val - lo;

    // Number of full spans from origin
    let periods = (v / span).floor() as i64;
    v -= periods as f64 * span;

    // If we ended up on an odd period, reflect
    if periods.rem_euclid(2) == 1 {
        v = span - v;
    }

    *val = lo + v;
}

// ---------------------------------------------------------------------------
// SpatialHashGrid — O(1) particle-to-cell lookup
// ---------------------------------------------------------------------------

/// A spatial hash grid that bins particles into uniform cubic cells for fast
/// neighbour queries. Building the grid is O(N) and point/radius queries
/// touch only the relevant cells.
pub struct SpatialHashGrid {
    /// Side length of each cubic cell.
    cell_size: f64,
    /// Map from integer cell coordinates to the indices of particles that
    /// fall inside that cell.
    grid: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl SpatialHashGrid {
    /// Create an empty grid with the given cell size.
    ///
    /// # Panics
    /// Panics if `cell_size` is not positive and finite.
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0 && cell_size.is_finite(), "cell_size must be positive and finite");
        Self {
            cell_size,
            grid: HashMap::new(),
        }
    }

    /// (Re-)build the grid from a slice of particles. Previous contents are
    /// cleared.
    pub fn build(&mut self, particles: &[Particle]) {
        self.grid.clear();
        let inv = 1.0 / self.cell_size;
        for (idx, p) in particles.iter().enumerate() {
            let key = (
                (p.position.x * inv).floor() as i64,
                (p.position.y * inv).floor() as i64,
                (p.position.z * inv).floor() as i64,
            );
            self.grid.entry(key).or_default().push(idx);
        }
    }

    /// Return the particle indices stored in the cell at integer coordinates
    /// `(cx, cy, cz)`. Returns an empty slice if the cell is unoccupied.
    pub fn query_cell(&self, cx: i64, cy: i64, cz: i64) -> &[usize] {
        match self.grid.get(&(cx, cy, cz)) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }

    /// Return all particle indices whose positions lie within `radius` of
    /// `center`. This inspects every cell that could overlap the query sphere.
    pub fn query_radius(&self, center: Vec3, radius: f64) -> Vec<usize> {
        let inv = 1.0 / self.cell_size;

        // Integer cell range that covers the query sphere.
        let lo_x = ((center.x - radius) * inv).floor() as i64;
        let hi_x = ((center.x + radius) * inv).floor() as i64;
        let lo_y = ((center.y - radius) * inv).floor() as i64;
        let hi_y = ((center.y + radius) * inv).floor() as i64;
        let lo_z = ((center.z - radius) * inv).floor() as i64;
        let hi_z = ((center.z + radius) * inv).floor() as i64;

        let mut result = Vec::new();
        for cx in lo_x..=hi_x {
            for cy in lo_y..=hi_y {
                for cz in lo_z..=hi_z {
                    if let Some(indices) = self.grid.get(&(cx, cy, cz)) {
                        result.extend(indices);
                    }
                }
            }
        }
        result
    }

    /// Convenience: return the integer cell key for a world-space position.
    pub fn cell_key(&self, pos: Vec3) -> (i64, i64, i64) {
        let inv = 1.0 / self.cell_size;
        (
            (pos.x * inv).floor() as i64,
            (pos.y * inv).floor() as i64,
            (pos.z * inv).floor() as i64,
        )
    }
}

// ---------------------------------------------------------------------------
// Element-aware velocity field computation (mesh-independent)
// ---------------------------------------------------------------------------

/// Compute the centroid of a mesh element given the node positions and the
/// element's node index list.
fn element_centroid(mesh_nodes: &[Vec3], node_indices: &[usize]) -> Vec3 {
    if node_indices.is_empty() {
        return Vec3::ZERO;
    }
    let mut sum = Vec3::ZERO;
    for &ni in node_indices {
        sum = sum + mesh_nodes[ni];
    }
    sum * (1.0 / node_indices.len() as f64)
}

/// Compute the bounding-sphere radius of a mesh element (max distance from
/// centroid to any of its nodes).
fn element_bounding_radius(mesh_nodes: &[Vec3], node_indices: &[usize], centroid: Vec3) -> f64 {
    let mut max_r2: f64 = 0.0;
    for &ni in node_indices {
        let d = mesh_nodes[ni] - centroid;
        let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    max_r2.sqrt()
}

impl ParticleSystem {
    /// Compute a velocity field at `mesh_nodes` using element connectivity
    /// and a [`SpatialHashGrid`] for fast particle lookup.
    ///
    /// For each element (given as a list of node indices into `mesh_nodes`),
    /// the routine finds particles near the element centroid (within its
    /// bounding-sphere radius) and computes the mean displacement from the
    /// centroid. That velocity is then assigned to every node of the element
    /// (accumulated and averaged across all elements sharing a node).
    ///
    /// `mesh_elements` is a slice of node-index lists (e.g.
    /// `&[vec![0,1,2], vec![1,3,2]]`). This avoids coupling to the mesh
    /// crate's `MeshElement` type.
    pub fn compute_velocity_field_with_mesh(
        &self,
        mesh_nodes: &[Vec3],
        mesh_elements: &[Vec<usize>],
        grid: &SpatialHashGrid,
    ) -> Vec<Vec3> {
        let n = mesh_nodes.len();

        // Phase 1: compute per-element velocities in parallel
        let elem_vels: Vec<Vec3> = mesh_elements
            .par_iter()
            .map(|elem_nodes| {
                let centroid = element_centroid(mesh_nodes, elem_nodes);
                let radius = element_bounding_radius(mesh_nodes, elem_nodes, centroid);
                let search_radius = if radius < grid.cell_size { grid.cell_size } else { radius };
                let nearby = grid.query_radius(centroid, search_radius);

                if nearby.is_empty() {
                    Vec3::ZERO
                } else {
                    let mut sum = Vec3::ZERO;
                    for &pi in &nearby {
                        sum = sum + (self.particles[pi].position - centroid);
                    }
                    sum * (1.0 / nearby.len() as f64)
                }
            })
            .collect();

        // Phase 2: scatter to nodes (sequential — cheap O(E * nodes_per_elem))
        let mut vel_sum = vec![Vec3::ZERO; n];
        let mut vel_count = vec![0u32; n];

        for (elem_nodes, elem_vel) in mesh_elements.iter().zip(elem_vels.iter()) {
            for &ni in elem_nodes {
                vel_sum[ni] = vel_sum[ni] + *elem_vel;
                vel_count[ni] += 1;
            }
        }

        vel_sum
            .into_iter()
            .zip(vel_count)
            .map(|(s, c)| if c == 0 { Vec3::ZERO } else { s * (1.0 / c as f64) })
            .collect()
    }

    /// Initialize a particle system by scattering particles inside a mesh
    /// domain.
    ///
    /// Candidate positions are generated uniformly inside `bounds` and kept
    /// only if they fall within at least one mesh element (using a simple
    /// point-in-element test). The routine over-samples to reach the target
    /// `count`.
    ///
    /// `mesh_nodes` and `mesh_elements` follow the same convention as
    /// [`compute_velocity_field_with_mesh`].
    pub fn initialize_in_mesh(
        bounds: BoundingBox3,
        count: usize,
        mesh_nodes: &[Vec3],
        mesh_elements: &[Vec<usize>],
        seed: u64,
    ) -> Self {
        let size = bounds.size();
        let mut rng = Xorshift64::new(seed);
        let mut particles = Vec::with_capacity(count);

        // Over-sample factor: generate many candidates per desired particle.
        let oversample = 4;
        let max_attempts = count * oversample.max(1) + count;

        let mut attempts = 0usize;
        while particles.len() < count && attempts < max_attempts {
            let x = bounds.min.x + rng.next_f64() * size.x;
            let y = bounds.min.y + rng.next_f64() * size.y;
            let z = bounds.min.z + rng.next_f64() * size.z;
            let pos = Vec3::new(x, y, z);
            attempts += 1;

            if point_in_any_element(pos, mesh_nodes, mesh_elements) {
                particles.push(Particle::at_rest(pos));
            }
        }

        // If we could not reach the target count (sparse mesh), accept what
        // we have rather than looping forever.
        Self { particles, bounds }
    }
}

/// Test whether `point` is inside any of the given mesh elements.
///
/// Supports triangles (3-node, using same-side test) and tetrahedra (4-node,
/// using barycentric sign test). Elements with other node counts are skipped.
fn point_in_any_element(point: Vec3, nodes: &[Vec3], elements: &[Vec<usize>]) -> bool {
    for elem in elements {
        match elem.len() {
            3 => {
                // Triangle: project onto the triangle's plane and use
                // barycentric coordinates (2D test, ignore z-thickness).
                let a = nodes[elem[0]];
                let b = nodes[elem[1]];
                let c = nodes[elem[2]];
                if point_in_triangle(point, a, b, c) {
                    return true;
                }
            }
            4 => {
                // Tetrahedron: barycentric coordinate sign test.
                let a = nodes[elem[0]];
                let b = nodes[elem[1]];
                let c = nodes[elem[2]];
                let d = nodes[elem[3]];
                if point_in_tetrahedron(point, a, b, c, d) {
                    return true;
                }
            }
            _ => continue,
        }
    }
    false
}

/// Point-in-triangle test using the cross-product (same-side) method.
/// Works in 3D by comparing cross products along the triangle normal.
fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
}

/// Point-in-tetrahedron test using signed volumes.
fn point_in_tetrahedron(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    let sign_total = signed_volume(a, b, c, d);
    if sign_total.abs() < 1e-15 {
        return false; // degenerate tetrahedron
    }
    let s1 = signed_volume(p, b, c, d);
    let s2 = signed_volume(a, p, c, d);
    let s3 = signed_volume(a, b, p, d);
    let s4 = signed_volume(a, b, c, p);

    // All sub-volumes must have the same sign as the total.
    let pos = sign_total > 0.0;
    if pos {
        s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0 && s4 >= 0.0
    } else {
        s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0 && s4 <= 0.0
    }
}

/// Signed volume of the tetrahedron formed by four points (1/6 of the scalar
/// triple product).
fn signed_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    ab.cross(&ac).dot(&ad) / 6.0
}

// ---------------------------------------------------------------------------
// FluidSimulation — high-level simulation pipeline
// ---------------------------------------------------------------------------

/// A complete fluid simulation pipeline that wraps a [`ParticleSystem`] with
/// a [`SpatialHashGrid`] and provides step/run/progress semantics.
pub struct FluidSimulation {
    /// The underlying particle system.
    pub particle_system: ParticleSystem,
    /// Spatial hash grid kept in sync with particle positions.
    pub spatial_grid: SpatialHashGrid,
    /// Total number of simulation steps to execute.
    pub step_count: usize,
    /// Number of steps completed so far.
    pub current_step: usize,
}

impl FluidSimulation {
    /// Create a new simulation. The spatial grid is built immediately from the
    /// current particle positions.
    pub fn new(system: ParticleSystem, cell_size: f64, steps: usize) -> Self {
        let mut spatial_grid = SpatialHashGrid::new(cell_size);
        spatial_grid.build(&system.particles);
        Self {
            particle_system: system,
            spatial_grid,
            step_count: steps,
            current_step: 0,
        }
    }

    /// Perform a single advection step: move particles, reflect at boundaries,
    /// rebuild the spatial grid, and increment the step counter.
    pub fn step(&mut self, velocity: Vec3, dt: f64) {
        self.particle_system.advect(velocity, dt);
        self.spatial_grid.build(&self.particle_system.particles);
        self.current_step += 1;
    }

    /// Run all remaining steps (from `current_step` to `step_count`).
    pub fn run(&mut self, velocity: Vec3, dt: f64) {
        while self.current_step < self.step_count {
            self.step(velocity, dt);
        }
    }

    /// Return the simulation progress as a fraction in `[0.0, 1.0]`.
    pub fn progress(&self) -> f32 {
        if self.step_count == 0 {
            return 1.0;
        }
        self.current_step as f32 / self.step_count as f32
    }

    /// Convenience wrapper: compute the velocity field at the given node
    /// positions using the current particle state (simple mean-displacement
    /// method from [`ParticleSystem::compute_velocity_field`]).
    pub fn velocity_field(&self, nodes: &[Vec3]) -> Vec<Vec3> {
        self.particle_system.compute_velocity_field(nodes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> BoundingBox3 {
        BoundingBox3::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0))
    }

    fn big_box() -> BoundingBox3 {
        BoundingBox3::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0))
    }

    // ----- Xorshift64 tests -----

    #[test]
    fn xorshift_produces_different_values() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = rng.next_u64();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn xorshift_f64_in_range() {
        let mut rng = Xorshift64::new(123);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
        }
    }

    #[test]
    fn xorshift_zero_seed_handled() {
        let mut rng = Xorshift64::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    // ----- ParticleSystem::initialize tests -----

    #[test]
    fn initialize_correct_count() {
        for count in [0, 1, 100, 5000] {
            let sys = ParticleSystem::initialize(unit_box(), count);
            assert_eq!(sys.particle_count(), count);
        }
    }

    #[test]
    fn initialize_particles_in_bounds() {
        let bounds = big_box();
        let sys = ParticleSystem::initialize(bounds, 10_000);

        for p in &sys.particles {
            assert!(
                bounds.contains(&p.position),
                "particle at {:?} is outside bounds",
                p.position
            );
        }
    }

    #[test]
    fn initialize_particles_at_rest() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        for p in &sys.particles {
            assert_eq!(p.velocity, Vec3::ZERO);
        }
    }

    #[test]
    fn initialize_deterministic() {
        let a = ParticleSystem::initialize(unit_box(), 500);
        let b = ParticleSystem::initialize(unit_box(), 500);
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    // ----- advect tests -----

    #[test]
    fn advect_updates_positions() {
        let mut sys = ParticleSystem::initialize(big_box(), 100);
        let initial_positions: Vec<Vec3> = sys.particles.iter().map(|p| p.position).collect();

        let vel = Vec3::new(1.0, 0.0, 0.0);
        sys.advect(vel, 0.5);

        for (i, p) in sys.particles.iter().enumerate() {
            let expected_x = initial_positions[i].x + 0.5;
            // May differ slightly if reflected, but within big_box should be fine
            if big_box().contains(&Vec3::new(expected_x, p.position.y, p.position.z)) {
                assert!(
                    (p.position.x - expected_x).abs() < 1e-10,
                    "particle {i} x: {} vs expected: {}",
                    p.position.x,
                    expected_x
                );
            }
        }
    }

    #[test]
    fn advect_sets_velocity() {
        let mut sys = ParticleSystem::initialize(unit_box(), 50);
        let vel = Vec3::new(2.0, -1.0, 0.5);
        sys.advect(vel, 0.1);

        for p in &sys.particles {
            assert_eq!(p.velocity, vel);
        }
    }

    #[test]
    fn advect_particles_remain_in_bounds() {
        let bounds = unit_box();
        let mut sys = ParticleSystem::initialize(bounds, 1000);

        // Advect with a large velocity to force boundary reflections
        let vel = Vec3::new(100.0, -50.0, 75.0);
        for _ in 0..10 {
            sys.advect(vel, 0.1);
        }

        for p in &sys.particles {
            assert!(
                bounds.contains(&p.position),
                "particle at {:?} escaped bounds after advection",
                p.position
            );
        }
    }

    #[test]
    fn advect_zero_velocity_no_change() {
        let mut sys = ParticleSystem::initialize(unit_box(), 100);
        let positions_before: Vec<Vec3> = sys.particles.iter().map(|p| p.position).collect();
        sys.advect(Vec3::ZERO, 1.0);

        for (i, p) in sys.particles.iter().enumerate() {
            assert_eq!(p.position, positions_before[i]);
        }
    }

    // ----- compute_velocity_field tests -----

    #[test]
    fn velocity_field_empty_particles() {
        let sys = ParticleSystem::initialize(unit_box(), 0);
        let nodes = vec![Vec3::new(0.5, 0.5, 0.5)];
        let field = sys.compute_velocity_field(&nodes);
        assert_eq!(field.len(), 1);
        assert_eq!(field[0], Vec3::ZERO);
    }

    #[test]
    fn velocity_field_empty_nodes() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let field = sys.compute_velocity_field(&[]);
        assert!(field.is_empty());
    }

    #[test]
    fn velocity_field_correct_count() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let nodes: Vec<Vec3> = (0..25).map(|i| Vec3::new(i as f64 * 0.04, 0.5, 0.5)).collect();
        let field = sys.compute_velocity_field(&nodes);
        assert_eq!(field.len(), 25);
    }

    #[test]
    fn velocity_field_single_particle() {
        // With a single particle at a known position, the velocity at a node
        // should be (particle_pos - node_pos).
        let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let sys = ParticleSystem {
            particles: vec![Particle::at_rest(Vec3::new(5.0, 5.0, 5.0))],
            bounds,
        };

        let nodes = vec![Vec3::new(1.0, 2.0, 3.0)];
        let field = sys.compute_velocity_field(&nodes);

        assert!((field[0].x - 4.0).abs() < 1e-10);
        assert!((field[0].y - 3.0).abs() < 1e-10);
        assert!((field[0].z - 2.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_field_symmetric_particles() {
        // Two particles symmetric around the origin: their mean position is
        // the origin, so the velocity at the origin should be ZERO.
        let bounds = BoundingBox3::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0));
        let sys = ParticleSystem {
            particles: vec![
                Particle::at_rest(Vec3::new(1.0, 0.0, 0.0)),
                Particle::at_rest(Vec3::new(-1.0, 0.0, 0.0)),
            ],
            bounds,
        };

        let nodes = vec![Vec3::ZERO];
        let field = sys.compute_velocity_field(&nodes);

        assert!((field[0].x).abs() < 1e-10);
        assert!((field[0].y).abs() < 1e-10);
        assert!((field[0].z).abs() < 1e-10);
    }

    // ----- reflect_axis tests -----

    #[test]
    fn reflect_axis_inside_unchanged() {
        let mut v = 0.5;
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reflect_axis_at_boundary() {
        let mut v = 1.0;
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_axis_beyond_upper() {
        let mut v = 1.3;
        reflect_axis(&mut v, 0.0, 1.0);
        // 1.3 -> reflects to 0.7
        assert!((v - 0.7).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_beyond_lower() {
        let mut v = -0.3;
        reflect_axis(&mut v, 0.0, 1.0);
        // -0.3 -> reflects to 0.3
        assert!((v - 0.3).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_multiple_reflections() {
        let mut v = 2.7; // 2.7 spans from 0: period 2 (even), remainder 0.7
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 0.7).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_negative_side_multiple() {
        let mut v = -1.4; // -1.4 from 0: normalize => -1.4, floor(-1.4) = -2, periods = -2
        reflect_axis(&mut v, 0.0, 1.0);
        // After normalization: v = -1.4 - (-2)*1 = 0.6, periods=-2 (even), so v=0.6
        assert!((v - 0.6).abs() < 1e-10, "reflected value: {v}");
    }

    // ----- particles_in_bounds test -----

    #[test]
    fn particles_in_bounds_after_init() {
        let sys = ParticleSystem::initialize(unit_box(), 500);
        assert_eq!(sys.particles_in_bounds(), 500);
    }

    // ----- Integration test: advect then compute field -----

    #[test]
    fn advect_then_compute_field() {
        let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let mut sys = ParticleSystem::initialize(bounds, 5000);

        // Advect a little
        sys.advect(Vec3::new(0.1, 0.2, 0.05), 1.0);

        // Compute velocity field at center
        let nodes = vec![Vec3::new(5.0, 5.0, 5.0)];
        let field = sys.compute_velocity_field(&nodes);

        // Should produce a finite, non-zero vector
        assert!(field[0].magnitude().is_finite());
    }

    // ===================================================================
    // SpatialHashGrid tests
    // ===================================================================

    #[test]
    fn spatial_grid_new() {
        let grid = SpatialHashGrid::new(1.0);
        assert_eq!(grid.cell_size, 1.0);
        assert!(grid.grid.is_empty());
    }

    #[test]
    #[should_panic]
    fn spatial_grid_zero_cell_size_panics() {
        SpatialHashGrid::new(0.0);
    }

    #[test]
    #[should_panic]
    fn spatial_grid_negative_cell_size_panics() {
        SpatialHashGrid::new(-1.0);
    }

    #[test]
    fn spatial_grid_build_single_particle() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![Particle::at_rest(Vec3::new(0.5, 0.5, 0.5))];
        grid.build(&particles);

        let cell = grid.query_cell(0, 0, 0);
        assert_eq!(cell, &[0]);
    }

    #[test]
    fn spatial_grid_build_multiple_particles_same_cell() {
        let mut grid = SpatialHashGrid::new(2.0);
        let particles = vec![
            Particle::at_rest(Vec3::new(0.1, 0.1, 0.1)),
            Particle::at_rest(Vec3::new(0.9, 0.9, 0.9)),
            Particle::at_rest(Vec3::new(1.5, 1.5, 1.5)),
        ];
        grid.build(&particles);

        // First two in cell (0,0,0), third in cell (0,0,0) as well (1.5 / 2.0 = 0.75 -> floor = 0)
        let cell = grid.query_cell(0, 0, 0);
        assert_eq!(cell.len(), 3);
    }

    #[test]
    fn spatial_grid_build_particles_different_cells() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![
            Particle::at_rest(Vec3::new(0.5, 0.5, 0.5)),
            Particle::at_rest(Vec3::new(1.5, 0.5, 0.5)),
            Particle::at_rest(Vec3::new(0.5, 1.5, 0.5)),
        ];
        grid.build(&particles);

        assert_eq!(grid.query_cell(0, 0, 0), &[0]);
        assert_eq!(grid.query_cell(1, 0, 0), &[1]);
        assert_eq!(grid.query_cell(0, 1, 0), &[2]);
    }

    #[test]
    fn spatial_grid_query_cell_empty() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![Particle::at_rest(Vec3::new(0.5, 0.5, 0.5))];
        grid.build(&particles);

        assert!(grid.query_cell(5, 5, 5).is_empty());
    }

    #[test]
    fn spatial_grid_query_radius_finds_nearby() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![
            Particle::at_rest(Vec3::new(0.0, 0.0, 0.0)),
            Particle::at_rest(Vec3::new(0.5, 0.0, 0.0)),
            Particle::at_rest(Vec3::new(5.0, 5.0, 5.0)),
        ];
        grid.build(&particles);

        let near = grid.query_radius(Vec3::new(0.25, 0.0, 0.0), 1.0);
        // Particles 0 and 1 are within radius 1.0 of (0.25, 0, 0)
        assert!(near.contains(&0));
        assert!(near.contains(&1));
        // Particle 2 is far away — should not appear
        assert!(!near.contains(&2));
    }

    #[test]
    fn spatial_grid_query_radius_empty() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![Particle::at_rest(Vec3::new(10.0, 10.0, 10.0))];
        grid.build(&particles);

        let near = grid.query_radius(Vec3::ZERO, 1.0);
        assert!(near.is_empty());
    }

    #[test]
    fn spatial_grid_query_radius_negative_coords() {
        let mut grid = SpatialHashGrid::new(1.0);
        let particles = vec![
            Particle::at_rest(Vec3::new(-0.5, -0.5, -0.5)),
            Particle::at_rest(Vec3::new(-1.5, -1.5, -1.5)),
        ];
        grid.build(&particles);

        let near = grid.query_radius(Vec3::new(-0.5, -0.5, -0.5), 0.1);
        assert!(near.contains(&0));
        assert!(!near.contains(&1));
    }

    #[test]
    fn spatial_grid_rebuild_clears_old() {
        let mut grid = SpatialHashGrid::new(1.0);
        let p1 = vec![Particle::at_rest(Vec3::new(0.5, 0.5, 0.5))];
        grid.build(&p1);
        assert_eq!(grid.query_cell(0, 0, 0).len(), 1);

        // Rebuild with different particles
        let p2 = vec![Particle::at_rest(Vec3::new(5.5, 5.5, 5.5))];
        grid.build(&p2);
        assert!(grid.query_cell(0, 0, 0).is_empty());
        assert_eq!(grid.query_cell(5, 5, 5).len(), 1);
    }

    #[test]
    fn spatial_grid_cell_key() {
        let grid = SpatialHashGrid::new(2.0);
        assert_eq!(grid.cell_key(Vec3::new(0.5, 0.5, 0.5)), (0, 0, 0));
        assert_eq!(grid.cell_key(Vec3::new(2.5, 4.5, -0.5)), (1, 2, -1));
    }

    // ===================================================================
    // Element-aware velocity field tests
    // ===================================================================

    /// Helper: build a simple triangulated quad in the XY plane.
    fn quad_mesh_nodes() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
    }

    fn quad_mesh_elements() -> Vec<Vec<usize>> {
        vec![vec![0, 1, 2], vec![0, 2, 3]]
    }

    #[test]
    fn velocity_field_with_mesh_correct_length() {
        let nodes = quad_mesh_nodes();
        let elems = quad_mesh_elements();
        let bounds = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 2.0, 2.0));
        let sys = ParticleSystem::initialize(bounds, 500);

        let mut grid = SpatialHashGrid::new(0.5);
        grid.build(&sys.particles);

        let field = sys.compute_velocity_field_with_mesh(&nodes, &elems, &grid);
        assert_eq!(field.len(), nodes.len());
    }

    #[test]
    fn velocity_field_with_mesh_no_particles() {
        let nodes = quad_mesh_nodes();
        let elems = quad_mesh_elements();
        let bounds = unit_box();
        let sys = ParticleSystem {
            particles: vec![],
            bounds,
        };

        let mut grid = SpatialHashGrid::new(0.5);
        grid.build(&sys.particles);

        let field = sys.compute_velocity_field_with_mesh(&nodes, &elems, &grid);
        for v in &field {
            assert_eq!(*v, Vec3::ZERO);
        }
    }

    #[test]
    fn velocity_field_with_mesh_single_particle_at_centroid() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let elems = vec![vec![0, 1, 2]];
        // Centroid = (1, 1, 0). Place a particle exactly there.
        let bounds = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(5.0, 5.0, 5.0));
        let sys = ParticleSystem {
            particles: vec![Particle::at_rest(Vec3::new(1.0, 1.0, 0.0))],
            bounds,
        };

        let mut grid = SpatialHashGrid::new(2.0);
        grid.build(&sys.particles);

        let field = sys.compute_velocity_field_with_mesh(&nodes, &elems, &grid);
        // The single particle is at the centroid, so displacement = ZERO
        for v in &field {
            assert!(v.magnitude() < 1e-10, "expected near-zero, got {:?}", v);
        }
    }

    // ===================================================================
    // Point-in-element tests
    // ===================================================================

    #[test]
    fn point_in_triangle_inside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 4.0, 0.0);
        assert!(point_in_triangle(Vec3::new(1.0, 1.0, 0.0), a, b, c));
    }

    #[test]
    fn point_in_triangle_outside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 4.0, 0.0);
        assert!(!point_in_triangle(Vec3::new(3.0, 3.0, 0.0), a, b, c));
    }

    #[test]
    fn point_in_triangle_on_edge() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        // Midpoint of edge AB
        assert!(point_in_triangle(Vec3::new(1.0, 0.0, 0.0), a, b, c));
    }

    #[test]
    fn point_in_tetrahedron_inside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 4.0, 0.0);
        let d = Vec3::new(0.0, 0.0, 4.0);
        assert!(point_in_tetrahedron(Vec3::new(0.5, 0.5, 0.5), a, b, c, d));
    }

    #[test]
    fn point_in_tetrahedron_outside() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 4.0, 0.0);
        let d = Vec3::new(0.0, 0.0, 4.0);
        assert!(!point_in_tetrahedron(Vec3::new(5.0, 5.0, 5.0), a, b, c, d));
    }

    #[test]
    fn point_in_any_element_triangle() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let elems = vec![vec![0, 1, 2]];
        assert!(point_in_any_element(Vec3::new(1.0, 1.0, 0.0), &nodes, &elems));
        assert!(!point_in_any_element(Vec3::new(3.0, 3.0, 0.0), &nodes, &elems));
    }

    #[test]
    fn point_in_any_element_tetrahedron() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
        ];
        let elems = vec![vec![0, 1, 2, 3]];
        assert!(point_in_any_element(Vec3::new(0.5, 0.5, 0.5), &nodes, &elems));
        assert!(!point_in_any_element(Vec3::new(5.0, 5.0, 5.0), &nodes, &elems));
    }

    // ===================================================================
    // initialize_in_mesh tests
    // ===================================================================

    #[test]
    fn initialize_in_mesh_particles_inside_triangle() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
        ];
        let elems = vec![vec![0, 1, 2]];
        let bounds = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(11.0, 11.0, 1.0));

        let sys = ParticleSystem::initialize_in_mesh(bounds, 50, &nodes, &elems, 42);

        // All particles must lie inside the triangle
        for p in &sys.particles {
            assert!(
                point_in_triangle(p.position, nodes[0], nodes[1], nodes[2]),
                "particle at {:?} is outside triangle",
                p.position
            );
        }
    }

    #[test]
    fn initialize_in_mesh_particles_inside_tetrahedron() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        ];
        let elems = vec![vec![0, 1, 2, 3]];
        let bounds = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(11.0, 11.0, 11.0));

        let sys = ParticleSystem::initialize_in_mesh(bounds, 30, &nodes, &elems, 99);

        for p in &sys.particles {
            assert!(
                point_in_tetrahedron(p.position, nodes[0], nodes[1], nodes[2], nodes[3]),
                "particle at {:?} is outside tetrahedron",
                p.position
            );
        }
    }

    #[test]
    fn initialize_in_mesh_deterministic() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        ];
        let elems = vec![vec![0, 1, 2, 3]];
        let bounds = BoundingBox3::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(11.0, 11.0, 11.0));

        let a = ParticleSystem::initialize_in_mesh(bounds, 20, &nodes, &elems, 77);
        let b = ParticleSystem::initialize_in_mesh(bounds, 20, &nodes, &elems, 77);
        assert_eq!(a.particles.len(), b.particles.len());
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    // ===================================================================
    // FluidSimulation pipeline tests
    // ===================================================================

    #[test]
    fn fluid_sim_new_builds_grid() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let sim = FluidSimulation::new(sys, 0.5, 10);
        assert_eq!(sim.current_step, 0);
        assert_eq!(sim.step_count, 10);
        assert!((sim.progress() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn fluid_sim_step_increments() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let mut sim = FluidSimulation::new(sys, 0.5, 10);

        sim.step(Vec3::new(0.1, 0.0, 0.0), 0.01);
        assert_eq!(sim.current_step, 1);
        assert!((sim.progress() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn fluid_sim_run_completes_all_steps() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let mut sim = FluidSimulation::new(sys, 0.5, 5);

        sim.run(Vec3::new(0.01, 0.0, 0.0), 0.01);
        assert_eq!(sim.current_step, 5);
        assert!((sim.progress() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fluid_sim_run_after_partial() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let mut sim = FluidSimulation::new(sys, 0.5, 5);

        sim.step(Vec3::new(0.01, 0.0, 0.0), 0.01);
        sim.step(Vec3::new(0.01, 0.0, 0.0), 0.01);
        assert_eq!(sim.current_step, 2);

        sim.run(Vec3::new(0.01, 0.0, 0.0), 0.01);
        assert_eq!(sim.current_step, 5);
    }

    #[test]
    fn fluid_sim_progress_zero_steps() {
        let sys = ParticleSystem::initialize(unit_box(), 10);
        let sim = FluidSimulation::new(sys, 0.5, 0);
        assert!((sim.progress() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fluid_sim_velocity_field() {
        let sys = ParticleSystem::initialize(unit_box(), 200);
        let sim = FluidSimulation::new(sys, 0.5, 5);

        let nodes = vec![Vec3::new(0.5, 0.5, 0.5)];
        let field = sim.velocity_field(&nodes);
        assert_eq!(field.len(), 1);
        assert!(field[0].magnitude().is_finite());
    }

    #[test]
    fn fluid_sim_particles_stay_in_bounds() {
        let bounds = unit_box();
        let sys = ParticleSystem::initialize(bounds, 500);
        let mut sim = FluidSimulation::new(sys, 0.25, 20);

        sim.run(Vec3::new(5.0, -3.0, 2.0), 0.05);

        for p in &sim.particle_system.particles {
            assert!(
                bounds.contains(&p.position),
                "particle at {:?} escaped bounds during simulation",
                p.position
            );
        }
    }

    #[test]
    fn fluid_sim_grid_updated_after_step() {
        let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let sys = ParticleSystem {
            particles: vec![Particle::at_rest(Vec3::new(0.5, 0.5, 0.5))],
            bounds,
        };
        let mut sim = FluidSimulation::new(sys, 1.0, 10);

        // Particle starts in cell (0,0,0)
        assert_eq!(sim.spatial_grid.query_cell(0, 0, 0).len(), 1);

        // Advect it to the right by 2 units
        sim.step(Vec3::new(2.0, 0.0, 0.0), 1.0);

        // Now particle should be at x=2.5, cell (2,0,0)
        assert_eq!(sim.spatial_grid.query_cell(2, 0, 0).len(), 1);
        assert!(sim.spatial_grid.query_cell(0, 0, 0).is_empty());
    }
}
