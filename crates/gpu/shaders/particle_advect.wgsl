// Particle advection compute shader.
//
// Each particle is stored as two vec4s (position + velocity) packed
// contiguously in a storage buffer. The uniform buffer provides a
// global velocity field vector and the timestep `dt`.
//
// Each invocation updates one particle:
//   position += velocity * dt

struct Particle {
    position: vec4<f32>,
    velocity: vec4<f32>,
};

struct Uniforms {
    velocity: vec3<f32>,
    dt: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn advect_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = arrayLength(&particles);

    if idx >= count {
        return;
    }

    let vel = uniforms.velocity;
    let dt  = uniforms.dt;

    let p = particles[idx].position;
    particles[idx].position = vec4<f32>(
        p.x + vel.x * dt,
        p.y + vel.y * dt,
        p.z + vel.z * dt,
        p.w,
    );
}
