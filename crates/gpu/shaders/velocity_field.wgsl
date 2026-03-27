// Velocity field compute shader.
//
// For each node position, compute the mean displacement from the node to
// every particle:
//
//   v_node = (1/N) * sum_i( particle_i.position - node_position )
//
// Strategy: one workgroup per node. Threads within the workgroup stride
// over all particles, accumulate partial sums in workgroup shared memory,
// then reduce and write the final mean to the output buffer.

struct GpuVec3 {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
};

struct Uniforms {
    num_particles: u32,
    num_nodes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       particles:  array<GpuVec3>;
@group(0) @binding(1) var<storage, read>       nodes:      array<GpuVec3>;
@group(0) @binding(2) var<storage, read_write> output:     array<GpuVec3>;
@group(0) @binding(3) var<uniform>             uniforms:   Uniforms;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_x: array<f32, 256>;
var<workgroup> shared_y: array<f32, 256>;
var<workgroup> shared_z: array<f32, 256>;

@compute @workgroup_size(256)
fn velocity_field_main(
    @builtin(workgroup_id)         wg_id:    vec3<u32>,
    @builtin(local_invocation_id)  local_id: vec3<u32>,
) {
    let node_idx = wg_id.x;
    let lid      = local_id.x;
    let num_p    = uniforms.num_particles;

    // Guard against over-dispatch (should not happen, but be safe).
    if node_idx >= uniforms.num_nodes {
        return;
    }

    // Load node position once.
    let node = nodes[node_idx];
    let nx = node.x;
    let ny = node.y;
    let nz = node.z;

    // Each thread accumulates a partial sum by striding over particles.
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    var sum_z: f32 = 0.0;

    var i: u32 = lid;
    loop {
        if i >= num_p {
            break;
        }
        let p = particles[i];
        sum_x += p.x - nx;
        sum_y += p.y - ny;
        sum_z += p.z - nz;
        i += WG_SIZE;
    }

    // Store partial sums in shared memory.
    shared_x[lid] = sum_x;
    shared_y[lid] = sum_y;
    shared_z[lid] = sum_z;

    workgroupBarrier();

    // Tree reduction within the workgroup.
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if lid < stride {
            shared_x[lid] += shared_x[lid + stride];
            shared_y[lid] += shared_y[lid + stride];
            shared_z[lid] += shared_z[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes the mean.
    if lid == 0u {
        let inv_n = 1.0 / f32(num_p);
        output[node_idx].x = shared_x[0] * inv_n;
        output[node_idx].y = shared_y[0] * inv_n;
        output[node_idx].z = shared_z[0] * inv_n;
        output[node_idx]._pad = 0.0;
    }
}
