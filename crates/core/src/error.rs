use thiserror::Error;

// ---------------------------------------------------------------------------
// Top-level error — composes all crate-specific errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SimuError {
    #[error("Physics error: {0}")]
    Physics(#[from] PhysicsError),

    #[error("Mesh error: {0}")]
    Mesh(#[from] MeshError),

    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    #[error("CAS error: {0}")]
    Cas(#[from] CasError),

    #[error("Audio error: {0}")]
    Audio(#[from] AudioError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

// ---------------------------------------------------------------------------
// Physics errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("Invalid parameter: {field} = {value} — {reason}")]
    InvalidParameter {
        field: &'static str,
        value: f64,
        reason: &'static str,
    },

    #[error("Simulation diverged at step {step}: {detail}")]
    Divergence { step: usize, detail: String },

    #[error("Integration exceeded maximum steps ({max_steps})")]
    MaxStepsExceeded { max_steps: usize },

    #[error("Negative mass is not physical: {0} kg")]
    NegativeMass(f64),

    #[error("Negative area is not physical: {0} m²")]
    NegativeArea(f64),
}

// ---------------------------------------------------------------------------
// Mesh errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum MeshError {
    #[error("Failed to parse mesh file: {0}")]
    ParseError(String),

    #[error("Unsupported mesh format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid mesh topology: {0}")]
    InvalidTopology(String),

    #[error("Mesh has no elements")]
    EmptyMesh,

    #[error("Node index {index} out of bounds (mesh has {node_count} nodes)")]
    NodeIndexOutOfBounds { index: usize, node_count: usize },

    #[error("IO error reading mesh: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// GPU errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,

    #[error("Failed to create GPU device: {0}")]
    DeviceCreation(String),

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Buffer operation failed: {0}")]
    BufferError(String),

    #[error("Compute dispatch failed: {0}")]
    DispatchError(String),

    #[error("GPU backend not available — falling back to CPU")]
    Unavailable,
}

// ---------------------------------------------------------------------------
// CAS errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CasError {
    #[error("Parse error at position {position}: {message}")]
    ParseError { position: usize, message: String },

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Domain error: {0}")]
    DomainError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

// ---------------------------------------------------------------------------
// Audio errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to decode audio: {0}")]
    DecodeError(String),

    #[error("Empty audio buffer — nothing to analyze")]
    EmptyBuffer,

    #[error("IO error reading audio file: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Convenience Result alias
// ---------------------------------------------------------------------------

pub type SimuResult<T> = Result<T, SimuError>;
