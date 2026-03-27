# SimuCADSuite

![SimuCADSuite Logo](https://github.com/JoshBaneyCS/SimuCADSuite/blob/main/simucadsuitelogo.png?raw=true)

SimuCADSuite is a comprehensive, modular scientific simulation desktop application rewritten from the ground up in **Rust**. Originally a Python (PyQt5) capstone project, it has been rebuilt as a high-performance, GPU-accelerated, multithreaded platform for physics simulation, fluid dynamics, symbolic mathematics, and signal processing.

## Features

- **Projectile Motion Kinematics** — Vacuum (analytical) and drag-based (Euler integration) trajectory solvers with 6 drag coefficient shapes, parameter sweeps, and trajectory sampling with velocity vector overlays.
- **Fluid Dynamics Simulation** — Lagrangian particle-based simulation supporting millions of particles, mesh-based velocity field computation, and Gmsh `.msh` file I/O.
- **Computer Algebra System** — Hand-written recursive descent expression parser, numeric evaluator, symbolic differentiation (sum/product/quotient/chain rules), algebraic simplification, and 2D/3D function plotting.
- **Audio Analysis** — FFT/DFT frequency decomposition via `rustfft`, audio file decoding (WAV/MP3/FLAC) via `symphonia`, and spectral visualization.
- **GPU Acceleration** — Cross-platform compute via `wgpu` (Metal on macOS, Vulkan on AMD/Linux, DX12 on Windows) with CPU fallback via `rayon`.
- **Multithreading** — Data-parallel workloads via `rayon` with background task coordination and progress reporting.
- **Desktop GUI** — Native cross-platform UI via `egui`/`eframe` with embedded scientific plotting via `egui_plot`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    APPLICATION (app)                  │
├─────────────────────────────────────────────────────┤
│                    PRESENTATION                      │
│         gui (egui/eframe)  ·  visualization           │
├─────────────────────────────────────────────────────┤
│                    COMPUTE                            │
│         gpu (wgpu/compute)  ·  cas (symbolic math)    │
│                   audio (FFT/DSP)                     │
├─────────────────────────────────────────────────────┤
│                    ENGINES                            │
│     physics (kinematics, fluid)  ·  mesh (I/O, CAD)  │
├─────────────────────────────────────────────────────┤
│                    FOUNDATION                        │
│   core (units, types, settings, constants, errors)   │
└─────────────────────────────────────────────────────┘
```

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `simucad-core` | Type-safe units, shared types, settings, physical constants, error hierarchy |
| `simucad-physics` | Integrators, kinematics solvers, drag models, trajectory analysis, fluid dynamics |
| `simucad-mesh` | Gmsh `.msh` v2.2 parser (nom), mesh types, validation, I/O |
| `simucad-gpu` | `ComputeBackend` trait, wgpu backend, CPU fallback, WGSL compute shaders |
| `simucad-cas` | Expression AST, parser, evaluator, symbolic differentiation, simplification, plotter |
| `simucad-audio` | Audio decoding (symphonia), FFT pipeline (rustfft), spectral data model |
| `simucad-gui` | egui application shell, pages, scientific plotting, background task runner |

## Getting Started

### Prerequisites

- **Rust 1.85+** (stable)
- A GPU supporting Metal, Vulkan, or DX12 (optional — CPU fallback available)

### Build

```bash
git clone https://github.com/JoshBaneyCS/SimuCADSuite.git
cd SimuCADSuite
cargo build --release
```

### Run

```bash
cargo run --release
```

### Test

```bash
cargo test --workspace
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `gpu` | on | wgpu compute backend |
| `cas` | on | Computer Algebra System |
| `audio` | on | Audio analysis module |
| `cuda` | off | CUDA specialization (future) |
| `dev-tools` | off | Debug panels, extra logging |

## GPU Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS | Metal | Supported via wgpu |
| Linux/AMD | Vulkan | Supported via wgpu |
| Windows | DX12 | Supported via wgpu |
| NVIDIA | CUDA | Planned (optional) |

## License

MIT

## Author

Josh Baney
