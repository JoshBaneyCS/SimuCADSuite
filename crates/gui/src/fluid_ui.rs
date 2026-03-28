//! Fluid dynamics simulation UI panel.
//!
//! Provides controls for loading a mesh file, setting flow parameters, and
//! running a background particle-based fluid simulation with progress
//! reporting and cancellation.

use std::collections::HashSet;
use std::path::PathBuf;

use egui::Ui;
use simucad_core::export::DataTable;
use simucad_core::progress::ProgressReporter;
use simucad_core::types::Vec3;
use simucad_mesh::io::{GmshLoader, MeshLoader};
use simucad_mesh::types::{ElementType, Mesh};
use simucad_physics::fluid::ParticleSystem;

#[cfg(feature = "gpu")]
use simucad_gpu::backend::{select_backend, ComputeBackend};
#[cfg(feature = "gpu")]
use simucad_physics::fluid::reflect_into_bounds;

use crate::task::{TaskRunner, TaskStatus};
use crate::viewport_3d::Viewport3D;

// ---------------------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------------------

/// Summary data returned from a completed fluid simulation.
#[derive(Debug, Clone)]
pub struct FluidResult {
    pub particle_count: usize,
    pub particles_in_bounds: usize,
    pub mesh_node_count: usize,
    pub mesh_element_count: usize,
    pub steps_completed: usize,
    pub velocity_magnitude: f64,
    /// Name of the compute backend used (e.g. "wgpu" or "cpu-rayon").
    pub backend_name: String,
    /// Downsampled particle data for 3D visualization: (x, y, z, speed).
    pub particle_viz: Vec<(f64, f64, f64, f64)>,
    /// Mesh wireframe edges for 3D visualization: (start, end).
    pub mesh_edges: Vec<([f64; 3], [f64; 3])>,
}

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the fluid dynamics simulation panel.
pub struct FluidPanel {
    /// Path to the mesh file to load.
    pub mesh_path: String,
    /// Inlet velocity components (m/s).
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
    /// Number of simulation particles.
    pub particle_count: usize,
    /// Number of simulation timesteps.
    pub num_steps: usize,
    /// Whether to use GPU acceleration (if available).
    pub use_gpu: bool,

    /// Whether a mesh has been loaded.
    pub mesh_loaded: bool,
    /// Info about the loaded mesh.
    pub mesh_info: Option<String>,

    /// Background simulation runner.
    pub task: TaskRunner<Result<FluidResult, String>>,
    /// Progress reporter shared with the background thread.
    pub progress: Option<ProgressReporter>,

    /// Result from the last completed simulation.
    pub last_result: Option<FluidResult>,
    /// Log of status messages.
    pub log_messages: Vec<String>,
    /// 3D viewport for particle/mesh visualization.
    pub viewport: Viewport3D,
    /// Export file path (without extension).
    pub export_path: String,
}

impl Default for FluidPanel {
    fn default() -> Self {
        Self {
            mesh_path: String::new(),
            velocity_x: 1.0,
            velocity_y: 0.0,
            velocity_z: 0.0,
            particle_count: 5000,
            num_steps: 200,
            use_gpu: cfg!(feature = "gpu"),
            mesh_loaded: false,
            mesh_info: None,
            task: TaskRunner::new(),
            progress: None,
            last_result: None,
            log_messages: Vec::new(),
            viewport: Viewport3D::default(),
            export_path: "fluid_results".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

impl FluidPanel {
    /// Render the fluid dynamics panel.
    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("Fluid Dynamics Simulator");
        ui.add_space(8.0);

        self.show_inputs(ui);
        ui.add_space(12.0);
        self.show_actions(ui);
        ui.add_space(12.0);
        self.show_status(ui);
        self.show_results(ui);
        self.show_3d_viewport(ui);
    }

    fn show_inputs(&mut self, ui: &mut Ui) {
        egui::Grid::new("fluid_inputs")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Mesh file (.msh):");
                ui.text_edit_singleline(&mut self.mesh_path);
                ui.end_row();

                ui.label("Velocity X (m/s):");
                ui.add(
                    egui::DragValue::new(&mut self.velocity_x)
                        .speed(0.1)
                        .range(-1000.0..=1000.0),
                );
                ui.end_row();

                ui.label("Velocity Y (m/s):");
                ui.add(
                    egui::DragValue::new(&mut self.velocity_y)
                        .speed(0.1)
                        .range(-1000.0..=1000.0),
                );
                ui.end_row();

                ui.label("Velocity Z (m/s):");
                ui.add(
                    egui::DragValue::new(&mut self.velocity_z)
                        .speed(0.1)
                        .range(-1000.0..=1000.0),
                );
                ui.end_row();

                ui.label("Particle count:");
                ui.add(
                    egui::DragValue::new(&mut self.particle_count)
                        .speed(100.0)
                        .range(10..=1_000_000),
                );
                ui.end_row();

                ui.label("Timesteps:");
                ui.add(
                    egui::DragValue::new(&mut self.num_steps)
                        .speed(1.0)
                        .range(1..=100_000),
                );
                ui.end_row();

                ui.label("GPU acceleration:");
                let gpu_available = cfg!(feature = "gpu");
                ui.add_enabled(gpu_available, egui::Checkbox::new(&mut self.use_gpu, if gpu_available { "Enabled" } else { "Not available" }));
                ui.end_row();
            });

        // Show mesh info if loaded.
        if let Some(ref info) = self.mesh_info {
            ui.add_space(4.0);
            ui.label(info);
        }
    }

    fn show_actions(&mut self, ui: &mut Ui) {
        let is_running = matches!(self.task.poll(), TaskStatus::Running { .. });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!is_running, egui::Button::new("Load Mesh"))
                .clicked()
            {
                self.load_mesh();
            }

            if ui
                .add_enabled(!is_running && self.mesh_loaded, egui::Button::new("Run Simulation"))
                .clicked()
            {
                self.run_simulation();
            }

            if is_running {
                if ui.button("Cancel").clicked() {
                    if let Some(ref prog) = self.progress {
                        prog.cancel();
                    }
                }
                ui.spinner();
                ui.label("Simulation running...");
            }
        });
    }

    fn show_status(&mut self, ui: &mut Ui) {
        // Show progress bar.
        if let Some(ref prog) = self.progress {
            if self.task.is_running() {
                let frac = prog.fraction() as f32;
                let msg = prog.message();
                ui.add(egui::ProgressBar::new(frac).show_percentage().text(msg));
            }
        }

        // Poll the background task.
        match self.task.poll() {
            TaskStatus::Idle => {}
            TaskStatus::Running { .. } => {}
            TaskStatus::Completed => {
                if let Some(result) = self.task.take_result() {
                    match result {
                        Ok(res) => {
                            self.log_messages.push(format!(
                                "Simulation complete: {} particles, {} steps [{}]",
                                res.particle_count, res.steps_completed, res.backend_name
                            ));
                            self.last_result = Some(res);
                        }
                        Err(msg) => {
                            self.log_messages.push(format!("Error: {msg}"));
                        }
                    }
                    self.progress = None;
                }
            }
            TaskStatus::Failed(msg) => {
                let msg = msg.clone();
                self.log_messages.push(format!("Task failed: {msg}"));
                self.progress = None;
            }
        }

        // Show log messages.
        if !self.log_messages.is_empty() {
            ui.separator();
            ui.label("Log:");
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    for msg in &self.log_messages {
                        ui.label(msg);
                    }
                });
        }
    }

    fn show_results(&mut self, ui: &mut Ui) {
        if let Some(ref res) = self.last_result {
            ui.add_space(8.0);
            ui.separator();
            ui.strong("Simulation Results");
            ui.add_space(4.0);

            egui::Grid::new("fluid_results")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Particles:");
                    ui.label(format!("{}", res.particle_count));
                    ui.end_row();

                    ui.label("Particles in bounds:");
                    ui.label(format!("{}", res.particles_in_bounds));
                    ui.end_row();

                    ui.label("Mesh nodes:");
                    ui.label(format!("{}", res.mesh_node_count));
                    ui.end_row();

                    ui.label("Mesh elements:");
                    ui.label(format!("{}", res.mesh_element_count));
                    ui.end_row();

                    ui.label("Steps completed:");
                    ui.label(format!("{}", res.steps_completed));
                    ui.end_row();

                    ui.label("Flow velocity magnitude:");
                    ui.label(format!("{:.4} m/s", res.velocity_magnitude));
                    ui.end_row();

                    ui.label("Compute backend:");
                    ui.label(&res.backend_name);
                    ui.end_row();
                });

            // Export controls.
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Export path:");
                ui.text_edit_singleline(&mut self.export_path);
                if ui.button("Save CSV").clicked() {
                    self.export_fluid_csv();
                }
                if ui.button("Save XLSX").clicked() {
                    self.export_fluid_xlsx();
                }
            });
        }
    }

    fn export_fluid_csv(&mut self) {
        let Some(ref res) = self.last_result else {
            return;
        };
        let table = fluid_result_to_table(res);
        let path = PathBuf::from(format!("{}.csv", self.export_path));
        match table.write_csv(&path) {
            Ok(()) => self.log_messages.push(format!("Saved to {}", path.display())),
            Err(e) => self.log_messages.push(format!("CSV error: {e}")),
        }
    }

    fn export_fluid_xlsx(&mut self) {
        let Some(ref res) = self.last_result else {
            return;
        };
        let table = fluid_result_to_table(res);
        let path = PathBuf::from(format!("{}.xlsx", self.export_path));
        match table.write_xlsx(&path) {
            Ok(()) => self.log_messages.push(format!("Saved to {}", path.display())),
            Err(e) => self.log_messages.push(format!("XLSX error: {e}")),
        }
    }

    fn show_3d_viewport(&mut self, ui: &mut Ui) {
        if let Some(ref res) = self.last_result {
            ui.add_space(12.0);
            ui.separator();
            ui.strong("3D Visualization");
            ui.add_space(4.0);

            let particles = if res.particle_viz.is_empty() {
                None
            } else {
                Some(res.particle_viz.as_slice())
            };
            let edges = if res.mesh_edges.is_empty() {
                None
            } else {
                Some(res.mesh_edges.as_slice())
            };
            self.viewport.show(ui, particles, edges);
        }
    }

    // -----------------------------------------------------------------------
    // Mesh loading
    // -----------------------------------------------------------------------

    fn load_mesh(&mut self) {
        let path = PathBuf::from(&self.mesh_path);
        if !path.exists() {
            self.log_messages
                .push(format!("Mesh file not found: {}", self.mesh_path));
            self.mesh_loaded = false;
            self.mesh_info = None;
            return;
        }

        match GmshLoader::load(&path) {
            Ok(mesh) => {
                let info = format!(
                    "Mesh loaded: {} nodes, {} elements, {}D",
                    mesh.node_count(),
                    mesh.element_count(),
                    mesh.dimension
                );
                self.log_messages.push(info.clone());
                self.mesh_info = Some(info);
                self.mesh_loaded = true;
            }
            Err(e) => {
                self.log_messages
                    .push(format!("Failed to load mesh: {e}"));
                self.mesh_loaded = false;
                self.mesh_info = None;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Simulation dispatch
    // -----------------------------------------------------------------------

    fn run_simulation(&mut self) {
        let mesh_path = PathBuf::from(&self.mesh_path);
        let velocity = Vec3::new(self.velocity_x, self.velocity_y, self.velocity_z);
        let particle_count = self.particle_count;
        let num_steps = self.num_steps;
        let use_gpu = self.use_gpu;

        let progress = ProgressReporter::new(num_steps as u64);
        self.progress = Some(progress.clone());

        self.log_messages.push(format!(
            "Starting simulation: particles={particle_count}, steps={num_steps}, \
             gpu={use_gpu}, v=({:.2},{:.2},{:.2})",
            velocity.x, velocity.y, velocity.z
        ));

        self.task.spawn(move || {
            // Load mesh.
            let mesh = GmshLoader::load(&mesh_path)
                .map_err(|e| format!("Failed to load mesh: {e}"))?;

            let mesh_node_count = mesh.node_count();
            let mesh_element_count = mesh.element_count();
            let bounds = mesh.bounding_box();

            // Select compute backend.
            #[cfg(feature = "gpu")]
            let backend: Box<dyn ComputeBackend> = if use_gpu {
                select_backend()
            } else {
                Box::new(simucad_gpu::cpu_backend::CpuBackend::new())
            };
            #[cfg(feature = "gpu")]
            let backend_name = backend.name().to_string();

            #[cfg(not(feature = "gpu"))]
            let backend_name = "cpu-rayon (built-in)".to_string();

            progress.set_message(format!("Initializing particles ({backend_name})..."));

            // Create particle system within mesh bounding box.
            let mut system = ParticleSystem::initialize(bounds, particle_count);

            let dt = 0.01;

            // Run advection steps.
            for step in 0..num_steps {
                if progress.is_cancelled() {
                    return Err("Simulation cancelled by user".to_string());
                }

                #[cfg(feature = "gpu")]
                {
                    // GPU/CPU backend: advect particles, then reflect into bounds.
                    backend
                        .advect_particles(&mut system.particles, velocity, dt)
                        .map_err(|e| format!("Compute error: {e}"))?;

                    // Set velocity on all particles and reflect into bounds.
                    for p in &mut system.particles {
                        p.velocity = velocity;
                        reflect_into_bounds(&mut p.position, &bounds);
                    }
                }

                #[cfg(not(feature = "gpu"))]
                {
                    system.advect(velocity, dt);
                }

                progress.set_progress((step + 1) as u64);

                if step % 50 == 0 {
                    progress.set_message(format!(
                        "[{backend_name}] Step {}/{num_steps}",
                        step + 1
                    ));
                }
            }

            progress.set_message("Complete".to_string());

            let particles_in_bounds = system.particles_in_bounds();
            let vel_mag = velocity.magnitude();

            // Extract particle data for 3D visualization (downsample to 100K).
            let all_particles = &system.particles;
            let step_viz = (all_particles.len() / 100_000).max(1);
            let particle_viz: Vec<(f64, f64, f64, f64)> = all_particles
                .iter()
                .step_by(step_viz)
                .map(|p| {
                    let speed = p.velocity.magnitude();
                    (p.position.x, p.position.y, p.position.z, speed)
                })
                .collect();

            // Extract mesh wireframe edges.
            let mesh_edges = extract_mesh_edges(&mesh);

            Ok(FluidResult {
                particle_count: system.particle_count(),
                particles_in_bounds,
                mesh_node_count,
                mesh_element_count,
                steps_completed: num_steps,
                velocity_magnitude: vel_mag,
                backend_name,
                particle_viz,
                mesh_edges,
            })
        });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`FluidResult`] to a [`DataTable`] of particle positions.
fn fluid_result_to_table(res: &FluidResult) -> DataTable {
    let n = res.particle_viz.len();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut speed = Vec::with_capacity(n);

    for &(px, py, pz, s) in &res.particle_viz {
        x.push(px);
        y.push(py);
        z.push(pz);
        speed.push(s);
    }

    let mut table = DataTable::new("Fluid Particles");
    table.add_column("x", "m", x);
    table.add_column("y", "m", y);
    table.add_column("z", "m", z);
    table.add_column("speed", "m/s", speed);
    table
}

/// Extract unique wireframe edges from a mesh for 3D rendering.
fn extract_mesh_edges(mesh: &Mesh) -> Vec<([f64; 3], [f64; 3])> {
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    let mut edges = Vec::new();

    for elem in &mesh.elements {
        let idx = &elem.node_indices;
        let element_edges: Vec<(usize, usize)> = match elem.element_type {
            ElementType::Line2 => {
                vec![(idx[0], idx[1])]
            }
            ElementType::Triangle3 => {
                vec![(idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])]
            }
            ElementType::Tetrahedron4 => {
                vec![
                    (idx[0], idx[1]),
                    (idx[0], idx[2]),
                    (idx[0], idx[3]),
                    (idx[1], idx[2]),
                    (idx[1], idx[3]),
                    (idx[2], idx[3]),
                ]
            }
        };

        for (a, b) in element_edges {
            let key = if a < b { (a, b) } else { (b, a) };
            if edge_set.insert(key) {
                let na = mesh.nodes[a];
                let nb = mesh.nodes[b];
                edges.push(([na.x, na.y, na.z], [nb.x, nb.y, nb.z]));
            }
        }
    }

    edges
}
