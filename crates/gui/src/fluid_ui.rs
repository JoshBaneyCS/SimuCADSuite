//! Fluid dynamics simulation UI panel.
//!
//! Provides controls for loading a mesh file, setting flow parameters, and
//! running a background particle-based fluid simulation with progress
//! reporting and cancellation.

use std::path::PathBuf;

use egui::Ui;
use simucad_core::progress::ProgressReporter;
use simucad_core::types::Vec3;
use simucad_mesh::io::{GmshLoader, MeshLoader};
use simucad_physics::fluid::ParticleSystem;

use crate::task::{TaskRunner, TaskStatus};

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
            mesh_loaded: false,
            mesh_info: None,
            task: TaskRunner::new(),
            progress: None,
            last_result: None,
            log_messages: Vec::new(),
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
                                "Simulation complete: {} particles, {} steps",
                                res.particle_count, res.steps_completed
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

    fn show_results(&self, ui: &mut Ui) {
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
                });
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

        let progress = ProgressReporter::new(num_steps as u64);
        self.progress = Some(progress.clone());

        self.log_messages.push(format!(
            "Starting simulation: particles={particle_count}, steps={num_steps}, v=({:.2},{:.2},{:.2})",
            velocity.x, velocity.y, velocity.z
        ));

        self.task.spawn(move || {
            // Load mesh.
            let mesh = GmshLoader::load(&mesh_path)
                .map_err(|e| format!("Failed to load mesh: {e}"))?;

            let mesh_node_count = mesh.node_count();
            let mesh_element_count = mesh.element_count();
            let bounds = mesh.bounding_box();

            progress.set_message("Initializing particles...".to_string());

            // Create particle system within mesh bounding box.
            let mut system = ParticleSystem::initialize(bounds, particle_count);

            let dt = 0.01;

            // Run advection steps.
            for step in 0..num_steps {
                if progress.is_cancelled() {
                    return Err("Simulation cancelled by user".to_string());
                }

                system.advect(velocity, dt);
                progress.set_progress((step + 1) as u64);

                if step % 50 == 0 {
                    progress.set_message(format!("Step {}/{num_steps}", step + 1));
                }
            }

            progress.set_message("Complete".to_string());

            let particles_in_bounds = system.particles_in_bounds();
            let vel_mag = velocity.magnitude();

            Ok(FluidResult {
                particle_count: system.particle_count(),
                particles_in_bounds,
                mesh_node_count,
                mesh_element_count,
                steps_completed: num_steps,
                velocity_magnitude: vel_mag,
            })
        });
    }
}
