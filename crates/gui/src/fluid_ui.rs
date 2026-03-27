//! Fluid dynamics simulation UI panel.
//!
//! Provides controls for loading a mesh file, setting flow parameters, and
//! running a background fluid simulation.

use egui::Ui;

use crate::task::{TaskRunner, TaskStatus};

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the fluid dynamics simulation panel.
pub struct FluidPanel {
    /// Path to the mesh file to load.
    pub mesh_path: String,
    /// Inlet velocity magnitude (m/s).
    pub velocity: f64,
    /// Fluid density (kg/m^3).
    pub density: f64,
    /// Dynamic viscosity (Pa*s).
    pub viscosity: f64,
    /// Number of simulation timesteps.
    pub num_steps: usize,
    /// Background simulation runner.
    pub task: TaskRunner<Result<String, String>>,
    /// Log of status messages from the last run.
    pub log_messages: Vec<String>,
}

impl Default for FluidPanel {
    fn default() -> Self {
        Self {
            mesh_path: String::new(),
            velocity: 1.0,
            density: 1.225,
            viscosity: 1.81e-5,
            num_steps: 100,
            task: TaskRunner::new(),
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

        // Input controls.
        self.show_inputs(ui);
        ui.add_space(12.0);

        // Action buttons.
        self.show_actions(ui);
        ui.add_space(12.0);

        // Progress / status.
        self.show_status(ui);
    }

    fn show_inputs(&mut self, ui: &mut Ui) {
        egui::Grid::new("fluid_inputs")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Mesh file path:");
                ui.text_edit_singleline(&mut self.mesh_path);
                ui.end_row();

                ui.label("Inlet velocity (m/s):");
                ui.add(
                    egui::DragValue::new(&mut self.velocity)
                        .speed(0.1)
                        .range(0.0..=1000.0),
                );
                ui.end_row();

                ui.label("Density (kg/m^3):");
                ui.add(
                    egui::DragValue::new(&mut self.density)
                        .speed(0.01)
                        .range(0.001..=100_000.0),
                );
                ui.end_row();

                ui.label("Viscosity (Pa*s):");
                ui.add(
                    egui::DragValue::new(&mut self.viscosity)
                        .speed(1e-6)
                        .range(0.0..=1.0),
                );
                ui.end_row();

                ui.label("Timesteps:");
                ui.add(
                    egui::DragValue::new(&mut self.num_steps)
                        .speed(1.0)
                        .range(1..=1_000_000),
                );
                ui.end_row();
            });
    }

    fn show_actions(&mut self, ui: &mut Ui) {
        let is_running = matches!(self.task.poll(), TaskStatus::Running { .. });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!is_running, egui::Button::new("Run Simulation"))
                .clicked()
            {
                self.run_simulation();
            }

            if is_running {
                ui.spinner();
                ui.label("Simulation running...");
            }
        });
    }

    fn show_status(&mut self, ui: &mut Ui) {
        // Poll the background task.
        match self.task.poll() {
            TaskStatus::Idle => {}
            TaskStatus::Running { progress } => {
                let progress = *progress;
                ui.add(egui::ProgressBar::new(progress).show_percentage());
            }
            TaskStatus::Completed => {
                if let Some(result) = self.task.take_result() {
                    match result {
                        Ok(msg) => {
                            self.log_messages.push(format!("Success: {msg}"));
                        }
                        Err(msg) => {
                            self.log_messages.push(format!("Error: {msg}"));
                        }
                    }
                }
            }
            TaskStatus::Failed(msg) => {
                let msg = msg.clone();
                self.log_messages.push(format!("Task failed: {msg}"));
            }
        }

        // Show log messages.
        if !self.log_messages.is_empty() {
            ui.separator();
            ui.label("Log:");
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    for msg in &self.log_messages {
                        ui.label(msg);
                    }
                });
        }
    }

    // -----------------------------------------------------------------------
    // Simulation dispatch
    // -----------------------------------------------------------------------

    fn run_simulation(&mut self) {
        let mesh_path = self.mesh_path.clone();
        let velocity = self.velocity;
        let density = self.density;
        let viscosity = self.viscosity;
        let num_steps = self.num_steps;

        self.log_messages
            .push(format!("Starting simulation: mesh={mesh_path}, v={velocity} m/s, steps={num_steps}"));

        self.task.spawn(move || {
            // Validate the mesh path exists.
            let path = std::path::Path::new(&mesh_path);
            if !path.exists() {
                return Err(format!("Mesh file not found: {mesh_path}"));
            }

            // Placeholder simulation loop.
            // In a full implementation this would call into the physics/mesh
            // crates to run an actual CFD solver.
            tracing::info!(
                "Running fluid simulation: density={density}, viscosity={viscosity}, steps={num_steps}"
            );

            for step in 0..num_steps {
                // Simulate work.
                if step % 100 == 0 {
                    tracing::debug!("Fluid sim step {step}/{num_steps}");
                }
            }

            Ok(format!(
                "Completed {num_steps} timesteps on mesh '{}'",
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
            ))
        });
    }
}
