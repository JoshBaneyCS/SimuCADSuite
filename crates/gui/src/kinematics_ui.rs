//! Projectile motion (kinematics) UI panel.
//!
//! Provides input fields for launch parameters and displays both vacuum and
//! drag trajectories using the physics crate solvers. Results include plots,
//! summary statistics, sampled-point tables, and CSV export.

use std::path::PathBuf;

use egui::Ui;
use simucad_core::export::{trajectory_to_table, write_tables_xlsx, DataTable};
use simucad_core::types::{SimulationConfig, Trajectory, TrajectoryPoint};
use simucad_physics::drag::{DragModel, DragShape};
use simucad_physics::integrator::{AdaptiveRK45Integrator, EulerIntegrator, Integrator, RK4Integrator};
use simucad_physics::kinematics;
use simucad_physics::trajectory::sample_trajectory;

use crate::plotting;

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the projectile motion simulation panel.
pub struct KinematicsPanel {
    /// Initial launch speed (m/s).
    pub velocity: f64,
    /// Launch angle above the horizontal (degrees).
    pub angle_deg: f64,
    /// Gravitational acceleration (m/s^2).
    pub gravity: f64,
    /// Projectile mass (kg).
    pub mass: f64,
    /// Cross-sectional area for drag calculation (m^2).
    pub area: f64,
    /// Initial launch height (m).
    pub initial_height: f64,
    /// Shape used for the drag coefficient.
    pub drag_shape: DragShape,
    /// Numerical integrator for drag trajectory.
    pub integrator_choice: IntegratorChoice,

    /// Computed vacuum trajectory (filled on "Calculate").
    pub vacuum_trajectory: Option<Trajectory>,
    /// Computed drag trajectory (filled on "Calculate").
    pub drag_trajectory: Option<Trajectory>,
    /// Sampled points from vacuum trajectory.
    pub vacuum_samples: Vec<TrajectoryPoint>,
    /// Sampled points from drag trajectory.
    pub drag_samples: Vec<TrajectoryPoint>,
    /// Error message from the last calculation attempt.
    pub error_message: Option<String>,
    /// CSV export text area content.
    pub csv_output: String,
    /// Whether the CSV output area is visible.
    pub show_csv: bool,
    /// Export file path.
    pub export_path: String,
}

impl Default for KinematicsPanel {
    fn default() -> Self {
        Self {
            velocity: 50.0,
            angle_deg: 45.0,
            gravity: 9.80665,
            mass: 1.0,
            area: 0.01,
            initial_height: 0.0,
            drag_shape: DragShape::Sphere,
            integrator_choice: IntegratorChoice::RK4,
            vacuum_trajectory: None,
            drag_trajectory: None,
            vacuum_samples: Vec::new(),
            drag_samples: Vec::new(),
            error_message: None,
            csv_output: String::new(),
            show_csv: false,
            export_path: "trajectory".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Integrator choice
// ---------------------------------------------------------------------------

/// Selection of numerical integrator for the drag trajectory solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorChoice {
    Euler,
    RK4,
    AdaptiveRK45,
}

const INTEGRATOR_CHOICES: &[(IntegratorChoice, &str)] = &[
    (IntegratorChoice::Euler, "Euler (1st order)"),
    (IntegratorChoice::RK4, "RK4 (4th order)"),
    (IntegratorChoice::AdaptiveRK45, "Adaptive RK4-5"),
];

fn integrator_label(choice: IntegratorChoice) -> &'static str {
    INTEGRATOR_CHOICES
        .iter()
        .find(|(c, _)| *c == choice)
        .map(|(_, label)| *label)
        .unwrap_or("Unknown")
}

// ---------------------------------------------------------------------------
// Drag shape helpers
// ---------------------------------------------------------------------------

const DRAG_SHAPES: &[(DragShape, &str)] = &[
    (DragShape::Sphere, "Sphere"),
    (DragShape::Circle, "Circle (flat disk)"),
    (DragShape::Square, "Square (flat plate)"),
    (DragShape::Rhombus, "Rhombus"),
    (DragShape::Airfoil, "Airfoil"),
    (DragShape::None, "None (vacuum)"),
];

fn drag_shape_label(shape: DragShape) -> &'static str {
    DRAG_SHAPES
        .iter()
        .find(|(s, _)| *s == shape)
        .map(|(_, label)| *label)
        .unwrap_or("Unknown")
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

impl KinematicsPanel {
    /// Render the kinematics panel into the given `Ui`.
    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("Projectile Motion Simulator");
        ui.add_space(8.0);

        // Two-column layout: inputs on the left, results on the right.
        ui.columns(2, |cols| {
            self.show_inputs(&mut cols[0]);
            self.show_results(&mut cols[1]);
        });
    }

    /// Render the input form.
    fn show_inputs(&mut self, ui: &mut Ui) {
        egui::Grid::new("kinematics_inputs")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Velocity (m/s):");
                ui.add(egui::DragValue::new(&mut self.velocity).speed(0.5).range(0.0..=10_000.0));
                ui.end_row();

                ui.label("Angle (degrees):");
                ui.add(egui::DragValue::new(&mut self.angle_deg).speed(0.5).range(0.0..=90.0));
                ui.end_row();

                ui.label("Gravity (m/s^2):");
                ui.add(egui::DragValue::new(&mut self.gravity).speed(0.1).range(0.01..=100.0));
                ui.end_row();

                ui.label("Mass (kg):");
                ui.add(egui::DragValue::new(&mut self.mass).speed(0.1).range(0.001..=10_000.0));
                ui.end_row();

                ui.label("Cross-section area (m^2):");
                ui.add(egui::DragValue::new(&mut self.area).speed(0.001).range(0.0..=100.0));
                ui.end_row();

                ui.label("Initial height (m):");
                ui.add(egui::DragValue::new(&mut self.initial_height).speed(0.5).range(0.0..=100_000.0));
                ui.end_row();

                ui.label("Drag shape:");
                egui::ComboBox::from_id_salt("drag_shape")
                    .selected_text(drag_shape_label(self.drag_shape))
                    .show_ui(ui, |ui| {
                        for (shape, label) in DRAG_SHAPES {
                            ui.selectable_value(&mut self.drag_shape, *shape, *label);
                        }
                    });
                ui.end_row();

                ui.label("Integrator:");
                egui::ComboBox::from_id_salt("integrator")
                    .selected_text(integrator_label(self.integrator_choice))
                    .show_ui(ui, |ui| {
                        for (choice, label) in INTEGRATOR_CHOICES {
                            ui.selectable_value(&mut self.integrator_choice, *choice, *label);
                        }
                    });
                ui.end_row();
            });

        ui.add_space(12.0);

        ui.horizontal(|ui| {
            if ui.button("Calculate").clicked() {
                self.calculate();
            }

            if self.vacuum_trajectory.is_some() || self.drag_trajectory.is_some() {
                if ui.button("Show CSV").clicked() {
                    self.export_csv();
                }
            }
        });

        // File export controls.
        if self.vacuum_trajectory.is_some() || self.drag_trajectory.is_some() {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Export path:");
                ui.text_edit_singleline(&mut self.export_path);
                if ui.button("Save CSV").clicked() {
                    self.save_csv_file();
                }
                if ui.button("Save XLSX").clicked() {
                    self.save_xlsx_file();
                }
            });
        }

        if let Some(ref err) = self.error_message {
            ui.add_space(4.0);
            ui.colored_label(egui::Color32::RED, err);
        }

        // CSV output area.
        if self.show_csv && !self.csv_output.is_empty() {
            ui.add_space(8.0);
            ui.collapsing("CSV Output", |ui| {
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        ui.monospace(&self.csv_output);
                    });
            });
        }
    }

    /// Render the trajectory results (plot, summary, sampled points).
    fn show_results(&mut self, ui: &mut Ui) {
        if self.vacuum_trajectory.is_none() && self.drag_trajectory.is_none() {
            ui.label("Enter parameters and click Calculate to see results.");
            return;
        }

        // Plot both trajectories.
        plotting::plot_trajectories(
            ui,
            self.vacuum_trajectory.as_ref(),
            self.drag_trajectory.as_ref(),
        );

        ui.add_space(12.0);

        // Summary comparison table.
        self.show_summary_table(ui);

        // Sampled points scatter plot.
        if !self.vacuum_samples.is_empty() {
            ui.add_space(8.0);
            ui.collapsing("Sampled Points Plot (Vacuum)", |ui| {
                plotting::plot_sampled_points(ui, &self.vacuum_samples);
            });
        }

        // Sampled points tables.
        if let Some(ref vac) = self.vacuum_trajectory {
            ui.add_space(4.0);
            ui.collapsing("Sampled Points Table (Vacuum)", |ui| {
                Self::show_points_table(ui, vac, "vacuum_pts");
            });
        }

        if let Some(ref drg) = self.drag_trajectory {
            ui.collapsing("Sampled Points Table (Drag)", |ui| {
                Self::show_points_table(ui, drg, "drag_pts");
            });
        }
    }

    /// Display the summary comparison table.
    fn show_summary_table(&self, ui: &mut Ui) {
        egui::Grid::new("trajectory_summary")
            .num_columns(3)
            .striped(true)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.strong("Metric");
                ui.strong("Vacuum");
                ui.strong("With Drag");
                ui.end_row();

                let vac = self.vacuum_trajectory.as_ref();
                let drg = self.drag_trajectory.as_ref();

                ui.label("Range (m)");
                ui.label(vac.map_or("--".to_string(), |t| format!("{:.2}", t.range)));
                ui.label(drg.map_or("--".to_string(), |t| format!("{:.2}", t.range)));
                ui.end_row();

                ui.label("Max Height (m)");
                ui.label(vac.map_or("--".to_string(), |t| format!("{:.2}", t.max_height)));
                ui.label(drg.map_or("--".to_string(), |t| format!("{:.2}", t.max_height)));
                ui.end_row();

                ui.label("Flight Time (s)");
                ui.label(vac.map_or("--".to_string(), |t| format!("{:.3}", t.flight_time)));
                ui.label(drg.map_or("--".to_string(), |t| format!("{:.3}", t.flight_time)));
                ui.end_row();
            });
    }

    /// Render a table of sampled trajectory points.
    fn show_points_table(ui: &mut Ui, trajectory: &Trajectory, id: &str) {
        let step = (trajectory.points.len() / 20).max(1);
        egui::Grid::new(id)
            .num_columns(5)
            .striped(true)
            .spacing([12.0, 2.0])
            .show(ui, |ui| {
                ui.strong("t (s)");
                ui.strong("x (m)");
                ui.strong("y (m)");
                ui.strong("vx (m/s)");
                ui.strong("vy (m/s)");
                ui.end_row();

                for (i, pt) in trajectory.points.iter().enumerate() {
                    if i % step != 0 && i != trajectory.points.len() - 1 {
                        continue;
                    }
                    ui.label(format!("{:.3}", pt.time));
                    ui.label(format!("{:.2}", pt.position.x));
                    ui.label(format!("{:.2}", pt.position.y));
                    ui.label(format!("{:.2}", pt.velocity.x));
                    ui.label(format!("{:.2}", pt.velocity.y));
                    ui.end_row();
                }
            });
    }

    // -----------------------------------------------------------------------
    // Simulation logic
    // -----------------------------------------------------------------------

    /// Run the vacuum and drag trajectory calculations using the physics API.
    fn calculate(&mut self) {
        self.error_message = None;
        self.vacuum_trajectory = None;
        self.drag_trajectory = None;
        self.vacuum_samples.clear();
        self.drag_samples.clear();
        self.show_csv = false;

        let angle_rad = self.angle_deg.to_radians();
        let num_points = 1000;

        // Vacuum trajectory (analytical).
        match kinematics::vacuum_trajectory(
            self.velocity,
            angle_rad,
            self.gravity,
            self.initial_height,
            num_points,
        ) {
            Ok(traj) => {
                self.vacuum_samples = sample_trajectory(&traj, 25);
                self.vacuum_trajectory = Some(traj);
            }
            Err(e) => {
                self.error_message = Some(format!("Vacuum trajectory error: {e}"));
                return;
            }
        }

        // Drag trajectory (numerical).
        let drag_model = DragModel::at_sea_level(self.drag_shape, self.area);
        let config = SimulationConfig {
            timestep: 0.001,
            max_steps: 500_000,
            ..SimulationConfig::default()
        };

        let integrator: Box<dyn Integrator> = match self.integrator_choice {
            IntegratorChoice::Euler => Box::new(EulerIntegrator::new()),
            IntegratorChoice::RK4 => Box::new(RK4Integrator::new()),
            IntegratorChoice::AdaptiveRK45 => Box::new(AdaptiveRK45Integrator::new(1e-6)),
        };

        match kinematics::drag_trajectory(
            self.velocity,
            angle_rad,
            self.gravity,
            &drag_model,
            self.mass,
            self.initial_height,
            &config,
            &*integrator,
        ) {
            Ok(traj) => {
                self.drag_samples = sample_trajectory(&traj, 25);
                self.drag_trajectory = Some(traj);
            }
            Err(e) => {
                self.error_message = Some(format!("Drag trajectory error: {e}"));
            }
        }
    }

    /// Collect trajectory tables for export.
    fn collect_tables(&self) -> Vec<DataTable> {
        let mut tables = Vec::new();
        if let Some(ref vac) = self.vacuum_trajectory {
            tables.push(trajectory_to_table(vac, "Vacuum"));
        }
        if let Some(ref drg) = self.drag_trajectory {
            tables.push(trajectory_to_table(drg, "Drag"));
        }
        tables
    }

    /// Save trajectories to a CSV file.
    fn save_csv_file(&mut self) {
        let tables = self.collect_tables();
        if tables.is_empty() {
            return;
        }

        let path = PathBuf::from(format!("{}.csv", self.export_path));
        // Concatenate all tables into one CSV string.
        let mut combined = String::new();
        for table in &tables {
            match table.to_csv() {
                Ok(csv) => {
                    combined.push_str(&format!("# {}\n", table.title));
                    combined.push_str(&csv);
                    combined.push('\n');
                }
                Err(e) => {
                    self.error_message = Some(format!("CSV error: {e}"));
                    return;
                }
            }
        }
        match std::fs::write(&path, combined) {
            Ok(()) => self.error_message = Some(format!("Saved to {}", path.display())),
            Err(e) => self.error_message = Some(format!("Write failed: {e}")),
        }
    }

    /// Save trajectories to an XLSX file.
    fn save_xlsx_file(&mut self) {
        let tables = self.collect_tables();
        if tables.is_empty() {
            return;
        }

        let path = PathBuf::from(format!("{}.xlsx", self.export_path));
        match write_tables_xlsx(&tables, &path) {
            Ok(()) => self.error_message = Some(format!("Saved to {}", path.display())),
            Err(e) => self.error_message = Some(format!("XLSX error: {e}")),
        }
    }

    /// Export both trajectories to CSV format and show in the text area.
    fn export_csv(&mut self) {
        let mut combined = String::new();

        if let Some(ref vac) = self.vacuum_trajectory {
            let table: DataTable = trajectory_to_table(vac, "Vacuum");
            match table.to_csv() {
                Ok(csv) => {
                    combined.push_str("# Vacuum Trajectory\n");
                    combined.push_str(&csv);
                    combined.push('\n');
                }
                Err(e) => {
                    self.error_message = Some(format!("CSV export error: {e}"));
                    return;
                }
            }
        }

        if let Some(ref drg) = self.drag_trajectory {
            let table: DataTable = trajectory_to_table(drg, "Drag");
            match table.to_csv() {
                Ok(csv) => {
                    combined.push_str("# Drag Trajectory\n");
                    combined.push_str(&csv);
                }
                Err(e) => {
                    self.error_message = Some(format!("CSV export error: {e}"));
                    return;
                }
            }
        }

        self.csv_output = combined;
        self.show_csv = true;
    }
}
