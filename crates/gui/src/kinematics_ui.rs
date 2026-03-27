//! Projectile motion (kinematics) UI panel.
//!
//! Provides input fields for launch parameters and displays both vacuum and
//! drag trajectories using the physics crate solvers.

use egui::Ui;
use simucad_core::types::Trajectory;
use simucad_physics::drag::{DragModel, DragShape};

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

    /// Computed vacuum trajectory (filled on "Calculate").
    pub vacuum_trajectory: Option<Trajectory>,
    /// Computed drag trajectory (filled on "Calculate").
    pub drag_trajectory: Option<Trajectory>,
    /// Error message from the last calculation attempt.
    pub error_message: Option<String>,
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
            vacuum_trajectory: None,
            drag_trajectory: None,
            error_message: None,
        }
    }
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
            });

        ui.add_space(12.0);

        if ui.button("Calculate").clicked() {
            self.calculate();
        }

        if let Some(ref err) = self.error_message {
            ui.add_space(4.0);
            ui.colored_label(egui::Color32::RED, err);
        }
    }

    /// Render the trajectory results (plot and table).
    fn show_results(&mut self, ui: &mut Ui) {
        match (&self.vacuum_trajectory, &self.drag_trajectory) {
            (Some(vacuum), Some(drag)) => {
                plotting::plot_trajectories(ui, vacuum, drag);
                ui.add_space(12.0);
                self.show_summary_table(ui, vacuum, drag);
            }
            _ => {
                ui.label("Enter parameters and click Calculate to see results.");
            }
        }
    }

    /// Display a summary comparison table.
    fn show_summary_table(&self, ui: &mut Ui, vacuum: &Trajectory, drag: &Trajectory) {
        egui::Grid::new("trajectory_summary")
            .num_columns(3)
            .striped(true)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.strong("Metric");
                ui.strong("Vacuum");
                ui.strong("With Drag");
                ui.end_row();

                ui.label("Range (m)");
                ui.label(format!("{:.2}", vacuum.range));
                ui.label(format!("{:.2}", drag.range));
                ui.end_row();

                ui.label("Max Height (m)");
                ui.label(format!("{:.2}", vacuum.max_height));
                ui.label(format!("{:.2}", drag.max_height));
                ui.end_row();

                ui.label("Flight Time (s)");
                ui.label(format!("{:.3}", vacuum.flight_time));
                ui.label(format!("{:.3}", drag.flight_time));
                ui.end_row();
            });

        // Show sampled points from the vacuum trajectory.
        ui.add_space(8.0);
        ui.collapsing("Sampled Points (Vacuum)", |ui| {
            Self::show_points_table(ui, vacuum, "vacuum_pts");
        });

        ui.collapsing("Sampled Points (Drag)", |ui| {
            Self::show_points_table(ui, drag, "drag_pts");
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

    /// Run the vacuum and drag trajectory calculations.
    fn calculate(&mut self) {
        self.error_message = None;

        let angle_rad = self.angle_deg.to_radians();
        let vx = self.velocity * angle_rad.cos();
        let vy = self.velocity * angle_rad.sin();

        let dt = 0.001;
        let max_steps = 500_000;

        // Vacuum trajectory (no drag).
        self.vacuum_trajectory = Some(Self::integrate_trajectory(
            vx,
            vy,
            self.initial_height,
            self.gravity,
            DragModel::vacuum(),
            self.mass,
            dt,
            max_steps,
        ));

        // Drag trajectory.
        let drag_model = DragModel::at_sea_level(self.drag_shape, self.area);
        self.drag_trajectory = Some(Self::integrate_trajectory(
            vx,
            vy,
            self.initial_height,
            self.gravity,
            drag_model,
            self.mass,
            dt,
            max_steps,
        ));
    }

    /// Integrate a trajectory using symplectic Euler with the given drag
    /// model. This is a self-contained integrator so the GUI crate does not
    /// require kinematics/trajectory modules that may not exist yet.
    fn integrate_trajectory(
        vx0: f64,
        vy0: f64,
        y0: f64,
        gravity: f64,
        drag_model: DragModel,
        mass: f64,
        dt: f64,
        max_steps: usize,
    ) -> Trajectory {
        use simucad_core::types::{TrajectoryPoint, Vec2};

        let mut points = Vec::with_capacity(max_steps.min(10_000));
        let mut x = 0.0_f64;
        let mut y = y0;
        let mut vx = vx0;
        let mut vy = vy0;
        let mut t = 0.0_f64;
        let mut max_height = y;

        // Record the initial point.
        points.push(TrajectoryPoint {
            time: t,
            position: Vec2::new(x, y),
            velocity: Vec2::new(vx, vy),
            speed: (vx * vx + vy * vy).sqrt(),
        });

        for _ in 0..max_steps {
            // Net acceleration: gravity + drag/mass.
            let vel = Vec2::new(vx, vy);
            let drag_force = drag_model.drag_force(vel);
            let ax = drag_force.x / mass;
            let ay = -gravity + drag_force.y / mass;

            // Symplectic Euler: update velocity first, then position.
            vx += ax * dt;
            vy += ay * dt;
            x += vx * dt;
            y += vy * dt;
            t += dt;

            if y > max_height {
                max_height = y;
            }

            points.push(TrajectoryPoint {
                time: t,
                position: Vec2::new(x, y),
                velocity: Vec2::new(vx, vy),
                speed: (vx * vx + vy * vy).sqrt(),
            });

            // Stop when the projectile hits the ground.
            if y <= 0.0 {
                break;
            }
        }

        let range = points.last().map(|p| p.position.x).unwrap_or(0.0);
        let flight_time = points.last().map(|p| p.time).unwrap_or(0.0);

        Trajectory {
            points,
            max_height,
            range,
            flight_time,
        }
    }
}
