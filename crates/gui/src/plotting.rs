//! Scientific plotting wrappers around `egui_plot`.
//!
//! Provides convenience functions for drawing trajectories, 2D function
//! curves, and vector fields inside an egui UI.

use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints, Arrows};
use simucad_core::types::{Trajectory, Vec2};

// ---------------------------------------------------------------------------
// Trajectory plotting
// ---------------------------------------------------------------------------

/// Plot vacuum and drag trajectories on the same axes.
///
/// The vacuum trajectory is drawn as a blue line and the drag trajectory as
/// a red line. Axes are labelled with distance in metres.
pub fn plot_trajectories(ui: &mut Ui, vacuum: &Trajectory, drag: &Trajectory) {
    let vacuum_points: PlotPoints = vacuum
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y])
        .collect();

    let drag_points: PlotPoints = drag
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y])
        .collect();

    let vacuum_line = Line::new(vacuum_points)
        .name("Vacuum")
        .color(egui::Color32::from_rgb(80, 140, 255));

    let drag_line = Line::new(drag_points)
        .name("With Drag")
        .color(egui::Color32::from_rgb(255, 100, 80));

    Plot::new("trajectory_plot")
        .legend(egui_plot::Legend::default())
        .x_axis_label("Horizontal Distance (m)")
        .y_axis_label("Height (m)")
        .height(300.0)
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(vacuum_line);
            plot_ui.line(drag_line);
        });
}

// ---------------------------------------------------------------------------
// 2D function plotting
// ---------------------------------------------------------------------------

/// Plot a set of (x, y) data points as a single curve.
pub fn plot_function_2d(ui: &mut Ui, points: &[(f64, f64)], label: &str) {
    let plot_points: PlotPoints = points.iter().map(|&(x, y)| [x, y]).collect();

    let line = Line::new(plot_points)
        .name(label)
        .color(egui::Color32::from_rgb(100, 200, 100));

    Plot::new("function_plot")
        .legend(egui_plot::Legend::default())
        .x_axis_label("x")
        .y_axis_label("f(x)")
        .height(300.0)
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

// ---------------------------------------------------------------------------
// Vector field plotting
// ---------------------------------------------------------------------------

/// Plot a 2D vector field as arrows.
///
/// `positions` gives the tail of each arrow, and `velocities` gives the
/// direction and magnitude. Both slices must have the same length.
pub fn plot_vector_field(ui: &mut Ui, positions: &[Vec2], velocities: &[Vec2]) {
    assert_eq!(
        positions.len(),
        velocities.len(),
        "positions and velocities must have the same length"
    );

    if positions.is_empty() {
        ui.label("(no vector data)");
        return;
    }

    // Find maximum velocity magnitude for scaling.
    let max_mag = velocities
        .iter()
        .map(|v| v.magnitude())
        .fold(0.0_f64, f64::max);

    let scale = if max_mag > f64::EPSILON {
        1.0 / max_mag
    } else {
        1.0
    };

    let origins: Vec<[f64; 2]> = positions.iter().map(|p| [p.x, p.y]).collect();
    // egui_plot::Arrows expects (origins, vectors) where vectors are the
    // direction offsets from each origin, not absolute tip positions.
    let vectors: Vec<[f64; 2]> = velocities
        .iter()
        .map(|v| [v.x * scale, v.y * scale])
        .collect();

    let arrows = Arrows::new(
        origins.into_iter().collect::<PlotPoints>(),
        vectors.into_iter().collect::<PlotPoints>(),
    )
    .name("Velocity Field")
    .color(egui::Color32::from_rgb(200, 200, 50));

    Plot::new("vector_field_plot")
        .legend(egui_plot::Legend::default())
        .x_axis_label("x (m)")
        .y_axis_label("y (m)")
        .height(300.0)
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            plot_ui.arrows(arrows);
        });
}
