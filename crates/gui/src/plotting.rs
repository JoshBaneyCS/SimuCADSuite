//! Scientific plotting wrappers around `egui_plot`.
//!
//! Provides convenience functions for drawing trajectories, 2D function
//! curves, vector fields, and sampled scatter plots inside an egui UI.

use egui::Ui;
use egui_plot::{Arrows, Line, Plot, PlotPoints, Points, Legend};
use simucad_core::types::{Trajectory, TrajectoryPoint, Vec2};

// ---------------------------------------------------------------------------
// Trajectory plotting
// ---------------------------------------------------------------------------

/// Plot vacuum and/or drag trajectories on the same axes.
///
/// Either trajectory may be `None`; at least one should be `Some` for a
/// meaningful plot. The vacuum trajectory is drawn as a blue line and the
/// drag trajectory as a red line. Axes are labelled with distance in metres.
pub fn plot_trajectories(
    ui: &mut Ui,
    vacuum: Option<&Trajectory>,
    drag: Option<&Trajectory>,
) {
    Plot::new("trajectory_plot")
        .legend(Legend::default())
        .x_axis_label("Horizontal Distance (m)")
        .y_axis_label("Height (m)")
        .height(300.0)
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            if let Some(vac) = vacuum {
                let points: PlotPoints = vac
                    .points
                    .iter()
                    .map(|p| [p.position.x, p.position.y])
                    .collect();
                let line = Line::new(points)
                    .name("Vacuum")
                    .color(egui::Color32::from_rgb(80, 140, 255));
                plot_ui.line(line);
            }

            if let Some(drg) = drag {
                let points: PlotPoints = drg
                    .points
                    .iter()
                    .map(|p| [p.position.x, p.position.y])
                    .collect();
                let line = Line::new(points)
                    .name("With Drag")
                    .color(egui::Color32::from_rgb(255, 100, 80));
                plot_ui.line(line);
            }
        });
}

// ---------------------------------------------------------------------------
// 2D function plotting
// ---------------------------------------------------------------------------

/// Plot a set of (x, y) data points as a single curve.
pub fn plot_function_2d(ui: &mut Ui, points: &[(f64, f64)], label: &str) {
    plot_function_2d_ex(ui, points, label, false);
}

/// Plot a set of (x, y) data points as a single curve, with optional equal
/// aspect ratio (useful for polar plots).
pub fn plot_function_2d_ex(
    ui: &mut Ui,
    points: &[(f64, f64)],
    label: &str,
    equal_aspect: bool,
) {
    let plot_points: PlotPoints = points.iter().map(|&(x, y)| [x, y]).collect();

    let line = Line::new(plot_points)
        .name(label)
        .color(egui::Color32::from_rgb(100, 200, 100));

    let mut plot = Plot::new(egui::Id::new(label).with("fn_plot"))
        .legend(Legend::default())
        .x_axis_label("x")
        .y_axis_label("f(x)")
        .height(300.0);

    if equal_aspect {
        plot = plot.data_aspect(1.0);
    }

    plot.show(ui, |plot_ui| {
        plot_ui.line(line);
    });
}

/// Plot two functions overlaid on the same axes.
pub fn plot_two_functions(
    ui: &mut Ui,
    points_a: &[(f64, f64)],
    label_a: &str,
    points_b: &[(f64, f64)],
    label_b: &str,
) {
    let pp_a: PlotPoints = points_a.iter().map(|&(x, y)| [x, y]).collect();
    let pp_b: PlotPoints = points_b.iter().map(|&(x, y)| [x, y]).collect();

    let line_a = Line::new(pp_a)
        .name(label_a)
        .color(egui::Color32::from_rgb(100, 200, 100));
    let line_b = Line::new(pp_b)
        .name(label_b)
        .color(egui::Color32::from_rgb(255, 150, 50));

    Plot::new("overlay_plot")
        .legend(Legend::default())
        .x_axis_label("x")
        .y_axis_label("y")
        .height(300.0)
        .show(ui, |plot_ui| {
            plot_ui.line(line_a);
            plot_ui.line(line_b);
        });
}

// ---------------------------------------------------------------------------
// Sampled trajectory points — scatter plot with velocity vectors
// ---------------------------------------------------------------------------

/// Plot sampled trajectory points as a scatter with tiny velocity arrows.
pub fn plot_sampled_points(ui: &mut Ui, points: &[TrajectoryPoint]) {
    if points.is_empty() {
        ui.label("(no sampled points)");
        return;
    }

    let positions: PlotPoints = points
        .iter()
        .map(|p| [p.position.x, p.position.y])
        .collect();

    let scatter = Points::new(positions)
        .name("Sample Points")
        .radius(4.0)
        .color(egui::Color32::from_rgb(255, 200, 50));

    // Velocity arrows — scale so they are visible but not overwhelming.
    let max_speed = points
        .iter()
        .map(|p| p.speed)
        .fold(0.0_f64, f64::max);

    let arrow_scale = if max_speed > f64::EPSILON {
        // Make longest arrow roughly 5% of the trajectory range.
        let x_range = points.last().map(|p| p.position.x).unwrap_or(1.0);
        (x_range * 0.05) / max_speed
    } else {
        1.0
    };

    let origins: Vec<[f64; 2]> = points.iter().map(|p| [p.position.x, p.position.y]).collect();
    let vectors: Vec<[f64; 2]> = points
        .iter()
        .map(|p| [p.velocity.x * arrow_scale, p.velocity.y * arrow_scale])
        .collect();

    let arrows = Arrows::new(
        origins.into_iter().collect::<PlotPoints>(),
        vectors.into_iter().collect::<PlotPoints>(),
    )
    .name("Velocity")
    .color(egui::Color32::from_rgb(150, 150, 255));

    Plot::new("sampled_points_plot")
        .legend(Legend::default())
        .x_axis_label("x (m)")
        .y_axis_label("y (m)")
        .height(250.0)
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            plot_ui.points(scatter);
            plot_ui.arrows(arrows);
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
        .legend(Legend::default())
        .x_axis_label("x (m)")
        .y_axis_label("y (m)")
        .height(300.0)
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            plot_ui.arrows(arrows);
        });
}
