//! Scientific plotting wrappers around `egui_plot`.
//!
//! Provides convenience functions for drawing trajectories, 2D function
//! curves, vector fields, and sampled scatter plots inside an egui UI.

use egui::Ui;
use egui_plot::{Arrows, Line, Plot, PlotPoints, Points, Legend};
use simucad_core::types::{Trajectory, TrajectoryPoint, Vec2};

// Re-export color mapping utilities for use in other modules.
pub use self::colormaps::{colormap_viridis, colormap_inferno, colormap_coolwarm};

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

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------

pub mod colormaps {
    use egui::Color32;

    /// Viridis-like colormap: dark purple -> blue -> green -> yellow.
    pub fn colormap_viridis(t: f64) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        let r = (255.0 * (1.5 * t - 0.5).clamp(0.0, 1.0)) as u8;
        let g = (255.0
            * (2.0 * t - 0.5)
                .clamp(0.0, 1.0)
                .min(1.0 - (2.0 * t - 1.5).max(0.0))) as u8;
        let b = (255.0 * (1.0 - 2.0 * t).clamp(0.0, 1.0)) as u8;
        Color32::from_rgb(r, g, b)
    }

    /// Inferno-like colormap: black -> magenta -> orange -> yellow.
    pub fn colormap_inferno(t: f64) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        let r = (255.0 * (1.5 * t).clamp(0.0, 1.0)) as u8;
        let g = (255.0 * (3.0 * t - 1.5).clamp(0.0, 1.0)) as u8;
        let b = (255.0 * (1.0 - (3.0 * t - 0.5).abs()).clamp(0.0, 1.0)) as u8;
        Color32::from_rgb(r, g, b)
    }

    /// Cool-to-warm (blue -> white -> red) diverging colormap.
    pub fn colormap_coolwarm(t: f64) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 {
            let s = t * 2.0;
            let r = (255.0 * s) as u8;
            let g = (255.0 * s) as u8;
            Color32::from_rgb(r, g, 255)
        } else {
            let s = (t - 0.5) * 2.0;
            let g = (255.0 * (1.0 - s)) as u8;
            let b = (255.0 * (1.0 - s)) as u8;
            Color32::from_rgb(255, g, b)
        }
    }
}

// ---------------------------------------------------------------------------
// 2D Heatmap (texture-based)
// ---------------------------------------------------------------------------

/// Render a 2D heatmap as an egui texture.
///
/// `data` is a row-major 2D grid `[rows][cols]`. The axes are labelled with
/// `x_label` / `y_label`, and the value range is auto-normalized to the
/// colormap.
pub fn plot_heatmap(
    ui: &mut Ui,
    data: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    x_label: &str,
    y_label: &str,
    title: &str,
    texture_handle: &mut Option<egui::TextureHandle>,
    colormap: fn(f64) -> egui::Color32,
) {
    if data.is_empty() || data[0].is_empty() {
        ui.label("(no heatmap data)");
        return;
    }

    let rows = data.len();
    let cols = data[0].len();

    // Auto-normalize.
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for row in data {
        for &v in row {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
    }
    let range = (max_val - min_val).max(1e-12);

    // Build texture (rebuild each frame for animation, or cache when static).
    let mut pixels = vec![egui::Color32::BLACK; cols * rows];
    for (row_idx, row) in data.iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            let t = if val.is_finite() {
                (val - min_val) / range
            } else {
                0.0
            };
            // Flip y so row 0 (bottom) is at the bottom of the image.
            let y = rows - 1 - row_idx;
            pixels[y * cols + col_idx] = colormap(t);
        }
    }

    let image = egui::ColorImage {
        size: [cols, rows],
        pixels,
    };

    let tex = ui.ctx().load_texture(
        title,
        image,
        egui::TextureOptions::LINEAR,
    );
    *texture_handle = Some(tex);

    ui.strong(title);

    // Draw axis labels and the texture image.
    ui.horizontal(|ui| {
        ui.label(format!(
            "{x_label}: [{:.2}, {:.2}]  {y_label}: [{:.2}, {:.2}]",
            x_range.0, x_range.1, y_range.0, y_range.1
        ));
        ui.label(format!(
            "  Range: [{:.4}, {:.4}]",
            min_val, max_val
        ));
    });

    if let &mut Some(ref tex) = texture_handle {
        let available = ui.available_width().min(800.0);
        let aspect = cols as f32 / rows as f32;
        let height = (available / aspect).min(300.0);
        ui.image(egui::load::SizedTexture::new(
            tex.id(),
            egui::vec2(available, height),
        ));
    }
}

/// Build a heatmap `ColorImage` for export (no UI needed).
pub fn build_heatmap_image(
    data: &[Vec<f64>],
    colormap: fn(f64) -> egui::Color32,
) -> Vec<u8> {
    if data.is_empty() || data[0].is_empty() {
        return Vec::new();
    }

    let rows = data.len();
    let cols = data[0].len();

    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for row in data {
        for &v in row {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
    }
    let range = (max_val - min_val).max(1e-12);

    let mut rgba = Vec::with_capacity(cols * rows * 4);
    for row_idx in 0..rows {
        let y = rows - 1 - row_idx;
        for col_idx in 0..cols {
            let val = data[y][col_idx];
            let t = if val.is_finite() {
                (val - min_val) / range
            } else {
                0.0
            };
            let c = colormap(t);
            rgba.push(c.r());
            rgba.push(c.g());
            rgba.push(c.b());
            rgba.push(255);
        }
    }
    rgba
}

// ---------------------------------------------------------------------------
// Contour plots (marching squares)
// ---------------------------------------------------------------------------

/// Extract contour lines at the given iso-values from a 2D grid.
///
/// Returns a list of polylines, where each polyline is a `Vec<(f64, f64)>`
/// of (x, y) coordinates in the data's coordinate space.
pub fn extract_contours(
    data: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    iso_values: &[f64],
) -> Vec<(f64, Vec<(f64, f64)>)> {
    if data.is_empty() || data[0].is_empty() {
        return Vec::new();
    }

    let rows = data.len();
    let cols = data[0].len();
    let dx = (x_range.1 - x_range.0) / (cols - 1).max(1) as f64;
    let dy = (y_range.1 - y_range.0) / (rows - 1).max(1) as f64;

    let mut contours = Vec::new();

    for &iso in iso_values {
        let mut segments: Vec<((f64, f64), (f64, f64))> = Vec::new();

        // Marching squares: process each cell.
        for row in 0..rows - 1 {
            for col in 0..cols - 1 {
                let v00 = data[row][col];
                let v10 = data[row][col + 1];
                let v01 = data[row + 1][col];
                let v11 = data[row + 1][col + 1];

                // Cell corners in coordinate space.
                let x0 = x_range.0 + col as f64 * dx;
                let x1 = x0 + dx;
                let y0 = y_range.0 + row as f64 * dy;
                let y1 = y0 + dy;

                // Classify corners: above (1) or below (0) the iso-value.
                let case = ((v00 >= iso) as u8)
                    | (((v10 >= iso) as u8) << 1)
                    | (((v11 >= iso) as u8) << 2)
                    | (((v01 >= iso) as u8) << 3);

                if case == 0 || case == 15 {
                    continue; // No contour through this cell.
                }

                // Linear interpolation along edges.
                let lerp = |a: f64, b: f64| -> f64 {
                    if (b - a).abs() < 1e-15 {
                        0.5
                    } else {
                        (iso - a) / (b - a)
                    }
                };

                // Edge midpoints (interpolated).
                let bottom = (x0 + lerp(v00, v10) * dx, y0); // edge 0-1
                let right = (x1, y0 + lerp(v10, v11) * dy);  // edge 1-2
                let top = (x0 + lerp(v01, v11) * dx, y1);    // edge 3-2
                let left = (x0, y0 + lerp(v00, v01) * dy);   // edge 0-3

                // Map case to line segments.
                let segs: &[((f64, f64), (f64, f64))] = match case {
                    1 | 14 => &[(bottom, left)],
                    2 | 13 => &[(bottom, right)],
                    3 | 12 => &[(left, right)],
                    4 | 11 => &[(right, top)],
                    5 => &[(bottom, right), (left, top)], // saddle
                    6 | 9 => &[(bottom, top)],
                    7 | 8 => &[(left, top)],
                    10 => &[(bottom, left), (right, top)], // saddle
                    _ => &[],
                };

                for &seg in segs {
                    segments.push(seg);
                }
            }
        }

        // Chain segments into polylines.
        let points: Vec<(f64, f64)> = segments
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect();

        if !points.is_empty() {
            contours.push((iso, points));
        }
    }

    contours
}

/// Plot contour lines on an egui_plot.
pub fn plot_contours(
    ui: &mut Ui,
    data: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    num_levels: usize,
    title: &str,
) {
    if data.is_empty() || data[0].is_empty() {
        ui.label("(no contour data)");
        return;
    }

    // Auto-compute iso-values.
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for row in data {
        for &v in row {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
    }

    let levels: Vec<f64> = (0..num_levels)
        .map(|i| min_val + (max_val - min_val) * (i as f64 + 0.5) / num_levels as f64)
        .collect();

    let contours = extract_contours(data, x_range, y_range, &levels);

    Plot::new(egui::Id::new(title).with("contour"))
        .legend(Legend::default())
        .x_axis_label("x")
        .y_axis_label("t")
        .height(300.0)
        .show(ui, |plot_ui| {
            let palette = [
                egui::Color32::from_rgb(68, 1, 84),
                egui::Color32::from_rgb(59, 82, 139),
                egui::Color32::from_rgb(33, 145, 140),
                egui::Color32::from_rgb(94, 201, 98),
                egui::Color32::from_rgb(253, 231, 37),
                egui::Color32::from_rgb(255, 180, 50),
                egui::Color32::from_rgb(255, 100, 50),
                egui::Color32::from_rgb(200, 50, 50),
            ];

            for (idx, (iso_val, pts)) in contours.iter().enumerate() {
                // Draw segments as individual 2-point lines.
                for chunk in pts.chunks(2) {
                    if chunk.len() == 2 {
                        let line = Line::new(PlotPoints::new(vec![
                            [chunk[0].0, chunk[0].1],
                            [chunk[1].0, chunk[1].1],
                        ]))
                        .color(palette[idx % palette.len()])
                        .width(1.5);
                        plot_ui.line(line);
                    }
                }

                // One labeled invisible point for the legend.
                let label_pt = Points::new(PlotPoints::new(vec![[f64::NAN, f64::NAN]]))
                    .name(format!("{iso_val:.3}"))
                    .color(palette[idx % palette.len()])
                    .radius(0.0);
                plot_ui.points(label_pt);
            }
        });
}
