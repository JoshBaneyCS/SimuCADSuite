//! Home dashboard page.
//!
//! Displays a grid of large navigation buttons that take the user to each
//! simulation module.

use egui::Ui;

use crate::app::Page;

/// Render the home dashboard.
///
/// Returns `Some(page)` if the user clicks a navigation button, or `None` if
/// no navigation occurred this frame.
pub fn show_home(ui: &mut Ui) -> Option<Page> {
    let mut target: Option<Page> = None;

    ui.vertical_centered(|ui| {
        ui.add_space(20.0);
        ui.heading("SimuCAD Suite");
        ui.label("Scientific Simulation Platform");
        ui.add_space(30.0);
    });

    // Navigation tiles laid out in a responsive grid.
    let tiles: &[(&str, &str, Page)] = &[
        (
            "Projectile Motion",
            "Simulate projectile trajectories with\nvacuum and drag models",
            Page::Kinematics,
        ),
        (
            "Fluid Dynamics",
            "Run CFD simulations on imported\nmesh geometries",
            Page::FluidDynamics,
        ),
        (
            "Scientific Calculator",
            "Evaluate expressions, differentiate\nsymbolic formulas, and plot functions",
            Page::Calculator,
        ),
        (
            "Settings",
            "Configure appearance, compute backend,\nand data paths",
            Page::Settings,
        ),
    ];

    let available_width = ui.available_width();
    // Try 2 columns, but fall back to 1 if the panel is too narrow.
    let columns = if available_width > 500.0 { 2 } else { 1 };
    let tile_width = (available_width / columns as f32) - 16.0;

    egui::Grid::new("home_grid")
        .num_columns(columns)
        .spacing([12.0, 12.0])
        .show(ui, |ui| {
            for (i, (title, description, page)) in tiles.iter().enumerate() {
                let response = ui.allocate_ui_with_layout(
                    egui::vec2(tile_width, 100.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        egui::Frame::group(ui.style())
                            .inner_margin(12.0)
                            .show(ui, |ui| {
                                ui.set_min_size(egui::vec2(tile_width - 28.0, 70.0));
                                ui.strong(*title);
                                ui.add_space(4.0);
                                ui.label(*description);
                            })
                    },
                );

                if response.response.interact(egui::Sense::click()).clicked() {
                    target = Some(*page);
                }

                // End the row after `columns` tiles.
                if (i + 1) % columns == 0 {
                    ui.end_row();
                }
            }
        });

    target
}
