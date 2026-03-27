//! Home dashboard page.
//!
//! Displays a visually appealing landing page with a centered heading and
//! a grid of navigation cards leading to each simulation module.

use egui::Ui;

use crate::app::Page;

/// Render the home dashboard.
///
/// Returns `Some(page)` if the user clicks a navigation card, or `None` if
/// no navigation occurred this frame.
pub fn show_home(ui: &mut Ui) -> Option<Page> {
    let mut target: Option<Page> = None;

    ui.vertical_centered(|ui| {
        ui.add_space(40.0);
        ui.heading(
            egui::RichText::new("SimuCADSuite")
                .size(36.0)
                .strong(),
        );
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new("Scientific Simulation Platform")
                .size(16.0)
                .weak(),
        );
        ui.add_space(40.0);
    });

    // Navigation tiles.
    let mut tiles: Vec<(&str, &str, Page)> = vec![
        (
            "Projectile Motion",
            "Simulate projectile trajectories with\nvacuum and aerodynamic drag models.\nCompare results side-by-side and export CSV.",
            Page::Kinematics,
        ),
        (
            "Fluid Dynamics",
            "Run particle-based fluid simulations on\nimported Gmsh mesh geometries with\nprogress tracking and cancellation.",
            Page::FluidDynamics,
        ),
        (
            "Scientific Calculator",
            "Evaluate expressions, differentiate,\nintegrate, find roots, and plot\nsymbolic formulas with CAS engine.",
            Page::Calculator,
        ),
    ];

    // Audio card is only shown when the audio feature is enabled.
    #[cfg(feature = "audio")]
    {
        tiles.push((
            "Audio Analyzer",
            "Load audio files, visualize waveforms,\nand perform spectral analysis with\nreal-time playback.",
            Page::AudioAnalyzer,
        ));
    }

    let available_width = ui.available_width();
    let columns = if available_width > 700.0 {
        3
    } else if available_width > 450.0 {
        2
    } else {
        1
    };
    let tile_width = (available_width / columns as f32) - 16.0;

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(12.0, 12.0);

        for (title, description, page) in &tiles {
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(tile_width, 120.0),
                egui::Sense::click(),
            );

            // Draw card background.
            let visuals = if response.hovered() {
                ui.visuals().widgets.hovered
            } else {
                ui.visuals().widgets.inactive
            };

            ui.painter().rect(
                rect,
                8.0,
                visuals.bg_fill,
                visuals.bg_stroke,
            );

            // Card content.
            let text_rect = rect.shrink(12.0);
            ui.painter().text(
                text_rect.left_top(),
                egui::Align2::LEFT_TOP,
                *title,
                egui::FontId::proportional(16.0),
                visuals.text_color(),
            );
            ui.painter().text(
                text_rect.left_top() + egui::vec2(0.0, 24.0),
                egui::Align2::LEFT_TOP,
                *description,
                egui::FontId::proportional(12.0),
                ui.visuals().text_color().gamma_multiply(0.7),
            );

            if response.clicked() {
                target = Some(*page);
            }
        }
    });

    target
}
