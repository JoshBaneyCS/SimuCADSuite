//! Settings dialog UI.
//!
//! Renders toggle switches and path editors for all [`AppSettings`] fields.

use egui::Ui;
use simucad_core::settings::AppSettings;

/// Render the settings editor into the given `Ui`.
pub fn show_settings(ui: &mut Ui, settings: &mut AppSettings) {
    ui.heading("Settings");
    ui.add_space(8.0);

    // -----------------------------------------------------------------------
    // Appearance
    // -----------------------------------------------------------------------
    ui.collapsing("Appearance", |ui| {
        ui.horizontal(|ui| {
            ui.label("Dark mode:");
            ui.checkbox(&mut settings.appearance.dark_mode, "");
        });
    });

    ui.add_space(4.0);

    // -----------------------------------------------------------------------
    // Compute
    // -----------------------------------------------------------------------
    ui.collapsing("Compute", |ui| {
        ui.horizontal(|ui| {
            ui.label("GPU acceleration:");
            ui.checkbox(&mut settings.compute.gpu_enabled, "");
        });

        ui.horizontal(|ui| {
            ui.label("Multithreading:");
            ui.checkbox(&mut settings.compute.multithreading, "");
        });

        ui.horizontal(|ui| {
            ui.label("Thread count (0 = auto):");
            let mut count = settings.compute.thread_count.unwrap_or(0);
            if ui
                .add(egui::DragValue::new(&mut count).speed(1.0).range(0..=256))
                .changed()
            {
                settings.compute.thread_count = if count == 0 { None } else { Some(count) };
            }
        });
    });

    ui.add_space(4.0);

    // -----------------------------------------------------------------------
    // Paths
    // -----------------------------------------------------------------------
    ui.collapsing("Data Paths", |ui| {
        let mut kinematics = settings.paths.kinematics_data_path.display().to_string();
        let mut fluid = settings.paths.fluid_data_path.display().to_string();
        let mut audio = settings.paths.audio_data_path.display().to_string();
        let mut mesh = settings.paths.mesh_output_dir.display().to_string();

        egui::Grid::new("path_settings")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label("Kinematics data:");
                if ui.text_edit_singleline(&mut kinematics).changed() {
                    settings.paths.kinematics_data_path = kinematics.into();
                }
                ui.end_row();

                ui.label("Fluid data:");
                if ui.text_edit_singleline(&mut fluid).changed() {
                    settings.paths.fluid_data_path = fluid.into();
                }
                ui.end_row();

                ui.label("Audio data:");
                if ui.text_edit_singleline(&mut audio).changed() {
                    settings.paths.audio_data_path = audio.into();
                }
                ui.end_row();

                ui.label("Mesh output:");
                if ui.text_edit_singleline(&mut mesh).changed() {
                    settings.paths.mesh_output_dir = mesh.into();
                }
                ui.end_row();
            });
    });

    ui.add_space(12.0);

    // -----------------------------------------------------------------------
    // Reset / Save
    // -----------------------------------------------------------------------
    ui.horizontal(|ui| {
        if ui.button("Reset to Defaults").clicked() {
            *settings = AppSettings::default();
        }

        if ui.button("Save Settings").clicked() {
            let path = std::path::Path::new("simucad_settings.toml");
            match settings.save(path) {
                Ok(()) => {
                    tracing::info!("Settings saved to {}", path.display());
                }
                Err(e) => {
                    tracing::error!("Failed to save settings: {e}");
                }
            }
        }
    });
}
