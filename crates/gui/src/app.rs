//! Main application shell for the SimuCAD Suite desktop GUI.
//!
//! This module defines the top-level [`SimuApp`] struct that implements
//! [`eframe::App`] and orchestrates page navigation, settings, and the
//! menu bar.

use std::path::PathBuf;

use eframe::egui;
use simucad_core::export::ProjectFile;
use simucad_core::settings::AppSettings;

use crate::calculator_ui::CalculatorPanel;
use crate::fluid_ui::FluidPanel;
use crate::home;
use crate::kinematics_ui::KinematicsPanel;
use crate::settings_ui;

#[cfg(feature = "audio")]
use crate::audio_ui::AudioPanel;

// ---------------------------------------------------------------------------
// Page enum
// ---------------------------------------------------------------------------

/// The pages (screens) the application can display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Kinematics,
    FluidDynamics,
    Calculator,
    AudioAnalyzer,
    Settings,
}

impl Page {
    /// Human-readable label for the page, used in the navigation bar.
    pub fn label(&self) -> &'static str {
        match self {
            Page::Home => "Home",
            Page::Kinematics => "Projectile Motion",
            Page::FluidDynamics => "Fluid Dynamics",
            Page::Calculator => "Scientific Calculator",
            Page::AudioAnalyzer => "Audio Analyzer",
            Page::Settings => "Settings",
        }
    }
}

// ---------------------------------------------------------------------------
// SimuApp
// ---------------------------------------------------------------------------

/// The root application state for the SimuCAD Suite GUI.
pub struct SimuApp {
    /// Currently displayed page.
    pub current_page: Page,
    /// Persistent application settings.
    pub settings: AppSettings,
    /// Kinematics (projectile motion) panel state.
    pub kinematics_panel: KinematicsPanel,
    /// Fluid dynamics panel state.
    pub fluid_panel: FluidPanel,
    /// Calculator panel state.
    pub calculator_panel: CalculatorPanel,
    /// Audio analyzer panel state.
    #[cfg(feature = "audio")]
    pub audio_panel: AudioPanel,
    /// Whether the settings dialog is open (as a floating window).
    pub settings_open: bool,
    /// Path of the currently open project file.
    pub project_path: Option<PathBuf>,
    /// Status message for file operations.
    pub file_status: String,
}

impl SimuApp {
    /// Create a new application instance, optionally restoring persisted
    /// state from the [`eframe::CreationContext`].
    pub fn new(cc: &eframe::CreationContext, settings: AppSettings) -> Self {
        // Initialize 3D renderer if wgpu backend is available.
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            let resources = crate::renderer_3d::Renderer3DResources::new(
                &render_state.device,
                render_state.target_format,
            );
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);
            tracing::info!("3D renderer initialized (wgpu)");
        }

        Self {
            current_page: Page::Home,
            settings,
            kinematics_panel: KinematicsPanel::default(),
            fluid_panel: FluidPanel::default(),
            calculator_panel: CalculatorPanel::default(),
            #[cfg(feature = "audio")]
            audio_panel: AudioPanel::default(),
            settings_open: false,
            project_path: None,
            file_status: String::new(),
        }
    }

    /// Save current panel parameters to a project file.
    fn save_project(&mut self) {
        let path = self
            .project_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("project.simucad"));

        let kin = &self.kinematics_panel;
        let project = ProjectFile {
            version: 1,
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            kinematics: Some(simucad_core::export::KinematicsProject {
                velocity: kin.velocity,
                angle_deg: kin.angle_deg,
                gravity: kin.gravity,
                mass: kin.mass,
                area: kin.area,
                initial_height: kin.initial_height,
                drag_shape: format!("{:?}", kin.drag_shape),
                integrator: format!("{:?}", kin.integrator_choice),
            }),
            fluid: Some(simucad_core::export::FluidProject {
                mesh_path: self.fluid_panel.mesh_path.clone(),
                velocity_x: self.fluid_panel.velocity_x,
                velocity_y: self.fluid_panel.velocity_y,
                velocity_z: self.fluid_panel.velocity_z,
                particle_count: self.fluid_panel.particle_count,
                num_steps: self.fluid_panel.num_steps,
                use_gpu: self.fluid_panel.use_gpu,
            }),
            calculator: Some(simucad_core::export::CalculatorProject {
                expression: self.calculator_panel.expression_input.clone(),
                variable: self.calculator_panel.diff_variable.clone(),
                plot_x_min: self.calculator_panel.plot_x_min,
                plot_x_max: self.calculator_panel.plot_x_max,
                plot_samples: self.calculator_panel.plot_samples,
                angle_mode: "Radians".into(),
            }),
        };

        match project.save(&path) {
            Ok(()) => {
                self.file_status = format!("Project saved to {}", path.display());
                self.project_path = Some(path);
            }
            Err(e) => {
                self.file_status = format!("Save failed: {e}");
            }
        }
    }

    /// Load panel parameters from a project file.
    fn load_project(&mut self) {
        let path = self
            .project_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("project.simucad"));

        match ProjectFile::load(&path) {
            Ok(project) => {
                if let Some(kin) = project.kinematics {
                    self.kinematics_panel.velocity = kin.velocity;
                    self.kinematics_panel.angle_deg = kin.angle_deg;
                    self.kinematics_panel.gravity = kin.gravity;
                    self.kinematics_panel.mass = kin.mass;
                    self.kinematics_panel.area = kin.area;
                    self.kinematics_panel.initial_height = kin.initial_height;
                }

                if let Some(fluid) = project.fluid {
                    self.fluid_panel.mesh_path = fluid.mesh_path;
                    self.fluid_panel.velocity_x = fluid.velocity_x;
                    self.fluid_panel.velocity_y = fluid.velocity_y;
                    self.fluid_panel.velocity_z = fluid.velocity_z;
                    self.fluid_panel.particle_count = fluid.particle_count;
                    self.fluid_panel.num_steps = fluid.num_steps;
                    self.fluid_panel.use_gpu = fluid.use_gpu;
                }

                if let Some(calc) = project.calculator {
                    self.calculator_panel.expression_input = calc.expression;
                    self.calculator_panel.diff_variable = calc.variable;
                    self.calculator_panel.plot_x_min = calc.plot_x_min;
                    self.calculator_panel.plot_x_max = calc.plot_x_max;
                    self.calculator_panel.plot_samples = calc.plot_samples;
                }

                self.file_status = format!("Project loaded from {}", path.display());
                self.project_path = Some(path);
            }
            Err(e) => {
                self.file_status = format!("Load failed: {e}");
            }
        }
    }
}

impl eframe::App for SimuApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply dark/light theme based on settings.
        if self.settings.appearance.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        // ---------------------------------------------------------------
        // Top menu bar
        // ---------------------------------------------------------------
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Home").clicked() {
                        self.current_page = Page::Home;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save Project...").clicked() {
                        self.save_project();
                        ui.close_menu();
                    }
                    if ui.button("Load Project...").clicked() {
                        self.load_project();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Settings", |ui| {
                    if ui.button("Open Settings...").clicked() {
                        self.settings_open = true;
                        ui.close_menu();
                    }
                });

                // Navigation breadcrumbs on the right side of the menu bar.
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        let mut pages = vec![
                            Page::Home,
                            Page::Kinematics,
                            Page::FluidDynamics,
                            Page::Calculator,
                        ];
                        #[cfg(feature = "audio")]
                        pages.push(Page::AudioAnalyzer);
                        for page in pages.iter().rev() {
                            let label = page.label();
                            let is_current = *page == self.current_page;
                            let button = egui::Button::new(label).selected(is_current);
                            if ui.add(button).clicked() {
                                self.current_page = *page;
                            }
                        }
                    },
                );
            });
        });

        // ---------------------------------------------------------------
        // Status bar at bottom
        // ---------------------------------------------------------------
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "SimuCADSuite v{}",
                        env!("CARGO_PKG_VERSION")
                    ))
                    .small()
                    .weak(),
                );
                if !self.file_status.is_empty() {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(&self.file_status).small().weak(),
                    );
                }
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(
                            egui::RichText::new(self.current_page.label())
                                .small()
                                .weak(),
                        );
                    },
                );
            });
        });

        // ---------------------------------------------------------------
        // Settings window (floating)
        // ---------------------------------------------------------------
        if self.settings_open {
            let mut open = self.settings_open;
            egui::Window::new("Settings")
                .open(&mut open)
                .resizable(true)
                .default_width(400.0)
                .show(ctx, |ui| {
                    settings_ui::show_settings(ui, &mut self.settings);
                });
            self.settings_open = open;
        }

        // ---------------------------------------------------------------
        // Central panel -- renders the current page
        // ---------------------------------------------------------------
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_page {
                Page::Home => {
                    if let Some(target) = home::show_home(ui) {
                        self.current_page = target;
                    }
                }
                Page::Kinematics => {
                    self.kinematics_panel.show(ui);
                }
                Page::FluidDynamics => {
                    self.fluid_panel.show(ui);
                }
                Page::Calculator => {
                    self.calculator_panel.show(ui);
                }
                Page::AudioAnalyzer => {
                    #[cfg(feature = "audio")]
                    self.audio_panel.show(ui);
                    #[cfg(not(feature = "audio"))]
                    ui.label("Audio feature not enabled. Rebuild with --features audio.");
                }
                Page::Settings => {
                    settings_ui::show_settings(ui, &mut self.settings);
                }
            }
        });
    }
}
