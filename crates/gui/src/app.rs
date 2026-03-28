//! Main application shell for the SimuCAD Suite desktop GUI.
//!
//! This module defines the top-level [`SimuApp`] struct that implements
//! [`eframe::App`] and orchestrates page navigation, settings, and the
//! menu bar.

use eframe::egui;
use simucad_core::settings::AppSettings;

use crate::calculator_ui::CalculatorPanel;
use crate::fluid_ui::FluidPanel;
use crate::home;
use crate::kinematics_ui::KinematicsPanel;
use crate::mesh_ui::MeshPanel;
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
    MeshViewer,
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
            Page::MeshViewer => "Mesh Viewer",
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
    /// Mesh viewer panel state.
    pub mesh_panel: MeshPanel,
    /// Audio analyzer panel state.
    #[cfg(feature = "audio")]
    pub audio_panel: AudioPanel,
    /// Whether the settings dialog is open (as a floating window).
    pub settings_open: bool,
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
            mesh_panel: MeshPanel::default(),
            #[cfg(feature = "audio")]
            audio_panel: AudioPanel::default(),
            settings_open: false,
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
                            Page::MeshViewer,
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
                Page::MeshViewer => {
                    self.mesh_panel.show(ui);
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
