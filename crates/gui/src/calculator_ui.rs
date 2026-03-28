//! Scientific calculator UI panel.
//!
//! Provides a text input for mathematical expressions with buttons for
//! evaluation, symbolic differentiation, integration, root-finding, and
//! function plotting. CAS-dependent features are gated behind
//! `#[cfg(feature = "cas")]`.

use std::path::PathBuf;

use egui::Ui;

use crate::animation::{self, ColormapChoice};
use crate::plotting;

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// 2D plot coordinate mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlotMode {
    /// y = f(x)
    #[default]
    Cartesian,
    /// r = f(theta), displayed in Cartesian coordinates.
    Polar,
    /// x = f(t), y = g(t)
    Parametric,
}

/// PDE solver mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PdeMode {
    #[default]
    Wave,
    Heat,
}

/// PDE visualization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PdeVizMode {
    /// Standard 1D line plot at each time step.
    #[default]
    LinePlot,
    /// Space-time heatmap (x-axis = space, y-axis = time).
    Heatmap,
    /// Contour lines over space-time grid.
    Contour,
}

/// Fourier analysis sub-mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FourierMode {
    #[default]
    Series,
    Dft,
}

/// State for the scientific calculator panel.
pub struct CalculatorPanel {
    /// The current expression string typed by the user.
    pub expression_input: String,
    /// The variable to differentiate/integrate with respect to.
    pub diff_variable: String,
    /// Result of the last operation (evaluation, differentiation, etc.).
    pub result: Option<String>,
    /// LaTeX preview string.
    pub latex_preview: Option<String>,
    /// Plot data for the most recently plotted function: `(x, y)` pairs.
    pub plot_data: Vec<(f64, f64)>,
    /// X-axis minimum for plotting.
    pub plot_x_min: f64,
    /// X-axis maximum for plotting.
    pub plot_x_max: f64,
    /// Number of sample points for plotting.
    pub plot_samples: usize,
    /// Taylor series expansion order.
    pub taylor_order: usize,
    /// Taylor series expansion center.
    pub taylor_center: f64,
    /// History of previous expressions and results.
    pub history: Vec<(String, String)>,

    // -- Angle mode --
    /// Radians or Degrees for trigonometric functions.
    #[cfg(feature = "cas")]
    pub angle_mode: simucad_cas::evaluator::AngleMode,

    // -- Plot mode (polar / parametric) --
    /// Coordinate mode for 2D plotting.
    pub plot_mode: PlotMode,
    /// Second expression for parametric mode: y(t).
    pub parametric_y_expr: String,
    /// Theta / t minimum for polar and parametric modes.
    pub param_min: f64,
    /// Theta / t maximum for polar and parametric modes.
    pub param_max: f64,

    // -- 3D graphing --
    /// Whether the plot is in 3D mode: z = f(x, y).
    pub show_3d: bool,
    /// Second variable name for 3D plotting.
    pub y_variable: String,
    /// Y-axis minimum for 3D plotting.
    pub plot_y_min: f64,
    /// Y-axis maximum for 3D plotting.
    pub plot_y_max: f64,
    /// Grid resolution for 3D plotting.
    pub plot_3d_resolution: usize,
    /// Raw 3D point data from CAS.
    pub plot_3d_data: Vec<(f64, f64, f64)>,
    /// 3D viewport widget.
    pub viewport_3d: crate::viewport_3d::Viewport3D,

    // -- PDE wave / heat --
    /// PDE solver mode.
    pub pde_mode: PdeMode,
    /// Wave speed (c) or thermal diffusivity (alpha).
    pub pde_coeff: f64,
    /// PDE spatial domain minimum.
    pub pde_x_min: f64,
    /// PDE spatial domain maximum.
    pub pde_x_max: f64,
    /// Number of spatial grid points.
    pub pde_nx: usize,
    /// Final simulation time.
    pub pde_t_final: f64,
    /// Number of time steps.
    pub pde_nt: usize,
    /// Initial condition expression string.
    pub pde_initial_expr: String,
    /// Initial velocity expression (wave equation only).
    pub pde_velocity_expr: String,
    /// Left boundary value (heat equation only).
    pub pde_left_bc: f64,
    /// Right boundary value (heat equation only).
    pub pde_right_bc: f64,
    /// Computed PDE solution grid.
    #[cfg(feature = "cas")]
    pub pde_solution: Option<simucad_cas::pde::PdeGridSolution>,
    /// Current time step index for animation.
    pub pde_time_index: usize,
    /// Whether the PDE animation is playing.
    pub pde_playing: bool,

    // -- PDE visualization --
    /// PDE visualization mode (line plot, heatmap, contour).
    pub pde_viz_mode: PdeVizMode,
    /// Colormap for heatmap / contour rendering.
    pub pde_colormap: ColormapChoice,
    /// Texture handle for PDE heatmap (reused across frames).
    pub pde_heatmap_texture: Option<egui::TextureHandle>,
    /// Number of contour levels.
    pub pde_contour_levels: usize,
    /// Export directory path for animation frames.
    pub pde_export_path: String,
    /// Status message for export operations.
    pub pde_export_status: Option<String>,

    // -- Fourier analysis --
    /// Fourier analysis sub-mode.
    pub fourier_mode: FourierMode,
    /// Number of Fourier series terms.
    pub fourier_num_terms: usize,
    /// Period for Fourier series.
    pub fourier_period: f64,
    /// Fourier series coefficients (computed).
    #[cfg(feature = "cas")]
    pub fourier_coefficients: Option<simucad_cas::fourier::FourierCoefficients>,
    /// Fourier partial sum plot data.
    pub fourier_plot_data: Vec<(f64, f64)>,
    /// DFT sample rate.
    pub dft_sample_rate: f64,
    /// DFT number of samples.
    pub dft_num_samples: usize,
    /// DFT magnitude spectrum: (frequency, magnitude) pairs.
    pub dft_spectrum: Vec<(f64, f64)>,
}

impl Default for CalculatorPanel {
    fn default() -> Self {
        Self {
            expression_input: String::new(),
            diff_variable: "x".into(),
            result: None,
            latex_preview: None,
            plot_data: Vec::new(),
            plot_x_min: -10.0,
            plot_x_max: 10.0,
            plot_samples: 500,
            taylor_order: 5,
            taylor_center: 0.0,
            history: Vec::new(),

            #[cfg(feature = "cas")]
            angle_mode: simucad_cas::evaluator::AngleMode::Radians,

            plot_mode: PlotMode::Cartesian,
            parametric_y_expr: String::new(),
            param_min: 0.0,
            param_max: std::f64::consts::TAU,

            show_3d: false,
            y_variable: "y".into(),
            plot_y_min: -10.0,
            plot_y_max: 10.0,
            plot_3d_resolution: 50,
            plot_3d_data: Vec::new(),
            viewport_3d: crate::viewport_3d::Viewport3D::default(),

            pde_mode: PdeMode::Wave,
            pde_coeff: 1.0,
            pde_x_min: 0.0,
            pde_x_max: 1.0,
            pde_nx: 50,
            pde_t_final: 1.0,
            pde_nt: 500,
            pde_initial_expr: "sin(pi*x)".into(),
            pde_velocity_expr: "0".into(),
            pde_left_bc: 0.0,
            pde_right_bc: 0.0,
            #[cfg(feature = "cas")]
            pde_solution: None,
            pde_time_index: 0,
            pde_playing: false,

            pde_viz_mode: PdeVizMode::LinePlot,
            pde_colormap: ColormapChoice::Viridis,
            pde_heatmap_texture: None,
            pde_contour_levels: 10,
            pde_export_path: String::new(),
            pde_export_status: None,

            fourier_mode: FourierMode::Series,
            fourier_num_terms: 10,
            fourier_period: std::f64::consts::TAU,
            #[cfg(feature = "cas")]
            fourier_coefficients: None,
            fourier_plot_data: Vec::new(),
            dft_sample_rate: 1000.0,
            dft_num_samples: 1024,
            dft_spectrum: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

impl CalculatorPanel {
    /// Render the calculator panel.
    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("Scientific Calculator");
        ui.add_space(8.0);

        #[cfg(not(feature = "cas"))]
        {
            self.show_no_cas(ui);
        }

        #[cfg(feature = "cas")]
        {
            self.show_cas(ui);
        }
    }

    /// Shown when the CAS feature is not enabled.
    #[cfg(not(feature = "cas"))]
    fn show_no_cas(&mut self, ui: &mut Ui) {
        ui.label("The CAS feature is not enabled. Rebuild with --features cas to use the scientific calculator.");
        ui.add_space(8.0);

        // Provide basic numeric evaluation.
        ui.horizontal(|ui| {
            ui.label("Expression:");
            let response = ui.text_edit_singleline(&mut self.expression_input);
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.evaluate_builtin();
            }
        });

        ui.add_space(4.0);
        if ui.button("Evaluate (basic)").clicked() {
            self.evaluate_builtin();
        }

        if let Some(ref result) = self.result {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.strong("Result:");
                ui.label(result);
            });
        }
    }

    /// Minimal built-in evaluator for simple arithmetic when CAS is off.
    #[cfg(not(feature = "cas"))]
    fn evaluate_builtin(&mut self) {
        let expr = self.expression_input.trim();
        if expr.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }
        if let Ok(val) = expr.parse::<f64>() {
            let result_str = format!("{val}");
            self.history.push((expr.to_string(), result_str.clone()));
            self.result = Some(result_str);
            return;
        }
        self.result = Some("Enable 'cas' feature for full expression evaluation".into());
    }

    /// Full CAS-powered calculator UI.
    #[cfg(feature = "cas")]
    fn show_cas(&mut self, ui: &mut Ui) {
        // Angle mode toggle.
        ui.horizontal(|ui| {
            ui.label("Angle:");
            ui.selectable_value(
                &mut self.angle_mode,
                simucad_cas::evaluator::AngleMode::Radians,
                "Rad",
            );
            ui.selectable_value(
                &mut self.angle_mode,
                simucad_cas::evaluator::AngleMode::Degrees,
                "Deg",
            );
            ui.add_space(16.0);
            ui.checkbox(&mut self.show_3d, "3D mode");
        });

        ui.add_space(4.0);

        // Expression input.
        let expr_label = match (self.show_3d, self.plot_mode) {
            (true, _) => "z(x,y) =",
            (false, PlotMode::Polar) => "r(\u{03b8}) =",
            (false, PlotMode::Parametric) => "x(t) =",
            _ => "f(x) =",
        };
        ui.horizontal(|ui| {
            ui.label(expr_label);
            let response = ui.text_edit_singleline(&mut self.expression_input);
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.evaluate();
            }
        });

        // Parametric second expression.
        if !self.show_3d && self.plot_mode == PlotMode::Parametric {
            ui.horizontal(|ui| {
                ui.label("y(t) =");
                ui.text_edit_singleline(&mut self.parametric_y_expr);
            });
        }

        ui.add_space(6.0);

        // Variable selector and plot mode.
        ui.horizontal(|ui| {
            ui.label("Variable:");
            ui.text_edit_singleline(&mut self.diff_variable);
            if self.show_3d {
                ui.label("Y var:");
                ui.text_edit_singleline(&mut self.y_variable);
            }
        });

        if !self.show_3d {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Plot mode:");
                ui.selectable_value(&mut self.plot_mode, PlotMode::Cartesian, "Cartesian");
                ui.selectable_value(&mut self.plot_mode, PlotMode::Polar, "Polar");
                ui.selectable_value(&mut self.plot_mode, PlotMode::Parametric, "Parametric");
            });
        }

        ui.add_space(4.0);

        // Action buttons.
        ui.horizontal(|ui| {
            if ui.button("Evaluate").clicked() {
                self.evaluate();
            }
            if ui.button("Differentiate").clicked() {
                self.differentiate();
            }
            if ui.button("Integrate").clicked() {
                self.integrate();
            }
            if ui.button("Find Roots").clicked() {
                self.find_roots();
            }
            if ui.button("Taylor").clicked() {
                self.taylor_expand();
            }
            if ui.button("Solve").clicked() {
                self.solve_equation();
            }
            if ui.button("Plot").clicked() {
                if self.show_3d {
                    self.generate_3d_plot();
                } else {
                    self.generate_plot();
                }
            }
            if ui.button("Clear").clicked() {
                self.expression_input.clear();
                self.result = None;
                self.latex_preview = None;
                self.plot_data.clear();
                self.plot_3d_data.clear();
            }
        });

        // Plot range controls.
        ui.add_space(4.0);
        if self.show_3d {
            ui.horizontal(|ui| {
                ui.label("x:");
                ui.add(egui::DragValue::new(&mut self.plot_x_min).speed(0.5).prefix("min: "));
                ui.add(egui::DragValue::new(&mut self.plot_x_max).speed(0.5).prefix("max: "));
                ui.label("y:");
                ui.add(egui::DragValue::new(&mut self.plot_y_min).speed(0.5).prefix("min: "));
                ui.add(egui::DragValue::new(&mut self.plot_y_max).speed(0.5).prefix("max: "));
                ui.label("res:");
                ui.add(egui::DragValue::new(&mut self.plot_3d_resolution).speed(1.0).range(5..=200));
            });
        } else if self.plot_mode == PlotMode::Cartesian {
            ui.horizontal(|ui| {
                ui.label("x range:");
                ui.add(egui::DragValue::new(&mut self.plot_x_min).speed(0.5).prefix("min: "));
                ui.add(egui::DragValue::new(&mut self.plot_x_max).speed(0.5).prefix("max: "));
                ui.label("samples:");
                ui.add(egui::DragValue::new(&mut self.plot_samples).speed(1.0).range(10..=10_000));
            });
        } else {
            // Polar or parametric range.
            let label = if self.plot_mode == PlotMode::Polar { "\u{03b8} range:" } else { "t range:" };
            ui.horizontal(|ui| {
                ui.label(label);
                ui.add(egui::DragValue::new(&mut self.param_min).speed(0.1).prefix("min: "));
                ui.add(egui::DragValue::new(&mut self.param_max).speed(0.1).prefix("max: "));
                ui.label("samples:");
                ui.add(egui::DragValue::new(&mut self.plot_samples).speed(1.0).range(10..=10_000));
            });
        }

        // Taylor series controls.
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Taylor:");
            ui.add(egui::DragValue::new(&mut self.taylor_order).speed(0.1).prefix("order: ").range(0..=20));
            ui.add(egui::DragValue::new(&mut self.taylor_center).speed(0.1).prefix("center: "));
        });

        ui.add_space(8.0);

        // Display result.
        if let Some(ref result) = self.result {
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Result:");
                ui.label(result);
            });
        }

        // LaTeX preview.
        if let Some(ref latex) = self.latex_preview {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.strong("LaTeX:");
                ui.monospace(latex);
            });
        }

        // Plot display — 2D or 3D.
        if self.show_3d && !self.plot_3d_data.is_empty() {
            ui.add_space(8.0);
            let (particles, edges) = build_surface_mesh(&self.plot_3d_data, self.plot_3d_resolution);
            self.viewport_3d.show(ui, Some(&particles), Some(&edges));
        } else if !self.plot_data.is_empty() {
            ui.add_space(8.0);
            let equal_aspect = self.plot_mode == PlotMode::Polar;
            plotting::plot_function_2d_ex(ui, &self.plot_data, &self.expression_input, equal_aspect);
        }

        // PDE wave / heat section.
        ui.add_space(8.0);
        self.show_pde_section(ui);

        // Fourier analysis section.
        ui.add_space(8.0);
        self.show_fourier_section(ui);

        // History.
        if !self.history.is_empty() {
            ui.add_space(12.0);
            ui.collapsing("History", |ui| {
                for (expr, result) in self.history.iter().rev() {
                    ui.horizontal(|ui| {
                        ui.monospace(expr);
                        ui.label("=");
                        ui.monospace(result);
                    });
                }
            });
        }
    }

    // -----------------------------------------------------------------------
    // CAS operations
    // -----------------------------------------------------------------------

    /// Evaluate the current expression numerically.
    #[cfg(feature = "cas")]
    fn evaluate(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                // Set LaTeX preview.
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                let env = simucad_cas::evaluator::Environment::new();
                match simucad_cas::evaluator::evaluate_with_angle_mode(
                    &ast,
                    &env,
                    self.angle_mode,
                ) {
                    Ok(val) => {
                        let result_str = format!("{val}");
                        self.history
                            .push((expr_str.to_string(), result_str.clone()));
                        self.result = Some(result_str);
                    }
                    Err(e) => {
                        self.result = Some(format!("Eval error: {e}"));
                    }
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Symbolically differentiate the expression.
    #[cfg(feature = "cas")]
    fn differentiate(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                match simucad_cas::derivative::differentiate(&ast, &self.diff_variable) {
                    Ok(derivative) => {
                        let simplified = simucad_cas::simplify::simplify(&derivative);
                        let result_str = format!("{simplified}");
                        let latex = simucad_cas::latex::to_latex(&simplified);
                        self.latex_preview = Some(latex);
                        self.history.push((
                            format!("d/d{} {}", self.diff_variable, expr_str),
                            result_str.clone(),
                        ));
                        self.result = Some(result_str);
                    }
                    Err(e) => {
                        self.result = Some(format!("Differentiation error: {e}"));
                    }
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Symbolically integrate the expression.
    #[cfg(feature = "cas")]
    fn integrate(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                match simucad_cas::integration::integrate(&ast, &self.diff_variable) {
                    Ok(integral) => {
                        let simplified = simucad_cas::simplify::simplify(&integral);
                        let result_str = format!("{simplified}");
                        let latex = simucad_cas::latex::to_latex(&simplified);
                        self.latex_preview = Some(latex);
                        self.history.push((
                            format!("int {} d{}", expr_str, self.diff_variable),
                            result_str.clone(),
                        ));
                        self.result = Some(result_str);
                    }
                    Err(e) => {
                        self.result = Some(format!("Integration error: {e}"));
                    }
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Find roots of the expression using Newton-Raphson.
    #[cfg(feature = "cas")]
    fn find_roots(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                // Try several initial guesses to find different roots.
                let guesses = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
                let tolerance = 1e-10;
                let max_iter = 200;
                let mut roots: Vec<f64> = Vec::new();

                for guess in guesses {
                    if let Ok(root) = simucad_cas::solver::solve_numeric(
                        &ast,
                        &self.diff_variable,
                        guess,
                        tolerance,
                        max_iter,
                    ) {
                        // Only add if not a duplicate (within tolerance).
                        let is_dup = roots.iter().any(|r| (r - root).abs() < 1e-6);
                        if !is_dup {
                            roots.push(root);
                        }
                    }
                }

                if roots.is_empty() {
                    self.result = Some("No roots found in the search range".into());
                } else {
                    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let result_str = roots
                        .iter()
                        .map(|r| format!("{:.6}", r))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.history.push((
                        format!("roots of {}", expr_str),
                        result_str.clone(),
                    ));
                    self.result = Some(format!("Roots: {result_str}"));
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Compute the Taylor series expansion of the expression.
    #[cfg(feature = "cas")]
    fn taylor_expand(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                match simucad_cas::taylor::taylor_expand(
                    &ast,
                    &self.diff_variable,
                    self.taylor_center,
                    self.taylor_order,
                ) {
                    Ok(expansion) => {
                        let simplified = simucad_cas::simplify::simplify(&expansion);
                        let result_str = format!("{simplified}");
                        let latex = simucad_cas::latex::to_latex(&simplified);
                        self.latex_preview = Some(latex);
                        self.history.push((
                            format!(
                                "taylor({}, {}, order={})",
                                expr_str, self.taylor_center, self.taylor_order
                            ),
                            result_str.clone(),
                        ));
                        self.result = Some(result_str);
                    }
                    Err(e) => {
                        self.result = Some(format!("Taylor error: {e}"));
                    }
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Solve the expression = 0 for the variable.
    #[cfg(feature = "cas")]
    fn solve_equation(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                // Try quadratic first (handles linear too).
                match simucad_cas::equations::solve_quadratic(&ast, &self.diff_variable) {
                    Ok(roots) => {
                        let result_str = roots
                            .iter()
                            .map(|r| format!("{r}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        self.history.push((
                            format!("solve {} = 0", expr_str),
                            result_str.clone(),
                        ));
                        self.result = Some(format!("Solutions: {result_str}"));
                    }
                    Err(_) => {
                        // Fall back to numeric root finding.
                        self.find_roots();
                    }
                }
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }

    /// Generate 2D plot data for the current expression.
    #[cfg(feature = "cas")]
    fn generate_plot(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            return;
        }

        match self.plot_mode {
            PlotMode::Cartesian => {
                match simucad_cas::parser::parse(expr_str) {
                    Ok(ast) => {
                        self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));
                        let mut points = Vec::with_capacity(self.plot_samples);
                        let dx = (self.plot_x_max - self.plot_x_min)
                            / (self.plot_samples - 1).max(1) as f64;

                        for i in 0..self.plot_samples {
                            let x = self.plot_x_min + i as f64 * dx;
                            let mut env = simucad_cas::evaluator::Environment::new();
                            env.set(&self.diff_variable, x);

                            if let Ok(y) = simucad_cas::evaluator::evaluate_with_angle_mode(
                                &ast,
                                &env,
                                self.angle_mode,
                            ) {
                                if y.is_finite() {
                                    points.push((x, y));
                                }
                            }
                        }
                        self.plot_data = points;
                        self.result = Some(format!("Plotted {} points", self.plot_data.len()));
                    }
                    Err(e) => self.result = Some(format!("Parse error: {e}")),
                }
            }
            PlotMode::Polar => {
                match simucad_cas::parser::parse(expr_str) {
                    Ok(ast) => {
                        self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));
                        match simucad_cas::plotter::generate_polar_points(
                            &ast,
                            &self.diff_variable,
                            self.param_min,
                            self.param_max,
                            self.plot_samples,
                        ) {
                            Ok(pts) => {
                                self.plot_data = pts;
                                self.result =
                                    Some(format!("Plotted {} polar points", self.plot_data.len()));
                            }
                            Err(e) => self.result = Some(format!("Plot error: {e}")),
                        }
                    }
                    Err(e) => self.result = Some(format!("Parse error: {e}")),
                }
            }
            PlotMode::Parametric => {
                let y_str = self.parametric_y_expr.trim();
                if y_str.is_empty() {
                    self.result = Some("Parametric mode requires a y(t) expression".into());
                    return;
                }
                match (
                    simucad_cas::parser::parse(expr_str),
                    simucad_cas::parser::parse(y_str),
                ) {
                    (Ok(x_ast), Ok(y_ast)) => {
                        match simucad_cas::plotter::generate_parametric_points(
                            &x_ast,
                            &y_ast,
                            &self.diff_variable,
                            self.param_min,
                            self.param_max,
                            self.plot_samples,
                        ) {
                            Ok(pts) => {
                                self.plot_data = pts;
                                self.result = Some(format!(
                                    "Plotted {} parametric points",
                                    self.plot_data.len()
                                ));
                            }
                            Err(e) => self.result = Some(format!("Plot error: {e}")),
                        }
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        self.result = Some(format!("Parse error: {e}"));
                    }
                }
            }
        }
    }

    /// Generate 3D surface plot data.
    #[cfg(feature = "cas")]
    fn generate_3d_plot(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));
                match simucad_cas::plotter::generate_3d_points(
                    &ast,
                    &self.diff_variable,
                    &self.y_variable,
                    (self.plot_x_min, self.plot_x_max),
                    (self.plot_y_min, self.plot_y_max),
                    self.plot_3d_resolution,
                ) {
                    Ok(pts) => {
                        self.result =
                            Some(format!("Plotted {} 3D points", pts.len()));
                        self.plot_3d_data = pts;
                    }
                    Err(e) => self.result = Some(format!("3D plot error: {e}")),
                }
            }
            Err(e) => self.result = Some(format!("Parse error: {e}")),
        }
    }

    // -----------------------------------------------------------------------
    // PDE wave / heat section
    // -----------------------------------------------------------------------

    #[cfg(feature = "cas")]
    fn show_pde_section(&mut self, ui: &mut Ui) {
        ui.collapsing("Wave / Heat Equation", |ui| {
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.pde_mode, PdeMode::Wave, "Wave");
                ui.selectable_value(&mut self.pde_mode, PdeMode::Heat, "Heat");
            });

            let coeff_label = match self.pde_mode {
                PdeMode::Wave => "Wave speed (c):",
                PdeMode::Heat => "Diffusivity (\u{03b1}):",
            };

            ui.horizontal(|ui| {
                ui.label(coeff_label);
                ui.add(egui::DragValue::new(&mut self.pde_coeff).speed(0.1).range(0.01..=100.0));
            });

            ui.horizontal(|ui| {
                ui.label("x range:");
                ui.add(egui::DragValue::new(&mut self.pde_x_min).speed(0.1).prefix("min: "));
                ui.add(egui::DragValue::new(&mut self.pde_x_max).speed(0.1).prefix("max: "));
                ui.label("nx:");
                ui.add(egui::DragValue::new(&mut self.pde_nx).speed(1.0).range(5..=500));
            });

            ui.horizontal(|ui| {
                ui.label("t_final:");
                ui.add(egui::DragValue::new(&mut self.pde_t_final).speed(0.1).range(0.01..=100.0));
                ui.label("nt:");
                ui.add(egui::DragValue::new(&mut self.pde_nt).speed(1.0).range(10..=50_000));
            });

            ui.horizontal(|ui| {
                ui.label("Initial u(x,0) =");
                ui.text_edit_singleline(&mut self.pde_initial_expr);
            });

            match self.pde_mode {
                PdeMode::Wave => {
                    ui.horizontal(|ui| {
                        ui.label("Initial velocity =");
                        ui.text_edit_singleline(&mut self.pde_velocity_expr);
                    });
                }
                PdeMode::Heat => {
                    ui.horizontal(|ui| {
                        ui.label("Left BC:");
                        ui.add(egui::DragValue::new(&mut self.pde_left_bc).speed(0.1));
                        ui.label("Right BC:");
                        ui.add(egui::DragValue::new(&mut self.pde_right_bc).speed(0.1));
                    });
                }
            }

            if ui.button("Solve PDE").clicked() {
                self.solve_pde();
            }

            // Solution display with animation.
            if let Some(ref solution) = self.pde_solution {
                let max_t = solution.data.len().saturating_sub(1);

                // Visualization mode selector.
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("View:");
                    ui.selectable_value(&mut self.pde_viz_mode, PdeVizMode::LinePlot, "Line Plot");
                    ui.selectable_value(&mut self.pde_viz_mode, PdeVizMode::Heatmap, "Heatmap");
                    ui.selectable_value(&mut self.pde_viz_mode, PdeVizMode::Contour, "Contour");
                });

                // Colormap selector (for heatmap / contour).
                if self.pde_viz_mode != PdeVizMode::LinePlot {
                    ui.horizontal(|ui| {
                        ui.label("Colormap:");
                        ui.selectable_value(&mut self.pde_colormap, ColormapChoice::Viridis, "Viridis");
                        ui.selectable_value(&mut self.pde_colormap, ColormapChoice::Inferno, "Inferno");
                        ui.selectable_value(&mut self.pde_colormap, ColormapChoice::CoolWarm, "Cool-Warm");
                    });
                }

                // Contour levels control.
                if self.pde_viz_mode == PdeVizMode::Contour {
                    ui.horizontal(|ui| {
                        ui.label("Contour levels:");
                        ui.add(egui::DragValue::new(&mut self.pde_contour_levels).speed(1.0).range(3..=50));
                    });
                }

                // Time-step controls (line plot mode only).
                if self.pde_viz_mode == PdeVizMode::LinePlot {
                    ui.horizontal(|ui| {
                        if ui.button(if self.pde_playing { "Pause" } else { "Play" }).clicked() {
                            self.pde_playing = !self.pde_playing;
                        }
                        ui.add(egui::Slider::new(&mut self.pde_time_index, 0..=max_t).text("time step"));
                    });

                    if max_t > 0 {
                        let t_val = solution.t_grid[self.pde_time_index.min(max_t)];
                        ui.label(format!("t = {t_val:.4}"));
                    }
                }

                // Render the selected visualization.
                match self.pde_viz_mode {
                    PdeVizMode::LinePlot => {
                        let idx = self.pde_time_index.min(max_t);
                        let profile: Vec<(f64, f64)> = solution.x_grid.iter().copied()
                            .zip(solution.data[idx].iter().copied())
                            .collect();
                        plotting::plot_function_2d(ui, &profile, "u(x, t)");
                    }
                    PdeVizMode::Heatmap => {
                        // Build space-time grid: rows = time steps, cols = spatial points.
                        let x_range = (
                            *solution.x_grid.first().unwrap_or(&0.0),
                            *solution.x_grid.last().unwrap_or(&1.0),
                        );
                        let t_range = (
                            *solution.t_grid.first().unwrap_or(&0.0),
                            *solution.t_grid.last().unwrap_or(&1.0),
                        );
                        plotting::plot_heatmap(
                            ui,
                            &solution.data,
                            x_range,
                            t_range,
                            "x",
                            "t",
                            "u(x, t) — Space-Time Heatmap",
                            &mut self.pde_heatmap_texture,
                            self.pde_colormap.function(),
                        );
                    }
                    PdeVizMode::Contour => {
                        let x_range = (
                            *solution.x_grid.first().unwrap_or(&0.0),
                            *solution.x_grid.last().unwrap_or(&1.0),
                        );
                        let t_range = (
                            *solution.t_grid.first().unwrap_or(&0.0),
                            *solution.t_grid.last().unwrap_or(&1.0),
                        );
                        plotting::plot_contours(
                            ui,
                            &solution.data,
                            x_range,
                            t_range,
                            self.pde_contour_levels,
                            "u(x, t) — Contour Plot",
                        );
                    }
                }

                // Animation line plot advance.
                if self.pde_viz_mode == PdeVizMode::LinePlot && self.pde_playing {
                    if self.pde_time_index < max_t {
                        self.pde_time_index += 1;
                    } else {
                        self.pde_playing = false;
                    }
                    ui.ctx().request_repaint();
                }

                // Export controls.
                ui.add_space(4.0);
                ui.separator();
                ui.label("Animation Export");
                ui.horizontal(|ui| {
                    ui.label("Output dir:");
                    ui.text_edit_singleline(&mut self.pde_export_path);
                });

                ui.horizontal(|ui| {
                    if ui.button("Export PNG Frames").clicked() {
                        self.export_pde_png_frames();
                    }
                    if ui.button("Export GIF").clicked() {
                        self.export_pde_gif();
                    }
                    if ui.button("Export Heatmap PNG").clicked() {
                        self.export_pde_heatmap_png();
                    }
                });

                if let Some(ref status) = self.pde_export_status {
                    ui.label(status.as_str());
                }
            }
        });
    }

    #[cfg(feature = "cas")]
    fn export_pde_png_frames(&mut self) {
        let Some(ref solution) = self.pde_solution else { return };
        let dir = if self.pde_export_path.is_empty() {
            std::env::temp_dir().join("simucad_pde_export")
        } else {
            PathBuf::from(&self.pde_export_path)
        };

        match animation::export_pde_frames_png(
            &solution.data,
            &dir,
            "pde",
            20,
            self.pde_colormap.function(),
        ) {
            Ok(n) => {
                self.pde_export_status = Some(format!("Exported {n} PNG frames to {}", dir.display()));
            }
            Err(e) => {
                self.pde_export_status = Some(format!("PNG export error: {e}"));
            }
        }
    }

    #[cfg(feature = "cas")]
    fn export_pde_gif(&mut self) {
        let Some(ref solution) = self.pde_solution else { return };
        let path = if self.pde_export_path.is_empty() {
            std::env::temp_dir().join("simucad_pde.gif")
        } else {
            PathBuf::from(&self.pde_export_path).join("pde_animation.gif")
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match animation::export_pde_gif(
            &solution.data,
            &path,
            20,
            5,
            self.pde_colormap.function(),
        ) {
            Ok(n) => {
                self.pde_export_status = Some(format!("Exported {n}-frame GIF to {}", path.display()));
            }
            Err(e) => {
                self.pde_export_status = Some(format!("GIF export error: {e}"));
            }
        }
    }

    #[cfg(feature = "cas")]
    fn export_pde_heatmap_png(&mut self) {
        let Some(ref solution) = self.pde_solution else { return };
        let path = if self.pde_export_path.is_empty() {
            std::env::temp_dir().join("simucad_pde_heatmap.png")
        } else {
            PathBuf::from(&self.pde_export_path).join("pde_heatmap.png")
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match animation::export_heatmap_png(
            &solution.data,
            &path,
            self.pde_colormap.function(),
        ) {
            Ok(()) => {
                self.pde_export_status = Some(format!("Exported heatmap to {}", path.display()));
            }
            Err(e) => {
                self.pde_export_status = Some(format!("Heatmap export error: {e}"));
            }
        }
    }

    #[cfg(feature = "cas")]
    fn solve_pde(&mut self) {
        let initial_str = self.pde_initial_expr.trim();
        if initial_str.is_empty() {
            self.result = Some("Initial condition expression required".into());
            return;
        }

        let initial_ast = match simucad_cas::parser::parse(initial_str) {
            Ok(a) => a,
            Err(e) => {
                self.result = Some(format!("Parse error (initial): {e}"));
                return;
            }
        };

        match self.pde_mode {
            PdeMode::Wave => {
                let vel_str = self.pde_velocity_expr.trim();
                let vel_ast = match simucad_cas::parser::parse(if vel_str.is_empty() { "0" } else { vel_str }) {
                    Ok(a) => a,
                    Err(e) => {
                        self.result = Some(format!("Parse error (velocity): {e}"));
                        return;
                    }
                };

                let config = simucad_cas::pde::WaveConfig {
                    c: self.pde_coeff,
                    x_range: (self.pde_x_min, self.pde_x_max),
                    nx: self.pde_nx,
                    t_final: self.pde_t_final,
                    nt: self.pde_nt,
                };

                match simucad_cas::pde::solve_wave_equation(
                    &config,
                    &initial_ast,
                    &vel_ast,
                ) {
                    Ok(sol) => {
                        self.result = Some(format!(
                            "Wave solved: {} time steps, {} spatial points",
                            sol.data.len(),
                            sol.x_grid.len()
                        ));
                        self.pde_solution = Some(sol);
                        self.pde_time_index = 0;
                    }
                    Err(e) => self.result = Some(format!("PDE error: {e}")),
                }
            }
            PdeMode::Heat => {
                let config = simucad_cas::pde::HeatConfig {
                    alpha: self.pde_coeff,
                    x_range: (self.pde_x_min, self.pde_x_max),
                    nx: self.pde_nx,
                    t_final: self.pde_t_final,
                    nt: self.pde_nt,
                };

                match simucad_cas::pde::solve_heat_equation(
                    &config,
                    &initial_ast,
                    self.pde_left_bc,
                    self.pde_right_bc,
                ) {
                    Ok(sol) => {
                        self.result = Some(format!(
                            "Heat solved: {} time steps, {} spatial points",
                            sol.data.len(),
                            sol.x_grid.len()
                        ));
                        self.pde_solution = Some(sol);
                        self.pde_time_index = 0;
                    }
                    Err(e) => self.result = Some(format!("PDE error: {e}")),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fourier analysis section
    // -----------------------------------------------------------------------

    #[cfg(feature = "cas")]
    fn show_fourier_section(&mut self, ui: &mut Ui) {
        ui.collapsing("Fourier Analysis", |ui| {
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.selectable_value(&mut self.fourier_mode, FourierMode::Series, "Series");
                ui.selectable_value(&mut self.fourier_mode, FourierMode::Dft, "DFT");
            });

            match self.fourier_mode {
                FourierMode::Series => {
                    ui.horizontal(|ui| {
                        ui.label("Terms:");
                        ui.add(egui::DragValue::new(&mut self.fourier_num_terms).speed(1.0).range(1..=100));
                        ui.label("Period:");
                        ui.add(egui::DragValue::new(&mut self.fourier_period).speed(0.1).range(0.01..=1000.0));
                    });

                    if ui.button("Compute Fourier Series").clicked() {
                        self.compute_fourier_series();
                    }

                    // Display coefficients.
                    if let Some(ref coeffs) = self.fourier_coefficients {
                        ui.add_space(4.0);
                        ui.label(format!("a\u{2080}/2 = {:.6}", coeffs.a0 / 2.0));
                        egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                            for n in 0..coeffs.a.len() {
                                ui.monospace(format!(
                                    "a{} = {:.6}  b{} = {:.6}",
                                    n + 1,
                                    coeffs.a[n],
                                    n + 1,
                                    coeffs.b[n]
                                ));
                            }
                        });
                    }

                    // Plot overlay: original + partial sum.
                    if !self.fourier_plot_data.is_empty() && !self.plot_data.is_empty() {
                        ui.add_space(4.0);
                        plotting::plot_two_functions(
                            ui,
                            &self.plot_data,
                            "Original",
                            &self.fourier_plot_data,
                            "Fourier Approx",
                        );
                    } else if !self.fourier_plot_data.is_empty() {
                        plotting::plot_function_2d(ui, &self.fourier_plot_data, "Fourier Approx");
                    }
                }
                FourierMode::Dft => {
                    ui.horizontal(|ui| {
                        ui.label("Sample rate:");
                        ui.add(egui::DragValue::new(&mut self.dft_sample_rate).speed(10.0).range(1.0..=100_000.0));
                        ui.label("Samples:");
                        ui.add(egui::DragValue::new(&mut self.dft_num_samples).speed(1.0).range(8..=65536));
                    });

                    if ui.button("Compute DFT").clicked() {
                        self.compute_dft();
                    }

                    if !self.dft_spectrum.is_empty() {
                        ui.add_space(4.0);
                        plotting::plot_function_2d_ex(ui, &self.dft_spectrum, "Magnitude Spectrum", false);
                    }
                }
            }
        });
    }

    #[cfg(feature = "cas")]
    fn compute_fourier_series(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("Expression required for Fourier series".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                match simucad_cas::fourier::compute_fourier_coefficients(
                    &ast,
                    &self.diff_variable,
                    self.fourier_period,
                    self.fourier_num_terms,
                    1024,
                ) {
                    Ok(coeffs) => {
                        // Build the partial sum and generate plot data.
                        let partial = simucad_cas::fourier::fourier_partial_sum(
                            &coeffs,
                            self.fourier_num_terms,
                            self.fourier_period,
                            &self.diff_variable,
                        );

                        // Generate plot points for the partial sum.
                        let mut pts = Vec::with_capacity(self.plot_samples);
                        let x_min = 0.0;
                        let x_max = self.fourier_period;
                        let dx = (x_max - x_min) / (self.plot_samples - 1).max(1) as f64;
                        for i in 0..self.plot_samples {
                            let x = x_min + i as f64 * dx;
                            let mut env = simucad_cas::evaluator::Environment::new();
                            env.set(&self.diff_variable, x);
                            if let Ok(y) = simucad_cas::evaluator::evaluate(&partial, &env) {
                                if y.is_finite() {
                                    pts.push((x, y));
                                }
                            }
                        }
                        self.fourier_plot_data = pts;

                        // Also generate original function plot for overlay.
                        let mut orig_pts = Vec::with_capacity(self.plot_samples);
                        for i in 0..self.plot_samples {
                            let x = x_min + i as f64 * dx;
                            let mut env = simucad_cas::evaluator::Environment::new();
                            env.set(&self.diff_variable, x);
                            if let Ok(y) = simucad_cas::evaluator::evaluate(&ast, &env) {
                                if y.is_finite() {
                                    orig_pts.push((x, y));
                                }
                            }
                        }
                        self.plot_data = orig_pts;

                        self.fourier_coefficients = Some(coeffs);
                        self.result = Some(format!(
                            "Fourier series computed with {} terms",
                            self.fourier_num_terms
                        ));
                    }
                    Err(e) => self.result = Some(format!("Fourier error: {e}")),
                }
            }
            Err(e) => self.result = Some(format!("Parse error: {e}")),
        }
    }

    #[cfg(feature = "cas")]
    fn compute_dft(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("Expression required for DFT".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                // Sample the expression at evenly spaced points.
                let n = self.dft_num_samples;
                let dt = 1.0 / self.dft_sample_rate;
                let mut samples = Vec::with_capacity(n);

                for i in 0..n {
                    let t = i as f64 * dt;
                    let mut env = simucad_cas::evaluator::Environment::new();
                    env.set(&self.diff_variable, t);
                    let val = simucad_cas::evaluator::evaluate(&ast, &env).unwrap_or(0.0);
                    samples.push(if val.is_finite() { val } else { 0.0 });
                }

                // Compute DFT using the CAS fourier module.
                let spectrum = simucad_cas::fourier::compute_dft(&samples, self.dft_sample_rate);
                self.dft_spectrum = spectrum;
                self.result = Some(format!("DFT computed: {} frequency bins", self.dft_spectrum.len()));
            }
            Err(e) => self.result = Some(format!("Parse error: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// 3D surface mesh builder
// ---------------------------------------------------------------------------

/// Particle + edge tuple type for surface mesh rendering.
#[cfg(feature = "cas")]
type SurfaceMesh = (Vec<(f64, f64, f64, f64)>, Vec<([f64; 3], [f64; 3])>);

/// Convert a flat list of 3D points (in row-major grid order) into particles
/// and wireframe edges suitable for `Viewport3D::show()`.
#[cfg(feature = "cas")]
fn build_surface_mesh(
    points: &[(f64, f64, f64)],
    resolution: usize,
) -> SurfaceMesh {
    // Build an Option-based grid to handle missing points.
    let mut grid: Vec<Vec<Option<(f64, f64, f64)>>> = vec![vec![None; resolution]; resolution];

    // Points come in row-major order: for each x_i, iterate over y_j.
    let mut idx = 0;
    for row in grid.iter_mut().take(resolution) {
        for cell in row.iter_mut().take(resolution) {
            if idx < points.len() {
                *cell = Some(points[idx]);
                idx += 1;
            }
        }
    }

    // Find z range for color mapping.
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for p in points {
        if p.2 < z_min { z_min = p.2; }
        if p.2 > z_max { z_max = p.2; }
    }
    let z_range = if (z_max - z_min).abs() < 1e-12 { 1.0 } else { z_max - z_min };

    // Build particles with z-based "speed" for color mapping.
    // Note: y and z swapped for viewport (y is up in the 3D view).
    let particles: Vec<(f64, f64, f64, f64)> = points
        .iter()
        .map(|&(x, y, z)| (x, z, y, (z - z_min) / z_range))
        .collect();

    // Build wireframe edges.
    let mut edges: Vec<([f64; 3], [f64; 3])> = Vec::new();
    for i in 0..resolution {
        for j in 0..resolution {
            if let Some((x1, y1, z1)) = grid[i][j] {
                // Connect to right neighbour.
                if let Some(Some((x2, y2, z2))) = (i + 1 < resolution).then(|| grid[i + 1][j]) {
                    edges.push(([x1, z1, y1], [x2, z2, y2]));
                }
                // Connect to bottom neighbour.
                if let Some(Some((x2, y2, z2))) = (j + 1 < resolution).then(|| grid[i][j + 1]) {
                    edges.push(([x1, z1, y1], [x2, z2, y2]));
                }
            }
        }
    }

    (particles, edges)
}
