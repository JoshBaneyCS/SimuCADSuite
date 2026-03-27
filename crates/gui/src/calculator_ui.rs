//! Scientific calculator UI panel.
//!
//! Provides a text input for mathematical expressions with buttons for
//! evaluation, symbolic differentiation, integration, root-finding, and
//! function plotting. CAS-dependent features are gated behind
//! `#[cfg(feature = "cas")]`.

use egui::Ui;

use crate::plotting;

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

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
        // Expression input.
        ui.horizontal(|ui| {
            ui.label("f(x) =");
            let response = ui.text_edit_singleline(&mut self.expression_input);
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.evaluate();
            }
        });

        ui.add_space(6.0);

        // Variable selector.
        ui.horizontal(|ui| {
            ui.label("Variable:");
            ui.text_edit_singleline(&mut self.diff_variable);
        });

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
                self.generate_plot();
            }
            if ui.button("Clear").clicked() {
                self.expression_input.clear();
                self.result = None;
                self.latex_preview = None;
                self.plot_data.clear();
            }
        });

        // Plot range controls.
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("x range:");
            ui.add(egui::DragValue::new(&mut self.plot_x_min).speed(0.5).prefix("min: "));
            ui.add(egui::DragValue::new(&mut self.plot_x_max).speed(0.5).prefix("max: "));
            ui.label("samples:");
            ui.add(egui::DragValue::new(&mut self.plot_samples).speed(1.0).range(10..=10_000));
        });

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

        // Plot display.
        if !self.plot_data.is_empty() {
            ui.add_space(8.0);
            plotting::plot_function_2d(ui, &self.plot_data, &self.expression_input);
        }

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
                match simucad_cas::evaluator::evaluate(&ast, &env) {
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

    /// Generate plot data for the current expression.
    #[cfg(feature = "cas")]
    fn generate_plot(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                self.latex_preview = Some(simucad_cas::latex::to_latex(&ast));

                let mut points = Vec::with_capacity(self.plot_samples);
                let dx =
                    (self.plot_x_max - self.plot_x_min) / (self.plot_samples - 1).max(1) as f64;

                for i in 0..self.plot_samples {
                    let x = self.plot_x_min + i as f64 * dx;
                    let mut env = simucad_cas::evaluator::Environment::new();
                    env.set(&self.diff_variable, x);

                    if let Ok(y) = simucad_cas::evaluator::evaluate(&ast, &env) {
                        if y.is_finite() {
                            points.push((x, y));
                        }
                    }
                }

                self.plot_data = points;
                self.result = Some(format!("Plotted {} points", self.plot_data.len()));
            }
            Err(e) => {
                self.result = Some(format!("Parse error: {e}"));
            }
        }
    }
}
