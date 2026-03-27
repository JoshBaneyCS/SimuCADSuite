//! Scientific calculator UI panel.
//!
//! Provides a text input for mathematical expressions with buttons for
//! evaluation, symbolic differentiation, and function plotting. CAS-dependent
//! features are gated behind `#[cfg(feature = "cas")]`.

use egui::Ui;

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the scientific calculator panel.
pub struct CalculatorPanel {
    /// The current expression string typed by the user.
    pub expression_input: String,
    /// The variable to differentiate with respect to.
    pub diff_variable: String,
    /// Result of the last operation (evaluation, differentiation, etc.).
    pub result: Option<String>,
    /// Plot data for the most recently plotted function: `(x, y)` pairs.
    pub plot_data: Vec<(f64, f64)>,
    /// X-axis range for plotting.
    pub plot_x_min: f64,
    /// X-axis range for plotting.
    pub plot_x_max: f64,
    /// Number of sample points for plotting.
    pub plot_samples: usize,
    /// History of previous expressions and results.
    pub history: Vec<(String, String)>,
}

impl Default for CalculatorPanel {
    fn default() -> Self {
        Self {
            expression_input: String::new(),
            diff_variable: "x".into(),
            result: None,
            plot_data: Vec::new(),
            plot_x_min: -10.0,
            plot_x_max: 10.0,
            plot_samples: 500,
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

        // Expression input.
        ui.horizontal(|ui| {
            ui.label("f(x) =");
            let response = ui.text_edit_singleline(&mut self.expression_input);
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.evaluate();
            }
        });

        ui.add_space(6.0);

        // Action buttons.
        ui.horizontal(|ui| {
            if ui.button("Evaluate").clicked() {
                self.evaluate();
            }

            #[cfg(feature = "cas")]
            {
                ui.separator();
                if ui.button("Differentiate").clicked() {
                    self.differentiate();
                }
                ui.label("w.r.t.");
                ui.text_edit_singleline(&mut self.diff_variable);
            }

            #[cfg(not(feature = "cas"))]
            {
                ui.separator();
                ui.add_enabled(false, egui::Button::new("Differentiate"))
                    .on_disabled_hover_text("Enable the 'cas' feature to use symbolic differentiation");
            }

            ui.separator();
            if ui.button("Plot").clicked() {
                self.generate_plot();
            }

            if ui.button("Clear").clicked() {
                self.expression_input.clear();
                self.result = None;
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

        ui.add_space(8.0);

        // Display result.
        if let Some(ref result) = self.result {
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("Result:");
                ui.label(result);
            });
        }

        // Plot display.
        if !self.plot_data.is_empty() {
            ui.add_space(8.0);
            crate::plotting::plot_function_2d(ui, &self.plot_data, &self.expression_input);
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
    // Operations
    // -----------------------------------------------------------------------

    /// Evaluate the current expression numerically.
    fn evaluate(&mut self) {
        #[cfg(feature = "cas")]
        {
            self.evaluate_with_cas();
        }

        #[cfg(not(feature = "cas"))]
        {
            self.evaluate_builtin();
        }
    }

    /// Minimal built-in evaluator for simple arithmetic when the CAS feature
    /// is not enabled. Supports basic Rust-parseable float expressions.
    #[cfg(not(feature = "cas"))]
    fn evaluate_builtin(&mut self) {
        let expr = self.expression_input.trim();
        if expr.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        // Attempt trivial numeric parse first.
        if let Ok(val) = expr.parse::<f64>() {
            let result_str = format!("{val}");
            self.history.push((expr.to_string(), result_str.clone()));
            self.result = Some(result_str);
            return;
        }

        self.result = Some("Enable 'cas' feature for full expression evaluation".into());
    }

    /// Evaluate using the CAS crate's parser and evaluator.
    #[cfg(feature = "cas")]
    fn evaluate_with_cas(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            self.result = Some("(empty expression)".into());
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                let env = simucad_cas::evaluator::Environment::new();
                match simucad_cas::evaluator::evaluate(&ast, &env) {
                    Ok(val) => {
                        let result_str = format!("{val}");
                        self.history.push((expr_str.to_string(), result_str.clone()));
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
                match simucad_cas::derivative::differentiate(&ast, &self.diff_variable) {
                    Ok(derivative) => {
                        let simplified = simucad_cas::simplify::simplify(&derivative);
                        let result_str = format!("{simplified}");
                        self.history.push((format!("d/d{} {}", self.diff_variable, expr_str), result_str.clone()));
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

    /// Generate plot data for the current expression.
    fn generate_plot(&mut self) {
        #[cfg(feature = "cas")]
        {
            self.generate_plot_cas();
        }

        #[cfg(not(feature = "cas"))]
        {
            self.generate_plot_builtin();
        }
    }

    /// Plot using the CAS evaluator.
    #[cfg(feature = "cas")]
    fn generate_plot_cas(&mut self) {
        let expr_str = self.expression_input.trim();
        if expr_str.is_empty() {
            return;
        }

        match simucad_cas::parser::parse(expr_str) {
            Ok(ast) => {
                let mut points = Vec::with_capacity(self.plot_samples);
                let dx = (self.plot_x_max - self.plot_x_min) / (self.plot_samples - 1).max(1) as f64;

                for i in 0..self.plot_samples {
                    let x = self.plot_x_min + i as f64 * dx;
                    let mut env = simucad_cas::evaluator::Environment::new();
                    env.set("x", x);

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

    /// Placeholder plot generation when CAS is not available.
    #[cfg(not(feature = "cas"))]
    fn generate_plot_builtin(&mut self) {
        self.result = Some("Enable 'cas' feature for function plotting".into());
    }
}
