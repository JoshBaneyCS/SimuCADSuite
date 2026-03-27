//! CAS round-trip integration tests.
//!
//! These tests exercise cross-module workflows: parsing expressions,
//! differentiating, integrating, simplifying, evaluating, finding roots,
//! generating plot points, and producing LaTeX output.

use simucad_cas::derivative::differentiate;
use simucad_cas::evaluator::{evaluate, Environment};
use simucad_cas::integration::integrate;
use simucad_cas::latex::to_latex;
use simucad_cas::parser::parse;
use simucad_cas::plotter::generate_2d_points;
use simucad_cas::simplify::simplify;
use simucad_cas::solver::find_roots;

/// Helper: parse, simplify, and evaluate at a given x.
fn parse_simplify_eval(input: &str, var: &str, val: f64) -> f64 {
    let expr = parse(input).unwrap();
    let s = simplify(&expr);
    let mut env = Environment::new();
    env.set(var, val);
    evaluate(&s, &env).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Parse "x^2 + 3*x + 2", differentiate, simplify, verify "2*x + 3"
// ---------------------------------------------------------------------------

#[test]
fn differentiate_polynomial_and_verify() {
    let expr = parse("x^2 + 3*x + 2").unwrap();
    let deriv = differentiate(&expr, "x").unwrap();
    let simplified = simplify(&deriv);

    // Evaluate the derivative at several points to verify it equals 2*x + 3
    for x in [-5.0, -1.0, 0.0, 1.0, 3.0, 10.0] {
        let mut env = Environment::new();
        env.set("x", x);
        let result = evaluate(&simplified, &env).unwrap();
        let expected = 2.0 * x + 3.0;
        assert!(
            (result - expected).abs() < 1e-10,
            "d/dx(x^2 + 3x + 2) at x={}: got {}, expected {}",
            x,
            result,
            expected,
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Parse "sin(x)", integrate, differentiate, simplify, verify "sin(x)"
// ---------------------------------------------------------------------------

#[test]
fn integrate_then_differentiate_sin_roundtrip() {
    let expr = parse("sin(x)").unwrap();

    // Integrate: sin(x) -> -cos(x)
    let integral = integrate(&expr, "x").unwrap();

    // Differentiate the integral: d/dx(-cos(x)) = sin(x)
    let deriv = differentiate(&integral, "x").unwrap();
    let simplified = simplify(&deriv);

    // Verify at several points
    for x in [0.0, 0.5, 1.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI] {
        let mut env = Environment::new();
        env.set("x", x);
        let result = evaluate(&simplified, &env).unwrap();
        let expected = x.sin();
        assert!(
            (result - expected).abs() < 1e-10,
            "d/dx(integral(sin(x))) at x={}: got {}, expected {}",
            x,
            result,
            expected,
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Parse "x^2 - 4", find roots, verify +/-2 within tolerance
// ---------------------------------------------------------------------------

#[test]
fn find_roots_quadratic() {
    let expr = parse("x^2 - 4").unwrap();
    let roots = find_roots(&expr, "x", -5.0, 5.0, 1000).unwrap();

    assert!(
        roots.len() >= 2,
        "x^2 - 4 should have 2 roots in [-5, 5], got {:?}",
        roots,
    );

    let has_neg2 = roots.iter().any(|r| (r - (-2.0)).abs() < 1e-6);
    let has_pos2 = roots.iter().any(|r| (r - 2.0).abs() < 1e-6);

    assert!(
        has_neg2,
        "should find root at -2, got {:?}",
        roots,
    );
    assert!(
        has_pos2,
        "should find root at +2, got {:?}",
        roots,
    );
}

// ---------------------------------------------------------------------------
// 4. Evaluate "2^10" -> verify 1024.0
// ---------------------------------------------------------------------------

#[test]
fn evaluate_power_expression() {
    let expr = parse("2^10").unwrap();
    let env = Environment::new();
    let result = evaluate(&expr, &env).unwrap();
    assert!(
        (result - 1024.0).abs() < 1e-10,
        "2^10 should be 1024.0, got {}",
        result,
    );
}

// ---------------------------------------------------------------------------
// 5. Generate 2D plot points for "x^2", verify parabolic shape
// ---------------------------------------------------------------------------

#[test]
fn generate_plot_points_parabola() {
    let expr = parse("x^2").unwrap();
    let points = generate_2d_points(&expr, "x", -3.0, 3.0, 61).unwrap();

    assert_eq!(points.len(), 61);

    // Verify each point matches y = x^2
    for &(x, y) in &points {
        let expected = x * x;
        assert!(
            (y - expected).abs() < 1e-10,
            "at x={}, y={} but expected {}",
            x,
            y,
            expected,
        );
    }

    // Verify parabolic shape: y values should be symmetric about x=0
    // and the minimum should be at x=0
    let (min_x, min_y) = points
        .iter()
        .copied()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    assert!(
        min_x.abs() < 0.2,
        "minimum should be near x=0, got x={}",
        min_x,
    );
    assert!(
        min_y.abs() < 0.01,
        "minimum y should be near 0, got y={}",
        min_y,
    );

    // First and last points should have y=9 (x=+-3)
    assert!(
        (points.first().unwrap().1 - 9.0).abs() < 1e-10,
        "y at x=-3 should be 9",
    );
    assert!(
        (points.last().unwrap().1 - 9.0).abs() < 1e-10,
        "y at x=3 should be 9",
    );
}

// ---------------------------------------------------------------------------
// 6. LaTeX output for complex expressions
// ---------------------------------------------------------------------------

#[test]
fn latex_output_for_complex_expressions() {
    // x^2 + sin(x)/2
    let expr = parse("x^2 + sin(x) / 2").unwrap();
    let latex = to_latex(&expr);
    assert!(
        latex.contains("x^{2}"),
        "should contain x^{{2}}, got: {}",
        latex,
    );
    assert!(
        latex.contains("\\sin"),
        "should contain \\sin, got: {}",
        latex,
    );
    assert!(
        latex.contains("\\frac"),
        "should contain \\frac, got: {}",
        latex,
    );

    // (x + 1)^2
    let expr2 = parse("(x + 1)^2").unwrap();
    let latex2 = to_latex(&expr2);
    assert!(
        latex2.contains("^{2}"),
        "should contain ^{{2}}, got: {}",
        latex2,
    );
    assert!(
        latex2.contains("\\left"),
        "should contain grouping, got: {}",
        latex2,
    );

    // sqrt(x)
    let expr3 = parse("sqrt(x)").unwrap();
    let latex3 = to_latex(&expr3);
    assert_eq!(latex3, "\\sqrt{x}");

    // exp(x)
    let expr4 = parse("exp(x)").unwrap();
    let latex4 = to_latex(&expr4);
    assert_eq!(latex4, "e^{x}");
}

// ---------------------------------------------------------------------------
// Additional round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn parse_evaluate_trig_identity() {
    // sin(x)^2 + cos(x)^2 = 1 for any x
    for x in [0.0, 0.5, 1.0, 2.0, -1.0, std::f64::consts::PI] {
        let result = parse_simplify_eval("sin(x)^2 + cos(x)^2", "x", x);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "sin(x)^2 + cos(x)^2 should be 1 at x={}, got {}",
            x,
            result,
        );
    }
}

#[test]
fn differentiate_then_evaluate_chain_rule() {
    // d/dx(sin(x^2)) = cos(x^2) * 2*x
    let expr = parse("sin(x^2)").unwrap();
    let deriv = differentiate(&expr, "x").unwrap();
    let simplified = simplify(&deriv);

    for x in [0.0, 0.5, 1.0, -1.0, 2.0] {
        let mut env = Environment::new();
        env.set("x", x);
        let result = evaluate(&simplified, &env).unwrap();
        let expected = (x * x).cos() * 2.0 * x;
        assert!(
            (result - expected).abs() < 1e-10,
            "d/dx(sin(x^2)) at x={}: got {}, expected {}",
            x,
            result,
            expected,
        );
    }
}

#[test]
fn find_roots_cubic_polynomial() {
    // x^3 - x = x(x-1)(x+1) has roots at -1, 0, 1
    let expr = parse("x^3 - x").unwrap();
    let roots = find_roots(&expr, "x", -2.0, 2.0, 1000).unwrap();

    assert!(
        roots.len() >= 3,
        "x^3 - x should have 3 roots in [-2, 2], got {:?}",
        roots,
    );

    let has_neg1 = roots.iter().any(|r| (r - (-1.0)).abs() < 1e-6);
    let has_zero = roots.iter().any(|r| r.abs() < 1e-6);
    let has_pos1 = roots.iter().any(|r| (r - 1.0).abs() < 1e-6);

    assert!(has_neg1, "should find root at -1, got {:?}", roots);
    assert!(has_zero, "should find root at 0, got {:?}", roots);
    assert!(has_pos1, "should find root at +1, got {:?}", roots);
}

#[test]
fn integrate_power_and_verify() {
    // integral(x^3) = x^4/4
    let expr = parse("x^3").unwrap();
    let integral = integrate(&expr, "x").unwrap();
    let simplified = simplify(&integral);

    for x in [1.0, 2.0, 3.0, -1.0] {
        let mut env = Environment::new();
        env.set("x", x);
        let result = evaluate(&simplified, &env).unwrap();
        let expected = x.powi(4) / 4.0;
        assert!(
            (result - expected).abs() < 1e-10,
            "integral(x^3) at x={}: got {}, expected {}",
            x,
            result,
            expected,
        );
    }
}
