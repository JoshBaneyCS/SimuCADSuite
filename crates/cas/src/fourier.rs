//! Fourier series computation and discrete Fourier transform.
//!
//! Provides numerical computation of Fourier series coefficients via
//! trapezoidal quadrature, symbolic reconstruction of partial sums, and
//! a basic DFT for sampled data.

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{evaluate, Environment};

// ---------------------------------------------------------------------------
// Fourier series coefficients
// ---------------------------------------------------------------------------

/// Computed Fourier series coefficients for a function over one period.
#[derive(Debug, Clone)]
pub struct FourierCoefficients {
    /// The DC component a_0 (note: the series uses a_0 / 2).
    pub a0: f64,
    /// Cosine coefficients a_1, a_2, ...
    pub a: Vec<f64>,
    /// Sine coefficients b_1, b_2, ...
    pub b: Vec<f64>,
}

/// Compute Fourier series coefficients for `expr` over one period `[0, period]`
/// using the trapezoidal rule with `quadrature_points` sample points.
///
/// The standard Fourier series is:
///
/// ```text
/// f(x) ≈ a_0/2 + Σ_{n=1}^{N} [a_n cos(2πnx/L) + b_n sin(2πnx/L)]
/// ```
///
/// where `L = period`.
pub fn compute_fourier_coefficients(
    expr: &Expr,
    var: &str,
    period: f64,
    num_terms: usize,
    quadrature_points: usize,
) -> Result<FourierCoefficients, CasError> {
    if period <= 0.0 {
        return Err(CasError::DomainError("period must be positive".into()));
    }
    if num_terms == 0 {
        return Err(CasError::DomainError(
            "num_terms must be at least 1".into(),
        ));
    }
    if quadrature_points < 4 {
        return Err(CasError::DomainError(
            "quadrature_points must be at least 4".into(),
        ));
    }

    let l = period;
    let n_pts = quadrature_points;
    let dx = l / n_pts as f64;
    let two_pi_over_l = std::f64::consts::TAU / l;

    // Sample the function at quadrature points.
    let mut f_vals = Vec::with_capacity(n_pts);
    let mut env = Environment::new();
    for i in 0..n_pts {
        let x = i as f64 * dx;
        env.set(var, x);
        let val = evaluate(expr, &env).unwrap_or(0.0);
        f_vals.push(if val.is_finite() { val } else { 0.0 });
    }

    // a_0 = (2/L) * integral_0^L f(x) dx  (trapezoidal)
    let a0 = 2.0 / l * trapezoidal_sum(&f_vals, dx);

    let mut a_coeffs = Vec::with_capacity(num_terms);
    let mut b_coeffs = Vec::with_capacity(num_terms);

    for n in 1..=num_terms {
        let nf = n as f64;

        // a_n = (2/L) * integral_0^L f(x) * cos(2*pi*n*x/L) dx
        let cos_vals: Vec<f64> = (0..n_pts)
            .map(|i| {
                let x = i as f64 * dx;
                f_vals[i] * (two_pi_over_l * nf * x).cos()
            })
            .collect();
        let a_n = 2.0 / l * trapezoidal_sum(&cos_vals, dx);

        // b_n = (2/L) * integral_0^L f(x) * sin(2*pi*n*x/L) dx
        let sin_vals: Vec<f64> = (0..n_pts)
            .map(|i| {
                let x = i as f64 * dx;
                f_vals[i] * (two_pi_over_l * nf * x).sin()
            })
            .collect();
        let b_n = 2.0 / l * trapezoidal_sum(&sin_vals, dx);

        a_coeffs.push(a_n);
        b_coeffs.push(b_n);
    }

    Ok(FourierCoefficients {
        a0,
        a: a_coeffs,
        b: b_coeffs,
    })
}

/// Trapezoidal rule sum for evenly spaced samples with spacing `dx`.
fn trapezoidal_sum(vals: &[f64], dx: f64) -> f64 {
    if vals.len() < 2 {
        return vals.first().copied().unwrap_or(0.0) * dx;
    }
    let mut sum = 0.5 * vals[0] + 0.5 * vals[vals.len() - 1];
    for v in &vals[1..vals.len() - 1] {
        sum += v;
    }
    sum * dx
}

// ---------------------------------------------------------------------------
// Fourier partial sum as symbolic Expr
// ---------------------------------------------------------------------------

/// Build a symbolic expression for the Fourier partial sum:
///
/// ```text
/// a_0/2 + Σ_{n=1}^{N} [a_n cos(2πnx/L) + b_n sin(2πnx/L)]
/// ```
pub fn fourier_partial_sum(
    coeffs: &FourierCoefficients,
    num_terms: usize,
    period: f64,
    var: &str,
) -> Expr {
    let two_pi_over_l = std::f64::consts::TAU / period;

    // Start with a_0 / 2.
    let mut result = Expr::num(coeffs.a0 / 2.0);

    let terms = num_terms.min(coeffs.a.len());
    for n in 0..terms {
        let nf = (n + 1) as f64;
        let omega_n = two_pi_over_l * nf;

        // a_n * cos(omega_n * x)
        if coeffs.a[n].abs() > 1e-15 {
            let cos_term = Expr::mul(
                Expr::num(coeffs.a[n]),
                Expr::func(
                    "cos",
                    vec![Expr::mul(Expr::num(omega_n), Expr::var(var))],
                ),
            );
            result = Expr::add(result, cos_term);
        }

        // b_n * sin(omega_n * x)
        if coeffs.b[n].abs() > 1e-15 {
            let sin_term = Expr::mul(
                Expr::num(coeffs.b[n]),
                Expr::func(
                    "sin",
                    vec![Expr::mul(Expr::num(omega_n), Expr::var(var))],
                ),
            );
            result = Expr::add(result, sin_term);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Discrete Fourier Transform (numerical)
// ---------------------------------------------------------------------------

/// Compute the DFT of real-valued samples and return `(frequency, magnitude)`
/// pairs for the positive-frequency half of the spectrum.
///
/// Uses a naive O(N^2) DFT. For large inputs consider using an FFT library
/// (the audio crate provides one via `rustfft`).
pub fn compute_dft(samples: &[f64], sample_rate: f64) -> Vec<(f64, f64)> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let nyquist_count = n / 2 + 1;
    let mut spectrum = Vec::with_capacity(nyquist_count);

    for k in 0..nyquist_count {
        let mut re = 0.0;
        let mut im = 0.0;
        let angle_base = -std::f64::consts::TAU * k as f64 / n as f64;

        for (i, &s) in samples.iter().enumerate() {
            let angle = angle_base * i as f64;
            re += s * angle.cos();
            im += s * angle.sin();
        }

        let magnitude = (re * re + im * im).sqrt() / n as f64;
        let frequency = k as f64 * sample_rate / n as f64;
        spectrum.push((frequency, magnitude));
    }

    spectrum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_constant_function() {
        // f(x) = 5 => a0 = 10, all other coefficients ≈ 0
        let expr = Expr::num(5.0);
        let coeffs =
            compute_fourier_coefficients(&expr, "x", std::f64::consts::TAU, 5, 4096).unwrap();
        assert!(
            (coeffs.a0 - 10.0).abs() < 0.05,
            "a0 should be ~10, got {}",
            coeffs.a0
        );
        for (i, a) in coeffs.a.iter().enumerate() {
            assert!(a.abs() < 0.01, "a{} should be ~0, got {}", i + 1, a);
        }
        for (i, b) in coeffs.b.iter().enumerate() {
            assert!(b.abs() < 0.01, "b{} should be ~0, got {}", i + 1, b);
        }
    }

    #[test]
    fn fourier_sin_coefficients() {
        // f(x) = sin(x), period = 2*pi
        // Expected: a0 = 0, a_n = 0, b_1 = 1, b_n = 0 for n > 1
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let coeffs =
            compute_fourier_coefficients(&expr, "x", std::f64::consts::TAU, 5, 4096).unwrap();
        assert!(coeffs.a0.abs() < 1e-4, "a0 should be ~0, got {}", coeffs.a0);
        assert!(
            (coeffs.b[0] - 1.0).abs() < 1e-3,
            "b1 should be ~1, got {}",
            coeffs.b[0]
        );
        for (i, b) in coeffs.b.iter().enumerate().skip(1) {
            assert!(b.abs() < 1e-3, "b{} should be ~0, got {}", i + 1, b);
        }
    }

    #[test]
    fn fourier_partial_sum_reproduces_sin() {
        // The partial sum of sin(x) with enough terms should evaluate ≈ sin(x).
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let coeffs =
            compute_fourier_coefficients(&expr, "x", std::f64::consts::TAU, 5, 1024).unwrap();
        let partial = fourier_partial_sum(&coeffs, 5, std::f64::consts::TAU, "x");

        let mut env = Environment::new();
        for x in [0.5, 1.0, 2.0, 3.0] {
            env.set("x", x);
            let approx = evaluate(&partial, &env).unwrap();
            let exact = x.sin();
            assert!(
                (approx - exact).abs() < 1e-3,
                "at x={x}: approx={approx}, exact={exact}"
            );
        }
    }

    #[test]
    fn dft_pure_sine() {
        // DFT of sin(2*pi*f0*t) at sample_rate should show peak at f0.
        let f0 = 10.0;
        let sample_rate = 256.0;
        let n = 256;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (std::f64::consts::TAU * f0 * t).sin()
            })
            .collect();

        let spectrum = compute_dft(&samples, sample_rate);
        assert!(!spectrum.is_empty());

        // Find the peak frequency.
        let (peak_freq, peak_mag) = spectrum
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        assert!(
            (peak_freq - f0).abs() < 2.0,
            "peak should be near {f0} Hz, got {peak_freq} Hz"
        );
        assert!(peak_mag > 0.3, "peak magnitude should be significant, got {peak_mag}");
    }

    #[test]
    fn dft_empty_input() {
        let spectrum = compute_dft(&[], 44100.0);
        assert!(spectrum.is_empty());
    }

    #[test]
    fn fourier_error_bad_period() {
        let expr = Expr::num(1.0);
        assert!(compute_fourier_coefficients(&expr, "x", -1.0, 5, 256).is_err());
    }
}
