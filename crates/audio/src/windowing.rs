//! Window functions for spectral analysis.
//!
//! Provides common window functions used to reduce spectral leakage when
//! performing FFT-based analysis. Each window can be generated as a vector
//! and applied to a sample buffer in-place.

use std::f64::consts::PI;

/// Supported window function types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// No windowing (all coefficients are 1.0).
    Rectangular,
    /// Hann (raised cosine) window.
    Hann,
    /// Hamming window (raised cosine with non-zero endpoints).
    Hamming,
    /// Blackman 3-term window.
    Blackman,
    /// Blackman-Harris 4-term minimum sidelobe window.
    BlackmanHarris,
    /// Flat-top window for accurate amplitude measurement.
    FlatTop,
    /// Kaiser window with adjustable `beta` parameter.
    Kaiser { beta: f64 },
}

/// Generate a window of the specified type and length.
///
/// Returns a `Vec<f64>` of `length` coefficients in the range `[0.0, 1.0]`
/// (except `FlatTop`, which can have negative lobes).
pub fn generate_window(window_type: WindowType, length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }

    let n_minus_1 = (length - 1) as f64;

    match window_type {
        WindowType::Rectangular => vec![1.0; length],

        WindowType::Hann => (0..length)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f64 / n_minus_1).cos()))
            .collect(),

        WindowType::Hamming => (0..length)
            .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f64 / n_minus_1).cos())
            .collect(),

        WindowType::Blackman => (0..length)
            .map(|n| {
                let x = n as f64 / n_minus_1;
                0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
            })
            .collect(),

        WindowType::BlackmanHarris => {
            let a0 = 0.35875;
            let a1 = 0.48829;
            let a2 = 0.14128;
            let a3 = 0.01168;
            (0..length)
                .map(|n| {
                    let x = n as f64 / n_minus_1;
                    a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                        - a3 * (6.0 * PI * x).cos()
                })
                .collect()
        }

        WindowType::FlatTop => {
            // 5-term flat-top window (from IEEE / Heinzel et al.)
            let a0 = 0.21557895;
            let a1 = 0.41663158;
            let a2 = 0.277263158;
            let a3 = 0.083578947;
            let a4 = 0.006947368;
            (0..length)
                .map(|n| {
                    let x = n as f64 / n_minus_1;
                    a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                        - a3 * (6.0 * PI * x).cos() + a4 * (8.0 * PI * x).cos()
                })
                .collect()
        }

        WindowType::Kaiser { beta } => (0..length)
            .map(|n| {
                let alpha = n_minus_1 / 2.0;
                let ratio = (n as f64 - alpha) / alpha;
                let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
                bessel_i0(arg) / bessel_i0(beta)
            })
            .collect(),
    }
}

/// Apply a window in-place to a sample buffer.
///
/// The window is applied element-wise: `samples[i] *= window[i]`.
/// If `window` is shorter than `samples`, only the first `window.len()`
/// samples are modified. If `window` is longer, the extra coefficients
/// are ignored.
pub fn apply_window(samples: &mut [f64], window: &[f64]) {
    let len = samples.len().min(window.len());
    for i in 0..len {
        samples[i] *= window[i];
    }
}

/// Compute the coherent gain (sum of window coefficients).
///
/// Dividing by coherent gain corrects the amplitude of spectral peaks.
pub fn coherent_gain(window: &[f64]) -> f64 {
    window.iter().sum()
}

/// Compute the noise bandwidth factor for power spectral density correction.
///
/// Defined as `N * sum(w[n]^2) / (sum(w[n]))^2` where N is the window length.
pub fn noise_bandwidth(window: &[f64]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let sum: f64 = window.iter().sum();
    if sum.abs() < f64::EPSILON {
        return 0.0;
    }
    let sum_sq: f64 = window.iter().map(|w| w * w).sum();
    let n = window.len() as f64;
    n * sum_sq / (sum * sum)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind, order zero (I_0).
///
/// Uses a series expansion that converges rapidly for moderate arguments.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;

    for k in 1..=50 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_window_is_all_ones() {
        let w = generate_window(WindowType::Rectangular, 128);
        assert_eq!(w.len(), 128);
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn hann_endpoints_are_zero() {
        let w = generate_window(WindowType::Hann, 1024);
        assert!(w[0].abs() < 1e-12);
        assert!(w[1023].abs() < 1e-12);
    }

    #[test]
    fn hann_midpoint_is_one() {
        let w = generate_window(WindowType::Hann, 1025);
        // For odd length, the exact midpoint should be 1.0
        assert!((w[512] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hamming_endpoints_nonzero() {
        let w = generate_window(WindowType::Hamming, 1024);
        // Hamming endpoints are 0.54 - 0.46 = 0.08
        assert!((w[0] - 0.08).abs() < 1e-12);
        assert!((w[1023] - 0.08).abs() < 1e-12);
    }

    #[test]
    fn window_symmetry() {
        // All symmetric windows should satisfy w[n] == w[N-1-n]
        let types = vec![
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::BlackmanHarris,
            WindowType::FlatTop,
            WindowType::Kaiser { beta: 5.0 },
        ];

        for wt in types {
            let w = generate_window(wt, 256);
            for i in 0..128 {
                assert!(
                    (w[i] - w[255 - i]).abs() < 1e-12,
                    "Symmetry failed for {:?} at index {}",
                    wt,
                    i
                );
            }
        }
    }

    #[test]
    fn blackman_harris_sum() {
        let w = generate_window(WindowType::BlackmanHarris, 1024);
        let sum: f64 = w.iter().sum();
        // Coherent gain should be roughly 0.35875 * N for BH
        let expected_approx = 0.35875 * 1024.0;
        // Allow generous tolerance (within 5%)
        assert!(
            (sum - expected_approx).abs() / expected_approx < 0.05,
            "BlackmanHarris sum {sum} not near expected {expected_approx}"
        );
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = generate_window(WindowType::Kaiser { beta: 0.0 }, 128);
        for &v in &w {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "Kaiser(beta=0) should be rectangular, got {v}"
            );
        }
    }

    #[test]
    fn apply_window_scales_samples() {
        let w = generate_window(WindowType::Hann, 8);
        let mut samples = vec![1.0; 8];
        apply_window(&mut samples, &w);
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                (s - w[i]).abs() < 1e-12,
                "apply_window mismatch at index {i}"
            );
        }
    }

    #[test]
    fn noise_bandwidth_rectangular() {
        let w = generate_window(WindowType::Rectangular, 1024);
        let nb = noise_bandwidth(&w);
        // For rectangular window, noise bandwidth should be 1.0
        assert!((nb - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_and_single_element() {
        let w0 = generate_window(WindowType::Hann, 0);
        assert!(w0.is_empty());

        let w1 = generate_window(WindowType::Hann, 1);
        assert_eq!(w1.len(), 1);
        assert!((w1[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn coherent_gain_rectangular() {
        let w = generate_window(WindowType::Rectangular, 100);
        assert!((coherent_gain(&w) - 100.0).abs() < 1e-12);
    }
}
