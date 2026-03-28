//! Spectral analysis feature extraction.
//!
//! Computes summary features from a frequency-domain [`Spectrum`] and
//! time-domain sample buffers: spectral centroid, bandwidth, roll-off,
//! flatness, zero-crossing rate, RMS energy, peak frequency, and total
//! energy.

use serde::{Deserialize, Serialize};

use crate::types::Spectrum;

/// Spectral analysis results computed from a single spectrum frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectralFeatures {
    /// Centre of mass of the spectrum (Hz).
    pub spectral_centroid: f64,
    /// Spread around the centroid (Hz).
    pub spectral_bandwidth: f64,
    /// Frequency below which 85% of the spectral energy lies (Hz).
    pub spectral_rolloff: f64,
    /// Ratio of geometric mean to arithmetic mean of magnitudes.
    /// 0 = perfectly tonal, 1 = white noise.
    pub spectral_flatness: f64,
    /// Rate of sign changes per second in the time-domain signal.
    pub zero_crossing_rate: f64,
    /// Root mean square energy of the time-domain signal.
    pub rms_energy: f64,
    /// Frequency of the highest-magnitude bin (Hz).
    pub peak_frequency: f64,
    /// Sum of squared magnitudes across all bins.
    pub total_energy: f64,
}

/// Compute spectral features from a [`Spectrum`].
///
/// The `zero_crossing_rate` and `rms_energy` fields are set to `0.0`
/// because they require time-domain data; use the standalone
/// [`compute_zero_crossing_rate`] and [`compute_rms`] functions to fill
/// them in.
pub fn compute_spectral_features(spectrum: &Spectrum) -> SpectralFeatures {
    let bins = &spectrum.bins;

    if bins.is_empty() {
        return SpectralFeatures {
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            spectral_rolloff: 0.0,
            spectral_flatness: 0.0,
            zero_crossing_rate: 0.0,
            rms_energy: 0.0,
            peak_frequency: 0.0,
            total_energy: 0.0,
        };
    }

    // Total energy = sum of magnitude^2
    let total_energy: f64 = bins.iter().map(|b| b.magnitude * b.magnitude).sum();

    // Sum of magnitudes (for centroid / rolloff weighting)
    let mag_sum: f64 = bins.iter().map(|b| b.magnitude).sum();

    // --- Spectral centroid ---
    let spectral_centroid = if mag_sum > 0.0 {
        bins.iter()
            .map(|b| b.frequency_hz * b.magnitude)
            .sum::<f64>()
            / mag_sum
    } else {
        0.0
    };

    // --- Spectral bandwidth ---
    let spectral_bandwidth = if mag_sum > 0.0 {
        let variance: f64 = bins
            .iter()
            .map(|b| {
                let diff = b.frequency_hz - spectral_centroid;
                diff * diff * b.magnitude
            })
            .sum::<f64>()
            / mag_sum;
        variance.sqrt()
    } else {
        0.0
    };

    // --- Spectral rolloff (85% energy threshold) ---
    let rolloff_threshold = 0.85 * total_energy;
    let mut cumulative = 0.0;
    let mut spectral_rolloff = 0.0;
    for bin in bins {
        cumulative += bin.magnitude * bin.magnitude;
        if cumulative >= rolloff_threshold {
            spectral_rolloff = bin.frequency_hz;
            break;
        }
    }

    // --- Spectral flatness (geometric mean / arithmetic mean) ---
    let n = bins.len() as f64;
    let spectral_flatness = if mag_sum > 0.0 {
        // Use log-domain to avoid overflow/underflow with geometric mean.
        let log_sum: f64 = bins
            .iter()
            .map(|b| {
                if b.magnitude > 0.0 {
                    b.magnitude.ln()
                } else {
                    -120.0_f64.ln() // floor for zero magnitudes
                }
            })
            .sum::<f64>();
        let log_geo_mean = log_sum / n;
        let arith_mean = mag_sum / n;
        let geo_mean = log_geo_mean.exp();
        (geo_mean / arith_mean).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // --- Peak frequency ---
    let peak_frequency = bins
        .iter()
        .max_by(|a, b| {
            a.magnitude
                .partial_cmp(&b.magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|b| b.frequency_hz)
        .unwrap_or(0.0);

    SpectralFeatures {
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_flatness,
        zero_crossing_rate: 0.0,
        rms_energy: 0.0,
        peak_frequency,
        total_energy,
    }
}

/// Compute the zero-crossing rate for a time-domain signal.
///
/// Returns the number of sign changes per second.
pub fn compute_zero_crossing_rate(samples: &[f64], sample_rate: u32) -> f64 {
    if samples.len() < 2 || sample_rate == 0 {
        return 0.0;
    }

    let crossings: usize = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();

    let duration = samples.len() as f64 / sample_rate as f64;
    crossings as f64 / duration
}

/// Compute the root mean square energy of a sample buffer.
pub fn compute_rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrequencyBin, Spectrum};

    fn make_spectrum(freqs_and_mags: &[(f64, f64)]) -> Spectrum {
        let bins: Vec<FrequencyBin> = freqs_and_mags
            .iter()
            .map(|&(f, m)| FrequencyBin {
                frequency_hz: f,
                magnitude: m,
                phase: 0.0,
            })
            .collect();
        Spectrum {
            bins,
            sample_rate: 44100,
            fft_size: 1024,
        }
    }

    #[test]
    fn centroid_of_single_peak() {
        let spectrum = make_spectrum(&[(440.0, 1.0)]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.spectral_centroid - 440.0).abs() < 1e-10);
    }

    #[test]
    fn centroid_of_two_equal_peaks() {
        let spectrum = make_spectrum(&[(200.0, 1.0), (400.0, 1.0)]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.spectral_centroid - 300.0).abs() < 1e-10);
    }

    #[test]
    fn peak_frequency_correct() {
        let spectrum = make_spectrum(&[(100.0, 0.5), (440.0, 2.0), (1000.0, 0.1)]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.peak_frequency - 440.0).abs() < 1e-10);
    }

    #[test]
    fn total_energy_is_sum_of_squared_magnitudes() {
        let spectrum = make_spectrum(&[(100.0, 2.0), (200.0, 3.0)]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.total_energy - 13.0).abs() < 1e-10); // 4 + 9
    }

    #[test]
    fn spectral_rolloff_single_bin() {
        let spectrum = make_spectrum(&[(1000.0, 1.0)]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.spectral_rolloff - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn flatness_equal_magnitudes() {
        // When all magnitudes are equal, geometric mean == arithmetic mean,
        // so flatness should be close to 1.0.
        let spectrum = make_spectrum(&[
            (100.0, 1.0),
            (200.0, 1.0),
            (300.0, 1.0),
            (400.0, 1.0),
        ]);
        let features = compute_spectral_features(&spectrum);
        assert!(
            (features.spectral_flatness - 1.0).abs() < 1e-10,
            "Equal magnitudes should give flatness ~1.0, got {}",
            features.spectral_flatness
        );
    }

    #[test]
    fn zero_crossing_rate_square_wave() {
        // A "square wave" alternating +1 / -1 should have high ZCR.
        let samples: Vec<f64> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let zcr = compute_zero_crossing_rate(&samples, 1000);
        // 999 crossings in 1 second
        assert!((zcr - 999.0).abs() < 1.0);
    }

    #[test]
    fn zero_crossing_rate_dc() {
        let samples = vec![1.0; 1000];
        let zcr = compute_zero_crossing_rate(&samples, 44100);
        assert!((zcr - 0.0).abs() < 1e-12);
    }

    #[test]
    fn rms_of_constant_signal() {
        let samples = vec![0.5; 100];
        let rms = compute_rms(&samples);
        assert!((rms - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rms_of_sine() {
        // RMS of a full-cycle sine wave should be 1/sqrt(2).
        let n = 44100;
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        let rms = compute_rms(&samples);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (rms - expected).abs() < 1e-4,
            "RMS of sine: {rms}, expected {expected}"
        );
    }

    #[test]
    fn rms_empty() {
        assert!((compute_rms(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn empty_spectrum_features() {
        let spectrum = make_spectrum(&[]);
        let features = compute_spectral_features(&spectrum);
        assert!((features.spectral_centroid - 0.0).abs() < 1e-12);
        assert!((features.total_energy - 0.0).abs() < 1e-12);
    }
}
