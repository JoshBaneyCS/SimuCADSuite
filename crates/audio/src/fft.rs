//! FFT analysis pipeline.
//!
//! Provides forward FFT computation with Hann windowing, magnitude/phase
//! extraction, and power spectrum conversion to dB scale.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use simucad_core::error::AudioError;

use crate::types::{AudioSignal, FrequencyBin, Spectrum};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute a forward FFT on the given audio signal.
///
/// The signal is first mixed to mono (if multi-channel), then the first
/// `fft_size` samples are windowed with a Hann function and transformed.
/// Only the positive-frequency half of the spectrum (up to Nyquist) is
/// returned.
///
/// # Arguments
///
/// * `signal` - The input audio signal.
/// * `fft_size` - Number of FFT points. If the signal has fewer samples than
///   `fft_size` the buffer is zero-padded.
///
/// # Errors
///
/// Returns [`AudioError::EmptyBuffer`] if the signal contains no samples.
pub fn compute_fft(signal: &AudioSignal, fft_size: usize) -> Result<Spectrum, AudioError> {
    if signal.samples.is_empty() {
        return Err(AudioError::EmptyBuffer);
    }

    // Mix to mono for spectral analysis.
    let mono = signal.to_mono();

    // Build the input buffer with Hann window applied.
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
    for i in 0..fft_size {
        let sample = if i < mono.samples.len() {
            mono.samples[i]
        } else {
            0.0 // zero-pad
        };
        let window = hann_window(i, fft_size);
        buffer.push(Complex::new(sample * window, 0.0));
    }

    // Perform the forward FFT in-place.
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);

    // Extract positive frequencies only (DC to Nyquist).
    let nyquist_count = fft_size / 2 + 1;
    let sample_rate = mono.sample_rate;

    let bins: Vec<FrequencyBin> = buffer[..nyquist_count]
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let frequency_hz = i as f64 * sample_rate as f64 / fft_size as f64;
            let magnitude = c.norm();
            let phase = c.im.atan2(c.re);
            FrequencyBin {
                frequency_hz,
                magnitude,
                phase,
            }
        })
        .collect();

    Ok(Spectrum {
        bins,
        sample_rate,
        fft_size,
    })
}

/// Compute the power spectrum in decibels (dB) from a [`Spectrum`].
///
/// Each value is `10 * log10(magnitude^2)`. A floor of -120 dB is applied
/// to avoid `-inf` for zero-magnitude bins.
pub fn compute_power_spectrum(spectrum: &Spectrum) -> Vec<f64> {
    const DB_FLOOR: f64 = -120.0;

    spectrum
        .bins
        .iter()
        .map(|bin| {
            let power = bin.magnitude * bin.magnitude;
            if power > 0.0 {
                10.0 * power.log10()
            } else {
                DB_FLOOR
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hann (raised cosine) window function.
///
/// `w(n) = 0.5 * (1 - cos(2 * pi * n / (N - 1)))`
#[inline]
fn hann_window(n: usize, size: usize) -> f64 {
    if size <= 1 {
        return 1.0;
    }
    0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (size - 1) as f64).cos())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioSignal;

    /// Generate a pure sine tone at the given frequency.
    fn sine_signal(freq_hz: f64, sample_rate: u32, num_samples: usize) -> AudioSignal {
        let samples: Vec<f64> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq_hz * t).sin()
            })
            .collect();
        AudioSignal::new(samples, sample_rate, 1)
    }

    #[test]
    fn fft_empty_signal_returns_error() {
        let signal = AudioSignal::new(vec![], 44100, 1);
        assert!(compute_fft(&signal, 1024).is_err());
    }

    #[test]
    fn fft_pure_tone_peak_near_frequency() {
        let freq = 440.0;
        let sr = 44100;
        let fft_size = 4096;
        let signal = sine_signal(freq, sr, fft_size);
        let spectrum = compute_fft(&signal, fft_size).unwrap();

        // The peak bin should be close to 440 Hz.
        let peak = spectrum.peak_bin().unwrap();
        let resolution = spectrum.frequency_resolution();
        assert!(
            (peak.frequency_hz - freq).abs() < resolution,
            "Peak at {} Hz, expected near {} Hz (resolution {} Hz)",
            peak.frequency_hz,
            freq,
            resolution
        );
    }

    #[test]
    fn fft_bin_count_is_nyquist_plus_one() {
        let signal = sine_signal(100.0, 44100, 1024);
        let spectrum = compute_fft(&signal, 1024).unwrap();
        assert_eq!(spectrum.bins.len(), 1024 / 2 + 1);
    }

    #[test]
    fn fft_zero_padded_short_signal() {
        let signal = AudioSignal::new(vec![1.0; 64], 44100, 1);
        let spectrum = compute_fft(&signal, 1024).unwrap();
        assert_eq!(spectrum.fft_size, 1024);
        assert_eq!(spectrum.bins.len(), 513);
    }

    #[test]
    fn power_spectrum_values_are_finite() {
        let signal = sine_signal(1000.0, 44100, 2048);
        let spectrum = compute_fft(&signal, 2048).unwrap();
        let power = compute_power_spectrum(&spectrum);

        for &val in &power {
            assert!(val.is_finite(), "Power spectrum contains non-finite value");
        }
    }

    #[test]
    fn power_spectrum_floor_for_silence() {
        let signal = AudioSignal::new(vec![0.0; 1024], 44100, 1);
        let spectrum = compute_fft(&signal, 1024).unwrap();
        let power = compute_power_spectrum(&spectrum);

        for &val in &power {
            assert!(val <= -119.0, "Expected dB floor for silent signal, got {val}");
        }
    }

    #[test]
    fn hann_window_endpoints() {
        let n = 1024;
        // Hann window should be zero at the endpoints.
        assert!((hann_window(0, n)).abs() < 1e-12);
        assert!((hann_window(n - 1, n)).abs() < 1e-12);
    }

    #[test]
    fn hann_window_midpoint() {
        let n = 1024;
        // Hann window should be 1.0 at the midpoint.
        let mid = hann_window(n / 2, n);
        assert!((mid - 1.0).abs() < 0.01);
    }
}
