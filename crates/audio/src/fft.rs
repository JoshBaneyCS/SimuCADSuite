//! FFT analysis pipeline.
//!
//! Provides forward FFT computation with configurable windowing,
//! magnitude/phase extraction, inverse FFT, and power spectrum conversion
//! to dB scale.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use simucad_core::error::AudioError;

use crate::types::{AudioSignal, FrequencyBin, Spectrum};
use crate::windowing::{generate_window, WindowType};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute a forward FFT on the given audio signal.
///
/// The signal is first mixed to mono (if multi-channel), then the first
/// `fft_size` samples are windowed with the specified window function and
/// transformed. Only the positive-frequency half of the spectrum (up to
/// Nyquist) is returned.
///
/// # Arguments
///
/// * `signal` - The input audio signal.
/// * `fft_size` - Number of FFT points. If the signal has fewer samples than
///   `fft_size` the buffer is zero-padded.
/// * `window_type` - The window function to apply before the FFT.
///
/// # Errors
///
/// Returns [`AudioError::EmptyBuffer`] if the signal contains no samples.
pub fn compute_fft(
    signal: &AudioSignal,
    fft_size: usize,
    window_type: WindowType,
) -> Result<Spectrum, AudioError> {
    if signal.samples.is_empty() {
        return Err(AudioError::EmptyBuffer);
    }

    // Mix to mono for spectral analysis.
    let mono = signal.to_mono();

    // Generate window coefficients.
    let window = generate_window(window_type, fft_size);

    // Build the input buffer with window applied.
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
    for i in 0..fft_size {
        let sample = if i < mono.samples.len() {
            mono.samples[i]
        } else {
            0.0 // zero-pad
        };
        buffer.push(Complex::new(sample * window[i], 0.0));
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

/// Reconstruct a time-domain signal from a [`Spectrum`] via inverse FFT.
///
/// The spectrum is assumed to contain only the positive-frequency half
/// (DC through Nyquist). The full complex buffer is reconstructed by
/// mirroring conjugate-symmetric bins, then an inverse FFT is performed.
///
/// # Errors
///
/// Returns [`AudioError::EmptyBuffer`] if the spectrum has no bins.
pub fn compute_inverse_fft(spectrum: &Spectrum) -> Result<Vec<f64>, AudioError> {
    if spectrum.bins.is_empty() {
        return Err(AudioError::EmptyBuffer);
    }

    let fft_size = spectrum.fft_size;
    let nyquist_count = fft_size / 2 + 1;

    // Build the full complex buffer from positive-frequency bins.
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(fft_size);

    // Positive frequencies (DC to Nyquist).
    for bin in spectrum.bins.iter().take(nyquist_count) {
        buffer.push(Complex::new(
            bin.magnitude * bin.phase.cos(),
            bin.magnitude * bin.phase.sin(),
        ));
    }

    // Mirror the conjugate-symmetric negative frequencies.
    for i in 1..fft_size - nyquist_count + 1 {
        let idx = nyquist_count - 1 - i;
        if idx < spectrum.bins.len() {
            let bin = &spectrum.bins[idx];
            buffer.push(Complex::new(
                bin.magnitude * bin.phase.cos(),
                -bin.magnitude * bin.phase.sin(),
            ));
        } else {
            buffer.push(Complex::new(0.0, 0.0));
        }
    }

    // Ensure buffer is exactly fft_size.
    buffer.resize(fft_size, Complex::new(0.0, 0.0));

    // Perform inverse FFT.
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut buffer);

    // rustfft does not normalise — divide by fft_size.
    let scale = 1.0 / fft_size as f64;
    let samples: Vec<f64> = buffer.iter().map(|c| c.re * scale).collect();

    Ok(samples)
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
        assert!(compute_fft(&signal, 1024, WindowType::Hann).is_err());
    }

    #[test]
    fn fft_pure_tone_peak_near_frequency() {
        let freq = 440.0;
        let sr = 44100;
        let fft_size = 4096;
        let signal = sine_signal(freq, sr, fft_size);
        let spectrum = compute_fft(&signal, fft_size, WindowType::Hann).unwrap();

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
        let spectrum = compute_fft(&signal, 1024, WindowType::Hann).unwrap();
        assert_eq!(spectrum.bins.len(), 1024 / 2 + 1);
    }

    #[test]
    fn fft_zero_padded_short_signal() {
        let signal = AudioSignal::new(vec![1.0; 64], 44100, 1);
        let spectrum = compute_fft(&signal, 1024, WindowType::Hann).unwrap();
        assert_eq!(spectrum.fft_size, 1024);
        assert_eq!(spectrum.bins.len(), 513);
    }

    #[test]
    fn fft_with_different_windows() {
        let signal = sine_signal(440.0, 44100, 4096);
        // Should succeed with various window types.
        assert!(compute_fft(&signal, 4096, WindowType::Rectangular).is_ok());
        assert!(compute_fft(&signal, 4096, WindowType::Hamming).is_ok());
        assert!(compute_fft(&signal, 4096, WindowType::Blackman).is_ok());
        assert!(compute_fft(&signal, 4096, WindowType::BlackmanHarris).is_ok());
        assert!(compute_fft(&signal, 4096, WindowType::Kaiser { beta: 6.0 }).is_ok());
    }

    #[test]
    fn power_spectrum_values_are_finite() {
        let signal = sine_signal(1000.0, 44100, 2048);
        let spectrum = compute_fft(&signal, 2048, WindowType::Hann).unwrap();
        let power = compute_power_spectrum(&spectrum);

        for &val in &power {
            assert!(val.is_finite(), "Power spectrum contains non-finite value");
        }
    }

    #[test]
    fn power_spectrum_floor_for_silence() {
        let signal = AudioSignal::new(vec![0.0; 1024], 44100, 1);
        let spectrum = compute_fft(&signal, 1024, WindowType::Hann).unwrap();
        let power = compute_power_spectrum(&spectrum);

        for &val in &power {
            assert!(val <= -119.0, "Expected dB floor for silent signal, got {val}");
        }
    }

    #[test]
    fn inverse_fft_empty_spectrum_errors() {
        let spectrum = Spectrum {
            bins: vec![],
            sample_rate: 44100,
            fft_size: 1024,
        };
        assert!(compute_inverse_fft(&spectrum).is_err());
    }

    #[test]
    fn fft_ifft_round_trip() {
        // Use a rectangular window so the original signal is not modified.
        let fft_size = 1024;
        let signal = sine_signal(440.0, 44100, fft_size);
        let spectrum = compute_fft(&signal, fft_size, WindowType::Rectangular).unwrap();
        let recovered = compute_inverse_fft(&spectrum).unwrap();

        assert_eq!(recovered.len(), fft_size);
        for (i, (&orig, &rec)) in signal.samples.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Round-trip mismatch at sample {i}: orig={orig}, recovered={rec}"
            );
        }
    }

    #[test]
    fn fft_ifft_round_trip_dc() {
        // Constant DC signal — should survive round-trip.
        let fft_size = 256;
        let signal = AudioSignal::new(vec![0.5; fft_size], 44100, 1);
        let spectrum = compute_fft(&signal, fft_size, WindowType::Rectangular).unwrap();
        let recovered = compute_inverse_fft(&spectrum).unwrap();

        for (i, &rec) in recovered.iter().enumerate() {
            assert!(
                (0.5 - rec).abs() < 1e-6,
                "DC round-trip mismatch at sample {i}: recovered={rec}"
            );
        }
    }
}
