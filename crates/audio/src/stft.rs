//! Short-Time Fourier Transform (STFT).
//!
//! Computes a time-frequency representation of an audio signal by sliding
//! a windowed FFT across the waveform with a configurable hop size.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use simucad_core::error::AudioError;

use crate::types::{AudioSignal, FrequencyBin, Spectrum};
use crate::windowing::{generate_window, WindowType};

/// STFT result -- a time-frequency representation (spectrogram).
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// One [`Spectrum`] per analysis frame.
    pub frames: Vec<Spectrum>,
    /// Number of samples between successive frames.
    pub hop_size: usize,
    /// Number of samples in each analysis window (= FFT size).
    pub window_size: usize,
    /// Sample rate of the source signal (Hz).
    pub sample_rate: u32,
    /// Total duration of the source signal in seconds.
    pub total_duration_secs: f64,
}

impl Spectrogram {
    /// Number of analysis frames in the spectrogram.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Time (in seconds) at the centre of the given frame.
    pub fn time_at_frame(&self, frame: usize) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        let centre_sample = frame * self.hop_size + self.window_size / 2;
        centre_sample as f64 / self.sample_rate as f64
    }

    /// Build a 2-D magnitude matrix (frames x frequency bins).
    ///
    /// Each inner vector corresponds to one frame and contains the
    /// magnitude of every frequency bin.
    pub fn magnitude_matrix(&self) -> Vec<Vec<f64>> {
        self.frames
            .iter()
            .map(|s| s.bins.iter().map(|b| b.magnitude).collect())
            .collect()
    }

    /// Build a 2-D matrix of magnitudes converted to decibels.
    ///
    /// Uses `20 * log10(magnitude)` with a floor of -120 dB.
    pub fn to_db_matrix(&self) -> Vec<Vec<f64>> {
        const DB_FLOOR: f64 = -120.0;
        self.frames
            .iter()
            .map(|s| {
                s.bins
                    .iter()
                    .map(|b| {
                        if b.magnitude > 0.0 {
                            20.0 * b.magnitude.log10()
                        } else {
                            DB_FLOOR
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Compute the Short-Time Fourier Transform of an audio signal.
///
/// # Arguments
///
/// * `signal` - The input audio signal (will be mixed to mono internally).
/// * `window_type` - Window function to apply to each frame.
/// * `window_size` - Length of each analysis window (and FFT size).
/// * `hop_size` - Number of samples to advance between frames.
///
/// # Errors
///
/// Returns [`AudioError::EmptyBuffer`] if the signal has no samples.
pub fn compute_stft(
    signal: &AudioSignal,
    window_type: WindowType,
    window_size: usize,
    hop_size: usize,
) -> Result<Spectrogram, AudioError> {
    if signal.samples.is_empty() {
        return Err(AudioError::EmptyBuffer);
    }

    let mono = signal.to_mono();
    let samples = &mono.samples;
    let sample_rate = mono.sample_rate;
    let total_duration_secs = mono.duration_secs;

    // Pre-generate window coefficients.
    let window = generate_window(window_type, window_size);

    // Set up FFT planner (reusable across frames).
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);

    let nyquist_count = window_size / 2 + 1;

    let mut frames: Vec<Spectrum> = Vec::new();
    let mut offset = 0;

    while offset + window_size <= samples.len() {
        // Build windowed buffer.
        let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(window_size);
        for i in 0..window_size {
            let s = samples[offset + i];
            buffer.push(Complex::new(s * window[i], 0.0));
        }

        fft.process(&mut buffer);

        // Extract positive frequencies.
        let bins: Vec<FrequencyBin> = buffer[..nyquist_count]
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let frequency_hz = i as f64 * sample_rate as f64 / window_size as f64;
                FrequencyBin {
                    frequency_hz,
                    magnitude: c.norm(),
                    phase: c.im.atan2(c.re),
                }
            })
            .collect();

        frames.push(Spectrum {
            bins,
            sample_rate,
            fft_size: window_size,
        });

        offset += hop_size;
    }

    Ok(Spectrogram {
        frames,
        hop_size,
        window_size,
        sample_rate,
        total_duration_secs,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioSignal;
    use std::f64::consts::PI;

    fn sine_signal(freq_hz: f64, sample_rate: u32, num_samples: usize) -> AudioSignal {
        let samples: Vec<f64> = (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate as f64).sin())
            .collect();
        AudioSignal::new(samples, sample_rate, 1)
    }

    #[test]
    fn stft_empty_signal_errors() {
        let signal = AudioSignal::new(vec![], 44100, 1);
        assert!(compute_stft(&signal, WindowType::Hann, 1024, 512).is_err());
    }

    #[test]
    fn stft_frame_count() {
        let signal = sine_signal(440.0, 44100, 44100); // 1 second
        let spec = compute_stft(&signal, WindowType::Hann, 1024, 512).unwrap();
        // Expected frames: floor((44100 - 1024) / 512) + 1
        let expected = (44100 - 1024) / 512 + 1;
        assert_eq!(spec.frame_count(), expected);
    }

    #[test]
    fn stft_bins_per_frame() {
        let signal = sine_signal(440.0, 44100, 4096);
        let spec = compute_stft(&signal, WindowType::Hann, 1024, 512).unwrap();
        assert!(!spec.frames.is_empty());
        // Each frame should have fft_size/2 + 1 bins
        assert_eq!(spec.frames[0].bins.len(), 513);
    }

    #[test]
    fn stft_time_at_frame() {
        let signal = sine_signal(440.0, 44100, 44100);
        let spec = compute_stft(&signal, WindowType::Hann, 1024, 512).unwrap();
        // Frame 0 centre is at sample window_size/2
        let t0 = spec.time_at_frame(0);
        let expected = 512.0 / 44100.0;
        assert!((t0 - expected).abs() < 1e-10);
    }

    #[test]
    fn stft_magnitude_matrix_shape() {
        let signal = sine_signal(440.0, 44100, 8192);
        let spec = compute_stft(&signal, WindowType::Hann, 1024, 512).unwrap();
        let mat = spec.magnitude_matrix();
        assert_eq!(mat.len(), spec.frame_count());
        for row in &mat {
            assert_eq!(row.len(), 513);
        }
    }

    #[test]
    fn stft_db_matrix_finite() {
        let signal = sine_signal(440.0, 44100, 4096);
        let spec = compute_stft(&signal, WindowType::Hann, 1024, 512).unwrap();
        let db = spec.to_db_matrix();
        for row in &db {
            for &v in row {
                assert!(v.is_finite(), "dB matrix contains non-finite value");
            }
        }
    }

    #[test]
    fn stft_peak_near_tone_frequency() {
        let freq = 1000.0;
        let signal = sine_signal(freq, 44100, 44100);
        let spec = compute_stft(&signal, WindowType::Hann, 4096, 2048).unwrap();

        // Check the middle frame for a peak near 1000 Hz.
        let mid = spec.frame_count() / 2;
        let peak = spec.frames[mid].peak_bin().unwrap();
        let resolution = spec.frames[mid].frequency_resolution();
        assert!(
            (peak.frequency_hz - freq).abs() < resolution * 2.0,
            "Peak at {} Hz, expected near {} Hz",
            peak.frequency_hz,
            freq,
        );
    }
}
