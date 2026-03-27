use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AudioSignal — time-domain representation of an audio waveform
// ---------------------------------------------------------------------------

/// A decoded audio signal stored as interleaved f64 samples.
///
/// For multi-channel audio the samples are interleaved:
/// `[ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]`.
/// Most analysis functions expect mono (single-channel) data; use
/// [`AudioSignal::to_mono`] to mix down before processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSignal {
    /// Raw sample values normalised to the range `[-1.0, 1.0]`.
    pub samples: Vec<f64>,
    /// Number of samples per second (Hz).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo, etc.).
    pub channels: u16,
    /// Total duration of the signal in seconds.
    pub duration_secs: f64,
}

impl AudioSignal {
    /// Create a new `AudioSignal`, computing `duration_secs` automatically.
    pub fn new(samples: Vec<f64>, sample_rate: u32, channels: u16) -> Self {
        let total_frames = if channels > 0 {
            samples.len() / channels as usize
        } else {
            0
        };
        let duration_secs = if sample_rate > 0 {
            total_frames as f64 / sample_rate as f64
        } else {
            0.0
        };
        Self {
            samples,
            sample_rate,
            channels,
            duration_secs,
        }
    }

    /// Return the number of sample frames (samples per channel).
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Mix the signal down to mono by averaging across channels.
    ///
    /// If the signal is already mono the original samples are returned
    /// unchanged inside a new `AudioSignal`.
    pub fn to_mono(&self) -> AudioSignal {
        if self.channels <= 1 {
            return self.clone();
        }

        let ch = self.channels as usize;
        let frames = self.frame_count();
        let mut mono = Vec::with_capacity(frames);

        for f in 0..frames {
            let mut sum = 0.0_f64;
            for c in 0..ch {
                sum += self.samples[f * ch + c];
            }
            mono.push(sum / ch as f64);
        }

        AudioSignal::new(mono, self.sample_rate, 1)
    }
}

// ---------------------------------------------------------------------------
// FrequencyBin — one bin in a discrete frequency spectrum
// ---------------------------------------------------------------------------

/// A single frequency bin produced by an FFT.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrequencyBin {
    /// Centre frequency of this bin (Hz).
    pub frequency_hz: f64,
    /// Magnitude (absolute value of the complex FFT coefficient).
    pub magnitude: f64,
    /// Phase angle in radians.
    pub phase: f64,
}

// ---------------------------------------------------------------------------
// Spectrum — the full frequency-domain representation
// ---------------------------------------------------------------------------

/// A discrete frequency spectrum computed from an FFT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    /// Frequency bins (positive frequencies only, up to Nyquist).
    pub bins: Vec<FrequencyBin>,
    /// Sample rate of the source signal (Hz).
    pub sample_rate: u32,
    /// FFT size (number of points used in the transform).
    pub fft_size: usize,
}

impl Spectrum {
    /// Return the frequency resolution (Hz per bin).
    pub fn frequency_resolution(&self) -> f64 {
        if self.fft_size == 0 {
            return 0.0;
        }
        self.sample_rate as f64 / self.fft_size as f64
    }

    /// Find the bin with the highest magnitude (the dominant frequency).
    pub fn peak_bin(&self) -> Option<&FrequencyBin> {
        self.bins
            .iter()
            .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_signal_duration() {
        let signal = AudioSignal::new(vec![0.0; 44100], 44100, 1);
        assert!((signal.duration_secs - 1.0).abs() < 1e-12);
    }

    #[test]
    fn audio_signal_stereo_frame_count() {
        let signal = AudioSignal::new(vec![0.0; 88200], 44100, 2);
        assert_eq!(signal.frame_count(), 44100);
        assert!((signal.duration_secs - 1.0).abs() < 1e-12);
    }

    #[test]
    fn to_mono_averages_channels() {
        // Stereo: left = 1.0, right = -1.0 for every frame
        let samples: Vec<f64> = (0..100).flat_map(|_| vec![1.0, -1.0]).collect();
        let stereo = AudioSignal::new(samples, 44100, 2);
        let mono = stereo.to_mono();

        assert_eq!(mono.channels, 1);
        assert_eq!(mono.frame_count(), 100);
        for &s in &mono.samples {
            assert!((s - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn to_mono_passthrough_for_mono() {
        let signal = AudioSignal::new(vec![0.5; 100], 44100, 1);
        let mono = signal.to_mono();
        assert_eq!(mono.samples.len(), 100);
    }

    #[test]
    fn spectrum_frequency_resolution() {
        let spectrum = Spectrum {
            bins: vec![],
            sample_rate: 44100,
            fft_size: 1024,
        };
        let expected = 44100.0 / 1024.0;
        assert!((spectrum.frequency_resolution() - expected).abs() < 1e-12);
    }

    #[test]
    fn spectrum_peak_bin() {
        let spectrum = Spectrum {
            bins: vec![
                FrequencyBin { frequency_hz: 100.0, magnitude: 0.5, phase: 0.0 },
                FrequencyBin { frequency_hz: 440.0, magnitude: 1.0, phase: 0.0 },
                FrequencyBin { frequency_hz: 1000.0, magnitude: 0.3, phase: 0.0 },
            ],
            sample_rate: 44100,
            fft_size: 1024,
        };
        let peak = spectrum.peak_bin().unwrap();
        assert!((peak.frequency_hz - 440.0).abs() < 1e-12);
    }
}
