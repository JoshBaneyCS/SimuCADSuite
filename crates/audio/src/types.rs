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

    /// Return the total duration of the signal in seconds.
    pub fn duration(&self) -> f64 {
        self.duration_secs
    }

    /// Extract a portion of the signal between two sample indices.
    ///
    /// Indices are in terms of *frames* (per-channel samples). The returned
    /// signal contains all channels within the specified range.
    ///
    /// If `end_sample` exceeds the frame count it is clamped.
    pub fn slice(&self, start_sample: usize, end_sample: usize) -> AudioSignal {
        let frames = self.frame_count();
        let start = start_sample.min(frames);
        let end = end_sample.min(frames);
        if start >= end {
            return AudioSignal::new(Vec::new(), self.sample_rate, self.channels);
        }

        let ch = self.channels as usize;
        let new_samples = self.samples[start * ch..end * ch].to_vec();
        AudioSignal::new(new_samples, self.sample_rate, self.channels)
    }

    /// Resample the signal to `target_rate` using linear interpolation.
    ///
    /// This is a basic resampler suitable for quick previews and analysis;
    /// for production quality consider a polyphase or sinc-based resampler.
    pub fn resample(&self, target_rate: u32) -> AudioSignal {
        if target_rate == self.sample_rate || self.samples.is_empty() || self.sample_rate == 0 {
            return self.clone();
        }

        let ch = self.channels as usize;
        let src_frames = self.frame_count();
        let ratio = target_rate as f64 / self.sample_rate as f64;
        let dst_frames = (src_frames as f64 * ratio).round() as usize;

        if dst_frames == 0 {
            return AudioSignal::new(Vec::new(), target_rate, self.channels);
        }

        let mut out = Vec::with_capacity(dst_frames * ch);

        for f in 0..dst_frames {
            let src_pos = f as f64 / ratio;
            let idx0 = (src_pos.floor() as usize).min(src_frames - 1);
            let idx1 = (idx0 + 1).min(src_frames - 1);
            let frac = src_pos - idx0 as f64;

            for c in 0..ch {
                let s0 = self.samples[idx0 * ch + c];
                let s1 = self.samples[idx1 * ch + c];
                out.push(s0 + frac * (s1 - s0));
            }
        }

        AudioSignal::new(out, target_rate, self.channels)
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
    fn duration_method() {
        let signal = AudioSignal::new(vec![0.0; 44100], 44100, 1);
        assert!((signal.duration() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn slice_extracts_range() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let signal = AudioSignal::new(samples, 44100, 1);
        let sliced = signal.slice(10, 20);
        assert_eq!(sliced.frame_count(), 10);
        assert!((sliced.samples[0] - 10.0).abs() < 1e-12);
        assert!((sliced.samples[9] - 19.0).abs() < 1e-12);
    }

    #[test]
    fn slice_clamped_range() {
        let signal = AudioSignal::new(vec![1.0; 50], 44100, 1);
        let sliced = signal.slice(40, 1000);
        assert_eq!(sliced.frame_count(), 10);
    }

    #[test]
    fn slice_empty_when_start_ge_end() {
        let signal = AudioSignal::new(vec![1.0; 50], 44100, 1);
        let sliced = signal.slice(30, 10);
        assert_eq!(sliced.frame_count(), 0);
    }

    #[test]
    fn slice_stereo() {
        // Stereo: [L0, R0, L1, R1, ...]
        let samples: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let signal = AudioSignal::new(samples, 44100, 2);
        assert_eq!(signal.frame_count(), 10);
        let sliced = signal.slice(2, 5);
        assert_eq!(sliced.frame_count(), 3);
        assert_eq!(sliced.channels, 2);
        // Frame 2 starts at index 4 in original => samples [4.0, 5.0]
        assert!((sliced.samples[0] - 4.0).abs() < 1e-12);
        assert!((sliced.samples[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn resample_same_rate_is_clone() {
        let signal = AudioSignal::new(vec![1.0; 100], 44100, 1);
        let resampled = signal.resample(44100);
        assert_eq!(resampled.samples.len(), 100);
    }

    #[test]
    fn resample_double_rate() {
        let signal = AudioSignal::new(vec![0.0, 1.0, 0.0, -1.0], 4, 1);
        let resampled = signal.resample(8);
        // Should roughly double the number of frames
        assert_eq!(resampled.frame_count(), 8);
        assert_eq!(resampled.sample_rate, 8);
    }

    #[test]
    fn resample_half_rate() {
        let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin()).collect();
        let signal = AudioSignal::new(samples, 1000, 1);
        let resampled = signal.resample(500);
        assert_eq!(resampled.frame_count(), 50);
        assert_eq!(resampled.sample_rate, 500);
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
