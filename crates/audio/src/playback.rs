//! Real-time audio playback via cpal.
//!
//! Provides an [`AudioPlayer`] that streams an [`AudioSignal`] to the default
//! output device. Supports play, pause, stop, and seek with a shared playback
//! position that can be read from the GUI thread for waveform cursor sync.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use simucad_core::error::AudioError;

use crate::types::AudioSignal;

// ---------------------------------------------------------------------------
// Shared playback state (lock-free, read from GUI thread)
// ---------------------------------------------------------------------------

/// Thread-safe playback state shared between the audio callback and the GUI.
pub struct PlaybackState {
    /// Current playback position in frames (atomic for lock-free reads).
    position_frames: AtomicU64,
    /// Whether playback is currently active (not paused).
    playing: AtomicBool,
    /// Whether the stream has been stopped permanently.
    stopped: AtomicBool,
    /// Sample rate of the loaded signal.
    sample_rate: u32,
    /// Total number of frames in the signal.
    total_frames: u64,
}

impl PlaybackState {
    fn new(sample_rate: u32, total_frames: u64) -> Self {
        Self {
            position_frames: AtomicU64::new(0),
            playing: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            sample_rate,
            total_frames,
        }
    }

    /// Current playback position in seconds.
    pub fn position_secs(&self) -> f64 {
        self.position_frames.load(Ordering::Relaxed) as f64 / self.sample_rate as f64
    }

    /// Current playback position in frames.
    pub fn position_frames(&self) -> u64 {
        self.position_frames.load(Ordering::Relaxed)
    }

    /// Whether playback is currently active.
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// Whether playback has been permanently stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    /// Total duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.total_frames as f64 / self.sample_rate as f64
    }

    /// Playback progress as a fraction in [0, 1].
    pub fn progress(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.position_frames.load(Ordering::Relaxed) as f64 / self.total_frames as f64
    }
}

// ---------------------------------------------------------------------------
// Audio player
// ---------------------------------------------------------------------------

/// A real-time audio player that streams samples to the default output device.
///
/// The player holds an `Arc<PlaybackState>` that can be cloned and read from
/// the GUI thread for cursor synchronization.
pub struct AudioPlayer {
    /// Shared state readable from any thread.
    state: Arc<PlaybackState>,
    /// The cpal output stream (must be kept alive).
    _stream: cpal::Stream,
}

impl AudioPlayer {
    /// Create a new player and begin streaming `signal` to the default output
    /// device. Playback starts **paused** — call [`play`] to begin.
    pub fn new(signal: &AudioSignal) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        // Mix to mono for consistent playback and convert to f32.
        let mono = signal.to_mono();
        let samples: Arc<Vec<f32>> = Arc::new(mono.samples.iter().map(|&s| s as f32).collect());
        let total_frames = samples.len() as u64;
        let sample_rate = mono.sample_rate;

        let state = Arc::new(PlaybackState::new(sample_rate, total_frames));

        // Build the output config matching the signal's sample rate.
        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let state_cb = Arc::clone(&state);
        let samples_cb = Arc::clone(&samples);

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let playing = state_cb.playing.load(Ordering::Relaxed);
                    let stopped = state_cb.stopped.load(Ordering::Relaxed);

                    if !playing || stopped {
                        // Output silence when paused or stopped.
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    let mut pos = state_cb.position_frames.load(Ordering::Relaxed) as usize;
                    let total = samples_cb.len();

                    for sample in data.iter_mut() {
                        if pos < total {
                            *sample = samples_cb[pos];
                            pos += 1;
                        } else {
                            *sample = 0.0;
                        }
                    }

                    state_cb.position_frames.store(pos as u64, Ordering::Relaxed);

                    // Auto-pause at end of file.
                    if pos >= total {
                        state_cb.playing.store(false, Ordering::Relaxed);
                    }
                },
                move |err| {
                    tracing::error!("audio output error: {err}");
                },
                None,
            )
            .map_err(|e| AudioError::PlaybackError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::PlaybackError(e.to_string()))?;

        Ok(Self {
            state,
            _stream: stream,
        })
    }

    /// Get a reference to the shared playback state.
    pub fn state(&self) -> &Arc<PlaybackState> {
        &self.state
    }

    /// Start or resume playback.
    pub fn play(&self) {
        // If at end, rewind first.
        if self.state.position_frames.load(Ordering::Relaxed) >= self.state.total_frames {
            self.state.position_frames.store(0, Ordering::Relaxed);
        }
        self.state.playing.store(true, Ordering::Relaxed);
    }

    /// Pause playback (can be resumed).
    pub fn pause(&self) {
        self.state.playing.store(false, Ordering::Relaxed);
    }

    /// Stop playback and rewind to the beginning.
    pub fn stop(&self) {
        self.state.playing.store(false, Ordering::Relaxed);
        self.state.position_frames.store(0, Ordering::Relaxed);
    }

    /// Seek to a specific position in seconds.
    pub fn seek(&self, time_secs: f64) {
        let frame = (time_secs * self.state.sample_rate as f64) as u64;
        let clamped = frame.min(self.state.total_frames);
        self.state
            .position_frames
            .store(clamped, Ordering::Relaxed);
    }

    /// Seek to a specific progress fraction in [0, 1].
    pub fn seek_fraction(&self, fraction: f64) {
        let frame = (fraction.clamp(0.0, 1.0) * self.state.total_frames as f64) as u64;
        self.state.position_frames.store(frame, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_state_defaults() {
        let state = PlaybackState::new(44100, 44100);
        assert_eq!(state.position_secs(), 0.0);
        assert!(!state.is_playing());
        assert!(!state.is_stopped());
        assert!((state.duration_secs() - 1.0).abs() < 1e-6);
        assert_eq!(state.progress(), 0.0);
    }

    #[test]
    fn playback_state_progress() {
        let state = PlaybackState::new(44100, 88200);
        state
            .position_frames
            .store(44100, Ordering::Relaxed);
        assert!((state.progress() - 0.5).abs() < 1e-6);
        assert!((state.position_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn playback_state_zero_total() {
        let state = PlaybackState::new(44100, 0);
        assert_eq!(state.progress(), 0.0);
    }
}
