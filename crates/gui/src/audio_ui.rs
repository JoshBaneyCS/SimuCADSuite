//! Audio analyzer UI panel.
//!
//! Provides controls for loading audio files, computing FFT and STFT analyses,
//! displaying waveforms, spectra, spectral features, and spectrograms.
//!
//! This entire module is gated behind the `audio` feature flag since
//! `simucad-audio` is an optional dependency.

#![cfg(feature = "audio")]

use std::path::Path;
use std::sync::Arc;

use egui::Ui;
use simucad_audio::fft::{compute_fft, compute_power_spectrum};
use simucad_audio::playback::{AudioPlayer, PlaybackState};
use simucad_audio::spectral::{compute_rms, compute_spectral_features, compute_zero_crossing_rate};
use simucad_audio::stft::compute_stft;
use simucad_audio::types::AudioSignal;
use simucad_audio::windowing::WindowType;

// ---------------------------------------------------------------------------
// Window type table
// ---------------------------------------------------------------------------

const WINDOW_TYPES: &[(&str, WindowType)] = &[
    ("Rectangular", WindowType::Rectangular),
    ("Hann", WindowType::Hann),
    ("Hamming", WindowType::Hamming),
    ("Blackman", WindowType::Blackman),
    ("Blackman-Harris", WindowType::BlackmanHarris),
    ("Flat Top", WindowType::FlatTop),
];

// ---------------------------------------------------------------------------
// Panel state
// ---------------------------------------------------------------------------

/// State for the audio analyzer panel.
pub struct AudioPanel {
    file_path: String,
    signal: Option<AudioSignal>,
    mono_signal: Option<AudioSignal>,
    spectrum: Option<simucad_audio::types::Spectrum>,
    power_spectrum_db: Vec<f64>,
    spectral_features: Option<simucad_audio::spectral::SpectralFeatures>,
    spectrogram_db: Vec<Vec<f64>>,
    spectrogram_time_labels: Vec<f64>,
    spectrogram_texture: Option<egui::TextureHandle>,
    fft_size: usize,
    window_type_index: usize,
    stft_window_size: usize,
    stft_hop_size: usize,
    status: String,
    waveform_points: Vec<(f64, f64)>,
    /// Active audio player (None when no file is playing).
    player: Option<AudioPlayer>,
    /// Shared playback state for reading position from the GUI thread.
    playback_state: Option<Arc<PlaybackState>>,
}

impl Default for AudioPanel {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            signal: None,
            mono_signal: None,
            spectrum: None,
            power_spectrum_db: Vec::new(),
            spectral_features: None,
            spectrogram_db: Vec::new(),
            spectrogram_time_labels: Vec::new(),
            spectrogram_texture: None,
            fft_size: 4096,
            window_type_index: 1, // Hann
            stft_window_size: 1024,
            stft_hop_size: 512,
            status: String::new(),
            waveform_points: Vec::new(),
            player: None,
            playback_state: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Downsample an audio signal to at most `max_points` (time, amplitude) pairs
/// suitable for waveform plotting.
fn downsample_waveform(signal: &AudioSignal, max_points: usize) -> Vec<(f64, f64)> {
    let mono = signal.to_mono();
    let step = (mono.samples.len() / max_points).max(1);
    mono.samples
        .iter()
        .step_by(step)
        .enumerate()
        .map(|(i, &s)| {
            let t = (i * step) as f64 / mono.sample_rate as f64;
            (t, s)
        })
        .collect()
}

/// Map a normalized 0..1 value to a viridis-like color gradient
/// (dark purple -> blue -> green -> yellow).
fn db_to_color(t: f64) -> egui::Color32 {
    let r = (255.0 * (1.5 * t - 0.5).clamp(0.0, 1.0)) as u8;
    let g = (255.0
        * (2.0 * t - 0.5)
            .clamp(0.0, 1.0)
            .min(1.0 - (2.0 * t - 1.5).max(0.0))) as u8;
    let b = (255.0 * (1.0 - 2.0 * t).clamp(0.0, 1.0)) as u8;
    egui::Color32::from_rgb(r, g, b)
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

impl AudioPanel {
    /// Render the audio analyzer panel.
    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("Audio Analyzer");
        ui.add_space(8.0);

        self.show_file_input(ui);
        ui.add_space(8.0);
        self.show_signal_info(ui);
        ui.add_space(8.0);
        self.show_playback_controls(ui);
        ui.add_space(8.0);
        self.show_fft_controls(ui);
        ui.add_space(4.0);
        self.show_stft_controls(ui);
        ui.add_space(12.0);
        self.show_waveform_plot(ui);
        ui.add_space(8.0);
        self.show_spectrum_plot(ui);
        ui.add_space(8.0);
        self.show_spectral_features(ui);
        ui.add_space(8.0);
        self.show_spectrogram(ui);
    }

    // -- File input ----------------------------------------------------------

    fn show_file_input(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Audio file:");
            ui.text_edit_singleline(&mut self.file_path);
            if ui.button("Load").clicked() {
                self.load_file();
            }
        });

        if !self.status.is_empty() {
            ui.label(&self.status);
        }
    }

    // -- Signal info ---------------------------------------------------------

    fn show_signal_info(&self, ui: &mut Ui) {
        let Some(ref sig) = self.signal else {
            return;
        };

        egui::Grid::new("audio_signal_info")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Sample rate:");
                ui.label(format!("{} Hz", sig.sample_rate));
                ui.end_row();

                ui.label("Channels:");
                ui.label(format!("{}", sig.channels));
                ui.end_row();

                ui.label("Duration:");
                ui.label(format!("{:.3} s", sig.duration_secs));
                ui.end_row();

                ui.label("Frame count:");
                ui.label(format!("{}", sig.frame_count()));
                ui.end_row();
            });
    }

    // -- FFT controls --------------------------------------------------------

    fn show_fft_controls(&mut self, ui: &mut Ui) {
        let has_signal = self.mono_signal.is_some();

        ui.horizontal(|ui| {
            ui.label("FFT size:");
            ui.add(
                egui::DragValue::new(&mut self.fft_size)
                    .speed(0.0)
                    .range(512..=16384),
            );
            // Snap to nearest power of 2.
            self.fft_size = self.fft_size.next_power_of_two();

            ui.label("Window:");
            egui::ComboBox::from_id_salt("window_type")
                .selected_text(WINDOW_TYPES[self.window_type_index].0)
                .show_ui(ui, |ui| {
                    for (idx, (name, _)) in WINDOW_TYPES.iter().enumerate() {
                        ui.selectable_value(&mut self.window_type_index, idx, *name);
                    }
                });

            if ui
                .add_enabled(has_signal, egui::Button::new("Analyze"))
                .clicked()
            {
                self.run_fft_analysis();
            }
        });
    }

    // -- STFT controls -------------------------------------------------------

    fn show_stft_controls(&mut self, ui: &mut Ui) {
        let has_signal = self.mono_signal.is_some();

        ui.horizontal(|ui| {
            ui.label("STFT window:");
            ui.add(
                egui::DragValue::new(&mut self.stft_window_size)
                    .speed(1.0)
                    .range(64..=8192),
            );

            ui.label("Hop:");
            ui.add(
                egui::DragValue::new(&mut self.stft_hop_size)
                    .speed(1.0)
                    .range(32..=4096),
            );

            if ui
                .add_enabled(has_signal, egui::Button::new("Compute STFT"))
                .clicked()
            {
                self.run_stft();
            }
        });
    }

    // -- Playback controls ---------------------------------------------------

    fn show_playback_controls(&mut self, ui: &mut Ui) {
        let has_signal = self.signal.is_some();

        ui.horizontal(|ui| {
            ui.strong("Playback");
            ui.add_space(8.0);

            let is_playing = self
                .playback_state
                .as_ref()
                .is_some_and(|s| s.is_playing());

            // Play / Pause toggle
            if is_playing {
                if ui.add_enabled(true, egui::Button::new("Pause")).clicked() {
                    if let Some(ref player) = self.player {
                        player.pause();
                    }
                }
            } else if ui
                .add_enabled(has_signal, egui::Button::new("Play"))
                .clicked()
            {
                // Create player on first play if needed.
                if self.player.is_none() {
                    self.create_player();
                }
                if let Some(ref player) = self.player {
                    player.play();
                }
            }

            // Stop
            if ui
                .add_enabled(self.player.is_some(), egui::Button::new("Stop"))
                .clicked()
            {
                if let Some(ref player) = self.player {
                    player.stop();
                }
            }

            // Position / duration label
            if let Some(ref state) = self.playback_state {
                let pos = state.position_secs();
                let dur = state.duration_secs();
                ui.label(format!(
                    "{:02}:{:04.1} / {:02}:{:04.1}",
                    pos as u32 / 60,
                    pos % 60.0,
                    dur as u32 / 60,
                    dur % 60.0,
                ));
            }
        });

        // Seek slider
        if let Some(ref state) = self.playback_state {
            let mut progress = state.progress() as f32;
            let slider = egui::Slider::new(&mut progress, 0.0..=1.0)
                .show_value(false)
                .text("Seek");
            if ui.add(slider).changed() {
                if let Some(ref player) = self.player {
                    player.seek_fraction(progress as f64);
                }
            }
        }

        // Request continuous repaint while playing so cursor updates.
        if self
            .playback_state
            .as_ref()
            .is_some_and(|s| s.is_playing())
        {
            ui.ctx().request_repaint();
        }
    }

    // -- Waveform plot -------------------------------------------------------

    fn show_waveform_plot(&self, ui: &mut Ui) {
        if self.waveform_points.is_empty() {
            return;
        }

        ui.strong("Waveform");
        let points: egui_plot::PlotPoints = self
            .waveform_points
            .iter()
            .map(|&(x, y)| [x, y])
            .collect();
        let line = egui_plot::Line::new(points).name("Waveform");

        // Build cursor line at current playback position.
        let cursor_pos = self.playback_state.as_ref().map(|s| s.position_secs());

        egui_plot::Plot::new("waveform_plot")
            .height(150.0)
            .x_axis_label("Time (s)")
            .y_axis_label("Amplitude")
            .show(ui, |plot_ui| {
                plot_ui.line(line);

                // Draw vertical cursor line at playback position.
                if let Some(t) = cursor_pos {
                    let cursor = egui_plot::Line::new(egui_plot::PlotPoints::new(vec![
                        [t, -1.0],
                        [t, 1.0],
                    ]))
                    .name("Cursor")
                    .color(egui::Color32::from_rgb(255, 220, 50))
                    .width(2.0);
                    plot_ui.line(cursor);
                }
            });
    }

    // -- Spectrum plot -------------------------------------------------------

    fn show_spectrum_plot(&self, ui: &mut Ui) {
        if self.power_spectrum_db.is_empty() || self.spectrum.is_none() {
            return;
        }

        let spectrum = self.spectrum.as_ref().unwrap();
        let freq_resolution = spectrum.frequency_resolution();

        ui.strong("Power Spectrum");
        let points: egui_plot::PlotPoints = self
            .power_spectrum_db
            .iter()
            .enumerate()
            .map(|(i, &db)| [i as f64 * freq_resolution, db])
            .collect();
        let line = egui_plot::Line::new(points).name("Spectrum (dB)");

        egui_plot::Plot::new("spectrum_plot")
            .height(150.0)
            .x_axis_label("Frequency (Hz)")
            .y_axis_label("Magnitude (dB)")
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });
    }

    // -- Spectral features ---------------------------------------------------

    fn show_spectral_features(&self, ui: &mut Ui) {
        let Some(ref feat) = self.spectral_features else {
            return;
        };

        ui.strong("Spectral Features");
        egui::Grid::new("spectral_features_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Spectral centroid:");
                ui.label(format!("{:.2} Hz", feat.spectral_centroid));
                ui.end_row();

                ui.label("Spectral bandwidth:");
                ui.label(format!("{:.2} Hz", feat.spectral_bandwidth));
                ui.end_row();

                ui.label("Spectral rolloff:");
                ui.label(format!("{:.2} Hz", feat.spectral_rolloff));
                ui.end_row();

                ui.label("Spectral flatness:");
                ui.label(format!("{:.6}", feat.spectral_flatness));
                ui.end_row();

                ui.label("Peak frequency:");
                ui.label(format!("{:.2} Hz", feat.peak_frequency));
                ui.end_row();

                ui.label("RMS energy:");
                ui.label(format!("{:.6}", feat.rms_energy));
                ui.end_row();

                ui.label("Zero crossing rate:");
                ui.label(format!("{:.6}", feat.zero_crossing_rate));
                ui.end_row();

                ui.label("Total energy:");
                ui.label(format!("{:.4}", feat.total_energy));
                ui.end_row();
            });
    }

    // -- Spectrogram ---------------------------------------------------------

    fn show_spectrogram(&mut self, ui: &mut Ui) {
        if self.spectrogram_db.is_empty() {
            return;
        }

        ui.strong("Spectrogram");

        // Build the texture only once (cleared when STFT is recomputed).
        if self.spectrogram_texture.is_none() {
            self.spectrogram_texture = Some(self.build_spectrogram_texture(ui));
        }

        if let Some(ref texture) = self.spectrogram_texture {
            let available = ui.available_size();
            let display_width = available.x.min(800.0);
            let display_height = 200.0;
            ui.image(egui::load::SizedTexture::new(
                texture.id(),
                egui::vec2(display_width, display_height),
            ));
        }
    }

    fn build_spectrogram_texture(&self, ui: &mut Ui) -> egui::TextureHandle {
        let num_frames = self.spectrogram_db.len();
        let num_bins = self.spectrogram_db[0].len();

        // Find min/max dB for normalization.
        let mut min_db = f64::MAX;
        let mut max_db = f64::MIN;
        for frame in &self.spectrogram_db {
            for &val in frame {
                if val < min_db {
                    min_db = val;
                }
                if val > max_db {
                    max_db = val;
                }
            }
        }
        let range = (max_db - min_db).max(1.0);

        // Build color image (flip y so low freq at bottom).
        let width = num_frames;
        let height = num_bins;
        let mut pixels = vec![egui::Color32::BLACK; width * height];
        for (frame_idx, frame) in self.spectrogram_db.iter().enumerate() {
            for (bin_idx, &db) in frame.iter().enumerate() {
                let t = ((db - min_db) / range).clamp(0.0, 1.0);
                let color = db_to_color(t);
                let y = height - 1 - bin_idx;
                pixels[y * width + frame_idx] = color;
            }
        }

        let image = egui::ColorImage {
            size: [width, height],
            pixels,
        };
        ui.ctx()
            .load_texture("spectrogram", image, egui::TextureOptions::NEAREST)
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------

    fn load_file(&mut self) {
        // Stop any existing playback before loading a new file.
        self.player = None;
        self.playback_state = None;

        let path = Path::new(&self.file_path);
        match simucad_audio::decoder::decode_file(path) {
            Ok(signal) => {
                self.status = format!(
                    "Loaded: {} Hz, {} ch, {:.2} s",
                    signal.sample_rate, signal.channels, signal.duration_secs
                );
                self.waveform_points = downsample_waveform(&signal, 2000);
                let mono = signal.to_mono();
                self.mono_signal = Some(mono);
                self.signal = Some(signal);

                // Clear previous analysis.
                self.spectrum = None;
                self.power_spectrum_db.clear();
                self.spectral_features = None;
                self.spectrogram_db.clear();
                self.spectrogram_time_labels.clear();
                self.spectrogram_texture = None;
            }
            Err(e) => {
                self.status = format!("Error loading file: {e}");
                self.signal = None;
                self.mono_signal = None;
                self.waveform_points.clear();
            }
        }
    }

    fn create_player(&mut self) {
        let Some(ref signal) = self.signal else {
            return;
        };
        match AudioPlayer::new(signal) {
            Ok(player) => {
                self.playback_state = Some(Arc::clone(player.state()));
                self.player = Some(player);
                self.status = "Player ready.".to_string();
            }
            Err(e) => {
                self.status = format!("Playback error: {e}");
            }
        }
    }

    fn run_fft_analysis(&mut self) {
        let Some(ref mono) = self.mono_signal else {
            return;
        };

        let window_type = WINDOW_TYPES[self.window_type_index].1;

        match compute_fft(mono, self.fft_size, window_type) {
            Ok(spectrum) => {
                self.power_spectrum_db = compute_power_spectrum(&spectrum);

                let mut features = compute_spectral_features(&spectrum);
                // Fill in time-domain features from dedicated functions.
                features.zero_crossing_rate = compute_zero_crossing_rate(&mono.samples, mono.sample_rate);
                features.rms_energy = compute_rms(&mono.samples);

                self.spectral_features = Some(features);
                self.spectrum = Some(spectrum);
                self.status = "FFT analysis complete.".to_string();
            }
            Err(e) => {
                self.status = format!("FFT error: {e}");
            }
        }
    }

    fn run_stft(&mut self) {
        let Some(ref mono) = self.mono_signal else {
            return;
        };

        let window_type = WINDOW_TYPES[self.window_type_index].1;

        match compute_stft(mono, window_type, self.stft_window_size, self.stft_hop_size) {
            Ok(spectrogram) => {
                let num_frames = spectrogram.frame_count();
                self.spectrogram_time_labels = (0..num_frames)
                    .map(|i| spectrogram.time_at_frame(i))
                    .collect();
                self.spectrogram_db = spectrogram.to_db_matrix();
                // Invalidate cached texture so it is rebuilt on next draw.
                self.spectrogram_texture = None;
                self.status = format!("STFT complete: {num_frames} frames.");
            }
            Err(e) => {
                self.status = format!("STFT error: {e}");
            }
        }
    }
}
