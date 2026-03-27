//! Audio file decoding via Symphonia.
//!
//! Supports WAV, MP3, and FLAC through the corresponding Symphonia feature
//! flags. The decoder reads all packets from the default (best) audio track
//! and converts the samples to interleaved `f64` values in `[-1.0, 1.0]`.

use std::path::Path;

use simucad_core::error::AudioError;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::types::AudioSignal;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decode an audio file at `path` into an [`AudioSignal`].
///
/// The function auto-detects the codec from the file extension and container
/// metadata. All packets in the default track are decoded and concatenated
/// into a single contiguous buffer of f64 samples.
///
/// # Errors
///
/// Returns [`AudioError::Io`] if the file cannot be opened,
/// [`AudioError::UnsupportedFormat`] if Symphonia cannot identify the codec,
/// and [`AudioError::DecodeError`] for packet-level decoding failures.
/// [`AudioError::EmptyBuffer`] is returned when the file contains no audio
/// samples.
pub fn decode_file(path: &Path) -> Result<AudioSignal, AudioError> {
    tracing::info!("Decoding audio file: {}", path.display());

    // Open the file as a media source.
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Provide a hint based on the file extension so Symphonia can pick the
    // right demuxer / codec quickly.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    // Probe the stream for a supported format.
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| AudioError::UnsupportedFormat(format!("{e}")))?;

    let mut format_reader = probed.format;

    // Select the default (best) audio track.
    let track = format_reader
        .default_track()
        .ok_or_else(|| AudioError::UnsupportedFormat("No audio track found".into()))?;

    let track_id = track.id;

    // Extract codec parameters we need.
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::DecodeError("Missing sample rate in codec params".into()))?;

    let channels = track
        .codec_params
        .channels
        .map(|ch| ch.count() as u16)
        .unwrap_or(1);

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::DecodeError(format!("Failed to create decoder: {e}")))?;

    // Decode all packets and collect samples.
    let mut all_samples: Vec<f64> = Vec::new();

    loop {
        let packet = match format_reader.next_packet() {
            Ok(pkt) => pkt,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // End of stream.
                break;
            }
            Err(e) => {
                tracing::warn!("Error reading packet: {e}");
                break;
            }
        };

        // Skip packets that do not belong to our track.
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                tracing::warn!("Decode error (skipping packet): {msg}");
                continue;
            }
            Err(e) => {
                return Err(AudioError::DecodeError(format!("{e}")));
            }
        };

        append_samples(&decoded, &mut all_samples);
    }

    if all_samples.is_empty() {
        return Err(AudioError::EmptyBuffer);
    }

    tracing::info!(
        "Decoded {} samples, {} channels, {} Hz",
        all_samples.len(),
        channels,
        sample_rate
    );

    Ok(AudioSignal::new(all_samples, sample_rate, channels))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a Symphonia `AudioBufferRef` (any sample format) to interleaved
/// `f64` values and append them to `out`.
fn append_samples(buf: &AudioBufferRef, out: &mut Vec<f64>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push(b.chan(ch)[frame] as f64);
                }
            }
        }
        AudioBufferRef::F64(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push(b.chan(ch)[frame]);
                }
            }
        }
        AudioBufferRef::S16(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push(b.chan(ch)[frame] as f64 / i16::MAX as f64);
                }
            }
        }
        AudioBufferRef::S32(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push(b.chan(ch)[frame] as f64 / i32::MAX as f64);
                }
            }
        }
        AudioBufferRef::U8(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    // u8 audio is unsigned with 128 as the zero point.
                    out.push((b.chan(ch)[frame] as f64 - 128.0) / 128.0);
                }
            }
        }
        AudioBufferRef::S24(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let max_val = (1i32 << 23) as f64;
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    let raw: i32 = b.chan(ch)[frame].inner();
                    out.push(raw as f64 / max_val);
                }
            }
        }
        AudioBufferRef::U16(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push((b.chan(ch)[frame] as f64 - 32768.0) / 32768.0);
                }
            }
        }
        AudioBufferRef::U24(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let half = (1u32 << 23) as f64;
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    let raw: u32 = b.chan(ch)[frame].inner();
                    out.push((raw as f64 - half) / half);
                }
            }
        }
        AudioBufferRef::U32(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let half = (1u64 << 31) as f64;
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push((b.chan(ch)[frame] as f64 - half) / half);
                }
            }
        }
        AudioBufferRef::S8(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            out.reserve(channels * frames);
            for frame in 0..frames {
                for ch in 0..channels {
                    out.push(b.chan(ch)[frame] as f64 / i8::MAX as f64);
                }
            }
        }
    }
}
