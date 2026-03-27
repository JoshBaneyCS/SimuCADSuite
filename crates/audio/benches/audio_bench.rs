use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simucad_audio::fft::{compute_fft, compute_power_spectrum};
use simucad_audio::stft::compute_stft;
use simucad_audio::types::AudioSignal;
use simucad_audio::windowing::WindowType;

/// Generate a mono sine-wave signal at the given sample rate and duration.
fn make_signal(sample_rate: u32, duration_secs: f64) -> AudioSignal {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let freq = 440.0; // A4
    let samples: Vec<f64> = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * freq * t).sin()
        })
        .collect();

    AudioSignal {
        samples,
        sample_rate,
        channels: 1,
        duration_secs,
    }
}

fn bench_compute_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_fft");

    for fft_size in [1024, 4096, 16384] {
        // Signal must have at least fft_size samples.
        let signal = make_signal(44100, (fft_size as f64 / 44100.0).max(0.1));
        group.bench_with_input(
            BenchmarkId::from_parameter(fft_size),
            &fft_size,
            |b, &size| {
                b.iter(|| {
                    compute_fft(black_box(&signal), black_box(size), WindowType::Hann).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_compute_stft(c: &mut Criterion) {
    let signal = make_signal(44100, 1.0);
    let window_size = 1024;
    let hop_size = 512;

    c.bench_function("compute_stft_1s_44100hz", |b| {
        b.iter(|| {
            compute_stft(
                black_box(&signal),
                WindowType::Hann,
                black_box(window_size),
                black_box(hop_size),
            )
            .unwrap();
        });
    });
}

fn bench_compute_power_spectrum(c: &mut Criterion) {
    let signal = make_signal(44100, 0.1);
    let spectrum = compute_fft(&signal, 4096, WindowType::Hann).unwrap();

    c.bench_function("compute_power_spectrum", |b| {
        b.iter(|| {
            compute_power_spectrum(black_box(&spectrum));
        });
    });
}

criterion_group!(
    benches,
    bench_compute_fft,
    bench_compute_stft,
    bench_compute_power_spectrum,
);
criterion_main!(benches);
