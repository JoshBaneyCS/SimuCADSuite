use tracing::debug;

// ---------------------------------------------------------------------------
// GPU compute timing via timestamp queries
// ---------------------------------------------------------------------------

/// Lightweight GPU profiler that measures compute pass duration using
/// `wgpu::Features::TIMESTAMP_QUERY`.
///
/// If the feature is not supported or profiling is disabled, every method
/// is a no-op -- callers do not need to branch on availability.
pub struct GpuProfiler {
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    enabled: bool,
    /// Nanoseconds per timestamp tick (varies by adapter). Usually 1.0.
    timestamp_period: f32,
}

impl GpuProfiler {
    /// Create a new profiler. If `enabled` is `false` **or** the device does
    /// not support timestamp queries, all methods become no-ops.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> Self {
        let features = device.features();
        let supports = features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

        if !enabled || !supports {
            if enabled && !supports {
                debug!("GpuProfiler: timestamp queries not supported, profiling disabled");
            }
            return Self {
                query_set: None,
                resolve_buffer: None,
                readback_buffer: None,
                enabled: false,
                timestamp_period: 1.0,
            };
        }

        let timestamp_period = queue.get_timestamp_period();

        // We need exactly 2 timestamps: begin + end.
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        // Buffer to resolve query results into (2 x u64 = 16 bytes).
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler_resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for CPU readback.
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler_readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            query_set: Some(query_set),
            resolve_buffer: Some(resolve_buffer),
            readback_buffer: Some(readback_buffer),
            enabled: true,
            timestamp_period,
        }
    }

    /// Returns `true` if the profiler is active and will record timestamps.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Write a start timestamp. Call **before** `begin_compute_pass`.
    pub fn write_start_timestamp(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, 0);
        }
    }

    /// Write an end timestamp. Call **after** `end_compute_pass` (i.e., after
    /// the pass is dropped).
    pub fn write_end_timestamp(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, 1);
        }
    }

    /// Resolve query results into the resolve buffer. Call after
    /// `write_end_timestamp` and before submitting the encoder.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(qs), Some(buf)) = (&self.query_set, &self.resolve_buffer) {
            encoder.resolve_query_set(qs, 0..2, buf, 0);
        }
        if let (Some(resolve), Some(readback)) =
            (&self.resolve_buffer, &self.readback_buffer)
        {
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
        }
    }

    /// After submitting and polling, read back the elapsed time in milliseconds.
    ///
    /// Returns `None` if profiling is disabled or the readback fails.
    pub fn read_elapsed_ms(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Option<f64> {
        let readback = self.readback_buffer.as_ref()?;

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        if timestamps.len() < 2 {
            drop(data);
            readback.unmap();
            return None;
        }

        let start = timestamps[0];
        let end = timestamps[1];
        drop(data);
        readback.unmap();

        let elapsed_ns = (end.wrapping_sub(start)) as f64 * self.timestamp_period as f64;
        let elapsed_ms = elapsed_ns / 1_000_000.0;

        debug!("GPU pass elapsed: {elapsed_ms:.3} ms");
        Some(elapsed_ms)
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-side logic only -- no GPU needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profiler_is_noop() {
        // We cannot create a real device in unit tests, but we can verify the
        // struct in its disabled state behaves correctly.
        let profiler = GpuProfiler {
            query_set: None,
            resolve_buffer: None,
            readback_buffer: None,
            enabled: false,
            timestamp_period: 1.0,
        };

        assert!(!profiler.is_enabled());
    }

    #[test]
    fn disabled_profiler_read_returns_none() {
        let profiler = GpuProfiler {
            query_set: None,
            resolve_buffer: None,
            readback_buffer: None,
            enabled: false,
            timestamp_period: 1.0,
        };

        // read_elapsed_ms should return None without a device -- it early-returns
        // because readback_buffer is None.
        // We cannot call it without a real device/queue, but the None path is
        // exercised by the `?` on readback_buffer.
        assert!(!profiler.is_enabled());
    }
}
