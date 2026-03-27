use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// ProgressReporter — thread-safe progress tracking for long simulations
// ---------------------------------------------------------------------------

/// Thread-safe progress reporter for long-running simulations.
///
/// All operations are lock-free (atomic) except for the human-readable
/// message, which uses a `Mutex<String>`.
#[derive(Clone)]
pub struct ProgressReporter {
    current: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
    message: Arc<std::sync::Mutex<String>>,
}

impl ProgressReporter {
    /// Create a new reporter with the given total step count.
    pub fn new(total: u64) -> Self {
        Self {
            current: Arc::new(AtomicU64::new(0)),
            total: Arc::new(AtomicU64::new(total)),
            cancelled: Arc::new(AtomicBool::new(false)),
            message: Arc::new(std::sync::Mutex::new(String::new())),
        }
    }

    /// Set the current progress to an absolute value.
    pub fn set_progress(&self, current: u64) {
        self.current.store(current, Ordering::Relaxed);
    }

    /// Increment the current progress by one.
    pub fn increment(&self) {
        self.current.fetch_add(1, Ordering::Relaxed);
    }

    /// Set a human-readable status message.
    pub fn set_message(&self, msg: impl Into<String>) {
        if let Ok(mut guard) = self.message.lock() {
            *guard = msg.into();
        }
    }

    /// Return the fraction of work completed, clamped to `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let current = self.current.load(Ordering::Relaxed);
        (current as f64 / total as f64).clamp(0.0, 1.0)
    }

    /// Signal cancellation.  Workers should poll [`is_cancelled`] and
    /// exit early when it returns `true`.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Return a clone of the current status message.
    pub fn message(&self) -> String {
        self.message
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let p = ProgressReporter::new(100);
        assert!((p.fraction() - 0.0).abs() < f64::EPSILON);
        assert!(!p.is_cancelled());
        assert!(p.message().is_empty());
    }

    #[test]
    fn set_and_read_progress() {
        let p = ProgressReporter::new(200);
        p.set_progress(100);
        assert!((p.fraction() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn increment() {
        let p = ProgressReporter::new(10);
        for _ in 0..10 {
            p.increment();
        }
        assert!((p.fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fraction_clamped_to_one() {
        let p = ProgressReporter::new(5);
        p.set_progress(999);
        assert!((p.fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fraction_zero_total() {
        let p = ProgressReporter::new(0);
        p.set_progress(10);
        assert!((p.fraction() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cancel_flag() {
        let p = ProgressReporter::new(100);
        assert!(!p.is_cancelled());
        p.cancel();
        assert!(p.is_cancelled());
    }

    #[test]
    fn message_set_and_get() {
        let p = ProgressReporter::new(100);
        p.set_message("Computing step 42...");
        assert_eq!(p.message(), "Computing step 42...");
    }

    #[test]
    fn clone_shares_state() {
        let p1 = ProgressReporter::new(100);
        let p2 = p1.clone();
        p1.set_progress(50);
        assert!((p2.fraction() - 0.5).abs() < 1e-12);
        p2.cancel();
        assert!(p1.is_cancelled());
    }

    #[test]
    fn thread_safety() {
        let p = ProgressReporter::new(1000);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let reporter = p.clone();
                std::thread::spawn(move || {
                    for _ in 0..250 {
                        reporter.increment();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!((p.fraction() - 1.0).abs() < 1e-12);
    }
}
