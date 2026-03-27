//! Background task runner for long-running simulations.
//!
//! [`TaskRunner`] wraps a `std::thread::JoinHandle` and exposes a
//! poll-based API suitable for use inside an immediate-mode GUI loop.

use std::thread::JoinHandle;

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

/// The current status of a background task.
#[derive(Debug, Clone)]
pub enum TaskStatus {
    /// No task has been started (or it has been consumed).
    Idle,
    /// A task is currently executing.
    Running {
        /// Estimated progress in `[0.0, 1.0]`. This is purely informational
        /// and may not be updated by all tasks.
        progress: f32,
    },
    /// The task finished successfully and its result is waiting to be taken.
    Completed,
    /// The task panicked or otherwise failed.
    Failed(String),
}

// ---------------------------------------------------------------------------
// TaskRunner
// ---------------------------------------------------------------------------

/// A lightweight wrapper that runs a closure on a background thread and lets
/// the GUI poll for completion each frame.
///
/// The generic parameter `T` is the return type of the closure. It must be
/// `Send + 'static` because it crosses a thread boundary.
pub struct TaskRunner<T: Send + 'static> {
    status: TaskStatus,
    handle: Option<JoinHandle<T>>,
    result: Option<T>,
}

impl<T: Send + 'static> TaskRunner<T> {
    /// Create an idle task runner.
    pub fn new() -> Self {
        Self {
            status: TaskStatus::Idle,
            handle: None,
            result: None,
        }
    }

    /// Spawn a new background task.
    ///
    /// If a task is already running it will be detached (its handle is
    /// dropped, but the thread continues to run). Call [`poll`] beforehand
    /// to check.
    pub fn spawn<F>(&mut self, f: F)
    where
        F: FnOnce() -> T + Send + 'static,
    {
        // Drop any previous handle (the thread will run to completion but
        // we won't collect its result).
        self.handle = None;
        self.result = None;

        self.status = TaskStatus::Running { progress: 0.0 };
        self.handle = Some(std::thread::spawn(f));
    }

    /// Poll the background task.
    ///
    /// If the task has finished, the status transitions to `Completed` (or
    /// `Failed` if the thread panicked). The actual result can then be
    /// retrieved with [`take_result`].
    ///
    /// This is cheap to call every frame.
    pub fn poll(&mut self) -> &TaskStatus {
        if let Some(handle) = &self.handle {
            if handle.is_finished() {
                // Take ownership of the handle so we can join it.
                let handle = self.handle.take().unwrap();
                match handle.join() {
                    Ok(value) => {
                        self.result = Some(value);
                        self.status = TaskStatus::Completed;
                    }
                    Err(panic_payload) => {
                        let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            (*s).to_string()
                        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        self.status = TaskStatus::Failed(msg);
                    }
                }
            }
        }
        &self.status
    }

    /// Take the result of a completed task, resetting the runner to `Idle`.
    ///
    /// Returns `None` if the task has not completed or the result has already
    /// been taken.
    pub fn take_result(&mut self) -> Option<T> {
        if matches!(self.status, TaskStatus::Completed) {
            self.status = TaskStatus::Idle;
            self.result.take()
        } else {
            None
        }
    }

    /// Returns `true` if a task is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self.status, TaskStatus::Running { .. })
    }

    /// Returns `true` if the runner is idle (no task, or result already taken).
    pub fn is_idle(&self) -> bool {
        matches!(self.status, TaskStatus::Idle)
    }
}

impl<T: Send + 'static> Default for TaskRunner<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_runner_starts_idle() {
        let runner: TaskRunner<i32> = TaskRunner::new();
        assert!(runner.is_idle());
    }

    #[test]
    fn task_runner_spawn_and_complete() {
        let mut runner: TaskRunner<i32> = TaskRunner::new();
        runner.spawn(|| 42);

        // Spin until done.
        loop {
            match runner.poll() {
                TaskStatus::Completed => break,
                TaskStatus::Failed(msg) => panic!("Task failed: {msg}"),
                _ => std::thread::yield_now(),
            }
        }

        assert_eq!(runner.take_result(), Some(42));
        assert!(runner.is_idle());
    }

    #[test]
    fn task_runner_take_result_returns_none_when_idle() {
        let mut runner: TaskRunner<i32> = TaskRunner::new();
        assert_eq!(runner.take_result(), None);
    }

    #[test]
    fn task_runner_handles_panic() {
        let mut runner: TaskRunner<i32> = TaskRunner::new();
        runner.spawn(|| panic!("intentional test panic"));

        loop {
            match runner.poll() {
                TaskStatus::Completed => panic!("Expected failure, not completion"),
                TaskStatus::Failed(_) => break,
                _ => std::thread::yield_now(),
            }
        }

        assert!(matches!(runner.poll(), TaskStatus::Failed(_)));
    }
}
