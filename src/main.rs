use anyhow::Result;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!("SimuCADSuite v{} starting", env!("CARGO_PKG_VERSION"));

    // Load application settings
    let settings_path = dirs_or_default().join("settings.toml");
    let settings = simucad_core::settings::AppSettings::load(&settings_path)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to load settings: {e}, using defaults");
            simucad_core::settings::AppSettings::default()
        });

    tracing::info!(
        gpu = settings.compute.gpu_enabled,
        multithreading = settings.compute.multithreading,
        dark_mode = settings.appearance.dark_mode,
        "Configuration loaded"
    );

    // Configure eframe native options
    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("SimuCADSuite")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    // Launch the GUI
    eframe::run_native(
        "SimuCADSuite",
        native_options,
        Box::new(move |cc| Ok(Box::new(simucad_gui::app::SimuApp::new(cc, settings)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}

/// Returns the application data directory, creating it if needed.
fn dirs_or_default() -> std::path::PathBuf {
    let dir = dirs_path();
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    dir
}

/// Platform-appropriate config directory.
fn dirs_path() -> std::path::PathBuf {
    if let Some(config) = dirs_impl() {
        config.join("SimuCADSuite")
    } else {
        std::path::PathBuf::from(".")
    }
}

#[cfg(target_os = "macos")]
fn dirs_impl() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join("Library/Application Support"))
}

#[cfg(target_os = "linux")]
fn dirs_impl() -> Option<std::path::PathBuf> {
    std::env::var("XDG_CONFIG_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(|h| std::path::PathBuf::from(h).join(".config")))
}

#[cfg(target_os = "windows")]
fn dirs_impl() -> Option<std::path::PathBuf> {
    std::env::var("APPDATA").ok().map(std::path::PathBuf::from)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn dirs_impl() -> Option<std::path::PathBuf> {
    None
}
