use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Application settings — serialized as TOML
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppSettings {
    pub appearance: AppearanceSettings,
    pub compute: ComputeSettings,
    pub paths: PathSettings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppearanceSettings {
    pub dark_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeSettings {
    pub multithreading: bool,
    pub gpu_enabled: bool,
    pub thread_count: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathSettings {
    pub kinematics_data_path: PathBuf,
    pub fluid_data_path: PathBuf,
    pub audio_data_path: PathBuf,
    pub mesh_output_dir: PathBuf,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            appearance: AppearanceSettings { dark_mode: true },
            compute: ComputeSettings {
                multithreading: true,
                gpu_enabled: true,
                thread_count: None,
            },
            paths: PathSettings {
                kinematics_data_path: PathBuf::from("data/kinematics"),
                fluid_data_path: PathBuf::from("data/fluid"),
                audio_data_path: PathBuf::from("data/audio"),
                mesh_output_dir: PathBuf::from("data/mesh"),
            },
        }
    }
}

impl AppSettings {
    /// Load settings from a TOML file. Returns defaults if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self, crate::error::SimuError> {
        if !path.exists() {
            tracing::info!("Settings file not found at {}, using defaults", path.display());
            return Ok(Self::default());
        }

        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content)
            .map_err(|e| crate::error::SimuError::Config(format!("Failed to parse settings: {e}")))
    }

    /// Save settings to a TOML file.
    pub fn save(&self, path: &Path) -> Result<(), crate::error::SimuError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)
            .map_err(|e| crate::error::SimuError::Config(format!("Failed to serialize settings: {e}")))?;

        std::fs::write(path, content)?;
        tracing::info!("Settings saved to {}", path.display());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_serialize_roundtrip() {
        let settings = AppSettings::default();
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let parsed: AppSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(settings, parsed);
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let result = AppSettings::load(Path::new("/nonexistent/settings.toml"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), AppSettings::default());
    }
}
