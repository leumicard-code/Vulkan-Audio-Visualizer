// ============================================================================
// config.rs — Persistent Configuration (Active Preset, Source, Theme, Paths)
// ============================================================================

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::theme::ThemeMode;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UserConfig {
    pub theme: ThemeMode,
    pub active_shader_id: String,
    pub monitor_source: Option<String>,
    pub auto_cycle_sec: u32,
    pub milkdrop_dirs: Vec<String>,
    pub show_overlay: bool,
}

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            theme: ThemeMode::StudioObsidian,
            active_shader_id: "01_lorenz_attractor".to_string(),
            monitor_source: None,
            auto_cycle_sec: 0,
            milkdrop_dirs: Vec::new(),
            show_overlay: true,
        }
    }
}

pub fn app_config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("vulkan_audio_studio")
}

fn config_file_path() -> PathBuf {
    app_config_dir().join("config.json")
}

impl UserConfig {
    pub fn load() -> Self {
        let path = config_file_path();
        fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let dir = app_config_dir();
        let _ = fs::create_dir_all(&dir);
        let path = config_file_path();
        let tmp = dir.join("config.json.tmp");
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if fs::write(&tmp, json).is_ok() {
                let _ = fs::rename(&tmp, &path);
            }
        }
    }
}
