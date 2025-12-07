use std::{
    env, fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use common::{SimConfig, SimConfigRaw};

pub struct Config {
    pub name: Arc<str>,
    pub data_dirs: Vec<PathBuf>,
    pub save: PathBuf,
    pub chunk_load_parallelism: u32,
    pub server: Option<SocketAddr>,
    pub local_simulation: SimConfig,
}

impl Config {
    pub fn load(dirs: &directories::ProjectDirs) -> Self {
        Self::load_with_raw(dirs).0
    }

    pub fn load_with_raw(dirs: &directories::ProjectDirs) -> (Self, RawConfig) {
        let path = Self::config_path(dirs);
        let raw = Self::read_raw_config(&path);
        let config = Self::from_raw(dirs, &raw);
        (config, raw)
    }

    pub fn config_path(dirs: &directories::ProjectDirs) -> PathBuf {
        dirs.config_dir().join("client.toml")
    }

    fn read_raw_config(path: &Path) -> RawConfig {
        match fs::read(path) {
            Ok(data) => {
                info!("found config at {}", path.display());
                match std::str::from_utf8(&data)
                    .map_err(anyhow::Error::from)
                    .and_then(|s| toml::from_str(s).map_err(anyhow::Error::from))
                {
                    Ok(raw) => raw,
                    Err(err) => {
                        error!("failed to parse config: {}", err);
                        RawConfig::default()
                    }
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
                info!("{} not found, using defaults", path.display());
                RawConfig::default()
            }
            Err(err) => {
                error!("failed to read config: {}: {}", path.display(), err);
                RawConfig::default()
            }
        }
    }

    fn from_raw(dirs: &directories::ProjectDirs, raw: &RawConfig) -> Self {
        let mut data_dirs = Vec::new();
        if let Some(dir) = raw.data_dir.clone() {
            data_dirs.push(dir);
        }
        data_dirs.push(dirs.data_dir().into());
        if let Ok(path) = env::current_exe()
            && let Some(dir) = path.parent()
        {
            data_dirs.push(dir.into());
        }
        #[cfg(feature = "use-repo-assets")]
        {
            data_dirs.push(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .join("assets"),
            );
        }

        Config {
            name: raw
                .name
                .clone()
                .unwrap_or_else(|| Arc::<str>::from(whoami::username())),
            data_dirs,
            save: raw.save.clone().unwrap_or_else(|| "default.save".into()),
            chunk_load_parallelism: raw.chunk_load_parallelism.unwrap_or(256),
            server: raw.server,
            local_simulation: SimConfig::from_raw(&raw.local_simulation),
        }
    }

    pub fn save_raw_config(path: &Path, raw: &RawConfig) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let serialized = toml::to_string_pretty(raw)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(path, serialized)?;
        Ok(())
    }

    pub fn find_asset(&self, path: &Path) -> Option<PathBuf> {
        for dir in &self.data_dirs {
            let full_path = dir.join(path);
            if full_path.exists() {
                debug!(path = ?path.display(), dir = ?dir.display(), "found asset");
                return Some(full_path);
            }
        }
        None
    }
}

/// Data as parsed directly out of the config file
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RawConfig {
    pub name: Option<Arc<str>>,
    pub data_dir: Option<PathBuf>,
    pub save: Option<PathBuf>,
    pub chunk_load_parallelism: Option<u32>,
    pub server: Option<SocketAddr>,
    #[serde(default)]
    pub local_simulation: SimConfigRaw,
}
