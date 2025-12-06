//! Configuration module

use crate::storage::{StorageBackend, mock};
use std::path::PathBuf;
use std::time::Duration;

/// Main configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub storage: StorageConfig,
    pub engine: EngineConfig,
    pub api: ApiConfig,
}

impl Config {
    /// Load config from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let storage_mode = std::env::var("STORAGE_MODE").unwrap_or_else(|_| "mock".to_string());

        let storage = match storage_mode.as_str() {
            "mock" => StorageConfig::Mock {
                ebs_root: std::env::var("EBS_ROOT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/tmp/vortex/ebs")),
                s3_root: std::env::var("S3_ROOT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/tmp/vortex/s3")),
                simulate_latency: std::env::var("SIMULATE_LATENCY")
                    .map(|v| v == "true")
                    .unwrap_or(false),
            },
            #[cfg(feature = "aws-storage")]
            "aws" => StorageConfig::Aws {
                ebs_mount_path: std::env::var("EBS_MOUNT_PATH")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/ebs")),
                s3_bucket: std::env::var("S3_BUCKET")
                    .unwrap_or_else(|_| "vortex-bucket".to_string()),
            },
            _ => anyhow::bail!("Unknown storage mode: {}", storage_mode),
        };

        let engine = EngineConfig {
            default_dims: std::env::var("DEFAULT_DIMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(384),
            hnsw_m: std::env::var("HNSW_M")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16),
            hnsw_ef_construction: std::env::var("HNSW_EF_CONSTRUCTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),
            hnsw_ef_search: std::env::var("HNSW_EF_SEARCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            flush_interval: Duration::from_secs(
                std::env::var("FLUSH_INTERVAL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10),
            ),
            backup_interval: Duration::from_secs(
                std::env::var("BACKUP_INTERVAL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(300),
            ),
        };

        let api = ApiConfig {
            host: std::env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("API_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
        };

        Ok(Self { storage, engine, api })
    }
}

/// Storage configuration
#[derive(Debug, Clone)]
pub enum StorageConfig {
    Mock {
        ebs_root: PathBuf,
        s3_root: PathBuf,
        simulate_latency: bool,
    },
    #[cfg(feature = "aws-storage")]
    Aws {
        ebs_mount_path: PathBuf,
        s3_bucket: String,
    },
}

impl StorageConfig {
    /// Create storage backend from config
    pub async fn create_backend(&self) -> anyhow::Result<StorageBackend> {
        match self {
            StorageConfig::Mock {
                ebs_root,
                s3_root,
                simulate_latency,
            } => {
                let config = if *simulate_latency {
                    mock::MockStorageConfig::realistic()
                } else {
                    mock::MockStorageConfig::fast()
                };

                Ok(mock::create_mock_storage(ebs_root, s3_root, config)?)
            }
            #[cfg(feature = "aws-storage")]
            StorageConfig::Aws {
                ebs_mount_path,
                s3_bucket,
            } => {
                use crate::storage::aws;
                Ok(aws::create_aws_storage(ebs_mount_path, s3_bucket).await?)
            }
        }
    }
}

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub default_dims: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub flush_interval: Duration,
    pub backup_interval: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            default_dims: 384,
            hnsw_m: 16,
            hnsw_ef_construction: 500, // Increased from 200 for better recall
            hnsw_ef_search: 200, // Increased from 100 for better search quality
            flush_interval: Duration::from_secs(10),
            backup_interval: Duration::from_secs(300),
        }
    }
}

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}
