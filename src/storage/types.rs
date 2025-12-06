//! Storage type definitions

use serde::{Deserialize, Serialize};

/// Object info from list operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectInfo {
    pub key: String,
    pub size: u64,
    pub last_modified: chrono::DateTime<chrono::Utc>,
}
