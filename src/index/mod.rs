//! ANN Index abstraction layer
//!
//! This module provides a trait-based abstraction over different ANN (Approximate Nearest Neighbor)
//! index implementations, allowing the system to work with multiple backends:
//!
//! - **HNSW** (default): In-memory Hierarchical Navigable Small World graph
//! - **DiskANN** (optional): Disk-based Vamana graph for large-scale deployments
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     TenantState                              │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │              Arc<dyn AnnIndex>                          ││
//! │  │  ┌───────────────────┐  ┌───────────────────────────┐  ││
//! │  │  │   HnswAdapter     │  │   DiskAnnIndex (optional) │  ││
//! │  │  │   (wraps HNSW)    │  │   (disk-based)            │  ││
//! │  │  └───────────────────┘  └───────────────────────────┘  ││
//! │  └─────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use vortex::index::{AnnIndex, AnnIndexConfig, create_index};
//!
//! // Create HNSW index (default)
//! let config = AnnIndexConfig::hnsw();
//! let index: Arc<dyn AnnIndex> = create_index(config);
//!
//! // Or with DiskANN (requires feature flag)
//! #[cfg(feature = "diskann-index")]
//! let config = AnnIndexConfig::diskann(384);
//! ```

mod config;
mod hnsw_adapter;
mod r#trait;

#[cfg(feature = "diskann-index")]
mod diskann;

pub use config::{AnnIndexConfig, HnswParams};
pub use hnsw_adapter::HnswAdapter;
pub use r#trait::{AnnIndex, AnnIndexExt};

#[cfg(feature = "diskann-index")]
pub use config::DiskAnnParams;
#[cfg(feature = "diskann-index")]
pub use diskann::DiskAnnIndex;

use std::sync::Arc;

use crate::storage::{BlockStorage, StorageResult};

/// Create a new ANN index from configuration
pub fn create_index(config: AnnIndexConfig) -> Arc<dyn AnnIndex> {
    match config {
        AnnIndexConfig::Hnsw(params) => Arc::new(HnswAdapter::new(params)),
        #[cfg(feature = "diskann-index")]
        AnnIndexConfig::DiskAnn(params) => Arc::new(DiskAnnIndex::new(params)),
    }
}

/// Load an ANN index from storage
pub async fn load_index(
    config: AnnIndexConfig,
    storage: &dyn BlockStorage,
    path: &str,
) -> StorageResult<Arc<dyn AnnIndex>> {
    match config {
        AnnIndexConfig::Hnsw(params) => {
            let adapter = HnswAdapter::load(storage, path, params).await?;
            Ok(Arc::new(adapter))
        }
        #[cfg(feature = "diskann-index")]
        AnnIndexConfig::DiskAnn(params) => {
            let index = DiskAnnIndex::load(storage, path, params).await?;
            Ok(Arc::new(index))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_hnsw_index() {
        let config = AnnIndexConfig::hnsw();
        let index = create_index(config);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.index_type_name(), "HNSW");
        assert!(index.supports_incremental_insert());
    }

    #[test]
    fn test_config_default() {
        let config = AnnIndexConfig::default();
        assert_eq!(config.index_type_name(), "HNSW");
    }

    #[test]
    fn test_hnsw_params_with_m() {
        let config = AnnIndexConfig::hnsw_with_m(32);
        if let AnnIndexConfig::Hnsw(params) = config {
            assert_eq!(params.m, 32);
            assert_eq!(params.m_max0, 64);
        } else {
            panic!("Expected HNSW config");
        }
    }
}
