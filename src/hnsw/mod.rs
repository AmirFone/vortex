//! HNSW (Hierarchical Navigable Small World) index
//!
//! Key features:
//! - External vector storage (vectors not stored in graph)
//! - Disk-based persistence
//! - Lock-free reads, locked writes

pub mod node;
pub mod search;
pub mod insert;
pub mod persistence;

use crate::index::HnswParams;
use crate::storage::BlockStorage;
use crate::vectors::VectorStore;
use node::HnswNode;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// HNSW configuration - type alias for HnswParams (single source of truth)
///
/// Use `HnswParams` directly for new code. This alias is provided for
/// backward compatibility within the hnsw module.
pub type HnswConfig = HnswParams;

/// HNSW index with external vector storage
pub struct HnswIndex {
    config: HnswConfig,
    /// All nodes in the graph
    nodes: RwLock<Vec<HnswNode>>,
    /// Entry point (node with highest level)
    entry_point: AtomicU32,
    /// Maximum level in the graph
    max_level: AtomicUsize,
}

impl HnswIndex {
    /// Create new empty index
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(Vec::new()),
            entry_point: AtomicU32::new(u32::MAX),
            max_level: AtomicUsize::new(0),
        }
    }

    /// Insert a new vector into the index
    /// `vector_index` is the index in the external VectorStore
    pub fn insert(&self, vector_index: u32, vectors: &VectorStore) {
        insert::insert_node(self, vector_index, vectors);
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
    ) -> Vec<(u32, f32)> {
        search::search_knn(self, query, k, ef.unwrap_or(self.config.ef_search), vectors)
    }

    /// Save index to storage
    pub async fn save(
        &self,
        storage: &dyn BlockStorage,
        path: &str,
    ) -> crate::storage::StorageResult<()> {
        persistence::save_index(self, storage, path).await
    }

    /// Load index from storage
    pub async fn load(
        storage: &dyn BlockStorage,
        path: &str,
        config: HnswConfig,
    ) -> crate::storage::StorageResult<Self> {
        persistence::load_index(storage, path, config).await
    }

    /// Number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    // Accessors for internal modules
    pub(crate) fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub(crate) fn nodes(&self) -> &RwLock<Vec<HnswNode>> {
        &self.nodes
    }

    pub(crate) fn entry_point(&self) -> u32 {
        self.entry_point.load(Ordering::SeqCst)
    }

    pub(crate) fn set_entry_point(&self, ep: u32) {
        self.entry_point.store(ep, Ordering::SeqCst);
    }

    pub(crate) fn max_level(&self) -> usize {
        self.max_level.load(Ordering::SeqCst)
    }

    pub(crate) fn set_max_level(&self, level: usize) {
        self.max_level.store(level, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{MockBlockStorage, MockStorageConfig};
    use std::sync::Arc;

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[tokio::test]
    async fn test_hnsw_insert_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let vectors = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        let config = HnswConfig::with_m(4);
        let index = HnswIndex::new(config);

        // Insert some vectors
        for i in 0..10 {
            let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
            let idx = vectors.append(&v).await.unwrap();
            index.insert(idx, &vectors);
        }

        assert_eq!(index.len(), 10);

        // Search
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = index.search(&query, 5, None, &vectors);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_hnsw_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().to_path_buf();

        let config = HnswConfig::with_m(4);

        // Create and populate index
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let index = HnswIndex::new(config.clone());

            for i in 0..5 {
                let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                let idx = vectors.append(&v).await.unwrap();
                index.insert(idx, &vectors);
            }

            index.save(&*storage, "index.hnsw").await.unwrap();
        }

        // Reload and verify
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let index = HnswIndex::load(&*storage, "index.hnsw", config.clone())
                .await
                .unwrap();

            assert_eq!(index.len(), 5);

            let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
            let results = index.search(&query, 3, None, &vectors);
            assert!(!results.is_empty());
        }
    }
}
