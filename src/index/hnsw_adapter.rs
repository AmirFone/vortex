//! HNSW Adapter
//!
//! Wraps the existing HnswIndex to implement the AnnIndex trait.

use async_trait::async_trait;
use std::fmt;

use crate::hnsw::HnswIndex;
use crate::storage::{BlockStorage, StorageResult};
use crate::vectors::VectorStore;

use super::config::HnswParams;
use super::AnnIndex;

/// Adapter that wraps HnswIndex to implement AnnIndex trait
pub struct HnswAdapter {
    inner: HnswIndex,
    params: HnswParams,
}

impl fmt::Debug for HnswAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HnswAdapter")
            .field("len", &self.inner.len())
            .field("params", &self.params)
            .finish()
    }
}

impl HnswAdapter {
    /// Create a new HNSW adapter with the given parameters
    pub fn new(params: HnswParams) -> Self {
        // HnswConfig is a type alias for HnswParams, so we can use clone directly
        Self {
            inner: HnswIndex::new(params.clone()),
            params,
        }
    }

    /// Create from an existing HnswIndex
    pub fn from_index(index: HnswIndex, params: HnswParams) -> Self {
        Self {
            inner: index,
            params,
        }
    }

    /// Get the underlying HnswIndex (for backwards compatibility)
    pub fn inner(&self) -> &HnswIndex {
        &self.inner
    }

    /// Get the parameters
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Load from storage
    pub async fn load(
        storage: &dyn BlockStorage,
        path: &str,
        params: HnswParams,
    ) -> StorageResult<Self> {
        // HnswConfig is a type alias for HnswParams, so we can use clone directly
        let index = HnswIndex::load(storage, path, params.clone()).await?;
        Ok(Self {
            inner: index,
            params,
        })
    }
}

#[async_trait]
impl AnnIndex for HnswAdapter {
    fn insert(&self, vector_index: u32, vectors: &VectorStore) {
        self.inner.insert(vector_index, vectors);
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
    ) -> Vec<(u32, f32)> {
        self.inner.search(query, k, ef, vectors)
    }

    async fn save(&self, storage: &dyn BlockStorage, path: &str) -> StorageResult<()> {
        self.inner.save(storage, path).await
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    async fn rebuild(&self, indices: &[u32], vectors: &VectorStore) -> StorageResult<()> {
        // HNSW supports incremental inserts, so we just insert each vector
        for &idx in indices {
            self.inner.insert(idx, vectors);
        }
        Ok(())
    }

    fn supports_incremental_insert(&self) -> bool {
        true
    }

    fn index_type_name(&self) -> &'static str {
        "HNSW"
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
    async fn test_hnsw_adapter_insert_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let vectors = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        let adapter = HnswAdapter::new(HnswParams::with_m(4));

        // Insert some vectors
        for i in 0..10 {
            let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
            let idx = vectors.append(&v).await.unwrap();
            adapter.insert(idx, &vectors);
        }

        assert_eq!(adapter.len(), 10);
        assert!(!adapter.is_empty());

        // Search
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = adapter.search(&query, 5, None, &vectors);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_hnsw_adapter_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().to_path_buf();

        let params = HnswParams::with_m(4);

        // Create and populate index
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let adapter = HnswAdapter::new(params.clone());

            for i in 0..5 {
                let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                let idx = vectors.append(&v).await.unwrap();
                adapter.insert(idx, &vectors);
            }

            adapter.save(&*storage, "index.hnsw").await.unwrap();
        }

        // Reload and verify
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let adapter = HnswAdapter::load(&*storage, "index.hnsw", params.clone())
                .await
                .unwrap();

            assert_eq!(adapter.len(), 5);
            assert_eq!(adapter.index_type_name(), "HNSW");
            assert!(adapter.supports_incremental_insert());

            let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
            let results = adapter.search(&query, 3, None, &vectors);
            assert!(!results.is_empty());
        }
    }

    #[tokio::test]
    async fn test_hnsw_adapter_rebuild() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let vectors = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        // First, add vectors to the store
        let mut indices = Vec::new();
        for i in 0..5 {
            let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            let idx = vectors.append(&v).await.unwrap();
            indices.push(idx);
        }

        // Create adapter and rebuild from indices
        let adapter = HnswAdapter::new(HnswParams::with_m(4));
        adapter.rebuild(&indices, &vectors).await.unwrap();

        assert_eq!(adapter.len(), 5);
    }
}
