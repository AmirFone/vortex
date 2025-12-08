//! DiskANN Index Implementation
//!
//! Provides a disk-based ANN index using the diskann-rs crate.
//! This implementation offers significantly lower memory usage compared to HNSW,
//! making it suitable for large-scale multi-tenant deployments.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use anndists::dist::DistCosine;
use diskann_rs::{DiskAnnParams as DiskAnnRsParams, IncrementalConfig, IncrementalDiskANN};

use crate::storage::{BlockStorage, StorageResult};
use crate::vectors::VectorStore;

use super::config::DiskAnnParams;
use super::AnnIndex;

/// DiskANN-based ANN index
///
/// Uses IncrementalDiskANN for efficient incremental inserts while maintaining
/// disk-based storage for low memory usage.
pub struct DiskAnnIndex {
    /// The underlying DiskANN index (wrapped for incremental operations)
    inner: RwLock<Option<IncrementalDiskANN<DistCosine>>>,
    /// Parameters for the index
    params: DiskAnnParams,
    /// Path for index storage
    index_path: RwLock<Option<PathBuf>>,
    /// Cached vectors for building/rebuilding (diskann needs direct access)
    vectors_cache: RwLock<Vec<Vec<f32>>>,
    /// Mapping from vortex index to diskann index
    index_mapping: RwLock<HashMap<u32, usize>>,
    /// Reverse mapping from diskann index to vortex index
    reverse_mapping: RwLock<Vec<u32>>,
    /// Count of vectors in the index
    count: AtomicUsize,
}

impl fmt::Debug for DiskAnnIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskAnnIndex")
            .field("len", &self.count.load(Ordering::Relaxed))
            .field("params", &self.params)
            .finish()
    }
}

impl DiskAnnIndex {
    /// Create a new DiskANN index with the given parameters
    pub fn new(params: DiskAnnParams) -> Self {
        Self {
            inner: RwLock::new(None),
            params,
            index_path: RwLock::new(None),
            vectors_cache: RwLock::new(Vec::new()),
            index_mapping: RwLock::new(HashMap::new()),
            reverse_mapping: RwLock::new(Vec::new()),
            count: AtomicUsize::new(0),
        }
    }

    /// Load index from storage
    pub async fn load(
        storage: &dyn BlockStorage,
        path: &str,
        params: DiskAnnParams,
    ) -> StorageResult<Self> {
        // Try to load the index metadata
        let meta_path = format!("{}.meta", path);

        // Read metadata (contains mappings and vector cache)
        let meta_data = storage.read(&meta_path).await?;
        let (index_mapping, reverse_mapping, vectors_cache): (
            HashMap<u32, usize>,
            Vec<u32>,
            Vec<Vec<f32>>,
        ) = bincode::deserialize(&meta_data).map_err(|e| {
            crate::storage::StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to deserialize DiskANN metadata: {}", e),
            ))
        })?;

        let count = reverse_mapping.len();

        // Create index from cached vectors if we have any
        let inner = if !vectors_cache.is_empty() {
            // Create a temporary file for the index
            let temp_dir = std::env::temp_dir();
            let index_path = temp_dir.join(format!("diskann_load_{}.index", rand::random::<u64>()));

            // Build the index from cached vectors
            let index = IncrementalDiskANN::<DistCosine>::build_with_config(
                &vectors_cache,
                index_path.to_str().unwrap(),
                DiskAnnIndex::to_incremental_config(&params),
            )
            .map_err(|e| {
                crate::storage::StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to build DiskANN index: {}", e),
                ))
            })?;
            Some(index)
        } else {
            None
        };

        Ok(Self {
            inner: RwLock::new(inner),
            params,
            index_path: RwLock::new(None),
            vectors_cache: RwLock::new(vectors_cache),
            index_mapping: RwLock::new(index_mapping),
            reverse_mapping: RwLock::new(reverse_mapping),
            count: AtomicUsize::new(count),
        })
    }

    /// Convert our params to diskann-rs IncrementalConfig
    fn to_incremental_config(params: &DiskAnnParams) -> IncrementalConfig {
        IncrementalConfig {
            delta_threshold: 10_000,
            tombstone_ratio_threshold: 0.1,
            delta_params: DiskAnnRsParams {
                max_degree: params.max_degree,
                build_beam_width: params.build_beam_width,
                alpha: params.alpha,
            },
        }
    }

    /// Build/rebuild the index from cached vectors
    fn rebuild_internal(&self) -> Result<(), String> {
        let vectors = self.vectors_cache.read();
        if vectors.is_empty() {
            *self.inner.write() = None;
            return Ok(());
        }

        // Create a temporary file for the index
        let temp_dir = std::env::temp_dir();
        let index_path = temp_dir.join(format!("diskann_{}.index", rand::random::<u64>()));

        let index = IncrementalDiskANN::<DistCosine>::build_with_config(
            &vectors,
            index_path.to_str().unwrap(),
            Self::to_incremental_config(&self.params),
        )
        .map_err(|e| format!("Failed to build DiskANN index: {}", e))?;

        *self.inner.write() = Some(index);
        *self.index_path.write() = Some(index_path);

        Ok(())
    }
}

#[async_trait]
impl AnnIndex for DiskAnnIndex {
    fn insert(&self, vector_index: u32, vectors: &VectorStore) {
        // Get the vector data
        if let Some(vector) = vectors.get(vector_index) {
            let mut cache = self.vectors_cache.write();
            let mut mapping = self.index_mapping.write();
            let mut reverse = self.reverse_mapping.write();

            // Check if already inserted
            if mapping.contains_key(&vector_index) {
                return;
            }

            // Add to cache
            let diskann_idx = cache.len();
            cache.push(vector.to_vec());
            mapping.insert(vector_index, diskann_idx);
            reverse.push(vector_index);

            self.count.fetch_add(1, Ordering::Relaxed);

            // For incremental index, we try to add to the existing index
            drop(cache);
            drop(mapping);
            drop(reverse);

            // Try to add to existing index, or mark for rebuild
            let mut inner = self.inner.write();
            if let Some(ref mut index) = *inner {
                // IncrementalDiskANN supports adding vectors
                if let Some(v) = vectors.get(vector_index) {
                    let _ = index.add_vectors(&[v.to_vec()]);
                }
            } else if self.count.load(Ordering::Relaxed) >= 10 {
                // Build initial index when we have enough vectors
                drop(inner);
                let _ = self.rebuild_internal();
            }
        }
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        _vectors: &VectorStore,
    ) -> Vec<(u32, f32)> {
        let inner = self.inner.read();

        if let Some(ref index) = *inner {
            // Use ef as beam width if provided
            let beam_width = ef.unwrap_or(self.params.search_beam_width);

            // Search the DiskANN index - returns Vec<(u32, f32)>
            let results = index.search_with_dists(query, k, beam_width);

            // Map DiskANN indices back to vortex indices
            let reverse = self.reverse_mapping.read();
            results
                .into_iter()
                .filter_map(|(diskann_idx, dist)| {
                    reverse
                        .get(diskann_idx as usize)
                        .map(|&vortex_idx| (vortex_idx, dist))
                })
                .collect()
        } else {
            // Fallback to brute force if index not built yet
            let cache = self.vectors_cache.read();
            let reverse = self.reverse_mapping.read();

            if cache.is_empty() {
                return Vec::new();
            }

            // Compute cosine distances
            let mut distances: Vec<(u32, f32)> = cache
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    let dist = cosine_distance(query, v);
                    let vortex_idx = *reverse.get(idx).unwrap_or(&(idx as u32));
                    (vortex_idx, dist)
                })
                .collect();

            // Sort by distance and take top k
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
            distances
        }
    }

    async fn save(&self, storage: &dyn BlockStorage, path: &str) -> StorageResult<()> {
        let meta_path = format!("{}.meta", path);

        // Serialize metadata
        let index_mapping = self.index_mapping.read().clone();
        let reverse_mapping = self.reverse_mapping.read().clone();
        let vectors_cache = self.vectors_cache.read().clone();

        let meta_data = bincode::serialize(&(index_mapping, reverse_mapping, vectors_cache))
            .map_err(|e| {
                crate::storage::StorageError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to serialize DiskANN metadata: {}", e),
                ))
            })?;

        storage.write(&meta_path, &meta_data).await?;
        storage.sync(&meta_path).await?;

        Ok(())
    }

    fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    fn is_empty(&self) -> bool {
        self.count.load(Ordering::Relaxed) == 0
    }

    async fn rebuild(&self, indices: &[u32], vectors: &VectorStore) -> StorageResult<()> {
        // Clear existing data
        {
            let mut cache = self.vectors_cache.write();
            let mut mapping = self.index_mapping.write();
            let mut reverse = self.reverse_mapping.write();

            cache.clear();
            mapping.clear();
            reverse.clear();
        }

        // Add all vectors
        for &idx in indices {
            if let Some(v) = vectors.get(idx) {
                let mut cache = self.vectors_cache.write();
                let mut mapping = self.index_mapping.write();
                let mut reverse = self.reverse_mapping.write();

                let diskann_idx = cache.len();
                cache.push(v.to_vec());
                mapping.insert(idx, diskann_idx);
                reverse.push(idx);
            }
        }

        self.count.store(indices.len(), Ordering::Relaxed);

        // Rebuild the index
        self.rebuild_internal().map_err(|e| {
            crate::storage::StorageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?;

        Ok(())
    }

    fn supports_incremental_insert(&self) -> bool {
        // IncrementalDiskANN supports incremental inserts
        true
    }

    fn index_type_name(&self) -> &'static str {
        "DiskANN"
    }
}

/// Compute cosine distance between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        1.0 - (dot / denom)
    } else {
        1.0
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
    async fn test_diskann_insert_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let vectors = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        let params = DiskAnnParams::new(4);
        let index = DiskAnnIndex::new(params);

        // Insert some vectors
        for i in 0..20 {
            let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
            let idx = vectors.append(&v).await.unwrap();
            index.insert(idx, &vectors);
        }

        assert_eq!(index.len(), 20);
        assert!(!index.is_empty());

        // Search
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = index.search(&query, 5, None, &vectors);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_diskann_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().to_path_buf();

        let params = DiskAnnParams::new(4);

        // Create and populate index
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let index = DiskAnnIndex::new(params.clone());

            for i in 0..15 {
                let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                let idx = vectors.append(&v).await.unwrap();
                index.insert(idx, &vectors);
            }

            index.save(&*storage, "index.diskann").await.unwrap();
        }

        // Reload and verify
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let vectors = VectorStore::open(storage.clone(), "vectors.bin", 4)
                .await
                .unwrap();
            let index = DiskAnnIndex::load(&*storage, "index.diskann", params.clone())
                .await
                .unwrap();

            assert_eq!(index.len(), 15);
            assert_eq!(index.index_type_name(), "DiskANN");

            let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
            let results = index.search(&query, 3, None, &vectors);
            assert!(!results.is_empty());
        }
    }

    #[tokio::test]
    async fn test_diskann_rebuild() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let vectors = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        // First, add vectors to the store
        let mut indices = Vec::new();
        for i in 0..15 {
            let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            let idx = vectors.append(&v).await.unwrap();
            indices.push(idx);
        }

        // Create index and rebuild from indices
        let params = DiskAnnParams::new(4);
        let index = DiskAnnIndex::new(params);
        index.rebuild(&indices, &vectors).await.unwrap();

        assert_eq!(index.len(), 15);
    }
}
