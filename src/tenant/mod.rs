//! Tenant management
//!
//! Each tenant has isolated:
//! - WAL (Write-Ahead Log)
//! - VectorStore
//! - HNSW Index
//! - Write Buffer (for recently written vectors not yet in HNSW)

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::storage::{BlockStorage, StorageError, StorageResult};
use crate::vectors::VectorStore;
use crate::wal::Wal;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-tenant state
pub struct TenantState {
    pub tenant_id: u64,
    pub dims: usize,
    pub wal: Wal,
    pub vectors: VectorStore,
    pub hnsw: HnswIndex,
    /// ID mapping: vector_id -> array_index in VectorStore
    pub id_map: RwLock<HashMap<u64, u32>>,
    /// Reverse ID mapping: array_index -> vector_id (for O(1) lookups in search)
    pub reverse_id_map: RwLock<HashMap<u32, u64>>,
    /// Write buffer: vectors written since last HNSW flush
    pub write_buffer: RwLock<Vec<u32>>,
    /// Last sequence flushed to HNSW
    pub last_flushed_seq: RwLock<u64>,
}

impl TenantState {
    /// Create or open a tenant
    pub async fn open(
        tenant_id: u64,
        dims: usize,
        storage: Arc<dyn BlockStorage>,
        hnsw_config: HnswConfig,
    ) -> StorageResult<Self> {
        let base_path = format!("tenant_{}", tenant_id);

        // Ensure directory exists
        storage.create_dir(&base_path).await?;

        let wal_path = format!("{}/wal.log", base_path);
        let vectors_path = format!("{}/vectors.bin", base_path);
        let hnsw_path = format!("{}/index.hnsw", base_path);
        let id_map_path = format!("{}/id_map.bin", base_path);
        let meta_path = format!("{}/meta.json", base_path);

        // Open WAL
        let wal = Wal::open(storage.clone(), &wal_path, dims)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        // Open VectorStore
        let vectors = VectorStore::open(storage.clone(), &vectors_path, dims).await?;

        // Try to load HNSW index
        let hnsw = if storage.exists(&hnsw_path).await? {
            HnswIndex::load(&*storage, &hnsw_path, hnsw_config.clone()).await?
        } else {
            HnswIndex::new(hnsw_config)
        };

        // Try to load id_map and build reverse map
        let id_map: HashMap<u64, u32> = if storage.exists(&id_map_path).await? {
            let data = storage.read(&id_map_path).await?;
            bincode::deserialize(&data)
                .map_err(|e| StorageError::Serialization(e.to_string()))?
        } else {
            HashMap::new()
        };

        // Build reverse map for O(1) lookups
        let reverse_id_map: HashMap<u32, u64> = id_map.iter().map(|(&k, &v)| (v, k)).collect();

        // Try to load metadata
        let last_flushed_seq = if storage.exists(&meta_path).await? {
            let data = storage.read(&meta_path).await?;
            let meta: TenantMeta =
                serde_json::from_slice(&data).map_err(|e| StorageError::Serialization(e.to_string()))?;
            meta.last_flushed_seq
        } else {
            0
        };

        // Replay WAL entries after last_flushed_seq to rebuild write_buffer
        let mut state = Self {
            tenant_id,
            dims,
            wal,
            vectors,
            hnsw,
            id_map: RwLock::new(id_map),
            reverse_id_map: RwLock::new(reverse_id_map),
            write_buffer: RwLock::new(Vec::new()),
            last_flushed_seq: RwLock::new(last_flushed_seq),
        };

        // Replay WAL to recover state
        state.replay_wal(storage.clone()).await?;

        Ok(state)
    }

    /// Replay WAL entries to recover state
    async fn replay_wal(&mut self, storage: Arc<dyn BlockStorage>) -> StorageResult<()> {
        let last_flushed = *self.last_flushed_seq.read();
        let entries = self
            .wal
            .replay_from(last_flushed)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        for entry in entries {
            // Check if already in id_map (scope ensures guard is dropped)
            let already_exists = {
                let id_map = self.id_map.read();
                id_map.contains_key(&entry.vector_id)
            };

            if already_exists {
                continue;
            }

            // Add to vector store if not already there
            let idx = self.vectors.append(&entry.vector).await?;

            // Update id_map and reverse_id_map
            self.id_map.write().insert(entry.vector_id, idx);
            self.reverse_id_map.write().insert(idx, entry.vector_id);

            // Add to write buffer (will be inserted to HNSW on flush)
            self.write_buffer.write().push(idx);
        }

        Ok(())
    }

    /// Upsert vectors
    /// Optimized: Uses batch append instead of individual appends
    pub async fn upsert(&self, vectors: Vec<(u64, Vec<f32>)>) -> StorageResult<UpsertResult> {
        if vectors.is_empty() {
            return Ok(UpsertResult {
                count: 0,
                sequence: self.wal.current_sequence(),
            });
        }

        // Write to WAL first (this is the durability guarantee)
        let wal_entries: Vec<(u64, u64, Vec<f32>)> = vectors
            .iter()
            .map(|(id, vec)| (self.tenant_id, *id, vec.clone()))
            .collect();

        let sequence = self
            .wal
            .append_batch(wal_entries)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        // Filter out duplicates and collect vectors for batch append
        let mut vectors_to_insert: Vec<(u64, Vec<f32>)> = Vec::with_capacity(vectors.len());
        {
            let id_map = self.id_map.read();
            for (vector_id, vector) in vectors {
                if !id_map.contains_key(&vector_id) {
                    vectors_to_insert.push((vector_id, vector));
                }
            }
        }

        if vectors_to_insert.is_empty() {
            return Ok(UpsertResult { count: 0, sequence });
        }

        // Batch append all vectors at once
        let vectors_data: Vec<Vec<f32>> = vectors_to_insert.iter().map(|(_, v)| v.clone()).collect();
        let start_idx = self.vectors.append_batch(&vectors_data).await?;

        // Update id_map, reverse_id_map, and write_buffer in single lock acquisitions
        let count = vectors_to_insert.len();
        {
            let mut id_map = self.id_map.write();
            let mut reverse_id_map = self.reverse_id_map.write();
            let mut write_buffer = self.write_buffer.write();

            for (i, (vector_id, _)) in vectors_to_insert.into_iter().enumerate() {
                let idx = start_idx + i as u32;
                id_map.insert(vector_id, idx);
                reverse_id_map.insert(idx, vector_id);
                write_buffer.push(idx);
            }
        }

        Ok(UpsertResult { count, sequence })
    }

    /// Search for k nearest neighbors
    /// Optimized: Uses reverse_id_map for O(1) lookups instead of O(n)
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Get reverse map once for all lookups
        let reverse_id_map = self.reverse_id_map.read();

        // Search HNSW index
        let hnsw_results = self.hnsw.search(query, k * 2, ef, &self.vectors);
        for (vector_index, similarity) in hnsw_results {
            // O(1) lookup using reverse_id_map
            if let Some(&vector_id) = reverse_id_map.get(&vector_index) {
                results.push(SearchResult {
                    id: vector_id,
                    similarity,
                });
            }
        }

        // Also search write buffer (brute force)
        let write_buffer = self.write_buffer.read();
        for &idx in write_buffer.iter() {
            if let Some(vec) = self.vectors.get(idx) {
                let similarity = cosine_similarity(query, &vec);

                // O(1) lookup using reverse_id_map
                if let Some(&vid) = reverse_id_map.get(&idx) {
                    // Check if already in results
                    if !results.iter().any(|r| r.id == vid) {
                        results.push(SearchResult { id: vid, similarity });
                    }
                }
            }
        }

        // Sort by similarity (descending) and take top k
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        results
    }

    /// Flush write buffer to HNSW index
    pub async fn flush_to_hnsw(&self, storage: &dyn BlockStorage) -> StorageResult<usize> {
        // Extract indices in a separate scope to ensure guard is dropped before await
        let indices: Vec<u32> = {
            let mut write_buffer = self.write_buffer.write();
            write_buffer.drain(..).collect()
        };

        if indices.is_empty() {
            return Ok(0);
        }

        let count = indices.len();

        // Insert each vector into HNSW
        for idx in &indices {
            self.hnsw.insert(*idx, &self.vectors);
        }

        // Save HNSW index
        let hnsw_path = format!("tenant_{}/index.hnsw", self.tenant_id);
        self.hnsw.save(storage, &hnsw_path).await?;

        // Save id_map
        let id_map_path = format!("tenant_{}/id_map.bin", self.tenant_id);
        let id_map_data = bincode::serialize(&*self.id_map.read())
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        storage.write(&id_map_path, &id_map_data).await?;
        storage.sync(&id_map_path).await?;

        // Update last_flushed_seq
        *self.last_flushed_seq.write() = self.wal.current_sequence();

        // Save metadata
        let meta_path = format!("tenant_{}/meta.json", self.tenant_id);
        let meta = TenantMeta {
            last_flushed_seq: *self.last_flushed_seq.read(),
            vector_count: self.vectors.count(),
        };
        let meta_data =
            serde_json::to_vec(&meta).map_err(|e| StorageError::Serialization(e.to_string()))?;
        storage.write(&meta_path, &meta_data).await?;
        storage.sync(&meta_path).await?;

        Ok(count)
    }

    /// Get stats
    pub fn stats(&self) -> TenantStats {
        TenantStats {
            tenant_id: self.tenant_id,
            vector_count: self.vectors.count(),
            hnsw_nodes: self.hnsw.len() as u64,
            write_buffer_size: self.write_buffer.read().len(),
            wal_sequence: self.wal.current_sequence(),
            last_flushed_seq: *self.last_flushed_seq.read(),
        }
    }
}

/// Cosine similarity (dot product for normalized vectors)
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Upsert result
#[derive(Debug)]
pub struct UpsertResult {
    pub count: usize,
    pub sequence: u64,
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub similarity: f32,
}

/// Tenant metadata
#[derive(serde::Serialize, serde::Deserialize)]
struct TenantMeta {
    last_flushed_seq: u64,
    vector_count: u64,
}

/// Tenant statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct TenantStats {
    pub tenant_id: u64,
    pub vector_count: u64,
    pub hnsw_nodes: u64,
    pub write_buffer_size: usize,
    pub wal_sequence: u64,
    pub last_flushed_seq: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{MockBlockStorage, MockStorageConfig};

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[tokio::test]
    async fn test_tenant_upsert_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = HnswConfig::new(4);

        let tenant = TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap();

        // Insert vectors
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| {
                let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
                (100 + i as u64, v)
            })
            .collect();

        let result = tenant.upsert(vectors).await.unwrap();
        assert_eq!(result.count, 10);

        // Search (should find in write buffer)
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = tenant.search(&query, 5, None);
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_flush_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = HnswConfig::new(4);

        let tenant = TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap();

        // Insert vectors
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| {
                let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
                (100 + i as u64, v)
            })
            .collect();

        tenant.upsert(vectors).await.unwrap();

        // Flush to HNSW
        let flushed = tenant.flush_to_hnsw(&*storage).await.unwrap();
        assert_eq!(flushed, 10);

        // Search (should find in HNSW)
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = tenant.search(&query, 5, None);
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_recovery() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().to_path_buf();
        let config = HnswConfig::new(4);

        // Create and populate tenant
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
                .await
                .unwrap();

            let vectors: Vec<(u64, Vec<f32>)> = (0..5)
                .map(|i| {
                    let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                    (100 + i as u64, v)
                })
                .collect();

            tenant.upsert(vectors).await.unwrap();

            // Flush to save id_map and metadata for recovery
            tenant.flush_to_hnsw(&*storage).await.unwrap();
        }

        // Reopen and verify recovery
        {
            let storage =
                Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
            let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
                .await
                .unwrap();

            let stats = tenant.stats();
            assert_eq!(stats.vector_count, 5);

            // Should be able to search
            let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
            let results = tenant.search(&query, 3, None);
            assert!(!results.is_empty());
        }
    }
}
