//! Tenant management
//!
//! Each tenant has isolated:
//! - WAL (Write-Ahead Log)
//! - VectorStore
//! - ANN Index (HNSW, DiskANN, etc.)
//! - Write Buffer (for recently written vectors not yet in index)

use crate::index::{AnnIndex, AnnIndexConfig, HnswParams};
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
    /// ANN index (HNSW, DiskANN, etc.)
    pub index: Arc<dyn AnnIndex>,
    /// Index configuration for persistence/reload
    index_config: AnnIndexConfig,
    /// ID mapping: vector_id -> array_index in VectorStore
    pub id_map: RwLock<HashMap<u64, u32>>,
    /// Reverse ID mapping: array_index -> vector_id (for O(1) lookups in search)
    pub reverse_id_map: RwLock<HashMap<u32, u64>>,
    /// Write buffer: vectors written since last index flush
    pub write_buffer: RwLock<Vec<u32>>,
    /// Last sequence flushed to index
    pub last_flushed_seq: RwLock<u64>,
}

impl TenantState {
    /// Create or open a tenant
    pub async fn open(
        tenant_id: u64,
        dims: usize,
        storage: Arc<dyn BlockStorage>,
        index_config: AnnIndexConfig,
    ) -> StorageResult<Self> {
        let base_path = format!("tenant_{}", tenant_id);

        // Ensure directory exists
        storage.create_dir(&base_path).await?;

        let wal_path = format!("{}/wal.log", base_path);
        let vectors_path = format!("{}/vectors.bin", base_path);
        let index_path = format!("{}/index.hnsw", base_path);
        let id_map_path = format!("{}/id_map.bin", base_path);
        let meta_path = format!("{}/meta.json", base_path);

        // Open WAL
        let wal = Wal::open(storage.clone(), &wal_path, dims)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        // Open VectorStore
        let vectors = VectorStore::open(storage.clone(), &vectors_path, dims).await?;

        // Try to load index
        let index: Arc<dyn AnnIndex> = if storage.exists(&index_path).await? {
            crate::index::load_index(index_config.clone(), &*storage, &index_path).await?
        } else {
            crate::index::create_index(index_config.clone())
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
            index,
            index_config,
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
    ///
    /// IMPORTANT: This function must NOT re-append vectors to VectorStore.
    /// Vectors are already persisted in VectorStore at indices [id_map.len()..count].
    /// We only need to rebuild the in-memory id_map, reverse_id_map, and write_buffer.
    async fn replay_wal(&mut self, _storage: Arc<dyn BlockStorage>) -> StorageResult<()> {
        let id_map_len = self.id_map.read().len();
        let vector_count = self.vectors.count() as usize;

        // Determine where to start replaying from:
        // - If id_map is empty but VectorStore has vectors, id_map was likely deleted
        //   → replay from the beginning (sequence 0) to rebuild all mappings
        // - Otherwise, replay from last_flushed_seq to rebuild only unflushed vectors
        let replay_from_seq = if id_map_len == 0 && vector_count > 0 {
            0 // id_map was deleted, rebuild from scratch
        } else {
            *self.last_flushed_seq.read()
        };

        let entries = self
            .wal
            .replay_from(replay_from_seq)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        // Track the next index for vectors being mapped
        // If rebuilding from scratch, start at 0; otherwise start at id_map.len()
        let mut current_idx = id_map_len as u32;

        for entry in entries {
            // Check if already in id_map (was flushed before crash or already processed)
            let already_exists = {
                let id_map = self.id_map.read();
                id_map.contains_key(&entry.vector_id)
            };

            if already_exists {
                continue;
            }

            // Vector is NOT in id_map but IS already in VectorStore
            // DON'T append - just rebuild the in-memory mappings
            let idx = current_idx;

            // Update id_map and reverse_id_map
            self.id_map.write().insert(entry.vector_id, idx);
            self.reverse_id_map.write().insert(idx, entry.vector_id);

            // Add to write buffer only if this vector wasn't flushed to index
            // (i.e., if idx >= index node count, it's still pending)
            let index_len = self.index.len() as u32;
            if idx >= index_len {
                self.write_buffer.write().push(idx);
            }

            current_idx += 1;
        }

        Ok(())
    }

    /// Upsert vectors
    ///
    /// IMPORTANT: Uses double-check pattern to handle concurrent upserts safely.
    /// - First check with read lock (fast path, filters obvious duplicates)
    /// - Append to storage (async operation, no locks held)
    /// - Second check with write lock (ensures atomicity, handles races)
    ///
    /// Race handling: If two concurrent upserts with same vector_id both pass the
    /// first check, both will append to storage, but only one will succeed in
    /// inserting to id_map. The "orphaned" vector data in storage is harmless.
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

        // FIRST CHECK: Filter obvious duplicates with read lock (fast path)
        // This is released before the await to satisfy Send bounds
        let vectors_to_insert: Vec<(u64, Vec<f32>)> = {
            let id_map = self.id_map.read();
            vectors
                .into_iter()
                .filter(|(vector_id, _)| !id_map.contains_key(vector_id))
                .collect()
        };

        if vectors_to_insert.is_empty() {
            return Ok(UpsertResult { count: 0, sequence });
        }

        // Batch append all vectors at once (no locks held during async I/O)
        let vectors_data: Vec<Vec<f32>> = vectors_to_insert.iter().map(|(_, v)| v.clone()).collect();
        let start_idx = self.vectors.append_batch(&vectors_data).await?;

        // SECOND CHECK: Acquire write locks and verify before inserting
        // This handles the race where another thread inserted the same vector_id
        let mut actually_inserted = 0;
        {
            let mut id_map = self.id_map.write();
            let mut reverse_id_map = self.reverse_id_map.write();
            let mut write_buffer = self.write_buffer.write();

            for (i, (vector_id, _)) in vectors_to_insert.into_iter().enumerate() {
                let idx = start_idx + i as u32;
                // Re-check under write lock to handle concurrent inserts
                if !id_map.contains_key(&vector_id) {
                    id_map.insert(vector_id, idx);
                    reverse_id_map.insert(idx, vector_id);
                    write_buffer.push(idx);
                    actually_inserted += 1;
                }
                // If duplicate detected here, vector is already in storage but won't
                // be in id_map - this is safe (orphaned data, no inconsistency)
            }
        }

        Ok(UpsertResult { count: actually_inserted, sequence })
    }

    /// Search for k nearest neighbors
    /// Async to enable cooperative scheduling with other tasks
    pub async fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<SearchResult> {
        // Yield to allow other tasks to run (cooperative multitasking)
        tokio::task::yield_now().await;

        // Phase 1: Index search (doesn't need reverse_id_map lock)
        // Use k * 4 instead of k * 2 to ensure enough candidates after deduplication
        // with write_buffer results (which may contain duplicates)
        let index_results = self.index.search(query, k * 4, ef, &self.vectors);

        // Yield after potentially expensive index search
        tokio::task::yield_now().await;

        // Phase 2: Get write buffer indices (short lock)
        let buffer_indices: Vec<u32> = {
            let write_buffer = self.write_buffer.read();
            write_buffer.clone()
        };

        // Phase 3: Compute buffer similarities
        // IMPORTANT: Acquire mmap lock ONCE for all vector reads to reduce contention
        let mut buffer_results: Vec<(u32, f32)> = Vec::new();
        {
            let mmap_guard = self.vectors.mmap();
            if let Some(mmap) = mmap_guard.as_ref() {
                for idx in buffer_indices {
                    // Use get_slice for zero-copy access with held lock
                    if let Some(vec) = self.vectors.get_slice(idx, mmap) {
                        buffer_results.push((idx, cosine_similarity(query, vec)));
                    }
                }
            }
        }

        // Phase 4: Map indices to vector IDs (short lock)
        let mut results = Vec::new();
        {
            let reverse_id_map = self.reverse_id_map.read();

            // Map index results
            for (vector_index, similarity) in index_results {
                if let Some(&vector_id) = reverse_id_map.get(&vector_index) {
                    results.push(SearchResult {
                        id: vector_id,
                        similarity,
                    });
                }
            }

            // Map buffer results (avoiding duplicates)
            let existing_ids: std::collections::HashSet<u64> =
                results.iter().map(|r| r.id).collect();

            for (idx, similarity) in buffer_results {
                if let Some(&vid) = reverse_id_map.get(&idx) {
                    if !existing_ids.contains(&vid) {
                        results.push(SearchResult { id: vid, similarity });
                    }
                }
            }
        }

        // Phase 5: Sort by similarity (descending) and take top k (no locks)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        results
    }

    /// Flush write buffer to index
    ///
    /// IMPORTANT: Write order is critical for crash safety:
    /// 1. id_map FIRST (source of truth for vector ID mappings)
    /// 2. meta.json SECOND (commits the id_map as valid, updates last_flushed_seq)
    /// 3. Index LAST (optimization only - can be rebuilt from id_map)
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

        // Insert each vector into index (in-memory only, not persisted yet)
        for idx in &indices {
            self.index.insert(*idx, &self.vectors);
        }

        // Capture sequence BEFORE any persistence
        let new_seq = self.wal.current_sequence();

        // STEP 1: Save id_map FIRST (source of truth for mappings)
        // If crash happens after this, id_map is valid but meta has old sequence
        // This causes harmless re-replay of already-processed WAL entries
        let id_map_path = format!("tenant_{}/id_map.bin", self.tenant_id);
        let id_map_data = bincode::serialize(&*self.id_map.read())
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        storage.write(&id_map_path, &id_map_data).await?;
        storage.sync(&id_map_path).await?;

        // STEP 2: Save metadata SECOND (commits the id_map as valid)
        // If crash happens after this, id_map and meta are consistent
        // HNSW will be rebuilt on next flush (no data loss)
        let meta_path = format!("tenant_{}/meta.json", self.tenant_id);
        let meta = TenantMeta {
            last_flushed_seq: new_seq,
            vector_count: self.vectors.count(),
        };
        let meta_data =
            serde_json::to_vec(&meta).map_err(|e| StorageError::Serialization(e.to_string()))?;
        storage.write(&meta_path, &meta_data).await?;
        storage.sync(&meta_path).await?;

        // STEP 3: Update in-memory sequence AFTER metadata is persisted
        *self.last_flushed_seq.write() = new_seq;

        // STEP 4: Save index LAST (optimization - recovery works without it)
        let index_path = format!("tenant_{}/index.hnsw", self.tenant_id);
        self.index.save(storage, &index_path).await?;

        Ok(count)
    }

    /// Get stats
    /// Async to enable cooperative scheduling with other tasks
    pub async fn stats(&self) -> TenantStats {
        // Yield to allow other tasks to run (cooperative multitasking)
        tokio::task::yield_now().await;

        TenantStats {
            tenant_id: self.tenant_id,
            vector_count: self.vectors.count(),
            index_nodes: self.index.len() as u64,
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
    pub index_nodes: u64,
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
        let config = AnnIndexConfig::hnsw_with_m(4);

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
        let results = tenant.search(&query, 5, None).await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_flush_and_search() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = AnnIndexConfig::hnsw_with_m(4);

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
        let results = tenant.search(&query, 5, None).await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_tenant_recovery() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage_path = temp_dir.path().to_path_buf();
        let config = AnnIndexConfig::hnsw_with_m(4);

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

            let stats = tenant.stats().await;
            assert_eq!(stats.vector_count, 5);

            // Should be able to search
            let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
            let results = tenant.search(&query, 3, None).await;
            assert!(!results.is_empty());
        }
    }
}
