//! Vector Engine
//!
//! Top-level orchestration of all components.
//! Manages tenants and coordinates background tasks.

use crate::config::EngineConfig;
use crate::hnsw::HnswConfig;
use crate::storage::StorageBackend;
use crate::tenant::{SearchResult, TenantState, TenantStats, UpsertResult};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Vector database engine
pub struct VectorEngine {
    config: EngineConfig,
    storage: Arc<StorageBackend>,
    tenants: DashMap<u64, Arc<TenantState>>,
    hnsw_config: HnswConfig,
    /// Shutdown signal
    shutdown: RwLock<bool>,
}

impl VectorEngine {
    /// Create a new engine
    pub async fn new(config: EngineConfig, storage: StorageBackend) -> anyhow::Result<Arc<Self>> {
        let hnsw_config = HnswConfig {
            m: config.hnsw_m,
            m_max0: config.hnsw_m * 2,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            ml: 1.0 / (config.hnsw_m as f64).ln(),
        };

        let engine = Arc::new(Self {
            config: config.clone(),
            storage: Arc::new(storage),
            tenants: DashMap::new(),
            hnsw_config,
            shutdown: RwLock::new(false),
        });

        // Start background flush task
        let engine_clone = engine.clone();
        let flush_interval = config.flush_interval;
        tokio::spawn(async move {
            engine_clone.background_flush_task(flush_interval).await;
        });

        Ok(engine)
    }

    /// Get or create a tenant
    pub async fn get_or_create_tenant(&self, tenant_id: u64) -> anyhow::Result<Arc<TenantState>> {
        // Check if already exists
        if let Some(tenant) = self.tenants.get(&tenant_id) {
            return Ok(tenant.clone());
        }

        // Create new tenant
        let tenant = TenantState::open(
            tenant_id,
            self.config.default_dims,
            Arc::new(BlockStorageWrapper(self.storage.clone())),
            self.hnsw_config.clone(),
        )
        .await?;

        let tenant = Arc::new(tenant);
        self.tenants.insert(tenant_id, tenant.clone());

        Ok(tenant)
    }

    /// Upsert vectors for a tenant
    pub async fn upsert(
        &self,
        tenant_id: u64,
        vectors: Vec<(u64, Vec<f32>)>,
    ) -> anyhow::Result<UpsertResult> {
        let tenant = self.get_or_create_tenant(tenant_id).await?;
        let result = tenant.upsert(vectors).await?;
        Ok(result)
    }

    /// Search for similar vectors
    pub async fn search(
        &self,
        tenant_id: u64,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let tenant = self.get_or_create_tenant(tenant_id).await?;
        Ok(tenant.search(&query, k, ef))
    }

    /// Get tenant stats
    pub async fn stats(&self, tenant_id: u64) -> anyhow::Result<TenantStats> {
        let tenant = self.get_or_create_tenant(tenant_id).await?;
        Ok(tenant.stats())
    }

    /// Flush all tenants
    pub async fn flush_all(&self) -> anyhow::Result<usize> {
        let mut total = 0;
        for entry in self.tenants.iter() {
            let tenant = entry.value();
            let count = tenant.flush_to_hnsw(&*self.storage.block).await?;
            total += count;
        }
        Ok(total)
    }

    /// Background flush task
    async fn background_flush_task(self: Arc<Self>, interval: Duration) {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            ticker.tick().await;

            if *self.shutdown.read().await {
                break;
            }

            for entry in self.tenants.iter() {
                let tenant = entry.value();
                if let Err(e) = tenant.flush_to_hnsw(&*self.storage.block).await {
                    tracing::error!(
                        tenant_id = tenant.tenant_id,
                        error = %e,
                        "Failed to flush tenant to HNSW"
                    );
                }
            }
        }
    }

    /// Shutdown the engine
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;

        // Final flush
        if let Err(e) = self.flush_all().await {
            tracing::error!(error = %e, "Failed to flush during shutdown");
        }
    }
}

/// Wrapper to make StorageBackend's block storage implement BlockStorage trait
struct BlockStorageWrapper(Arc<StorageBackend>);

#[async_trait::async_trait]
impl crate::storage::BlockStorage for BlockStorageWrapper {
    async fn write(&self, path: &str, data: &[u8]) -> crate::storage::StorageResult<()> {
        self.0.block.write(path, data).await
    }

    async fn write_at(
        &self,
        path: &str,
        offset: usize,
        data: &[u8],
    ) -> crate::storage::StorageResult<()> {
        self.0.block.write_at(path, offset, data).await
    }

    async fn append(&self, path: &str, data: &[u8]) -> crate::storage::StorageResult<u64> {
        self.0.block.append(path, data).await
    }

    async fn read(&self, path: &str) -> crate::storage::StorageResult<Vec<u8>> {
        self.0.block.read(path).await
    }

    async fn read_range(
        &self,
        path: &str,
        offset: u64,
        length: usize,
    ) -> crate::storage::StorageResult<Vec<u8>> {
        self.0.block.read_range(path, offset, length).await
    }

    async fn exists(&self, path: &str) -> crate::storage::StorageResult<bool> {
        self.0.block.exists(path).await
    }

    async fn size(&self, path: &str) -> crate::storage::StorageResult<u64> {
        self.0.block.size(path).await
    }

    async fn sync(&self, path: &str) -> crate::storage::StorageResult<()> {
        self.0.block.sync(path).await
    }

    async fn delete(&self, path: &str) -> crate::storage::StorageResult<()> {
        self.0.block.delete(path).await
    }

    async fn list(&self, prefix: &str) -> crate::storage::StorageResult<Vec<String>> {
        self.0.block.list(prefix).await
    }

    async fn create_dir(&self, path: &str) -> crate::storage::StorageResult<()> {
        self.0.block.create_dir(path).await
    }

    fn mmap(&self, path: &str) -> crate::storage::StorageResult<Option<memmap2::Mmap>> {
        self.0.block.mmap(path)
    }

    fn root_path(&self) -> &std::path::Path {
        self.0.block.root_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{create_temp_storage, MockStorageConfig};

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[tokio::test]
    async fn test_engine_basic() {
        let storage = create_temp_storage(MockStorageConfig::fast()).unwrap();
        let mut config = EngineConfig::default();
        config.default_dims = 4;  // Use 4 dimensions for test

        let engine = VectorEngine::new(config, storage).await.unwrap();

        // Upsert
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| {
                let v = normalize(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
                (100 + i as u64, v)
            })
            .collect();

        let result = engine.upsert(1, vectors).await.unwrap();
        assert_eq!(result.count, 10);

        // Search
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = engine.search(1, query, 5, None).await.unwrap();
        assert!(!results.is_empty());

        // Stats
        let stats = engine.stats(1).await.unwrap();
        assert_eq!(stats.vector_count, 10);
    }
}
