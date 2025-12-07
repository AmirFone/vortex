//! Simulation Framework for Vortex Benchmarks
//!
//! Provides a framework for running realistic benchmarks with:
//! - Configurable vector counts and dimensions
//! - Multiple storage backends (mock, S3)
//! - Automatic resource cleanup
//! - Detailed performance metrics

use crate::config::EngineConfig;
use crate::storage::mock::{create_temp_storage, MockStorageConfig};
use crate::storage::StorageBackend;
use crate::VectorEngine;
use rand::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

#[cfg(feature = "aws-storage")]
use crate::storage::aws::AwsResourceManager;

/// Storage type for simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    /// Mock storage (local temp files, no AWS costs)
    Mock,
    /// S3 only storage (requires AWS credentials)
    #[cfg(feature = "aws-storage")]
    AwsS3,
}

impl std::str::FromStr for StorageType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mock" => Ok(StorageType::Mock),
            #[cfg(feature = "aws-storage")]
            "s3" | "aws" | "aws-s3" => Ok(StorageType::AwsS3),
            _ => Err(format!("Unknown storage type: {}", s)),
        }
    }
}

/// Configuration for simulation
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of vectors to insert
    pub vector_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Batch size for upserts
    pub batch_size: usize,
    /// Number of search queries to run
    pub search_queries: usize,
    /// Top-k results to retrieve
    pub k: usize,
    /// Search ef parameter
    pub ef_search: usize,
    /// Storage backend type
    pub storage_type: StorageType,
    /// AWS region (only used for AWS storage)
    pub aws_region: String,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            vector_count: 100_000,
            dimensions: 384,
            batch_size: 1000,
            search_queries: 1000,
            k: 10,
            ef_search: 100,
            storage_type: StorageType::Mock,
            aws_region: "us-east-1".to_string(),
        }
    }
}

/// Results from a simulation run
#[derive(Debug, Clone)]
pub struct SimulationResults {
    /// Total vectors inserted
    pub vectors_inserted: usize,
    /// Upsert throughput (vectors/sec)
    pub upsert_throughput: f64,
    /// Average upsert latency per batch
    pub upsert_latency_avg: Duration,
    /// P50 upsert latency
    pub upsert_latency_p50: Duration,
    /// P99 upsert latency
    pub upsert_latency_p99: Duration,
    /// Total search queries executed
    pub search_queries: usize,
    /// Search throughput (queries/sec)
    pub search_throughput: f64,
    /// Average search latency
    pub search_latency_avg: Duration,
    /// P50 search latency
    pub search_latency_p50: Duration,
    /// P99 search latency
    pub search_latency_p99: Duration,
    /// Total simulation duration
    pub total_duration: Duration,
    /// Estimated AWS cost (USD)
    pub aws_cost_estimate: f64,
}

impl SimulationResults {
    /// Print results in a formatted way
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    SIMULATION RESULTS                         ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Vectors Inserted:     {:>10}                             ║", self.vectors_inserted);
        println!("║ Search Queries:       {:>10}                             ║", self.search_queries);
        println!("║ Total Duration:       {:>10.2?}                         ║", self.total_duration);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                      UPSERT METRICS                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Throughput:           {:>10.2} vectors/sec                 ║", self.upsert_throughput);
        println!("║ Avg Latency:          {:>10.2?}                         ║", self.upsert_latency_avg);
        println!("║ P50 Latency:          {:>10.2?}                         ║", self.upsert_latency_p50);
        println!("║ P99 Latency:          {:>10.2?}                         ║", self.upsert_latency_p99);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                      SEARCH METRICS                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Throughput:           {:>10.2} queries/sec                 ║", self.search_throughput);
        println!("║ Avg Latency:          {:>10.2?}                         ║", self.search_latency_avg);
        println!("║ P50 Latency:          {:>10.2?}                         ║", self.search_latency_p50);
        println!("║ P99 Latency:          {:>10.2?}                         ║", self.search_latency_p99);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Estimated AWS Cost:   ${:>9.4}                             ║", self.aws_cost_estimate);
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

/// Simulation runner
pub struct SimulationRunner {
    config: SimulationConfig,
    #[cfg(feature = "aws-storage")]
    resource_manager: Option<Arc<AwsResourceManager>>,
}

impl SimulationRunner {
    /// Create a new simulation runner
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "aws-storage")]
            resource_manager: None,
        }
    }

    /// Run the simulation
    pub async fn run(&mut self) -> Result<SimulationResults, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        info!(config = ?self.config, "Starting simulation");

        // Create storage backend based on config
        let storage = self.create_storage().await?;

        // Create engine config
        let mut engine_config = EngineConfig::default();
        engine_config.default_dims = self.config.dimensions;
        engine_config.hnsw_ef_search = self.config.ef_search;
        engine_config.flush_interval = Duration::from_secs(30);

        // Create engine
        let engine = VectorEngine::new(engine_config, storage).await?;

        // Run upsert benchmark
        let (upsert_latencies, vectors_inserted) = self.run_upsert_benchmark(&engine).await?;

        // Flush all data to HNSW
        info!("Flushing data to HNSW index...");
        engine.flush_all().await?;

        // Run search benchmark
        let search_latencies = self.run_search_benchmark(&engine).await?;

        let total_duration = start_time.elapsed();

        // Calculate metrics
        let results = self.calculate_results(
            vectors_inserted,
            upsert_latencies,
            search_latencies,
            total_duration,
        );

        // Cleanup AWS resources if used
        #[cfg(feature = "aws-storage")]
        if let Some(ref manager) = self.resource_manager {
            info!("Cleaning up AWS resources...");
            manager.cleanup_all().await?;
        }

        Ok(results)
    }

    async fn create_storage(&mut self) -> Result<StorageBackend, Box<dyn std::error::Error + Send + Sync>> {
        match self.config.storage_type {
            StorageType::Mock => {
                info!("Using mock storage (local temp files)");
                let storage = create_temp_storage(MockStorageConfig::fast())?;
                Ok(storage)
            }
            #[cfg(feature = "aws-storage")]
            StorageType::AwsS3 => {
                info!(region = %self.config.aws_region, "Using AWS S3 storage");

                // Create resource manager
                let manager = AwsResourceManager::new(&self.config.aws_region).await
                    .map_err(|e| format!("Failed to create AWS resource manager: {}", e))?;

                // Setup signal handlers for cleanup
                manager.clone().setup_signal_handlers();
                self.resource_manager = Some(manager.clone());

                // For S3-only mode, we still need local block storage for vectors/HNSW
                // S3 would be used for backups. For this simulation, use mock block storage.
                let storage = create_temp_storage(MockStorageConfig::fast())?;
                Ok(storage)
            }
        }
    }

    async fn run_upsert_benchmark(
        &self,
        engine: &VectorEngine,
    ) -> Result<(Vec<Duration>, usize), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            vector_count = self.config.vector_count,
            batch_size = self.config.batch_size,
            "Starting upsert benchmark"
        );

        let mut latencies = Vec::new();
        let num_batches = self.config.vector_count / self.config.batch_size;
        let tenant_id = 1u64;

        for batch_idx in 0..num_batches {
            let vectors: Vec<(u64, Vec<f32>)> = (0..self.config.batch_size)
                .map(|i| {
                    let id = (batch_idx * self.config.batch_size + i) as u64;
                    let vector = generate_random_vector(self.config.dimensions);
                    (id, vector)
                })
                .collect();

            let start = Instant::now();
            engine.upsert(tenant_id, vectors).await?;
            latencies.push(start.elapsed());

            if (batch_idx + 1) % 10 == 0 {
                info!(
                    progress = format!("{}/{}", batch_idx + 1, num_batches),
                    vectors = (batch_idx + 1) * self.config.batch_size,
                    "Upsert progress"
                );
            }
        }

        let total_vectors = num_batches * self.config.batch_size;
        Ok((latencies, total_vectors))
    }

    async fn run_search_benchmark(
        &self,
        engine: &VectorEngine,
    ) -> Result<Vec<Duration>, Box<dyn std::error::Error + Send + Sync>> {
        info!(
            queries = self.config.search_queries,
            k = self.config.k,
            ef = self.config.ef_search,
            "Starting search benchmark"
        );

        let mut latencies = Vec::new();
        let tenant_id = 1u64;

        for i in 0..self.config.search_queries {
            let query = generate_random_vector(self.config.dimensions);

            let start = Instant::now();
            let _results = engine
                .search(tenant_id, query, self.config.k, Some(self.config.ef_search))
                .await?;
            latencies.push(start.elapsed());

            if (i + 1) % 100 == 0 {
                info!(
                    progress = format!("{}/{}", i + 1, self.config.search_queries),
                    "Search progress"
                );
            }
        }

        Ok(latencies)
    }

    fn calculate_results(
        &self,
        vectors_inserted: usize,
        upsert_latencies: Vec<Duration>,
        search_latencies: Vec<Duration>,
        total_duration: Duration,
    ) -> SimulationResults {
        // Calculate upsert metrics
        let upsert_total: Duration = upsert_latencies.iter().sum();
        let upsert_throughput = vectors_inserted as f64 / upsert_total.as_secs_f64();
        let upsert_latency_avg = upsert_total / upsert_latencies.len() as u32;

        let mut sorted_upsert = upsert_latencies.clone();
        sorted_upsert.sort();
        let upsert_latency_p50 = sorted_upsert[sorted_upsert.len() / 2];
        let upsert_latency_p99 = sorted_upsert[(sorted_upsert.len() as f64 * 0.99) as usize];

        // Calculate search metrics
        let search_total: Duration = search_latencies.iter().sum();
        let search_throughput = self.config.search_queries as f64 / search_total.as_secs_f64();
        let search_latency_avg = search_total / search_latencies.len() as u32;

        let mut sorted_search = search_latencies.clone();
        sorted_search.sort();
        let search_latency_p50 = sorted_search[sorted_search.len() / 2];
        let search_latency_p99 = sorted_search[(sorted_search.len() as f64 * 0.99) as usize];

        // Estimate AWS costs
        let aws_cost_estimate = self.estimate_aws_cost();

        SimulationResults {
            vectors_inserted,
            upsert_throughput,
            upsert_latency_avg,
            upsert_latency_p50,
            upsert_latency_p99,
            search_queries: self.config.search_queries,
            search_throughput,
            search_latency_avg,
            search_latency_p50,
            search_latency_p99,
            total_duration,
            aws_cost_estimate,
        }
    }

    fn estimate_aws_cost(&self) -> f64 {
        match self.config.storage_type {
            StorageType::Mock => 0.0,
            #[cfg(feature = "aws-storage")]
            StorageType::AwsS3 => {
                // Rough estimates based on AWS pricing
                let storage_gb = (self.config.vector_count * self.config.dimensions * 4) as f64 / 1_000_000_000.0;
                let put_requests = (self.config.vector_count / self.config.batch_size) as f64;
                let get_requests = self.config.search_queries as f64 * 10.0; // Estimate reads per search

                let storage_cost = storage_gb * 0.023; // $0.023 per GB
                let put_cost = put_requests / 1000.0 * 0.005; // $0.005 per 1000 PUT
                let get_cost = get_requests / 1000.0 * 0.0004; // $0.0004 per 1000 GET

                storage_cost + put_cost + get_cost
            }
        }
    }
}

/// Generate a random normalized vector
fn generate_random_vector(dims: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
    normalize(&v)
}

/// Normalize a vector
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
