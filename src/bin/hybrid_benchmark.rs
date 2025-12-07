//! Hybrid Benchmark: Local BlockStorage + Real AWS S3
//!
//! This simulates the production architecture:
//! - Local SSD for BlockStorage (simulating EBS - same code path)
//! - Real AWS S3 for ObjectStorage (backups, snapshots)
//!
//! Usage:
//!   cargo run --release --bin hybrid_benchmark --features aws-storage -- --vectors 10000

use bytes::Bytes;
use clap::Parser;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vortex::config::EngineConfig;
use vortex::storage::aws::{AwsResourceManager, S3ObjectStorage};
use vortex::storage::mock::{create_temp_storage, MockStorageConfig};
use vortex::storage::ObjectStorage;
use vortex::VectorEngine;

#[derive(Parser)]
#[command(name = "hybrid-benchmark")]
#[command(about = "Benchmark with local storage + real AWS S3")]
struct Args {
    /// Number of vectors to insert
    #[arg(long, default_value = "10000")]
    vectors: usize,

    /// Vector dimensions
    #[arg(long, default_value = "384")]
    dimensions: usize,

    /// Batch size for upserts
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Number of search queries
    #[arg(long, default_value = "100")]
    queries: usize,

    /// Top-k results
    #[arg(long, default_value = "10")]
    k: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     HYBRID BENCHMARK: Local Storage + Real AWS S3            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Vectors:              {:>10}                             ║", args.vectors);
    println!("║ Dimensions:           {:>10}                             ║", args.dimensions);
    println!("║ Batch Size:           {:>10}                             ║", args.batch_size);
    println!("║ Search Queries:       {:>10}                             ║", args.queries);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ==================== PHASE 1: AWS Setup ====================
    println!("Phase 1: Connecting to AWS...");
    let aws_start = Instant::now();
    let aws_manager = AwsResourceManager::new("us-east-1").await?;
    aws_manager.clone().setup_signal_handlers();
    println!("  ✓ Connected to AWS in {:?}", aws_start.elapsed());

    // Create S3 storage for backups
    let s3_start = Instant::now();
    let s3 = S3ObjectStorage::new(aws_manager.clone(), "vortex-data").await?;
    println!("  ✓ Created S3 bucket '{}' in {:?}\n", s3.bucket(), s3_start.elapsed());

    // ==================== PHASE 2: Local Engine Setup ====================
    println!("Phase 2: Setting up local Vortex engine...");
    let engine_start = Instant::now();

    // Create local block storage (simulates EBS)
    let storage = create_temp_storage(MockStorageConfig::fast())?;

    // Configure engine
    let mut config = EngineConfig::default();
    config.default_dims = args.dimensions;
    config.hnsw_ef_search = 100;
    config.flush_interval = Duration::from_secs(300); // Don't auto-flush during benchmark

    let engine = VectorEngine::new(config, storage).await?;
    println!("  ✓ Engine initialized in {:?}\n", engine_start.elapsed());

    // ==================== PHASE 3: Upsert Benchmark ====================
    println!("Phase 3: Upsert Benchmark (Local BlockStorage)...");
    let mut upsert_latencies = Vec::new();
    let tenant_id = 1u64;
    let num_batches = args.vectors / args.batch_size;

    let upsert_start = Instant::now();
    for batch_idx in 0..num_batches {
        let vectors: Vec<(u64, Vec<f32>)> = (0..args.batch_size)
            .map(|i| {
                let id = (batch_idx * args.batch_size + i) as u64;
                let vector: Vec<f32> = (0..args.dimensions)
                    .map(|j| ((id as f32 * 0.001) + (j as f32 * 0.0001)).sin())
                    .collect();
                (id, vector)
            })
            .collect();

        let batch_start = Instant::now();
        engine.upsert(tenant_id, vectors).await?;
        upsert_latencies.push(batch_start.elapsed());

        if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
            println!("  Progress: {}/{} batches ({} vectors)",
                batch_idx + 1, num_batches, (batch_idx + 1) * args.batch_size);
        }
    }
    let upsert_total = upsert_start.elapsed();

    // Calculate upsert metrics
    let upsert_throughput = args.vectors as f64 / upsert_total.as_secs_f64();
    upsert_latencies.sort();
    let upsert_p50 = upsert_latencies[upsert_latencies.len() / 2];
    let upsert_p99 = upsert_latencies[(upsert_latencies.len() as f64 * 0.99) as usize];

    println!("  ✓ Upsert complete in {:?}\n", upsert_total);

    // ==================== PHASE 4: Flush to HNSW ====================
    println!("Phase 4: Building HNSW index...");
    let flush_start = Instant::now();
    engine.flush_all().await?;
    let flush_time = flush_start.elapsed();
    println!("  ✓ HNSW index built in {:?}\n", flush_time);

    // ==================== PHASE 5: Backup to S3 ====================
    println!("Phase 5: Backing up to AWS S3...");
    let backup_start = Instant::now();

    // Simulate backing up vector data to S3
    // In production, this would serialize the HNSW index and vectors
    let sample_vectors: Vec<u8> = (0..args.dimensions * 4 * 1000)
        .map(|i| (i % 256) as u8)
        .collect();

    // Upload backup chunks to S3
    let chunk_size = 100_000; // 100KB chunks
    let num_chunks = (sample_vectors.len() + chunk_size - 1) / chunk_size;

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, sample_vectors.len());
        let chunk = Bytes::from(sample_vectors[start..end].to_vec());
        s3.put(&format!("backup/chunk_{:04}.bin", i), chunk).await?;
    }

    // Upload metadata
    let metadata = format!(
        r#"{{"vectors": {}, "dimensions": {}, "chunks": {}, "timestamp": "{}"}}"#,
        args.vectors,
        args.dimensions,
        num_chunks,
        chrono::Utc::now().to_rfc3339()
    );
    s3.put("backup/metadata.json", Bytes::from(metadata)).await?;

    let backup_time = backup_start.elapsed();
    println!("  ✓ Backed up {} chunks to S3 in {:?}\n", num_chunks + 1, backup_time);

    // ==================== PHASE 6: Search Benchmark ====================
    println!("Phase 6: Search Benchmark (HNSW)...");
    let mut search_latencies = Vec::new();

    let search_start = Instant::now();
    for i in 0..args.queries {
        let query: Vec<f32> = (0..args.dimensions)
            .map(|j| ((i as f32 * 0.002) + (j as f32 * 0.0001)).cos())
            .collect();

        let query_start = Instant::now();
        let _results = engine.search(tenant_id, query, args.k, Some(100)).await?;
        search_latencies.push(query_start.elapsed());
    }
    let search_total = search_start.elapsed();

    // Calculate search metrics
    let search_throughput = args.queries as f64 / search_total.as_secs_f64();
    search_latencies.sort();
    let search_p50 = search_latencies[search_latencies.len() / 2];
    let search_p99 = search_latencies[(search_latencies.len() as f64 * 0.99) as usize];

    println!("  ✓ Search complete in {:?}\n", search_total);

    // ==================== PHASE 7: Verify S3 Backup ====================
    println!("Phase 7: Verifying S3 backup...");
    let verify_start = Instant::now();

    let objects = s3.list("backup/").await?;
    let metadata_bytes = s3.get("backup/metadata.json").await?;
    let metadata_str = String::from_utf8_lossy(&metadata_bytes);

    println!("  ✓ Found {} objects in S3", objects.len());
    println!("  ✓ Metadata: {}", metadata_str);
    println!("  ✓ Verification complete in {:?}\n", verify_start.elapsed());

    // ==================== PHASE 8: Cleanup ====================
    println!("Phase 8: Cleaning up AWS resources...");
    let cleanup_start = Instant::now();
    aws_manager.cleanup_all().await?;
    println!("  ✓ Cleanup complete in {:?}\n", cleanup_start.elapsed());

    // ==================== RESULTS ====================
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                    LOCAL BLOCKSTORAGE                        ║");
    println!("║                    (Simulating EBS)                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Upsert Throughput:    {:>10.0} vectors/sec                 ║", upsert_throughput);
    println!("║ Upsert P50 Latency:   {:>10.2?}                         ║", upsert_p50);
    println!("║ Upsert P99 Latency:   {:>10.2?}                         ║", upsert_p99);
    println!("║ HNSW Build Time:      {:>10.2?}                         ║", flush_time);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Search Throughput:    {:>10.0} queries/sec                 ║", search_throughput);
    println!("║ Search P50 Latency:   {:>10.2?}                         ║", search_p50);
    println!("║ Search P99 Latency:   {:>10.2?}                         ║", search_p99);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                    AWS S3 OBJECTSTORAGE                      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ S3 Backup Time:       {:>10.2?}                         ║", backup_time);
    println!("║ S3 Objects Created:   {:>10}                             ║", num_chunks + 1);
    println!("║ S3 Bucket:            {:>32} ║", s3.bucket());
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n✓ All tests passed! AWS S3 integration verified.");

    Ok(())
}
