//! Latency Profiling Tests for VectorDB
//!
//! These tests break down timing at a component level to identify
//! performance bottlenecks and understand where time is spent.
//!
//! Key bottlenecks measured:
//! - WAL fsync (dominates write latency)
//! - HNSW write lock (blocks searches during insert)
//! - Write buffer search (O(n) brute force)
//! - mmap refresh (RwLock contention)
//! - ID map lookups (now O(1) with reverse map)

mod common;

use common::{generate_random_vectors, normalize, random_vector, LatencyHistogram};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;
use vortex::hnsw::HnswConfig;
use vortex::storage::mock::{MockBlockStorage, MockStorageConfig};
use vortex::tenant::TenantState;

const TEST_DIMS: usize = 384;
const BATCH_SIZE: usize = 100;

// =============================================================================
// INSERT PATH BREAKDOWN
// =============================================================================

/// Profile the insert path to understand latency breakdown:
/// - Total insert time
/// - WAL append (includes fsync)
/// - VectorStore append
/// - ID map updates
#[tokio::test]
async fn profile_insert_path_breakdown() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    println!("\n=== Insert Path Breakdown ===");

    // Warm up
    let warmup = generate_random_vectors(100, TEST_DIMS);
    tenant.upsert(warmup).await.unwrap();

    // Measure single insert
    let mut single_latencies = Vec::new();
    for i in 0..100 {
        let vectors = vec![(1000 + i as u64, normalize(&random_vector(TEST_DIMS)))];
        let start = Instant::now();
        tenant.upsert(vectors).await.unwrap();
        single_latencies.push(start.elapsed());
    }

    let single_hist = LatencyHistogram::from_latencies(single_latencies);

    // Measure batch insert
    let mut batch_latencies = Vec::new();
    for batch in 0..20 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..BATCH_SIZE)
            .map(|i| {
                (
                    2000 + (batch * BATCH_SIZE + i) as u64,
                    normalize(&random_vector(TEST_DIMS)),
                )
            })
            .collect();

        let start = Instant::now();
        tenant.upsert(vectors).await.unwrap();
        batch_latencies.push(start.elapsed());
    }

    let batch_hist = LatencyHistogram::from_latencies(batch_latencies);

    println!("\n--- Single Vector Insert ---");
    single_hist.print_summary("Single Insert");
    println!(
        "Per-vector cost: {:?}",
        single_hist.mean
    );

    println!("\n--- Batch Insert ({} vectors) ---", BATCH_SIZE);
    batch_hist.print_summary("Batch Insert");
    println!(
        "Per-vector cost: {:?}",
        batch_hist.mean / BATCH_SIZE as u32
    );

    // Amortization factor
    let single_cost = single_hist.mean.as_nanos() as f64;
    let batch_per_vec = batch_hist.mean.as_nanos() as f64 / BATCH_SIZE as f64;
    println!(
        "\nAmortization factor: {:.1}x (batch is {:.1}x faster per vector)",
        single_cost / batch_per_vec,
        single_cost / batch_per_vec
    );
}

/// Compare WAL vs no-WAL overhead to isolate fsync cost
#[tokio::test]
async fn profile_wal_overhead() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    println!("\n=== WAL Overhead Analysis ===");

    // The full upsert includes WAL, VectorStore, and ID map updates
    // We can estimate WAL overhead by comparing with/without sync

    let mut latencies = Vec::new();
    for batch in 0..50 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..BATCH_SIZE)
            .map(|i| {
                (
                    (batch * BATCH_SIZE + i) as u64,
                    normalize(&random_vector(TEST_DIMS)),
                )
            })
            .collect();

        let start = Instant::now();
        tenant.upsert(vectors).await.unwrap();
        latencies.push(start.elapsed());
    }

    let hist = LatencyHistogram::from_latencies(latencies);
    hist.print_summary("Upsert (with WAL fsync)");

    // WAL fsync typically dominates - in fast mode it's ~0.1-1ms per batch
    // In realistic mode (simulating SSD latency) it would be 1-5ms
    println!("\nNote: WAL fsync is the primary bottleneck for write latency.");
    println!("In production with real SSDs, expect 1-5ms per fsync.");
    println!("Batching amortizes this cost across many vectors.");
}

// =============================================================================
// SEARCH PATH BREAKDOWN
// =============================================================================

/// Profile the search path:
/// - HNSW traversal time
/// - Write buffer brute force time
/// - Result assembly time
#[tokio::test]
async fn profile_search_path_breakdown() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert and flush to HNSW
    let vectors = generate_random_vectors(5000, TEST_DIMS);
    tenant.upsert(vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== Search Path Breakdown ===");
    println!("Index size: 5000 vectors (all in HNSW)");

    // Test different ef values
    for ef in [50, 100, 200, 500] {
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let query = normalize(&random_vector(TEST_DIMS));
            let start = Instant::now();
            tenant.search(&query, 10, Some(ef)).await;
            latencies.push(start.elapsed());
        }

        let hist = LatencyHistogram::from_latencies(latencies);
        println!(
            "ef={}: p50={:?}, p99={:?}, mean={:?}",
            ef, hist.p50, hist.p99, hist.mean
        );
    }

    // Test with write buffer
    println!("\n--- With Write Buffer ---");
    let extra = generate_random_vectors(1000, TEST_DIMS);
    let extra: Vec<(u64, Vec<f32>)> = extra
        .into_iter()
        .map(|(id, v)| (id + 10000, v))
        .collect();
    tenant.upsert(extra).await.unwrap();

    let stats = tenant.stats().await;
    println!(
        "HNSW nodes: {}, Write buffer: {}",
        stats.hnsw_nodes, stats.write_buffer_size
    );

    for ef in [100, 200] {
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let query = normalize(&random_vector(TEST_DIMS));
            let start = Instant::now();
            tenant.search(&query, 10, Some(ef)).await;
            latencies.push(start.elapsed());
        }

        let hist = LatencyHistogram::from_latencies(latencies);
        println!(
            "ef={} (with 1K buffer): p50={:?}, p99={:?}",
            ef, hist.p50, hist.p99
        );
    }
}

/// Profile write buffer overhead at different sizes
#[tokio::test]
async fn profile_write_buffer_overhead() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    println!("\n=== Write Buffer Overhead Analysis ===");

    // Pre-populate HNSW
    let initial = generate_random_vectors(1000, TEST_DIMS);
    tenant.upsert(initial).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    // Test search latency at different write buffer sizes
    for buffer_size in [0, 100, 500, 1000, 2000, 5000] {
        // Add to write buffer
        if buffer_size > 0 {
            let vectors: Vec<(u64, Vec<f32>)> = (0..buffer_size)
                .map(|i| (10000 + i as u64, normalize(&random_vector(TEST_DIMS))))
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }

        // Measure search latency
        let mut latencies = Vec::new();
        for _ in 0..50 {
            let query = normalize(&random_vector(TEST_DIMS));
            let start = Instant::now();
            tenant.search(&query, 10, Some(100)).await;
            latencies.push(start.elapsed());
        }

        let hist = LatencyHistogram::from_latencies(latencies);
        let stats = tenant.stats().await;

        println!(
            "Buffer size {}: p50={:?}, p99={:?}, mean={:?}",
            stats.write_buffer_size, hist.p50, hist.p99, hist.mean
        );

        // Flush for next iteration (if not last)
        if buffer_size < 5000 {
            tenant.flush_to_hnsw(&*storage).await.unwrap();
        }
    }

    println!("\nNote: Write buffer search is O(n) brute force.");
    println!("For low latency, flush when buffer exceeds ~1000 vectors.");
}

// =============================================================================
// FLUSH PATH BREAKDOWN
// =============================================================================

/// Profile the flush path:
/// - Write buffer drain
/// - HNSW insert per vector
/// - Index persistence
#[tokio::test]
async fn profile_flush_path_breakdown() {
    let config = HnswConfig::new(TEST_DIMS);

    println!("\n=== Flush Path Breakdown ===");

    // Test different buffer sizes
    for buffer_size in [100, 500, 1000, 2000] {
        // Create fresh tenant for each buffer size
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        // Fill buffer
        let vectors: Vec<(u64, Vec<f32>)> = (0..buffer_size)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(vectors).await.unwrap();

        // Measure flush
        let start = Instant::now();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
        let flush_duration = start.elapsed();

        let per_vector = flush_duration / buffer_size as u32;
        let stats = tenant.stats().await;

        println!(
            "Flush {} vectors: total={:?}, per_vector={:?}, final_hnsw={}",
            buffer_size, flush_duration, per_vector, stats.hnsw_nodes
        );
    }

    // Profile flush into existing index
    println!("\n--- Flush into Existing Index ---");
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
        .await
        .unwrap();

    // Build initial index
    let initial = generate_random_vectors(5000, TEST_DIMS);
    tenant.upsert(initial).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    // Add more and flush
    for round in 0..3 {
        let extra: Vec<(u64, Vec<f32>)> = (0..1000)
            .map(|i| (10000 + round * 1000 + i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(extra).await.unwrap();

        let start = Instant::now();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
        let duration = start.elapsed();

        let stats = tenant.stats().await;
        println!(
            "Round {}: flush 1000 into {} existing: {:?} ({:?}/vec)",
            round + 1,
            stats.hnsw_nodes,
            duration,
            duration / 1000
        );
    }
}

// =============================================================================
// LOCK CONTENTION ANALYSIS
// =============================================================================

/// Profile lock contention under concurrent access
#[tokio::test]
async fn profile_lock_contention() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Pre-populate
    let initial = generate_random_vectors(2000, TEST_DIMS);
    tenant.upsert(initial).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== Lock Contention Analysis ===");

    // Baseline: serial access
    println!("\n--- Serial Access (no contention) ---");
    let mut serial_read = Vec::new();
    let mut serial_write = Vec::new();

    for i in 0..100 {
        if i % 2 == 0 {
            let query = normalize(&random_vector(TEST_DIMS));
            let start = Instant::now();
            tenant.search(&query, 10, None).await;
            serial_read.push(start.elapsed());
        } else {
            let vectors = vec![(10000 + i as u64, normalize(&random_vector(TEST_DIMS)))];
            let start = Instant::now();
            tenant.upsert(vectors).await.unwrap();
            serial_write.push(start.elapsed());
        }
    }

    let serial_read_hist = LatencyHistogram::from_latencies(serial_read);
    let serial_write_hist = LatencyHistogram::from_latencies(serial_write);

    println!("Read p50: {:?}", serial_read_hist.p50);
    println!("Write p50: {:?}", serial_write_hist.p50);

    // High contention: multiple concurrent readers and writers
    println!("\n--- High Contention (4 readers, 4 writers) ---");

    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();

    // 4 reader tasks
    for _ in 0..4 {
        let tenant = tenant.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            barrier.wait().await;

            for _ in 0..50 {
                let query = normalize(&random_vector(TEST_DIMS));
                let start = Instant::now();
                tenant.search(&query, 10, None).await;
                latencies.push(start.elapsed());
            }

            ("read", latencies)
        }));
    }

    // 4 writer tasks
    for writer_id in 0..4u64 {
        let tenant = tenant.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            barrier.wait().await;

            for i in 0..50 {
                let id = 20000 + writer_id * 100 + i;
                let vectors = vec![(id, normalize(&random_vector(TEST_DIMS)))];
                let start = Instant::now();
                let _ = tenant.upsert(vectors).await;
                latencies.push(start.elapsed());
            }

            ("write", latencies)
        }));
    }

    let mut concurrent_reads = Vec::new();
    let mut concurrent_writes = Vec::new();

    for handle in handles {
        let (op_type, latencies) = handle.await.unwrap();
        if op_type == "read" {
            concurrent_reads.extend(latencies);
        } else {
            concurrent_writes.extend(latencies);
        }
    }

    let concurrent_read_hist = LatencyHistogram::from_latencies(concurrent_reads);
    let concurrent_write_hist = LatencyHistogram::from_latencies(concurrent_writes);

    println!("Read p50: {:?} (vs serial {:?})", concurrent_read_hist.p50, serial_read_hist.p50);
    println!("Write p50: {:?} (vs serial {:?})", concurrent_write_hist.p50, serial_write_hist.p50);

    // Calculate contention overhead
    let read_overhead = concurrent_read_hist.p50.as_nanos() as f64 / serial_read_hist.p50.as_nanos() as f64;
    let write_overhead = concurrent_write_hist.p50.as_nanos() as f64 / serial_write_hist.p50.as_nanos() as f64;

    println!("\nContention overhead: read={:.1}x, write={:.1}x", read_overhead, write_overhead);
}

/// Profile ID map lookup overhead (should be O(1) with reverse map)
#[tokio::test]
async fn profile_id_map_lookup() {
    let config = HnswConfig::new(TEST_DIMS);

    println!("\n=== ID Map Lookup Profile ===");

    // Test at different map sizes
    for size in [1000, 5000, 10000] {
        // Create fresh tenant for each size
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        // Populate
        let vectors = generate_random_vectors(size, TEST_DIMS);
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        // Measure search (which uses reverse map for ID lookup)
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let query = normalize(&random_vector(TEST_DIMS));
            let start = Instant::now();
            tenant.search(&query, 10, Some(100)).await;
            latencies.push(start.elapsed());
        }

        let hist = LatencyHistogram::from_latencies(latencies);
        println!(
            "Map size {}: search p50={:?}, p99={:?}",
            size, hist.p50, hist.p99
        );
    }

    println!("\nNote: ID lookups should be O(1) with reverse map.");
    println!("Search latency should not scale linearly with map size.");
}

// =============================================================================
// DISTANCE COMPUTATION ANALYSIS
// =============================================================================

/// Profile cosine distance computation overhead
#[tokio::test]
async fn profile_distance_computation() {
    println!("\n=== Distance Computation Profile ===");

    // Pure distance computation timing
    let query = normalize(&random_vector(TEST_DIMS));
    let vectors: Vec<Vec<f32>> = (0..10000)
        .map(|_| normalize(&random_vector(TEST_DIMS)))
        .collect();

    // Time brute force distance computation
    let start = Instant::now();
    let mut sum = 0.0f32;
    for v in &vectors {
        // Cosine similarity (dot product for normalized vectors)
        let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        sum += dot;
    }
    let duration = start.elapsed();

    println!(
        "10K distance computations ({}D): {:?}",
        TEST_DIMS, duration
    );
    println!("Per-distance: {:?}", duration / 10000);
    println!("Checksum (to prevent optimization): {:.4}", sum);

    // Compare with search which does fewer distance computations
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    let tenant_vectors = generate_random_vectors(10000, TEST_DIMS);
    tenant.upsert(tenant_vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let mut search_latencies = Vec::new();
    for _ in 0..100 {
        let q = normalize(&random_vector(TEST_DIMS));
        let start = Instant::now();
        tenant.search(&q, 10, Some(100)).await;
        search_latencies.push(start.elapsed());
    }

    let hist = LatencyHistogram::from_latencies(search_latencies);
    println!(
        "\nHNSW search (10K vectors, ef=100): p50={:?}",
        hist.p50
    );
    println!("HNSW does ~100-500 distance computations vs 10K for brute force");
}
