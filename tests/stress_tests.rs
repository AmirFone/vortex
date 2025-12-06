//! Stress Tests for VectorDB
//!
//! This module contains stress tests that push the system to its limits
//! to identify bottlenecks, concurrency issues, and performance degradation.
//!
//! Target scale: 100K vectors
//! Organization: All tests in one suite (runs with `cargo test`)

mod common;

use common::{
    generate_random_vectors, measure_latency_async, normalize, random_vector,
    LatencyHistogram, ThroughputResult,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;
use vectordb::hnsw::HnswConfig;
use vectordb::storage::mock::{MockBlockStorage, MockStorageConfig};
use vectordb::tenant::TenantState;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Number of dimensions for test vectors
const TEST_DIMS: usize = 384;

/// Target scale for large tests
const LARGE_SCALE_COUNT: usize = 100_000;

/// Medium scale for faster tests
const MEDIUM_SCALE_COUNT: usize = 10_000;

/// Small scale for quick tests
const SMALL_SCALE_COUNT: usize = 1_000;

/// Batch size for inserts
const BATCH_SIZE: usize = 100;

// =============================================================================
// CONCURRENT WRITE STRESS TESTS
// =============================================================================

/// Helper to run concurrent writer stress test
async fn run_concurrent_writers_test(num_threads: usize, vectors_per_thread: usize) -> StressTestResult {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    let barrier = Arc::new(Barrier::new(num_threads));
    let total_inserted = Arc::new(AtomicUsize::new(0));
    let mut latencies: Vec<Duration> = Vec::new();

    let start = Instant::now();

    let mut handles = Vec::new();
    for thread_id in 0..num_threads {
        let tenant = tenant.clone();
        let barrier = barrier.clone();
        let total_inserted = total_inserted.clone();

        let handle = tokio::spawn(async move {
            let mut thread_latencies = Vec::new();

            // Wait for all threads to be ready
            barrier.wait().await;

            for batch_num in 0..vectors_per_thread / BATCH_SIZE {
                let vectors: Vec<(u64, Vec<f32>)> = (0..BATCH_SIZE)
                    .map(|i| {
                        let id = (thread_id * vectors_per_thread + batch_num * BATCH_SIZE + i) as u64;
                        (id, normalize(&random_vector(TEST_DIMS)))
                    })
                    .collect();

                let (latency, result) = measure_latency_async(|| tenant.upsert(vectors)).await;
                thread_latencies.push(latency);

                if let Ok(r) = result {
                    total_inserted.fetch_add(r.count, Ordering::Relaxed);
                }
            }

            thread_latencies
        });

        handles.push(handle);
    }

    // Collect all latencies
    for handle in handles {
        if let Ok(thread_latencies) = handle.await {
            latencies.extend(thread_latencies);
        }
    }

    let total_duration = start.elapsed();
    let total_vectors = total_inserted.load(Ordering::Relaxed);

    StressTestResult {
        total_vectors,
        total_duration,
        latency_histogram: LatencyHistogram::from_latencies(latencies),
        throughput: ThroughputResult::new(
            total_vectors / BATCH_SIZE,
            total_vectors,
            total_vectors * TEST_DIMS * 4,
            total_duration,
        ),
    }
}

#[tokio::test]
async fn stress_concurrent_writers_2_threads() {
    let result = run_concurrent_writers_test(2, SMALL_SCALE_COUNT).await;

    println!("\n=== 2 Concurrent Writers Stress Test ===");
    println!("Total vectors inserted: {}", result.total_vectors);
    println!("Total duration: {:?}", result.total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        result.throughput.vectors_per_second
    );
    result.latency_histogram.print_summary("Batch Insert");

    // Assertions
    assert!(result.total_vectors > 0, "Should insert some vectors");
    assert!(
        result.latency_histogram.p99 < Duration::from_secs(1),
        "p99 latency should be under 1 second"
    );
}

#[tokio::test]
async fn stress_concurrent_writers_4_threads() {
    let result = run_concurrent_writers_test(4, SMALL_SCALE_COUNT / 2).await;

    println!("\n=== 4 Concurrent Writers Stress Test ===");
    println!("Total vectors inserted: {}", result.total_vectors);
    println!("Total duration: {:?}", result.total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        result.throughput.vectors_per_second
    );
    result.latency_histogram.print_summary("Batch Insert");

    assert!(result.total_vectors > 0, "Should insert some vectors");
}

#[tokio::test]
async fn stress_concurrent_writers_8_threads() {
    let result = run_concurrent_writers_test(8, SMALL_SCALE_COUNT / 4).await;

    println!("\n=== 8 Concurrent Writers Stress Test ===");
    println!("Total vectors inserted: {}", result.total_vectors);
    println!("Total duration: {:?}", result.total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        result.throughput.vectors_per_second
    );
    result.latency_histogram.print_summary("Batch Insert");

    assert!(result.total_vectors > 0, "Should insert some vectors");
}

#[tokio::test]
async fn stress_concurrent_writers_16_threads() {
    let result = run_concurrent_writers_test(16, SMALL_SCALE_COUNT / 8).await;

    println!("\n=== 16 Concurrent Writers Stress Test ===");
    println!("Total vectors inserted: {}", result.total_vectors);
    println!("Total duration: {:?}", result.total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        result.throughput.vectors_per_second
    );
    result.latency_histogram.print_summary("Batch Insert");

    assert!(result.total_vectors > 0, "Should insert some vectors");
}

// =============================================================================
// READ-WRITE MIXED WORKLOAD TESTS
// =============================================================================

/// Helper to run mixed read-write workload
async fn run_mixed_workload_test(
    write_percentage: usize,
    duration_secs: u64,
) -> MixedWorkloadResult {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Pre-populate with some vectors
    let initial_vectors = generate_random_vectors(1000, TEST_DIMS);
    tenant.upsert(initial_vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let read_percentage = 100 - write_percentage;
    let deadline = Instant::now() + Duration::from_secs(duration_secs);

    let mut write_latencies = Vec::new();
    let mut read_latencies = Vec::new();
    let mut next_id = 1000u64;
    let mut rng = rand::thread_rng();

    while Instant::now() < deadline {
        let is_write = rand::Rng::gen_range(&mut rng, 0..100) < write_percentage;

        if is_write {
            let vectors = vec![(next_id, normalize(&random_vector(TEST_DIMS)))];
            next_id += 1;

            let (latency, _) = measure_latency_async(|| tenant.upsert(vectors)).await;
            write_latencies.push(latency);
        } else {
            let query = normalize(&random_vector(TEST_DIMS));
            let (latency, _) = common::measure_latency(|| tenant.search(&query, 10, None));
            read_latencies.push(latency);
        }
    }

    MixedWorkloadResult {
        write_percentage,
        read_percentage,
        write_count: write_latencies.len(),
        read_count: read_latencies.len(),
        write_latencies: LatencyHistogram::from_latencies(write_latencies),
        read_latencies: LatencyHistogram::from_latencies(read_latencies),
    }
}

#[tokio::test]
async fn stress_mixed_10_write_90_read() {
    let result = run_mixed_workload_test(10, 5).await;

    println!("\n=== Mixed Workload: 10% Write, 90% Read ===");
    println!("Write operations: {}", result.write_count);
    println!("Read operations: {}", result.read_count);
    result.write_latencies.print_summary("Write");
    result.read_latencies.print_summary("Read");

    assert!(result.read_count > result.write_count, "Should have more reads than writes");
}

#[tokio::test]
async fn stress_mixed_50_write_50_read() {
    let result = run_mixed_workload_test(50, 5).await;

    println!("\n=== Mixed Workload: 50% Write, 50% Read ===");
    println!("Write operations: {}", result.write_count);
    println!("Read operations: {}", result.read_count);
    result.write_latencies.print_summary("Write");
    result.read_latencies.print_summary("Read");
}

#[tokio::test]
async fn stress_mixed_90_write_10_read() {
    let result = run_mixed_workload_test(90, 5).await;

    println!("\n=== Mixed Workload: 90% Write, 10% Read ===");
    println!("Write operations: {}", result.write_count);
    println!("Read operations: {}", result.read_count);
    result.write_latencies.print_summary("Write");
    result.read_latencies.print_summary("Read");

    assert!(result.write_count > result.read_count, "Should have more writes than reads");
}

// =============================================================================
// LARGE SCALE TESTS
// =============================================================================

#[tokio::test]
async fn stress_insert_10k_vectors() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    let mut latencies = Vec::new();
    let start = Instant::now();

    // Insert in batches
    for batch_start in (0..MEDIUM_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();

        let (latency, result) = measure_latency_async(|| tenant.upsert(vectors)).await;
        latencies.push(latency);
        assert!(result.is_ok());
    }

    let total_duration = start.elapsed();
    let histogram = LatencyHistogram::from_latencies(latencies);

    println!("\n=== Insert 10K Vectors Test ===");
    println!("Total duration: {:?}", total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        MEDIUM_SCALE_COUNT as f64 / total_duration.as_secs_f64()
    );
    histogram.print_summary("Batch Insert");

    // Check stats
    let stats = tenant.stats();
    assert_eq!(stats.vector_count, MEDIUM_SCALE_COUNT as u64);

    // Performance target: >1000 vec/s
    let throughput = MEDIUM_SCALE_COUNT as f64 / total_duration.as_secs_f64();
    println!("Target: >1000 vec/s, Actual: {:.2} vec/s", throughput);
}

#[tokio::test]
#[ignore] // Run with: cargo test stress_insert_100k_vectors -- --ignored --nocapture
async fn stress_insert_100k_vectors() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    let mut latencies = Vec::new();
    let start = Instant::now();

    // Insert in batches
    for batch_start in (0..LARGE_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();

        let (latency, result) = measure_latency_async(|| tenant.upsert(vectors)).await;
        latencies.push(latency);
        assert!(result.is_ok());

        // Progress indicator
        if batch_start % 10000 == 0 {
            println!("Inserted {} vectors...", batch_start);
        }
    }

    let total_duration = start.elapsed();
    let histogram = LatencyHistogram::from_latencies(latencies);

    println!("\n=== Insert 100K Vectors Test ===");
    println!("Total duration: {:?}", total_duration);
    println!(
        "Throughput: {:.2} vectors/sec",
        LARGE_SCALE_COUNT as f64 / total_duration.as_secs_f64()
    );
    histogram.print_summary("Batch Insert");

    let stats = tenant.stats();
    assert_eq!(stats.vector_count, LARGE_SCALE_COUNT as u64);
}

#[tokio::test]
async fn stress_search_10k_index() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert vectors
    println!("Inserting {} vectors...", MEDIUM_SCALE_COUNT);
    for batch_start in (0..MEDIUM_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Flush to HNSW
    println!("Flushing to HNSW index...");
    let flush_start = Instant::now();
    tenant.flush_to_hnsw(&*storage).await.unwrap();
    let flush_duration = flush_start.elapsed();
    println!("Flush duration: {:?}", flush_duration);

    // Perform searches
    let num_searches = 1000;
    let mut latencies = Vec::new();

    println!("Running {} searches...", num_searches);
    let search_start = Instant::now();

    for _ in 0..num_searches {
        let query = normalize(&random_vector(TEST_DIMS));
        let (latency, results) = common::measure_latency(|| tenant.search(&query, 10, None));
        latencies.push(latency);
        assert!(!results.is_empty());
    }

    let search_duration = search_start.elapsed();
    let histogram = LatencyHistogram::from_latencies(latencies);

    println!("\n=== Search 10K Index Test ===");
    println!("Search duration: {:?}", search_duration);
    println!(
        "Throughput: {:.2} searches/sec",
        num_searches as f64 / search_duration.as_secs_f64()
    );
    histogram.print_summary("Search");

    // Performance targets
    println!("\nPerformance Targets:");
    println!("  p50 target: <10ms, actual: {:?}", histogram.p50);
    println!("  p99 target: <50ms, actual: {:?}", histogram.p99);
}

// =============================================================================
// WRITE BUFFER STRESS TESTS
// =============================================================================

#[tokio::test]
async fn stress_large_write_buffer_10k() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert many vectors WITHOUT flushing to HNSW
    let start = Instant::now();
    for batch_start in (0..MEDIUM_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }
    let insert_duration = start.elapsed();

    println!("\n=== Large Write Buffer (10K) Test ===");
    println!("Insert {} vectors (no flush): {:?}", MEDIUM_SCALE_COUNT, insert_duration);

    // Search with large write buffer (brute force)
    let mut search_latencies = Vec::new();
    for _ in 0..100 {
        let query = normalize(&random_vector(TEST_DIMS));
        let (latency, _) = common::measure_latency(|| tenant.search(&query, 10, None));
        search_latencies.push(latency);
    }

    let histogram = LatencyHistogram::from_latencies(search_latencies);
    histogram.print_summary("Search (with 10K write buffer)");

    // This should be slow due to brute force - document the cost
    println!("\nNote: Search is O(n) brute force on write buffer");
}

#[tokio::test]
async fn stress_search_during_flush() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Insert some vectors
    for batch_start in (0..SMALL_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    let tenant_search = tenant.clone();
    let storage_flush = storage.clone();

    // Spawn search task
    let search_handle = tokio::spawn(async move {
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let query = normalize(&random_vector(TEST_DIMS));
            let (latency, _) = common::measure_latency(|| tenant_search.search(&query, 10, None));
            latencies.push(latency);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        latencies
    });

    // Run flush concurrently
    let flush_handle = tokio::spawn(async move {
        let start = Instant::now();
        tenant.flush_to_hnsw(&*storage_flush).await.unwrap();
        start.elapsed()
    });

    let (search_latencies, flush_duration) = tokio::join!(search_handle, flush_handle);
    let search_latencies = search_latencies.unwrap();
    let flush_duration = flush_duration.unwrap();

    let histogram = LatencyHistogram::from_latencies(search_latencies);

    println!("\n=== Search During Flush Test ===");
    println!("Flush duration: {:?}", flush_duration);
    histogram.print_summary("Search (during flush)");
}

#[tokio::test]
async fn stress_concurrent_flush() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Insert vectors
    for batch_start in (0..SMALL_SCALE_COUNT).step_by(BATCH_SIZE) {
        let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
            .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Try concurrent flushes (should be safe)
    let tenant1 = tenant.clone();
    let tenant2 = tenant.clone();
    let storage1 = storage.clone();
    let storage2 = storage.clone();

    let (result1, result2) = tokio::join!(
        async { tenant1.flush_to_hnsw(&*storage1).await },
        async { tenant2.flush_to_hnsw(&*storage2).await }
    );

    println!("\n=== Concurrent Flush Test ===");
    println!("Flush 1 result: {:?}", result1);
    println!("Flush 2 result: {:?}", result2);

    // Both should succeed (one gets all vectors, other gets 0)
    assert!(result1.is_ok());
    assert!(result2.is_ok());

    let stats = tenant.stats();
    println!("Final stats: {} vectors in HNSW", stats.hnsw_nodes);
}

// =============================================================================
// MULTI-TENANT STRESS TESTS
// =============================================================================

#[tokio::test]
async fn stress_10_tenants_concurrent() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let num_tenants = 10;
    let vectors_per_tenant = 500;

    let mut handles = Vec::new();
    let start = Instant::now();

    for tenant_id in 0..num_tenants {
        let storage = storage.clone();
        let handle = tokio::spawn(async move {
            let config = HnswConfig::new(TEST_DIMS);
            let tenant = TenantState::open(tenant_id as u64, TEST_DIMS, storage.clone(), config)
                .await
                .unwrap();

            let mut latencies = Vec::new();
            for batch_start in (0..vectors_per_tenant).step_by(BATCH_SIZE) {
                let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
                    .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
                    .collect();

                let (latency, _) = measure_latency_async(|| tenant.upsert(vectors)).await;
                latencies.push(latency);
            }

            // Flush
            tenant.flush_to_hnsw(&*storage).await.unwrap();

            // Search
            let mut search_latencies = Vec::new();
            for _ in 0..50 {
                let query = normalize(&random_vector(TEST_DIMS));
                let (latency, _) = common::measure_latency(|| tenant.search(&query, 10, None));
                search_latencies.push(latency);
            }

            (latencies, search_latencies, tenant.stats())
        });

        handles.push(handle);
    }

    let mut all_insert_latencies = Vec::new();
    let mut all_search_latencies = Vec::new();
    let mut total_vectors = 0u64;

    for handle in handles {
        let (insert_lat, search_lat, stats) = handle.await.unwrap();
        all_insert_latencies.extend(insert_lat);
        all_search_latencies.extend(search_lat);
        total_vectors += stats.vector_count;
    }

    let total_duration = start.elapsed();

    println!("\n=== 10 Tenants Concurrent Test ===");
    println!("Total duration: {:?}", total_duration);
    println!("Total vectors across all tenants: {}", total_vectors);

    let insert_histogram = LatencyHistogram::from_latencies(all_insert_latencies);
    let search_histogram = LatencyHistogram::from_latencies(all_search_latencies);

    insert_histogram.print_summary("Insert (all tenants)");
    search_histogram.print_summary("Search (all tenants)");

    assert_eq!(total_vectors, (num_tenants * vectors_per_tenant) as u64);
}

#[tokio::test]
async fn stress_tenant_isolation_under_load() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());

    // Create two tenants with different data
    let config = HnswConfig::new(TEST_DIMS);
    let tenant1 = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
        .await
        .unwrap();
    let tenant2 = TenantState::open(2, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert distinct vectors
    let vectors1: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| (i, normalize(&random_vector(TEST_DIMS))))
        .collect();
    let vectors2: Vec<(u64, Vec<f32>)> = (1000..1100)
        .map(|i| (i, normalize(&random_vector(TEST_DIMS))))
        .collect();

    tenant1.upsert(vectors1).await.unwrap();
    tenant2.upsert(vectors2).await.unwrap();

    tenant1.flush_to_hnsw(&*storage).await.unwrap();
    tenant2.flush_to_hnsw(&*storage).await.unwrap();

    // Verify isolation
    let query = normalize(&random_vector(TEST_DIMS));
    let results1 = tenant1.search(&query, 10, None);
    let results2 = tenant2.search(&query, 10, None);

    println!("\n=== Tenant Isolation Test ===");
    println!("Tenant 1 results: {:?}", results1.iter().map(|r| r.id).collect::<Vec<_>>());
    println!("Tenant 2 results: {:?}", results2.iter().map(|r| r.id).collect::<Vec<_>>());

    // Verify tenant 1 only returns IDs 0-99
    for r in &results1 {
        assert!(r.id < 100, "Tenant 1 should only have IDs 0-99");
    }

    // Verify tenant 2 only returns IDs 1000-1099
    for r in &results2 {
        assert!(r.id >= 1000 && r.id < 1100, "Tenant 2 should only have IDs 1000-1099");
    }

    println!("Tenant isolation verified!");
}

#[tokio::test]
async fn stress_background_flush_all_tenants() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let num_tenants = 5;

    let mut tenants = Vec::new();
    for tenant_id in 0..num_tenants {
        let config = HnswConfig::new(TEST_DIMS);
        let tenant = Arc::new(
            TenantState::open(tenant_id, TEST_DIMS, storage.clone(), config)
                .await
                .unwrap(),
        );

        // Insert vectors
        for batch_start in (0..500).step_by(BATCH_SIZE) {
            let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
                .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }

        tenants.push(tenant);
    }

    // Flush all tenants concurrently
    let start = Instant::now();
    let mut handles = Vec::new();

    for tenant in &tenants {
        let tenant = tenant.clone();
        let storage = storage.clone();
        handles.push(tokio::spawn(async move {
            tenant.flush_to_hnsw(&*storage).await
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    let flush_duration = start.elapsed();

    println!("\n=== Background Flush All Tenants Test ===");
    println!("Flush {} tenants concurrently: {:?}", num_tenants, flush_duration);

    for (i, result) in results.iter().enumerate() {
        println!("Tenant {} flush result: {:?}", i, result);
        assert!(result.is_ok());
    }
}

// =============================================================================
// HELPER STRUCTURES
// =============================================================================

#[derive(Debug)]
struct StressTestResult {
    total_vectors: usize,
    total_duration: Duration,
    latency_histogram: LatencyHistogram,
    throughput: ThroughputResult,
}

#[derive(Debug)]
struct MixedWorkloadResult {
    write_percentage: usize,
    read_percentage: usize,
    write_count: usize,
    read_count: usize,
    write_latencies: LatencyHistogram,
    read_latencies: LatencyHistogram,
}
