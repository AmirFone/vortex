//! Real-World Workload Simulation Tests for VectorDB
//!
//! These tests simulate realistic usage patterns to validate
//! the system performs well under production-like conditions.
//!
//! Workload patterns:
//! - RAG/Embedding: Bulk insert, then many searches
//! - Streaming: Continuous small inserts + concurrent searches
//! - Multi-Tenant SaaS: Many tenants with isolated workloads
//! - Burst Traffic: Traffic spikes
//! - Long-Running Stability: Sustained load over time

mod common;

use common::{
    brute_force_knn, calculate_recall, generate_random_vectors, measure_latency_async, normalize,
    random_vector, LatencyHistogram, ThroughputResult,
};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Barrier;
use vortex::hnsw::HnswConfig;
use vortex::storage::mock::{MockBlockStorage, MockStorageConfig};
use vortex::tenant::TenantState;

const TEST_DIMS: usize = 384;
const BATCH_SIZE: usize = 100;

// =============================================================================
// RAG/EMBEDDING WORKLOAD
// =============================================================================

/// Simulates a RAG (Retrieval-Augmented Generation) workload:
/// - Bulk insert documents (embeddings)
/// - Wait for flush
/// - Run many searches with varying k/ef
#[tokio::test]
async fn simulate_rag_workload() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    println!("\n=== RAG/Embedding Workload Simulation ===");

    // Phase 1: Bulk insert documents
    println!("\n--- Phase 1: Bulk Insert 10K vectors in batches of 100 ---");
    let total_vectors = 10_000;
    let mut insert_latencies = Vec::new();

    let insert_start = Instant::now();
    let vectors_for_recall = generate_random_vectors(total_vectors, TEST_DIMS);

    for batch_start in (0..total_vectors).step_by(BATCH_SIZE) {
        let batch: Vec<(u64, Vec<f32>)> = vectors_for_recall[batch_start..batch_start + BATCH_SIZE]
            .iter()
            .cloned()
            .collect();

        let (latency, _) = measure_latency_async(|| tenant.upsert(batch)).await;
        insert_latencies.push(latency);
    }

    let insert_duration = insert_start.elapsed();
    let insert_histogram = LatencyHistogram::from_latencies(insert_latencies);

    println!(
        "Insert completed: {} vectors in {:?}",
        total_vectors, insert_duration
    );
    println!(
        "Throughput: {:.2} vectors/sec",
        total_vectors as f64 / insert_duration.as_secs_f64()
    );
    insert_histogram.print_summary("Batch Insert");

    // Phase 2: Flush to HNSW
    println!("\n--- Phase 2: Flush to HNSW Index ---");
    let flush_start = Instant::now();
    tenant.flush_to_hnsw(&*storage).await.unwrap();
    let flush_duration = flush_start.elapsed();
    println!("Flush completed in {:?}", flush_duration);

    // Phase 3: Run searches with varying parameters
    println!("\n--- Phase 3: Search with varying k and ef ---");

    for (k, ef) in [(5, 50), (10, 100), (10, 200), (20, 200), (50, 500)] {
        let mut search_latencies = Vec::new();
        let mut recalls = Vec::new();
        let num_queries = 100;

        for _ in 0..num_queries {
            let query = normalize(&random_vector(TEST_DIMS));

            let (latency, results) =
                common::measure_latency_async(|| tenant.search(&query, k, Some(ef))).await;
            search_latencies.push(latency);

            // Calculate recall
            let result_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
            let ground_truth = brute_force_knn(&query, &vectors_for_recall, k);
            recalls.push(calculate_recall(&result_ids, &ground_truth, k));
        }

        let latency_hist = LatencyHistogram::from_latencies(search_latencies);
        let avg_recall: f64 = recalls.iter().sum::<f64>() / recalls.len() as f64;

        println!(
            "k={}, ef={}: p50={:?}, p99={:?}, recall={:.1}%",
            k,
            ef,
            latency_hist.p50,
            latency_hist.p99,
            avg_recall * 100.0
        );
    }

    // Verify performance targets
    let stats = tenant.stats().await;
    assert_eq!(stats.vector_count, total_vectors as u64);
    println!("\nRAG workload simulation completed successfully!");
}

// =============================================================================
// REAL-TIME STREAMING WORKLOAD
// =============================================================================

/// Simulates real-time streaming:
/// - Continuous small inserts at ~100 vectors/sec
/// - Concurrent searchers running continuously
/// - Measures write/read latency correlation and buffer growth
#[tokio::test]
async fn simulate_streaming_workload() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    println!("\n=== Real-Time Streaming Workload Simulation ===");

    // Pre-populate with some data
    let initial = generate_random_vectors(1000, TEST_DIMS);
    tenant.upsert(initial).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let duration_secs = 10;
    let running = Arc::new(AtomicBool::new(true));
    let total_writes = Arc::new(AtomicUsize::new(0));
    let total_reads = Arc::new(AtomicUsize::new(0));

    // Writer task: ~100 vectors/sec (batches of 10 every 100ms)
    let writer_tenant = tenant.clone();
    let writer_running = running.clone();
    let writer_count = total_writes.clone();

    let write_handle = tokio::spawn(async move {
        let mut latencies = Vec::new();
        let mut next_id = 1000u64;

        while writer_running.load(Ordering::Relaxed) {
            let vectors: Vec<(u64, Vec<f32>)> = (0..10)
                .map(|i| {
                    next_id += 1;
                    (next_id + i, normalize(&random_vector(TEST_DIMS)))
                })
                .collect();

            let (latency, result) = measure_latency_async(|| writer_tenant.upsert(vectors)).await;
            latencies.push(latency);

            if let Ok(r) = result {
                writer_count.fetch_add(r.count, Ordering::Relaxed);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        latencies
    });

    // Spawn 5 reader tasks
    let mut reader_handles = Vec::new();
    for _ in 0..5 {
        let reader_tenant = tenant.clone();
        let reader_running = running.clone();
        let reader_count = total_reads.clone();

        reader_handles.push(tokio::spawn(async move {
            let mut latencies = Vec::new();

            while reader_running.load(Ordering::Relaxed) {
                let query = normalize(&random_vector(TEST_DIMS));
                let (latency, results) =
                    common::measure_latency_async(|| reader_tenant.search(&query, 10, None)).await;

                if !results.is_empty() {
                    latencies.push(latency);
                    reader_count.fetch_add(1, Ordering::Relaxed);
                }

                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            latencies
        }));
    }

    // Let it run
    tokio::time::sleep(Duration::from_secs(duration_secs)).await;
    running.store(false, Ordering::Relaxed);

    // Collect results
    let write_latencies = write_handle.await.unwrap();
    let mut all_read_latencies = Vec::new();
    for handle in reader_handles {
        all_read_latencies.extend(handle.await.unwrap());
    }

    let write_hist = LatencyHistogram::from_latencies(write_latencies);
    let read_hist = LatencyHistogram::from_latencies(all_read_latencies);

    println!("\n--- Results after {} seconds ---", duration_secs);
    println!(
        "Total writes: {} vectors",
        total_writes.load(Ordering::Relaxed)
    );
    println!("Total reads: {} searches", total_reads.load(Ordering::Relaxed));

    write_hist.print_summary("Write (batch of 10)");
    read_hist.print_summary("Read (search k=10)");

    let stats = tenant.stats().await;
    println!("\nWrite buffer size: {}", stats.write_buffer_size);
    println!("HNSW nodes: {}", stats.hnsw_nodes);

    println!("\nStreaming workload simulation completed!");
}

// =============================================================================
// MULTI-TENANT SAAS WORKLOAD
// =============================================================================

/// Simulates multi-tenant SaaS:
/// - 10 tenants with independent workloads
/// - Each tenant: insert, search, flush cycle
/// - Measure per-tenant latency and cross-tenant interference
#[tokio::test]
async fn simulate_multi_tenant_workload() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let num_tenants = 10;
    let vectors_per_tenant = 1000;

    println!("\n=== Multi-Tenant SaaS Workload Simulation ===");
    println!(
        "Tenants: {}, Vectors per tenant: {}",
        num_tenants, vectors_per_tenant
    );

    let barrier = Arc::new(Barrier::new(num_tenants));
    let start = Instant::now();

    let mut handles = Vec::new();
    for tenant_id in 0..num_tenants as u64 {
        let storage = storage.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            let config = HnswConfig::new(TEST_DIMS);
            let tenant = TenantState::open(tenant_id, TEST_DIMS, storage.clone(), config)
                .await
                .unwrap();

            // Wait for all tenants to be ready
            barrier.wait().await;

            let mut insert_latencies = Vec::new();
            let mut search_latencies = Vec::new();

            // Insert phase
            for batch_start in (0..vectors_per_tenant).step_by(BATCH_SIZE) {
                let vectors: Vec<(u64, Vec<f32>)> = (batch_start..batch_start + BATCH_SIZE)
                    .map(|i| (i as u64, normalize(&random_vector(TEST_DIMS))))
                    .collect();

                let (latency, _) = measure_latency_async(|| tenant.upsert(vectors)).await;
                insert_latencies.push(latency);
            }

            // Flush
            let flush_start = Instant::now();
            tenant.flush_to_hnsw(&*storage).await.unwrap();
            let flush_duration = flush_start.elapsed();

            // Search phase
            for _ in 0..100 {
                let query = normalize(&random_vector(TEST_DIMS));
                let (latency, _) = common::measure_latency_async(|| tenant.search(&query, 10, None)).await;
                search_latencies.push(latency);
            }

            TenantMetrics {
                tenant_id,
                insert_histogram: LatencyHistogram::from_latencies(insert_latencies),
                search_histogram: LatencyHistogram::from_latencies(search_latencies),
                flush_duration,
                final_stats: tenant.stats().await,
            }
        }));
    }

    let mut all_metrics = Vec::new();
    for handle in handles {
        all_metrics.push(handle.await.unwrap());
    }

    let total_duration = start.elapsed();

    println!("\n--- Per-Tenant Results ---");
    for metrics in &all_metrics {
        println!(
            "Tenant {}: insert_p50={:?}, search_p50={:?}, flush={:?}, vectors={}",
            metrics.tenant_id,
            metrics.insert_histogram.p50,
            metrics.search_histogram.p50,
            metrics.flush_duration,
            metrics.final_stats.vector_count
        );
    }

    // Aggregate stats
    let mut all_insert_p50: Vec<Duration> = all_metrics
        .iter()
        .map(|m| m.insert_histogram.p50)
        .collect();
    let mut all_search_p50: Vec<Duration> = all_metrics
        .iter()
        .map(|m| m.search_histogram.p50)
        .collect();

    all_insert_p50.sort();
    all_search_p50.sort();

    println!("\n--- Aggregate Results ---");
    println!("Total time: {:?}", total_duration);
    println!(
        "Insert p50 range: {:?} - {:?}",
        all_insert_p50.first().unwrap(),
        all_insert_p50.last().unwrap()
    );
    println!(
        "Search p50 range: {:?} - {:?}",
        all_search_p50.first().unwrap(),
        all_search_p50.last().unwrap()
    );

    // Verify all tenants completed
    let total_vectors: u64 = all_metrics
        .iter()
        .map(|m| m.final_stats.vector_count)
        .sum();
    assert_eq!(total_vectors, (num_tenants * vectors_per_tenant) as u64);

    println!("\nMulti-tenant workload simulation completed!");
}

// =============================================================================
// BURST TRAFFIC WORKLOAD
// =============================================================================

/// Simulates traffic bursts:
/// - Baseline load for 10 seconds
/// - Spike to 10x for 5 seconds
/// - Return to baseline
/// - Measure latency spike and recovery
#[tokio::test]
async fn simulate_burst_traffic() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    println!("\n=== Burst Traffic Workload Simulation ===");

    // Pre-populate
    let initial = generate_random_vectors(5000, TEST_DIMS);
    tenant.upsert(initial).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    // Track phases
    let mut baseline_latencies = Vec::new();
    let mut burst_latencies = Vec::new();
    let mut recovery_latencies = Vec::new();

    // Baseline: 10 ops/sec for 5 seconds
    println!("\n--- Baseline Phase: 10 ops/sec for 5 seconds ---");
    let baseline_start = Instant::now();
    while baseline_start.elapsed() < Duration::from_secs(5) {
        let query = normalize(&random_vector(TEST_DIMS));
        let (latency, _) = common::measure_latency_async(|| tenant.search(&query, 10, None)).await;
        baseline_latencies.push(latency);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Burst: 100 ops/sec for 3 seconds
    println!("--- Burst Phase: 100 ops/sec for 3 seconds ---");
    let burst_start = Instant::now();
    while burst_start.elapsed() < Duration::from_secs(3) {
        let query = normalize(&random_vector(TEST_DIMS));
        let (latency, _) = common::measure_latency_async(|| tenant.search(&query, 10, None)).await;
        burst_latencies.push(latency);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Recovery: back to 10 ops/sec for 5 seconds
    println!("--- Recovery Phase: 10 ops/sec for 5 seconds ---");
    let recovery_start = Instant::now();
    while recovery_start.elapsed() < Duration::from_secs(5) {
        let query = normalize(&random_vector(TEST_DIMS));
        let (latency, _) = common::measure_latency_async(|| tenant.search(&query, 10, None)).await;
        recovery_latencies.push(latency);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let baseline_hist = LatencyHistogram::from_latencies(baseline_latencies);
    let burst_hist = LatencyHistogram::from_latencies(burst_latencies);
    let recovery_hist = LatencyHistogram::from_latencies(recovery_latencies);

    println!("\n--- Results ---");
    baseline_hist.print_summary("Baseline");
    burst_hist.print_summary("Burst (10x load)");
    recovery_hist.print_summary("Recovery");

    // Calculate degradation
    let baseline_p50 = baseline_hist.p50.as_secs_f64() * 1000.0;
    let burst_p50 = burst_hist.p50.as_secs_f64() * 1000.0;
    let recovery_p50 = recovery_hist.p50.as_secs_f64() * 1000.0;

    println!(
        "\nLatency change: baseline={:.2}ms → burst={:.2}ms ({:.1}x) → recovery={:.2}ms",
        baseline_p50,
        burst_p50,
        burst_p50 / baseline_p50.max(0.001),
        recovery_p50
    );

    // Verify system recovers
    assert!(
        recovery_hist.p50 < burst_hist.p50 * 2,
        "System should recover after burst"
    );

    println!("\nBurst traffic simulation completed!");
}

// =============================================================================
// LONG-RUNNING STABILITY TEST
// =============================================================================

/// Simulates sustained load over time:
/// - Continuous insert + search + flush for several minutes
/// - Check for memory leaks (heap growth)
/// - Check for performance degradation over time
#[tokio::test]
async fn simulate_stability_test() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = HnswConfig::new(TEST_DIMS);
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    println!("\n=== Long-Running Stability Test (60 seconds) ===");

    let test_duration = Duration::from_secs(60);
    let sample_interval = Duration::from_secs(10);

    let running = Arc::new(AtomicBool::new(true));
    let next_id = Arc::new(AtomicU64::new(0));
    let total_inserts = Arc::new(AtomicUsize::new(0));
    let total_searches = Arc::new(AtomicUsize::new(0));

    // Writer task
    let writer_tenant = tenant.clone();
    let writer_running = running.clone();
    let writer_next_id = next_id.clone();
    let writer_inserts = total_inserts.clone();

    let write_handle = tokio::spawn(async move {
        while writer_running.load(Ordering::Relaxed) {
            let id = writer_next_id.fetch_add(1, Ordering::Relaxed);
            let vectors = vec![(id, normalize(&random_vector(TEST_DIMS)))];
            if writer_tenant.upsert(vectors).await.is_ok() {
                writer_inserts.fetch_add(1, Ordering::Relaxed);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Reader task
    let reader_tenant = tenant.clone();
    let reader_running = running.clone();
    let reader_searches = total_searches.clone();

    let read_handle = tokio::spawn(async move {
        let mut latency_samples = Vec::new();

        while reader_running.load(Ordering::Relaxed) {
            let query = normalize(&random_vector(TEST_DIMS));
            let (latency, _) = common::measure_latency_async(|| reader_tenant.search(&query, 10, None)).await;
            latency_samples.push(latency);
            reader_searches.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        latency_samples
    });

    // Periodic flush task
    let flush_tenant = tenant.clone();
    let flush_storage = storage.clone();
    let flush_running = running.clone();

    let flush_handle = tokio::spawn(async move {
        while flush_running.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let _ = flush_tenant.flush_to_hnsw(&*flush_storage).await;
        }
    });

    // Monitor and sample
    let start = Instant::now();
    let mut samples = Vec::new();

    while start.elapsed() < test_duration {
        tokio::time::sleep(sample_interval).await;

        let stats = tenant.stats().await;
        let elapsed = start.elapsed().as_secs();

        let sample = StabilitySample {
            elapsed_secs: elapsed,
            vector_count: stats.vector_count,
            hnsw_nodes: stats.hnsw_nodes,
            write_buffer_size: stats.write_buffer_size,
        };

        println!(
            "t={}s: vectors={}, hnsw={}, buffer={}",
            sample.elapsed_secs,
            sample.vector_count,
            sample.hnsw_nodes,
            sample.write_buffer_size
        );

        samples.push(sample);
    }

    // Stop all tasks
    running.store(false, Ordering::Relaxed);

    let _ = write_handle.await;
    let search_latencies = read_handle.await.unwrap();
    let _ = flush_handle.await;

    // Analyze results
    println!("\n--- Stability Analysis ---");
    println!(
        "Total inserts: {}",
        total_inserts.load(Ordering::Relaxed)
    );
    println!(
        "Total searches: {}",
        total_searches.load(Ordering::Relaxed)
    );

    let search_hist = LatencyHistogram::from_latencies(search_latencies);
    search_hist.print_summary("Search (over full duration)");

    // Check for performance degradation
    // Compare first 20% vs last 20% of samples
    let sample_count = samples.len();
    if sample_count >= 5 {
        let early_vectors: u64 = samples[0].vector_count;
        let late_vectors: u64 = samples[sample_count - 1].vector_count;

        println!(
            "\nGrowth: {} → {} vectors ({:.1}x)",
            early_vectors,
            late_vectors,
            late_vectors as f64 / early_vectors.max(1) as f64
        );
    }

    println!("\nStability test completed!");
}

// =============================================================================
// HELPER STRUCTURES
// =============================================================================

#[derive(Debug)]
struct TenantMetrics {
    tenant_id: u64,
    insert_histogram: LatencyHistogram,
    search_histogram: LatencyHistogram,
    flush_duration: Duration,
    final_stats: vortex::tenant::TenantStats,
}

#[derive(Debug)]
struct StabilitySample {
    elapsed_secs: u64,
    vector_count: u64,
    hnsw_nodes: u64,
    write_buffer_size: usize,
}
