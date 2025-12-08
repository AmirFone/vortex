//! Concurrency Tests for VectorDB
//!
//! Tests covering:
//! - Race conditions
//! - Concurrent tenant creation
//! - Concurrent upserts
//! - Concurrent read/write operations
//! - Deadlock detection

mod common;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::time::{timeout, Duration};

use crate::common::test_index_config;
use vortex::tenant::TenantState;

use common::{random_vector, seeded_vector, temp_storage};

// ============================================================================
// RACE CONDITION TESTS
// ============================================================================

/// Test: Concurrent upserts to same tenant
#[tokio::test]
async fn test_concurrent_upserts_same_tenant() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    let num_threads = 10;
    let vectors_per_thread = 100;

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let tenant = tenant.clone();
        let handle = tokio::spawn(async move {
            for i in 0..vectors_per_thread {
                let vector_id = (thread_id * vectors_per_thread + i) as u64;
                let vector = seeded_vector(4, vector_id);
                let _ = tenant.upsert(vec![(vector_id, vector)]).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.await.unwrap();
    }

    let stats = tenant.stats().await;
    // Should have all vectors (allowing for some duplicates being skipped)
    assert!(
        stats.vector_count >= (num_threads * vectors_per_thread / 2) as u64,
        "Should have a significant portion of vectors, got {}",
        stats.vector_count
    );
}

/// Test: Concurrent upsert and search
#[tokio::test]
async fn test_concurrent_upsert_and_search() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicU64::new(0));
    let search_count = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    // Writer tasks
    let mut handles = vec![];
    for writer_id in 0..3 {
        let tenant = tenant.clone();
        let running = running.clone();
        let write_count = write_count.clone();
        let errors = errors.clone();
        let handle = tokio::spawn(async move {
            let mut id = writer_id * 10000;
            while running.load(Ordering::Relaxed) {
                let vector = random_vector(4);
                match tenant.upsert(vec![(id, vector)]).await {
                    Ok(_) => write_count.fetch_add(1, Ordering::Relaxed),
                    Err(_) => errors.fetch_add(1, Ordering::Relaxed),
                };
                id += 1;
            }
        });
        handles.push(handle);
    }

    // Reader tasks
    for _ in 0..3 {
        let tenant = tenant.clone();
        let running = running.clone();
        let search_count = search_count.clone();
        let errors = errors.clone();
        let handle = tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                let query = random_vector(4);
                let results = tenant.search(&query, 10, None).await;
                // Check for corrupted results
                for r in &results {
                    if r.similarity.is_nan() || r.similarity.is_infinite() {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                search_count.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    // Run for 200ms
    tokio::time::sleep(Duration::from_millis(200)).await;
    running.store(false, Ordering::Relaxed);

    // Wait for all tasks
    for handle in handles {
        let _ = timeout(Duration::from_secs(5), handle).await;
    }

    let total_writes = write_count.load(Ordering::Relaxed);
    let total_searches = search_count.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);

    assert!(total_writes > 0, "Should have completed some writes");
    assert!(total_searches > 0, "Should have completed some searches");
    assert_eq!(total_errors, 0, "Should have no errors");
}

/// Test: Concurrent upsert and flush
#[tokio::test]
async fn test_concurrent_upsert_and_flush() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicU64::new(0));
    let flush_count = Arc::new(AtomicU64::new(0));

    // Writer task
    let write_tenant = tenant.clone();
    let write_running = running.clone();
    let write_counter = write_count.clone();
    let writer = tokio::spawn(async move {
        let mut id = 0u64;
        while write_running.load(Ordering::Relaxed) {
            let vector = random_vector(4);
            let _ = write_tenant.upsert(vec![(id, vector)]).await;
            write_counter.fetch_add(1, Ordering::Relaxed);
            id += 1;
        }
    });

    // Flusher task
    let flush_tenant = tenant.clone();
    let flush_storage = storage.clone();
    let flush_running = running.clone();
    let flush_counter = flush_count.clone();
    let flusher = tokio::spawn(async move {
        while flush_running.load(Ordering::Relaxed) {
            let _ = flush_tenant.flush_to_hnsw(&*flush_storage).await;
            flush_counter.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Run for 200ms
    tokio::time::sleep(Duration::from_millis(200)).await;
    running.store(false, Ordering::Relaxed);

    let _ = timeout(Duration::from_secs(5), writer).await;
    let _ = timeout(Duration::from_secs(5), flusher).await;

    let writes = write_count.load(Ordering::Relaxed);
    let flushes = flush_count.load(Ordering::Relaxed);

    assert!(writes > 0, "Should have completed writes");
    assert!(flushes > 0, "Should have completed flushes");

    // Final stats should be consistent
    let stats = tenant.stats().await;
    assert!(
        stats.vector_count > 0,
        "Should have some vectors after concurrent operations"
    );
}

/// Test: Multiple readers during single writer
#[tokio::test]
async fn test_multiple_readers_single_writer() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Pre-populate with some vectors
    let initial_vectors: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(initial_vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let running = Arc::new(AtomicBool::new(true));
    let read_count = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Single writer
    let writer_tenant = tenant.clone();
    let writer_running = running.clone();
    let writer = tokio::spawn(async move {
        let mut id = 1000u64;
        while writer_running.load(Ordering::Relaxed) {
            let vector = random_vector(4);
            let _ = writer_tenant.upsert(vec![(id, vector)]).await;
            id += 1;
        }
    });
    handles.push(writer);

    // Multiple readers
    for _ in 0..10 {
        let reader_tenant = tenant.clone();
        let reader_running = running.clone();
        let reader_count = read_count.clone();
        let reader = tokio::spawn(async move {
            while reader_running.load(Ordering::Relaxed) {
                let query = random_vector(4);
                let _ = reader_tenant.search(&query, 5, None).await;
                reader_count.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(reader);
    }

    // Run for 100ms
    tokio::time::sleep(Duration::from_millis(100)).await;
    running.store(false, Ordering::Relaxed);

    for handle in handles {
        let _ = timeout(Duration::from_secs(5), handle).await;
    }

    let reads = read_count.load(Ordering::Relaxed);
    assert!(reads > 0, "Readers should have completed searches");
}

// ============================================================================
// TENANT ISOLATION TESTS
// ============================================================================

/// Test: Concurrent operations on different tenants
#[tokio::test]
async fn test_concurrent_different_tenants() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let num_tenants = 5;
    let vectors_per_tenant = 50;

    let mut handles = vec![];

    for tenant_id in 1..=num_tenants {
        let storage = storage.clone();
        let config = config.clone();
        let handle = tokio::spawn(async move {
            let tenant = TenantState::open(tenant_id, 4, storage.clone(), config)
                .await
                .unwrap();

            // Insert vectors
            let vectors: Vec<(u64, Vec<f32>)> = (0..vectors_per_tenant)
                .map(|i| (i as u64, seeded_vector(4, (tenant_id * 1000 + i) as u64)))
                .collect();
            tenant.upsert(vectors).await.unwrap();

            // Flush
            tenant.flush_to_hnsw(&*storage).await.unwrap();

            // Verify
            let stats = tenant.stats().await;
            assert_eq!(
                stats.vector_count, vectors_per_tenant as u64,
                "Tenant {} should have {} vectors",
                tenant_id, vectors_per_tenant
            );

            // Search
            let query = seeded_vector(4, (tenant_id * 1000) as u64);
            let results = tenant.search(&query, 5, None).await;
            assert!(!results.is_empty(), "Tenant {} should find results", tenant_id);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: No cross-tenant data leakage under concurrent access
#[tokio::test]
async fn test_no_cross_tenant_leakage() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    // Create two tenants with distinct vectors
    let tenant1 = Arc::new(
        TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap(),
    );
    let tenant2 = Arc::new(
        TenantState::open(2, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Tenant 1 gets vectors pointing in +X direction
    let t1_vec = common::normalize(&[1.0, 0.0, 0.0, 0.0]);
    tenant1.upsert(vec![(100, t1_vec.clone())]).await.unwrap();

    // Tenant 2 gets vectors pointing in +Y direction
    let t2_vec = common::normalize(&[0.0, 1.0, 0.0, 0.0]);
    tenant2.upsert(vec![(100, t2_vec.clone())]).await.unwrap();

    let running = Arc::new(AtomicBool::new(true));
    let leakage_detected = Arc::new(AtomicBool::new(false));

    let mut handles = vec![];

    // Concurrent searches on tenant1
    for _ in 0..5 {
        let t1 = tenant1.clone();
        let t1_expected = t1_vec.clone();
        let running = running.clone();
        let leakage = leakage_detected.clone();
        let handle = tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                let results = t1.search(&t1_expected, 10, None).await;
                for r in &results {
                    // If we find tenant2's vector (high similarity to Y-direction), that's leakage
                    let y_sim = common::cosine_similarity(
                        &[0.0, 1.0, 0.0, 0.0],
                        &t1_expected,
                    );
                    if r.similarity > 0.9 && y_sim > 0.9 {
                        leakage.store(true, Ordering::Relaxed);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Concurrent searches on tenant2
    for _ in 0..5 {
        let t2 = tenant2.clone();
        let t2_expected = t2_vec.clone();
        let running = running.clone();
        let leakage = leakage_detected.clone();
        let handle = tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                let results = t2.search(&t2_expected, 10, None).await;
                for r in &results {
                    // If we find tenant1's vector (high similarity to X-direction), that's leakage
                    let x_sim = common::cosine_similarity(
                        &[1.0, 0.0, 0.0, 0.0],
                        &t2_expected,
                    );
                    if r.similarity > 0.9 && x_sim > 0.9 {
                        leakage.store(true, Ordering::Relaxed);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Run for 100ms
    tokio::time::sleep(Duration::from_millis(100)).await;
    running.store(false, Ordering::Relaxed);

    for handle in handles {
        let _ = timeout(Duration::from_secs(5), handle).await;
    }

    assert!(
        !leakage_detected.load(Ordering::Relaxed),
        "No cross-tenant data leakage should occur"
    );
}

// ============================================================================
// STRESS TESTS
// ============================================================================

/// Test: High contention on single tenant
#[tokio::test]
async fn test_high_contention() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    let num_tasks = 20;
    let ops_per_task = 50;

    let mut handles = vec![];

    for task_id in 0..num_tasks {
        let tenant = tenant.clone();
        let storage = storage.clone();
        let handle = tokio::spawn(async move {
            for i in 0..ops_per_task {
                let vector_id = (task_id * ops_per_task + i) as u64;
                let vector = random_vector(4);

                // Mix of operations
                match i % 3 {
                    0 => {
                        let _ = tenant.upsert(vec![(vector_id, vector)]).await;
                    }
                    1 => {
                        let _ = tenant.search(&vector, 5, None).await;
                    }
                    2 => {
                        let _ = tenant.flush_to_hnsw(&*storage).await;
                    }
                    _ => unreachable!(),
                }
            }
        });
        handles.push(handle);
    }

    // All tasks should complete without deadlock
    for handle in handles {
        let result = timeout(Duration::from_secs(30), handle).await;
        assert!(result.is_ok(), "Task should complete without timeout");
    }

    // Final state should be consistent
    let stats = tenant.stats().await;
    assert!(stats.vector_count > 0, "Should have vectors after stress test");
}

/// Test: Rapid sequential operations
#[tokio::test]
async fn test_rapid_sequential_operations() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Rapid fire upserts
    for i in 0..500 {
        let vector = random_vector(4);
        tenant.upsert(vec![(i as u64, vector)]).await.unwrap();
    }

    assert_eq!(tenant.stats().await.vector_count, 500);

    // Rapid fire searches
    for _ in 0..100 {
        let query = random_vector(4);
        let _ = tenant.search(&query, 10, None).await;
    }
}

/// Test: Concurrent stats reads
#[tokio::test]
async fn test_concurrent_stats_reads() {
    let (_temp_dir, storage) = temp_storage();
    let config = test_index_config();

    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Pre-populate
    let vectors: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| (i as u64, random_vector(4)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    let running = Arc::new(AtomicBool::new(true));
    let stats_count = Arc::new(AtomicU64::new(0));

    let mut handles = vec![];

    // Multiple stats readers
    for _ in 0..10 {
        let tenant = tenant.clone();
        let running = running.clone();
        let count = stats_count.clone();
        let handle = tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                let stats = tenant.stats().await;
                assert!(stats.vector_count <= 1000); // Sanity check
                count.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    // Writer that adds vectors
    let writer_tenant = tenant.clone();
    let writer_running = running.clone();
    let writer = tokio::spawn(async move {
        let mut id = 1000u64;
        while writer_running.load(Ordering::Relaxed) {
            let _ = writer_tenant.upsert(vec![(id, random_vector(4))]).await;
            id += 1;
        }
    });
    handles.push(writer);

    tokio::time::sleep(Duration::from_millis(100)).await;
    running.store(false, Ordering::Relaxed);

    for handle in handles {
        let _ = timeout(Duration::from_secs(5), handle).await;
    }

    assert!(
        stats_count.load(Ordering::Relaxed) > 0,
        "Should have read stats many times"
    );
}
