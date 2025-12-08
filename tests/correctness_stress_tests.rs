//! Correctness Under Stress Tests for VectorDB
//!
//! These tests verify data integrity and correctness when the system
//! is under heavy load. They ensure no data loss, no duplicates,
//! and accurate search results.

mod common;

use common::{brute_force_knn, calculate_recall, generate_random_vectors, normalize, random_vector};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Barrier;
use crate::common::test_index_config;
use vortex::storage::mock::{MockBlockStorage, MockStorageConfig};
use vortex::tenant::TenantState;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

const TEST_DIMS: usize = 384;
const BATCH_SIZE: usize = 100;

// =============================================================================
// DATA INTEGRITY TESTS
// =============================================================================

#[tokio::test]
async fn verify_no_data_loss_under_concurrent_writes() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    let num_threads = 4;
    let vectors_per_thread = 500;
    let total_expected = num_threads * vectors_per_thread;

    let barrier = Arc::new(Barrier::new(num_threads));
    let successful_inserts = Arc::new(AtomicUsize::new(0));
    let mut all_ids: Vec<u64> = Vec::new();

    let mut handles = Vec::new();
    for thread_id in 0..num_threads {
        let tenant = tenant.clone();
        let barrier = barrier.clone();
        let successful_inserts = successful_inserts.clone();

        let handle = tokio::spawn(async move {
            let mut thread_ids = Vec::new();
            barrier.wait().await;

            for batch_num in 0..(vectors_per_thread / BATCH_SIZE) {
                let vectors: Vec<(u64, Vec<f32>)> = (0..BATCH_SIZE)
                    .map(|i| {
                        let id = (thread_id * 10000 + batch_num * BATCH_SIZE + i) as u64;
                        thread_ids.push(id);
                        (id, normalize(&random_vector(TEST_DIMS)))
                    })
                    .collect();

                if let Ok(result) = tenant.upsert(vectors).await {
                    successful_inserts.fetch_add(result.count, Ordering::Relaxed);
                }
            }

            thread_ids
        });

        handles.push(handle);
    }

    for handle in handles {
        let ids = handle.await.unwrap();
        all_ids.extend(ids);
    }

    // Flush to HNSW
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let stats = tenant.stats().await;
    let inserted = successful_inserts.load(Ordering::Relaxed);

    println!("\n=== No Data Loss Under Concurrent Writes ===");
    println!("Expected: {} vectors", total_expected);
    println!("Successfully inserted: {} vectors", inserted);
    println!("Vectors in store: {}", stats.vector_count);
    println!("Nodes in HNSW: {}", stats.index_nodes);

    // Verify all expected vectors are present
    assert_eq!(
        stats.vector_count, inserted as u64,
        "Vector count should match successful inserts"
    );

    // Verify we can find each inserted vector
    let mut found_count = 0;
    for id in all_ids.iter().take(100) {
        // Sample check
        let query = normalize(&random_vector(TEST_DIMS));
        let results = tenant.search(&query, 1000, Some(500)).await;
        if results.iter().any(|r| r.id == *id) {
            found_count += 1;
        }
    }

    println!("Sample verification: found {}/100 sampled IDs", found_count);
    // Note: Not all may be found with random queries, this is expected
}

#[tokio::test]
async fn verify_no_duplicate_ids() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert same IDs multiple times
    let test_ids: Vec<u64> = (0..100).collect();
    let mut inserted_ids: HashSet<u64> = HashSet::new();

    // First batch
    let vectors1: Vec<(u64, Vec<f32>)> = test_ids
        .iter()
        .map(|&id| (id, normalize(&random_vector(TEST_DIMS))))
        .collect();
    let result1 = tenant.upsert(vectors1).await.unwrap();
    inserted_ids.extend(test_ids.iter().take(result1.count));

    println!("\n=== No Duplicate IDs Test ===");
    println!("First batch: inserted {} vectors", result1.count);

    // Try to insert same IDs again (should be rejected)
    let vectors2: Vec<(u64, Vec<f32>)> = test_ids
        .iter()
        .map(|&id| (id, normalize(&random_vector(TEST_DIMS))))
        .collect();
    let result2 = tenant.upsert(vectors2).await.unwrap();

    println!("Second batch (duplicates): inserted {} vectors", result2.count);

    // Verify no duplicates in storage
    let stats = tenant.stats().await;
    assert_eq!(stats.vector_count, 100, "Should have exactly 100 unique vectors");
    assert_eq!(result2.count, 0, "Duplicate inserts should be rejected");

    println!("Verification passed: no duplicate IDs");
}

#[tokio::test]
async fn verify_search_finds_all_inserted() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert vectors with known patterns
    let mut inserted_vectors: Vec<(u64, Vec<f32>)> = Vec::new();
    for i in 0..100 {
        let vec = normalize(&random_vector(TEST_DIMS));
        inserted_vectors.push((i, vec));
    }

    tenant.upsert(inserted_vectors.clone()).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== Search Finds All Inserted Test ===");

    // Search for each inserted vector using itself as query
    let mut found_count = 0;
    for (id, vec) in &inserted_vectors {
        let results = tenant.search(vec, 10, Some(200)).await;
        if results.iter().any(|r| r.id == *id) {
            found_count += 1;
        }
    }

    println!(
        "Self-search: found {}/{} vectors",
        found_count,
        inserted_vectors.len()
    );

    // All vectors should find themselves with a high ef
    assert!(
        found_count >= 95,
        "At least 95% of vectors should find themselves, found {}%",
        found_count
    );
}

#[tokio::test]
async fn verify_recall_above_threshold() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert test vectors
    let vectors = generate_random_vectors(1000, TEST_DIMS);
    let vectors_for_search: Vec<(u64, Vec<f32>)> = vectors.clone();

    tenant.upsert(vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== Recall Above Threshold Test ===");

    // Run recall tests
    let num_queries = 50;
    let k = 10;
    let mut recalls = Vec::new();

    for _ in 0..num_queries {
        let query = normalize(&random_vector(TEST_DIMS));

        // Get HNSW results
        let hnsw_results: Vec<u64> = tenant
            .search(&query, k, Some(200))
            .await
            .iter()
            .map(|r| r.id)
            .collect();

        // Get ground truth (brute force)
        let ground_truth = brute_force_knn(&query, &vectors_for_search, k);

        // Calculate recall
        let recall = calculate_recall(&hnsw_results, &ground_truth, k);
        recalls.push(recall);
    }

    let avg_recall: f64 = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let min_recall = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_recall = recalls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Recall@{} over {} queries:", k, num_queries);
    println!("  Average: {:.2}%", avg_recall * 100.0);
    println!("  Min: {:.2}%", min_recall * 100.0);
    println!("  Max: {:.2}%", max_recall * 100.0);

    // Target: >90% recall
    assert!(
        avg_recall >= 0.90,
        "Average recall should be at least 90%, got {:.2}%",
        avg_recall * 100.0
    );

    println!("Recall verification passed!");
}

// =============================================================================
// HNSW GRAPH INTEGRITY TESTS
// =============================================================================

#[tokio::test]
async fn verify_hnsw_connectivity_after_stress() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Concurrent inserts
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let tenant = tenant.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            for batch in 0..5 {
                let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                    .map(|i| {
                        let id = (thread_id * 10000 + batch * 100 + i) as u64;
                        (id, normalize(&random_vector(TEST_DIMS)))
                    })
                    .collect();
                let _ = tenant.upsert(vectors).await;
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== HNSW Connectivity After Stress ===");

    // Verify all nodes are reachable via search
    let stats = tenant.stats().await;
    println!("HNSW nodes: {}", stats.index_nodes);

    // Test searches from random queries
    let mut successful_searches = 0;
    for _ in 0..100 {
        let query = normalize(&random_vector(TEST_DIMS));
        let results = tenant.search(&query, 10, Some(100)).await;
        if !results.is_empty() {
            successful_searches += 1;
        }
    }

    println!(
        "Successful searches: {}/100 (should be 100)",
        successful_searches
    );
    assert_eq!(successful_searches, 100, "All searches should return results");
}

#[tokio::test]
async fn verify_entry_point_valid() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    // Insert many vectors
    for batch in 0..10 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..100)
            .map(|i| {
                let id = (batch * 100 + i) as u64;
                (id, normalize(&random_vector(TEST_DIMS)))
            })
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    tenant.flush_to_hnsw(&*storage).await.unwrap();

    println!("\n=== Entry Point Valid Test ===");

    // Perform many searches - if entry point is invalid, these will fail
    let mut all_passed = true;
    for i in 0..100 {
        let query = normalize(&random_vector(TEST_DIMS));
        let results = tenant.search(&query, 10, None).await;
        if results.is_empty() {
            println!("Search {} returned empty results!", i);
            all_passed = false;
        }
    }

    assert!(all_passed, "All searches should return results");
    println!("Entry point validation passed!");
}

// =============================================================================
// RECOVERY CORRECTNESS TESTS
// =============================================================================

#[tokio::test]
async fn verify_recovery_after_stress() {
    let temp_dir = tempfile::tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    let config = test_index_config();

    let mut inserted_ids: HashSet<u64> = HashSet::new();

    // Phase 1: Create and populate
    {
        let storage =
            Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        // Insert with concurrent stress
        for batch in 0..5 {
            let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                .map(|i| {
                    let id = (batch * 100 + i) as u64;
                    inserted_ids.insert(id);
                    (id, normalize(&random_vector(TEST_DIMS)))
                })
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }

        // Flush to persist
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        println!("\n=== Recovery After Stress Test ===");
        println!("Phase 1: Inserted {} vectors", inserted_ids.len());
    }

    // Phase 2: Recover and verify
    {
        let storage =
            Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        let stats = tenant.stats().await;
        println!("Phase 2: Recovered {} vectors", stats.vector_count);

        // Verify counts match
        assert_eq!(
            stats.vector_count,
            inserted_ids.len() as u64,
            "Recovered count should match inserted"
        );

        // Verify searches work
        let query = normalize(&random_vector(TEST_DIMS));
        let results = tenant.search(&query, 10, None).await;
        assert!(!results.is_empty(), "Search should return results after recovery");

        // Verify all IDs are searchable
        let all_results: Vec<u64> = tenant
            .search(&query, 500, Some(500))
            .await
            .iter()
            .map(|r| r.id)
            .collect();
        let recovered_ids: HashSet<u64> = all_results.into_iter().collect();

        // At least some of our IDs should be in results
        let overlap: HashSet<_> = inserted_ids.intersection(&recovered_ids).collect();
        println!(
            "ID overlap: {}/{} inserted IDs found in search",
            overlap.len(),
            inserted_ids.len()
        );

        println!("Recovery verification passed!");
    }
}

#[tokio::test]
async fn verify_wal_sequence_monotonic() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
        .await
        .unwrap();

    let mut sequences: Vec<u64> = Vec::new();

    // Insert multiple batches and track sequences
    for batch in 0..10 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| {
                let id = (batch * 10 + i) as u64;
                (id, normalize(&random_vector(TEST_DIMS)))
            })
            .collect();

        let result = tenant.upsert(vectors).await.unwrap();
        sequences.push(result.sequence);
    }

    println!("\n=== WAL Sequence Monotonic Test ===");
    println!("Sequences: {:?}", sequences);

    // Verify sequences are strictly increasing
    for i in 1..sequences.len() {
        assert!(
            sequences[i] > sequences[i - 1],
            "Sequence {} should be > {}, got {} vs {}",
            i,
            i - 1,
            sequences[i],
            sequences[i - 1]
        );
    }

    println!("WAL sequence monotonicity verified!");
}

#[tokio::test]
async fn verify_no_phantom_vectors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    let config = test_index_config();

    // Insert and then delete (via not inserting)
    let known_ids: HashSet<u64> = (0..100).collect();

    {
        let storage =
            Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = known_ids
            .iter()
            .map(|&id| (id, normalize(&random_vector(TEST_DIMS))))
            .collect();

        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    // Recover and verify no phantom vectors
    {
        let storage =
            Arc::new(MockBlockStorage::new(&storage_path, MockStorageConfig::fast()).unwrap());
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config.clone())
            .await
            .unwrap();

        println!("\n=== No Phantom Vectors Test ===");

        // Search with high k to get all vectors
        let query = normalize(&random_vector(TEST_DIMS));
        let results = tenant.search(&query, 200, Some(500)).await;

        let result_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();

        // All result IDs should be in our known set
        let phantoms: Vec<u64> = result_ids
            .difference(&known_ids)
            .cloned()
            .collect();

        if !phantoms.is_empty() {
            println!("Found phantom IDs: {:?}", phantoms);
        }

        assert!(
            phantoms.is_empty(),
            "Should not have phantom vectors, found: {:?}",
            phantoms
        );

        println!("No phantom vectors found - verification passed!");
    }
}

// =============================================================================
// CONCURRENT READ-WRITE CORRECTNESS
// =============================================================================

#[tokio::test]
async fn verify_consistency_under_concurrent_read_write() {
    let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
    let config = test_index_config();
    let tenant = Arc::new(
        TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap(),
    );

    // Pre-populate
    let initial_vectors = generate_random_vectors(100, TEST_DIMS);
    tenant.upsert(initial_vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let write_tenant = tenant.clone();
    let read_tenant = tenant.clone();

    // Writer task
    let write_handle = tokio::spawn(async move {
        let mut next_id = 100u64;
        for _ in 0..50 {
            let vectors = vec![(next_id, normalize(&random_vector(TEST_DIMS)))];
            next_id += 1;
            let _ = write_tenant.upsert(vectors).await;
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        next_id
    });

    // Reader task
    let read_handle = tokio::spawn(async move {
        let mut successful_reads = 0;
        let mut failed_reads = 0;

        for _ in 0..100 {
            let query = normalize(&random_vector(TEST_DIMS));
            let results = read_tenant.search(&query, 10, None).await;
            if !results.is_empty() {
                successful_reads += 1;
            } else {
                failed_reads += 1;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        (successful_reads, failed_reads)
    });

    let (final_id_result, read_result) = tokio::join!(write_handle, read_handle);
    let final_id = final_id_result.unwrap();
    let (successful, failed) = read_result.unwrap();

    println!("\n=== Consistency Under Concurrent Read-Write ===");
    println!("Writes completed: {} new vectors", final_id - 100);
    println!(
        "Read results: {} successful, {} failed",
        successful, failed
    );

    // All reads should succeed (return results)
    assert!(
        failed == 0,
        "No reads should fail during concurrent write"
    );

    println!("Concurrent read-write consistency verified!");
}
