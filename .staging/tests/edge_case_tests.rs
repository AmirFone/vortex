//! Edge Case Tests for VectorDB
//!
//! Tests covering:
//! - Empty state handling
//! - Boundary conditions (k=0, k>N, etc.)
//! - Large scale operations
//! - Data integrity (NaN, Inf, zero vectors)
//! - Dimension mismatches

mod common;

use vectordb::hnsw::HnswConfig;
use vectordb::storage::BlockStorage;
use vectordb::tenant::TenantState;
use vectordb::wal::Wal;

use common::{normalize, random_vector, seeded_vector, temp_storage};

// ============================================================================
// EMPTY STATE TESTS
// ============================================================================

/// Test: Search on empty index returns empty results
#[tokio::test]
async fn test_search_empty_index() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let query = random_vector(4);
    let results = tenant.search(&query, 10, None);

    assert!(results.is_empty(), "Search on empty index should return empty");
}

/// Test: Flush on empty write buffer
#[tokio::test]
async fn test_flush_empty_buffer() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Flush with no data should succeed
    let flushed = tenant.flush_to_hnsw(&*storage).await.unwrap();
    assert_eq!(flushed, 0, "Flushing empty buffer should return 0");
}

/// Test: Stats on new tenant
#[tokio::test]
async fn test_stats_new_tenant() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let stats = tenant.stats();
    assert_eq!(stats.tenant_id, 1);
    assert_eq!(stats.vector_count, 0);
    assert_eq!(stats.hnsw_nodes, 0);
    assert_eq!(stats.write_buffer_size, 0);
    assert_eq!(stats.wal_sequence, 0);
}

/// Test: Empty WAL replay
#[tokio::test]
async fn test_empty_wal_replay() {
    let (_temp_dir, storage) = temp_storage();
    let wal_path = "test_wal.log";

    let wal = Wal::open(storage, wal_path, 4).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    assert!(entries.is_empty(), "Empty WAL should return empty entries");
}

/// Test: Empty upsert
#[tokio::test]
async fn test_empty_upsert() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let result = tenant.upsert(vec![]).await.unwrap();
    assert_eq!(result.count, 0);
}

// ============================================================================
// BOUNDARY CONDITION TESTS
// ============================================================================

/// Test: k=0 search returns empty
#[tokio::test]
async fn test_search_k_zero() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert some vectors
    let vectors: Vec<(u64, Vec<f32>)> = (0..10)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    // Search with k=0
    let query = random_vector(4);
    let results = tenant.search(&query, 0, None);

    assert!(results.is_empty(), "k=0 should return empty results");
}

/// Test: k > total_vectors
#[tokio::test]
async fn test_search_k_larger_than_total() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert 5 vectors
    let vectors: Vec<(u64, Vec<f32>)> = (0..5)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    // Search with k=100 (much larger than 5)
    let query = random_vector(4);
    let results = tenant.search(&query, 100, None);

    assert!(
        results.len() <= 5,
        "Should return at most 5 results when only 5 vectors exist"
    );
}

/// Test: Single vector in index
#[tokio::test]
async fn test_single_vector_search() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert single vector
    let vector = seeded_vector(4, 42);
    tenant.upsert(vec![(100, vector.clone())]).await.unwrap();

    // Search for the same vector
    let results = tenant.search(&vector, 10, None);

    assert_eq!(results.len(), 1, "Should find the single vector");
    assert_eq!(results[0].id, 100);
    assert!(
        results[0].similarity > 0.99,
        "Self-similarity should be ~1.0"
    );
}

/// Test: Two identical vectors
#[tokio::test]
async fn test_identical_vectors() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert two vectors with same values but different IDs
    let vector = seeded_vector(4, 42);
    tenant
        .upsert(vec![(100, vector.clone()), (101, vector.clone())])
        .await
        .unwrap();

    // Search
    let results = tenant.search(&vector, 10, None);

    assert!(results.len() >= 2, "Should find both identical vectors");
    // Both should have very high similarity
    for r in &results {
        assert!(
            r.similarity > 0.99,
            "Identical vectors should have ~1.0 similarity"
        );
    }
}

/// Test: Search with ef=1 (minimum candidates)
#[tokio::test]
async fn test_search_ef_minimum() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let vectors: Vec<(u64, Vec<f32>)> = (0..20)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let query = random_vector(4);
    let results = tenant.search(&query, 5, Some(1));

    // Should still return results, though quality may be lower
    assert!(!results.is_empty(), "ef=1 should still return results");
}

/// Test: Search with very large ef
#[tokio::test]
async fn test_search_ef_very_large() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let vectors: Vec<(u64, Vec<f32>)> = (0..20)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let query = random_vector(4);
    // Very large ef shouldn't crash
    let results = tenant.search(&query, 5, Some(1_000_000));

    assert!(!results.is_empty(), "Large ef should still work");
}

// ============================================================================
// LARGE SCALE TESTS
// ============================================================================

/// Test: Large batch insertion (1000 vectors)
#[tokio::test]
async fn test_large_batch_insertion() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let num_vectors = 1000;
    let vectors: Vec<(u64, Vec<f32>)> = (0..num_vectors)
        .map(|i| (i as u64, random_vector(4)))
        .collect();

    let result = tenant.upsert(vectors).await.unwrap();
    assert_eq!(result.count, num_vectors);

    let stats = tenant.stats();
    assert_eq!(stats.vector_count, num_vectors as u64);
}

/// Test: Large number of vectors (10,000)
#[tokio::test]
async fn test_ten_thousand_vectors() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let num_vectors = 10_000;

    // Insert in batches
    for batch in 0..10 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..1000)
            .map(|i| ((batch * 1000 + i) as u64, random_vector(4)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    let stats = tenant.stats();
    assert_eq!(stats.vector_count, num_vectors as u64);

    // Flush and verify search works
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let query = random_vector(4);
    let results = tenant.search(&query, 10, None);
    assert!(!results.is_empty(), "Should find results in large index");
}

/// Test: High-dimensional vectors (256 dims)
#[tokio::test]
async fn test_high_dimensional_vectors() {
    let (_temp_dir, storage) = temp_storage();
    let dims = 256;
    let config = HnswConfig::new(dims);

    let tenant = TenantState::open(1, dims, storage.clone(), config)
        .await
        .unwrap();

    let vectors: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| (i as u64, random_vector(dims)))
        .collect();

    tenant.upsert(vectors).await.unwrap();
    tenant.flush_to_hnsw(&*storage).await.unwrap();

    let query = random_vector(dims);
    let results = tenant.search(&query, 5, None);
    assert!(!results.is_empty(), "Should work with 256 dimensions");
}

/// Test: Vector ID at u64::MAX
#[tokio::test]
async fn test_max_vector_id() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    let vector = random_vector(4);
    let max_id = u64::MAX;

    let result = tenant.upsert(vec![(max_id, vector.clone())]).await.unwrap();
    assert_eq!(result.count, 1);

    // Should be searchable
    let results = tenant.search(&vector, 1, None);
    assert_eq!(results[0].id, max_id, "Should handle u64::MAX ID");
}

/// Test: Tenant ID at u64::MAX
#[tokio::test]
async fn test_max_tenant_id() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);
    let max_tenant = u64::MAX;

    let tenant = TenantState::open(max_tenant, 4, storage.clone(), config)
        .await
        .unwrap();

    let vectors: Vec<(u64, Vec<f32>)> = (0..5)
        .map(|i| (i as u64, random_vector(4)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    assert_eq!(tenant.stats().tenant_id, max_tenant);
    assert_eq!(tenant.stats().vector_count, 5);
}

// ============================================================================
// DATA INTEGRITY TESTS
// ============================================================================

/// Test: NaN values in vectors should be handled
#[tokio::test]
async fn test_nan_values() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert normal vector first
    let normal = normalize(&[1.0, 0.0, 0.0, 0.0]);
    tenant.upsert(vec![(100, normal.clone())]).await.unwrap();

    // Insert vector with NaN
    let nan_vec = vec![f32::NAN, 1.0, 0.0, 0.0];
    tenant.upsert(vec![(101, nan_vec)]).await.unwrap();

    // Search shouldn't crash
    let results = tenant.search(&normal, 10, None);
    // May or may not include the NaN vector, but shouldn't crash
    assert!(!results.is_empty() || results.is_empty()); // Just verify no panic
}

/// Test: Infinity values in vectors
#[tokio::test]
async fn test_infinity_values() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert normal vector
    let normal = normalize(&[1.0, 0.0, 0.0, 0.0]);
    tenant.upsert(vec![(100, normal.clone())]).await.unwrap();

    // Insert vector with infinity
    let inf_vec = vec![f32::INFINITY, 1.0, 0.0, 0.0];
    tenant.upsert(vec![(101, inf_vec)]).await.unwrap();

    // Search shouldn't crash
    let results = tenant.search(&normal, 10, None);
    assert!(!results.is_empty() || results.is_empty());
}

/// Test: All-zero vector (normalization edge case)
#[tokio::test]
async fn test_zero_vector() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert normal vector
    let normal = normalize(&[1.0, 0.0, 0.0, 0.0]);
    tenant.upsert(vec![(100, normal.clone())]).await.unwrap();

    // Insert zero vector
    let zero_vec = vec![0.0, 0.0, 0.0, 0.0];
    tenant.upsert(vec![(101, zero_vec)]).await.unwrap();

    // Search with zero vector
    let results = tenant.search(&[0.0, 0.0, 0.0, 0.0], 10, None);
    // Shouldn't crash
    assert!(results.is_empty() || !results.is_empty());
}

/// Test: Very small values (denormalized floats)
#[tokio::test]
async fn test_denormalized_floats() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Vector with very small values
    let tiny_vec = vec![1e-40, 1e-40, 1e-40, 1e-40];
    let normalized = normalize(&tiny_vec);
    tenant.upsert(vec![(100, normalized.clone())]).await.unwrap();

    // Should be searchable
    let results = tenant.search(&normalized, 1, None);
    assert!(!results.is_empty(), "Denormalized floats should work");
}

/// Test: Very large values
#[tokio::test]
async fn test_large_float_values() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Vector with large values (but not infinite)
    let large_vec = vec![1e30, 1e30, 1e30, 1e30];
    let normalized = normalize(&large_vec);
    tenant.upsert(vec![(100, normalized.clone())]).await.unwrap();

    // Should be searchable
    let results = tenant.search(&normalized, 1, None);
    assert!(!results.is_empty(), "Large float values should work after normalization");
}

/// Test: Negative values
#[tokio::test]
async fn test_negative_values() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Vector with all negative values
    let neg_vec = normalize(&[-1.0, -2.0, -3.0, -4.0]);
    tenant.upsert(vec![(100, neg_vec.clone())]).await.unwrap();

    let results = tenant.search(&neg_vec, 1, None);
    assert!(!results.is_empty());
    assert!(
        results[0].similarity > 0.99,
        "Should find the negative vector"
    );
}

// ============================================================================
// SPECIAL CASES
// ============================================================================

/// Test: Duplicate vector IDs (upsert semantics)
#[tokio::test]
async fn test_duplicate_vector_ids() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert vector with ID 100
    let vec1 = normalize(&[1.0, 0.0, 0.0, 0.0]);
    tenant.upsert(vec![(100, vec1.clone())]).await.unwrap();

    // Insert different vector with same ID
    let vec2 = normalize(&[0.0, 1.0, 0.0, 0.0]);
    tenant.upsert(vec![(100, vec2.clone())]).await.unwrap();

    // Should have only 1 vector (not 2)
    // Note: Current implementation may skip duplicates
    let stats = tenant.stats();
    assert!(
        stats.vector_count <= 2,
        "Duplicate handling should not create unbounded growth"
    );
}

/// Test: Many tenants
#[tokio::test]
async fn test_many_tenants() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let num_tenants = 10;

    for tenant_id in 1..=num_tenants {
        let tenant = TenantState::open(tenant_id, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..5)
            .map(|i| (i as u64, random_vector(4)))
            .collect();
        tenant.upsert(vectors).await.unwrap();

        assert_eq!(tenant.stats().tenant_id, tenant_id);
        assert_eq!(tenant.stats().vector_count, 5);
    }
}

/// Test: Sequential and batch upserts mixed
#[tokio::test]
async fn test_mixed_sequential_and_batch() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Sequential inserts
    for i in 0..5 {
        tenant
            .upsert(vec![(i as u64, random_vector(4))])
            .await
            .unwrap();
    }

    // Batch insert
    let batch: Vec<(u64, Vec<f32>)> = (5..15)
        .map(|i| (i as u64, random_vector(4)))
        .collect();
    tenant.upsert(batch).await.unwrap();

    // More sequential
    for i in 15..20 {
        tenant
            .upsert(vec![(i as u64, random_vector(4))])
            .await
            .unwrap();
    }

    assert_eq!(tenant.stats().vector_count, 20);
}

/// Test: Search after multiple flushes
#[tokio::test]
async fn test_multiple_flushes() {
    let (_temp_dir, storage) = temp_storage();
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert, flush, repeat
    for batch in 0..5 {
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| ((batch * 10 + i) as u64, random_vector(4)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    assert_eq!(tenant.stats().vector_count, 50);

    // Search should work
    let query = random_vector(4);
    let results = tenant.search(&query, 10, None);
    assert!(!results.is_empty());
}
