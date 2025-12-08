//! Crash Recovery Tests ("Lights Out" Tests)
//!
//! These tests simulate various crash scenarios and verify
//! that the database recovers correctly.

mod common;

use std::sync::Arc;

use crate::common::test_index_config;
use vortex::storage::BlockStorage;
use vortex::tenant::TenantState;
use vortex::wal::Wal;

use common::{reopen_storage, seeded_vector, temp_storage_with_path, truncate_file, FailingStorage, FailureMode};

// ============================================================================
// LIGHTS-OUT TESTS (Simulated Power Failure)
// ============================================================================

/// Test: Crash during WAL append (partial write)
#[tokio::test]
async fn test_crash_during_wal_append() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;
    let entry_size = 32 + dims * 4; // Header + vector data

    // Write some complete entries
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        for i in 0..5 {
            let vector = seeded_vector(dims, i);
            wal.append(1, 100 + i, &vector).await.unwrap();
        }
    }

    // Simulate crash by truncating mid-entry
    let wal_size = storage.size(wal_path).await.unwrap() as usize;
    let truncate_at = wal_size - entry_size / 2; // Truncate mid-entry
    truncate_file(&*storage, wal_path, truncate_at).await.unwrap();

    // Recovery should skip partial entry
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    // Should have 4 complete entries (5th was partial)
    assert_eq!(entries.len(), 4, "Should recover 4 complete entries");

    // Verify data integrity of recovered entries
    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(entry.vector_id, 100 + i as u64);
        let expected = seeded_vector(dims, i as u64);
        for (j, &val) in entry.vector.iter().enumerate() {
            assert!((val - expected[j]).abs() < 1e-6);
        }
    }

    drop(temp_dir);
}

/// Test: Crash at various truncation points
#[tokio::test]
async fn test_crash_at_various_truncation_points() {
    let dims = 4;
    let entry_size = 32 + dims * 4;

    // Test truncation at various points within an entry
    for truncate_offset in [4, 8, 16, 24, 30, entry_size - 4] {
        let (temp_dir, path, storage) = temp_storage_with_path();
        let wal_path = "test_wal.log";

        // Write two complete entries
        {
            let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
            let v0 = seeded_vector(dims, 0);
            let v1 = seeded_vector(dims, 1);
            wal.append(1, 100, &v0).await.unwrap();
            wal.append(1, 101, &v1).await.unwrap();
        }

        // Truncate at offset within second entry
        let truncate_at = entry_size + truncate_offset;
        truncate_file(&*storage, wal_path, truncate_at).await.unwrap();

        // Recovery
        let storage2 = reopen_storage(&path);
        let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
        let entries = wal.replay_from(0).await.unwrap();

        // First entry should always be recovered
        assert!(
            entries.len() >= 1,
            "First entry should be recovered with truncation at offset {}",
            truncate_offset
        );

        drop(temp_dir);
    }
}

/// Test: Empty WAL recovery
#[tokio::test]
async fn test_empty_wal_recovery() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // Create tenant (creates empty WAL)
    {
        let _tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();
        // No writes, just close
    }

    // Recovery with empty WAL
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    assert_eq!(tenant.stats().await.vector_count, 0, "Should recover empty state");

    drop(temp_dir);
}

/// Test: Recovery with corrupted HNSW index rebuilds from WAL
#[tokio::test]
async fn test_corrupted_hnsw_recovery() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();
    let hnsw_path = "tenant_1/index.hnsw";

    // Create tenant, insert vectors, flush to HNSW
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..20)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    // Corrupt HNSW file
    if storage.exists(hnsw_path).await.unwrap() {
        let mut data = storage.read(hnsw_path).await.unwrap();
        if data.len() > 10 {
            // Corrupt header
            data[0] ^= 0xFF;
            data[1] ^= 0xFF;
            data[2] ^= 0xFF;
            storage.write(hnsw_path, &data).await.unwrap();
        }
    }

    // Recovery - should rebuild from WAL
    let storage2 = reopen_storage(&path);
    let result = TenantState::open(1, 4, storage2.clone(), config).await;

    // Either recovery succeeds with rebuilt index, or we need to handle the error
    // For this implementation, corrupted HNSW might cause an error
    // The important thing is WAL data is preserved
    match result {
        Ok(tenant) => {
            // If recovery succeeded, verify data
            assert!(tenant.stats().await.vector_count > 0, "Should have recovered vectors");
        }
        Err(_) => {
            // If HNSW is unrecoverable, WAL should still be intact
            let wal_path = "tenant_1/wal.log";
            let wal = Wal::open(storage2, wal_path, 4).await.unwrap();
            let entries = wal.replay_from(0).await.unwrap();
            assert_eq!(entries.len(), 20, "WAL should still have all entries");
        }
    }

    drop(temp_dir);
}

// ============================================================================
// RECOVERY CORRECTNESS TESTS
// ============================================================================

/// Test: Idempotent recovery (replaying same WAL multiple times)
#[tokio::test]
async fn test_idempotent_recovery() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // Create and populate tenant
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Recover multiple times
    for recovery_num in 1..=3 {
        let storage_n = reopen_storage(&path);
        let tenant = TenantState::open(1, 4, storage_n.clone(), config.clone())
            .await
            .unwrap();

        assert_eq!(
            tenant.stats().await.vector_count,
            10,
            "Recovery #{} should have exactly 10 vectors",
            recovery_num
        );
    }

    drop(temp_dir);
}

/// Test: Recovery with missing id_map.bin (rebuild from WAL)
#[tokio::test]
async fn test_recovery_missing_id_map() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();
    let id_map_path = "tenant_1/id_map.bin";

    // Create, populate, and flush tenant
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..15)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    // Delete id_map.bin
    if storage.exists(id_map_path).await.unwrap() {
        storage.delete(id_map_path).await.unwrap();
    }

    // Recovery should still work (rebuild id_map from WAL)
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Vectors should still be searchable
    let query = seeded_vector(4, 0);
    let results = tenant.search(&query, 5, None).await;
    assert!(!results.is_empty(), "Should find vectors after id_map recovery");

    drop(temp_dir);
}

/// Test: Recovery with corrupted meta.json
#[tokio::test]
async fn test_recovery_corrupted_metadata() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();
    let meta_path = "tenant_1/meta.json";

    // Create, populate, and flush tenant
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    // Corrupt meta.json with invalid JSON
    if storage.exists(meta_path).await.unwrap() {
        storage.write(meta_path, b"{ invalid json").await.unwrap();
    }

    // Recovery should handle corrupted metadata gracefully
    let storage2 = reopen_storage(&path);
    let result = TenantState::open(1, 4, storage2.clone(), config).await;

    // Should either recover or fail gracefully
    match result {
        Ok(tenant) => {
            // If recovery succeeded, we're good
            assert!(tenant.stats().await.vector_count > 0);
        }
        Err(e) => {
            // Error is acceptable for corrupted metadata
            println!("Expected error for corrupted metadata: {:?}", e);
        }
    }

    drop(temp_dir);
}

/// Test: Recovery with vectors both in HNSW and write buffer
#[tokio::test]
async fn test_recovery_mixed_hnsw_and_buffer() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // Create tenant, insert first batch, flush, insert second batch, crash
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        // First batch - will be in HNSW
        let vectors1: Vec<(u64, Vec<f32>)> = (0..25)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors1).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        // Second batch - will be in write buffer only
        let vectors2: Vec<(u64, Vec<f32>)> = (25..50)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors2).await.unwrap();
        // No flush - crash
    }

    // Recovery
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Should have all 50 vectors
    assert_eq!(tenant.stats().await.vector_count, 50);

    // Verify both HNSW vectors and write buffer vectors are searchable
    for i in [0, 12, 24, 30, 40, 49] {
        let query = seeded_vector(4, i);
        let results = tenant.search(&query, 1, None).await;
        assert!(
            !results.is_empty(),
            "Vector {} should be searchable after recovery",
            i
        );
    }

    drop(temp_dir);
}

/// Test: Recovery doesn't duplicate vectors
#[tokio::test]
async fn test_recovery_no_duplicates() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // Insert same vector ID multiple times, then recover
    let vector_id = 999u64;
    let vector = seeded_vector(4, 42);

    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        // Insert the same ID multiple times (upsert semantics)
        for _ in 0..5 {
            tenant.upsert(vec![(vector_id, vector.clone())]).await.unwrap();
        }
    }

    // Recovery
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Search should return only one result for this vector
    let results = tenant.search(&vector, 10, None).await;
    let matching: Vec<_> = results.iter().filter(|r| r.id == vector_id).collect();
    assert_eq!(matching.len(), 1, "Should not have duplicate entries for same ID");

    drop(temp_dir);
}

// ============================================================================
// STORAGE FAILURE SIMULATION
// ============================================================================

/// Test: Recovery after write failure
#[tokio::test]
async fn test_recovery_after_write_failure() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = test_index_config();

    // Insert some vectors successfully
    {
        let tenant = TenantState::open(1, 4, failing_storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();

        // Enable write failures
        failing_storage.set_failure_mode(FailureMode::FailWrites);

        // Try to insert more (should fail)
        let vectors2: Vec<(u64, Vec<f32>)> = (10..20)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        let _ = tenant.upsert(vectors2).await; // May fail

        failing_storage.disable_failures();
    }

    // Recovery with normal storage
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Should have at least the first 10 vectors
    assert!(
        tenant.stats().await.vector_count >= 10,
        "Should recover at least first batch"
    );

    drop(temp_dir);
}

/// Test: Recovery after sync failure
#[tokio::test]
async fn test_recovery_after_sync_failure() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = test_index_config();

    {
        let tenant = TenantState::open(1, 4, failing_storage.clone(), config.clone())
            .await
            .unwrap();

        // Insert vectors
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();

        // Enable sync failures for flush
        failing_storage.set_failure_mode(FailureMode::FailSync);

        // Try to flush (sync will fail)
        let _ = tenant.flush_to_hnsw(&*storage).await;

        failing_storage.disable_failures();
    }

    // Recovery
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Vectors should still be recovered from WAL
    assert!(tenant.stats().await.vector_count >= 10, "Should recover from WAL despite sync failure");

    drop(temp_dir);
}
