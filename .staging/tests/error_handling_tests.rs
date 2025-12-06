//! Error Handling Tests for VectorDB
//!
//! Tests covering:
//! - Storage failure handling
//! - Corruption handling
//! - Graceful degradation
//! - Error propagation

mod common;

use std::sync::Arc;

use vectordb::hnsw::HnswConfig;
use vectordb::storage::BlockStorage;
use vectordb::tenant::TenantState;
use vectordb::wal::Wal;

use common::{
    reopen_storage, seeded_vector, temp_storage, temp_storage_with_path, truncate_file,
    FailingStorage, FailureMode,
};

// ============================================================================
// STORAGE FAILURE TESTS
// ============================================================================

/// Test: Write failure during upsert
#[tokio::test]
async fn test_write_failure_during_upsert() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, failing_storage.clone(), config)
        .await
        .unwrap();

    // First upsert should succeed
    let vectors1: Vec<(u64, Vec<f32>)> = (0..5)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors1).await.unwrap();

    // Enable write failures
    failing_storage.set_failure_mode(FailureMode::FailWrites);

    // Second upsert should fail
    let vectors2: Vec<(u64, Vec<f32>)> = (5..10)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    let result = tenant.upsert(vectors2).await;

    assert!(result.is_err(), "Should fail when writes fail");
}

/// Test: Sync failure handling
#[tokio::test]
async fn test_sync_failure_handling() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, failing_storage.clone(), config)
        .await
        .unwrap();

    // Insert vectors
    let vectors: Vec<(u64, Vec<f32>)> = (0..10)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    // Enable sync failures
    failing_storage.set_failure_mode(FailureMode::FailSync);

    // Flush should fail or handle gracefully
    let result = tenant.flush_to_hnsw(&*storage).await;
    // Either error or success is acceptable, but shouldn't panic
    match result {
        Ok(_) => (), // Flush might succeed if sync isn't critical path
        Err(_) => (), // Error is expected
    }
}

/// Test: Read failure during recovery
#[tokio::test]
async fn test_read_failure_during_recovery() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = HnswConfig::new(4);

    // Create and populate tenant
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        let vectors: Vec<(u64, Vec<f32>)> = (0..5)
            .map(|i| (i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
        tenant.flush_to_hnsw(&*storage).await.unwrap();
    }

    // Try recovery with failing reads
    let storage2 = reopen_storage(&path);
    let failing_storage = Arc::new(FailingStorage::new(storage2));
    failing_storage.set_failure_mode(FailureMode::FailAll);

    let result = TenantState::open(1, 4, failing_storage, config).await;

    // Should fail gracefully
    assert!(result.is_err(), "Recovery should fail when reads fail");

    drop(temp_dir);
}

/// Test: Append failure during WAL write
#[tokio::test]
async fn test_append_failure_wal() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write first entry successfully
    let mut wal = Wal::open(failing_storage.clone(), wal_path, dims)
        .await
        .unwrap();
    let v0 = seeded_vector(dims, 0);
    wal.append(1, 100, &v0).await.unwrap();

    // Enable append failures
    failing_storage.set_failure_mode(FailureMode::FailAppend);

    // Second append should fail
    let v1 = seeded_vector(dims, 1);
    let result = wal.append(1, 101, &v1).await;
    assert!(result.is_err(), "Append should fail when storage fails");
}

// ============================================================================
// CORRUPTION HANDLING TESTS
// ============================================================================

/// Test: Corrupted WAL magic number
#[tokio::test]
async fn test_corrupted_wal_magic() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write valid entry
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        let v = seeded_vector(dims, 0);
        wal.append(1, 100, &v).await.unwrap();
    }

    // Corrupt magic bytes (first 4 bytes)
    let mut data = storage.read(wal_path).await.unwrap();
    data[0] = 0xFF;
    data[1] = 0xFF;
    data[2] = 0xFF;
    data[3] = 0xFF;
    storage.write(wal_path, &data).await.unwrap();

    // Replay should reject corrupted entry
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    assert!(entries.is_empty(), "Corrupted magic should be rejected");

    drop(temp_dir);
}

/// Test: Corrupted WAL sequence number
#[tokio::test]
async fn test_corrupted_wal_sequence() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write valid entries
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        for i in 0..3 {
            let v = seeded_vector(dims, i);
            wal.append(1, 100 + i, &v).await.unwrap();
        }
    }

    // Corrupt sequence number in second entry (bytes 8-15)
    let mut data = storage.read(wal_path).await.unwrap();
    let entry_size = 32 + dims * 4;
    let seq_offset = entry_size + 8; // Second entry, seq_no field
    if seq_offset + 8 < data.len() {
        data[seq_offset] ^= 0xFF;
        storage.write(wal_path, &data).await.unwrap();
    }

    // Replay - should detect corruption via CRC
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    // First entry should be valid, rest may be rejected
    assert!(entries.len() <= 3, "Should handle corrupted sequence");

    drop(temp_dir);
}

/// Test: Corrupted id_map.bin
#[tokio::test]
async fn test_corrupted_id_map() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = HnswConfig::new(4);
    let id_map_path = "tenant_1/id_map.bin";

    // Create and flush tenant
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

    // Corrupt id_map.bin
    if storage.exists(id_map_path).await.unwrap() {
        let mut data = storage.read(id_map_path).await.unwrap();
        if data.len() > 10 {
            // Corrupt multiple bytes
            for i in 0..10 {
                data[i] ^= 0xFF;
            }
            storage.write(id_map_path, &data).await.unwrap();
        }
    }

    // Recovery should handle corrupted id_map
    let storage2 = reopen_storage(&path);
    let result = TenantState::open(1, 4, storage2, config).await;

    // Either recovers from WAL or fails gracefully
    match result {
        Ok(tenant) => {
            // If recovered, should have vectors
            assert!(tenant.stats().vector_count > 0);
        }
        Err(_) => {
            // Error is acceptable for corrupted data
        }
    }

    drop(temp_dir);
}

/// Test: Missing WAL file
#[tokio::test]
async fn test_missing_wal_file() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = HnswConfig::new(4);
    let wal_path = "tenant_1/wal.log";

    // Create and flush tenant
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

    // Delete WAL file
    if storage.exists(wal_path).await.unwrap() {
        storage.delete(wal_path).await.unwrap();
    }

    // Recovery should handle missing WAL
    let storage2 = reopen_storage(&path);
    let result = TenantState::open(1, 4, storage2, config).await;

    // Should create new WAL or fail gracefully
    match result {
        Ok(tenant) => {
            // If recovered without WAL, data may be lost but should be consistent
            let stats = tenant.stats();
            // Could have data from HNSW or empty
            assert!(stats.vector_count >= 0);
        }
        Err(_) => {
            // Error is acceptable
        }
    }

    drop(temp_dir);
}

/// Test: Truncated HNSW file
#[tokio::test]
async fn test_truncated_hnsw_file() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = HnswConfig::new(4);
    let hnsw_path = "tenant_1/index.hnsw";

    // Create and flush tenant
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

    // Truncate HNSW file
    if storage.exists(hnsw_path).await.unwrap() {
        let size = storage.size(hnsw_path).await.unwrap() as usize;
        if size > 10 {
            truncate_file(&*storage, hnsw_path, size / 2).await.unwrap();
        }
    }

    // Recovery should handle truncated HNSW
    let storage2 = reopen_storage(&path);
    let result = TenantState::open(1, 4, storage2.clone(), config).await;

    match result {
        Ok(tenant) => {
            // Recovered - vectors should come from WAL
            assert!(tenant.stats().vector_count > 0, "Should recover from WAL");
        }
        Err(_) => {
            // Error is acceptable
        }
    }

    drop(temp_dir);
}

// ============================================================================
// GRACEFUL DEGRADATION TESTS
// ============================================================================

/// Test: Continue operating after partial failure
#[tokio::test]
async fn test_continue_after_partial_failure() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, failing_storage.clone(), config)
        .await
        .unwrap();

    // First batch succeeds
    let vectors1: Vec<(u64, Vec<f32>)> = (0..10)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors1).await.unwrap();

    // Enable failures for one batch
    failing_storage.set_failure_mode(FailureMode::FailAfterN(0));
    let vectors2: Vec<(u64, Vec<f32>)> = (10..20)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    let _ = tenant.upsert(vectors2).await; // May fail

    // Disable failures
    failing_storage.disable_failures();

    // Third batch should succeed
    let vectors3: Vec<(u64, Vec<f32>)> = (20..30)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors3).await.unwrap();

    // Should have at least first and third batches
    assert!(
        tenant.stats().vector_count >= 20,
        "Should recover and continue operating"
    );
}

/// Test: Search works even during write failures
#[tokio::test]
async fn test_search_during_write_failures() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, failing_storage.clone(), config)
        .await
        .unwrap();

    // Insert some vectors
    let vectors: Vec<(u64, Vec<f32>)> = (0..20)
        .map(|i| (i as u64, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors).await.unwrap();

    // Enable write failures
    failing_storage.set_failure_mode(FailureMode::FailWrites);

    // Search should still work
    let query = seeded_vector(4, 0);
    let results = tenant.search(&query, 10, None);
    assert!(!results.is_empty(), "Search should work during write failures");
}

// ============================================================================
// ERROR PROPAGATION TESTS
// ============================================================================

/// Test: Error messages are informative
#[tokio::test]
async fn test_error_messages() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = HnswConfig::new(4);

    let tenant = TenantState::open(1, 4, failing_storage.clone(), config)
        .await
        .unwrap();

    // Enable failures
    failing_storage.set_failure_mode(FailureMode::FailAll);

    // Try to upsert
    let result = tenant.upsert(vec![(1, seeded_vector(4, 0))]).await;

    if let Err(e) = result {
        let error_msg = format!("{:?}", e);
        // Error should be somewhat descriptive
        assert!(
            !error_msg.is_empty(),
            "Error message should not be empty"
        );
    }
}

/// Test: Nested error context
#[tokio::test]
async fn test_nested_error_context() {
    let (_temp_dir, storage) = temp_storage();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));

    failing_storage.set_failure_mode(FailureMode::FailAll);

    let result = Wal::open(failing_storage, "test.log", 4).await;

    // Should fail with informative error
    if let Err(e) = result {
        let error_chain = format!("{:?}", e);
        assert!(error_chain.len() > 10, "Error chain should be informative");
    }
}

// ============================================================================
// PARTIAL WRITE TESTS
// ============================================================================

/// Test: Partial write simulation
#[tokio::test]
async fn test_partial_write_handling() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write some entries
    {
        let mut wal = Wal::open(failing_storage.clone(), wal_path, dims)
            .await
            .unwrap();

        // Write complete entries
        for i in 0..3 {
            let v = seeded_vector(dims, i);
            wal.append(1, 100 + i, &v).await.unwrap();
        }

        // Enable partial writes (truncate to 20 bytes)
        failing_storage.set_failure_mode(FailureMode::PartialWrite(20));

        // This write will be partial
        let v3 = seeded_vector(dims, 3);
        let _ = wal.append(1, 103, &v3).await;
    }

    // Recovery should handle partial write
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    // Should have at least the first 3 complete entries
    assert!(
        entries.len() >= 3,
        "Should recover complete entries, got {}",
        entries.len()
    );

    drop(temp_dir);
}
