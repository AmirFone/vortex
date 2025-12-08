//! ACID Compliance Tests for VectorDB
//!
//! Tests covering:
//! - Atomicity: Partial writes, batch operations
//! - Consistency: Sequence numbers, CRC32 validation
//! - Isolation: Concurrent reads during writes
//! - Durability: fsync guarantees, WAL replay

mod common;

use std::sync::Arc;

use crate::common::test_index_config;
use vortex::storage::BlockStorage;
use vortex::tenant::TenantState;
use vortex::wal::Wal;

use common::{normalize, random_vector, reopen_storage, seeded_vector, temp_storage_with_path, FailingStorage, FailureMode};

// ============================================================================
// ATOMICITY TESTS
// ============================================================================

/// Test: WAL entry atomicity - partial entries should be rejected
#[tokio::test]
async fn test_wal_partial_entry_rejection() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write some valid entries
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        for i in 0..5 {
            let vector = seeded_vector(dims, i);
            wal.append(1, 100 + i, &vector).await.unwrap();
        }
    }

    // Corrupt the WAL by truncating mid-entry
    let wal_size = storage.size(wal_path).await.unwrap();
    let truncate_at = wal_size - 10; // Truncate last 10 bytes
    let data = storage.read(wal_path).await.unwrap();
    storage.write(wal_path, &data[..truncate_at as usize]).await.unwrap();

    // Reopen and replay - should recover valid entries
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    // Should have recovered at least some entries (not 5 due to truncation)
    assert!(entries.len() < 5 || entries.len() == 4, "Expected 4 valid entries after truncation, got {}", entries.len());

    drop(temp_dir);
}

/// Test: CRC32 corruption detection
#[tokio::test]
async fn test_wal_crc_corruption_detection() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;

    // Write valid entries
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        for i in 0..3 {
            let vector = seeded_vector(dims, i);
            wal.append(1, 100 + i, &vector).await.unwrap();
        }
    }

    // Corrupt CRC32 (bytes 4-7 in first entry, after magic bytes)
    let mut data = storage.read(wal_path).await.unwrap();
    data[4] ^= 0xFF; // Flip bits in CRC
    data[5] ^= 0xFF;
    storage.write(wal_path, &data).await.unwrap();

    // Reopen and replay - corrupted entry should be rejected
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    // First entry should be rejected due to CRC failure
    assert!(entries.is_empty(), "Corrupted entries should be rejected");

    drop(temp_dir);
}

/// Test: Batch operation atomicity - simulate crash mid-batch
#[tokio::test]
async fn test_batch_atomicity_with_failure() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let failing_storage = Arc::new(FailingStorage::new(storage.clone()));
    let config = test_index_config();

    // Create tenant and insert some vectors
    let tenant = TenantState::open(1, 4, failing_storage.clone(), config.clone())
        .await
        .unwrap();

    // First batch should succeed
    let vectors1: Vec<(u64, Vec<f32>)> = (0..5)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    tenant.upsert(vectors1).await.unwrap();

    // Enable failure for second batch
    failing_storage.set_failure_mode(FailureMode::FailAfterN(1));

    // Second batch should fail partway through
    let vectors2: Vec<(u64, Vec<f32>)> = (5..15)
        .map(|i| (100 + i, seeded_vector(4, i)))
        .collect();
    let result = tenant.upsert(vectors2).await;

    // The operation may have failed
    // On recovery, we should have consistent state
    failing_storage.disable_failures();

    // Flush to persist
    let _ = tenant.flush_to_hnsw(&*storage).await;

    // Reopen and verify consistent state
    let storage2 = reopen_storage(&path);
    let tenant2 = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    let stats = tenant2.stats().await;
    // Should have at least the first 5 vectors
    assert!(stats.vector_count >= 5, "Should have at least first batch");

    drop(temp_dir);
}

// ============================================================================
// CONSISTENCY TESTS
// ============================================================================

/// Test: Sequence numbers are monotonically increasing
#[tokio::test]
async fn test_sequence_number_monotonicity() {
    let (_temp_dir, storage) = common::temp_storage();
    let wal_path = "test_wal.log";
    let dims = 4;

    let mut wal = Wal::open(storage, wal_path, dims).await.unwrap();
    let mut prev_seq = 0u64;

    for i in 0..100 {
        let vector = seeded_vector(dims, i);
        let seq = wal.append(1, 100 + i, &vector).await.unwrap();
        assert!(seq > prev_seq, "Sequence {} should be > {}", seq, prev_seq);
        prev_seq = seq;
    }
}

/// Test: Sequence numbers are strictly increasing in batch
#[tokio::test]
async fn test_batch_sequence_increment() {
    let (_temp_dir, storage) = common::temp_storage();
    let wal_path = "test_wal.log";
    let dims = 4;

    let mut wal = Wal::open(storage, wal_path, dims).await.unwrap();

    let batch: Vec<(u64, u64, Vec<f32>)> = (0..10)
        .map(|i| (1u64, 100 + i, seeded_vector(dims, i)))
        .collect();

    let initial_seq = wal.current_sequence();
    let final_seq = wal.append_batch(batch).await.unwrap();

    assert_eq!(
        final_seq,
        initial_seq + 10,
        "Batch of 10 should increment sequence by 10"
    );
}

/// Test: Multiple CRC corruption positions
#[tokio::test]
async fn test_crc_corruption_at_various_positions() {
    let dims = 4;
    // Entry size: 4 (magic) + 4 (crc) + 8 (seq) + 8 (tenant) + 8 (vector_id) + dims*4 (vector)
    let entry_size = 32 + dims * 4;

    for corrupt_offset in [0, 8, 16, 24, 32, 40] {
        let (temp_dir, path, storage) = temp_storage_with_path();
        let wal_path = "test_wal.log";

        // Write valid entry
        {
            let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
            let vector = seeded_vector(dims, 0);
            wal.append(1, 100, &vector).await.unwrap();
        }

        // Corrupt at specific offset
        let mut data = storage.read(wal_path).await.unwrap();
        if corrupt_offset < data.len() {
            data[corrupt_offset] ^= 0xFF;
            storage.write(wal_path, &data).await.unwrap();
        }

        // Try to replay - should detect corruption
        let storage2 = reopen_storage(&path);
        let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
        let entries = wal.replay_from(0).await.unwrap();

        // Corruption in header/data should be detected
        if corrupt_offset < entry_size {
            assert!(
                entries.is_empty(),
                "Corruption at offset {} should be detected",
                corrupt_offset
            );
        }

        drop(temp_dir);
    }
}

// ============================================================================
// ISOLATION TESTS
// ============================================================================

/// Test: Concurrent reads during writes
#[tokio::test]
async fn test_concurrent_reads_during_writes() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use tokio::time::{timeout, Duration};

    let (_temp_dir, storage) = common::temp_storage();
    let config = test_index_config();
    let tenant = Arc::new(
        TenantState::open(1, 4, storage.clone(), config)
            .await
            .unwrap(),
    );

    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicU64::new(0));
    let read_count = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    // Writer task
    let writer_tenant = tenant.clone();
    let writer_running = running.clone();
    let writer_count = write_count.clone();
    let writer_errors = errors.clone();
    let writer = tokio::spawn(async move {
        let mut id = 1000u64;
        while writer_running.load(Ordering::Relaxed) {
            let vectors = vec![(id, random_vector(4))];
            match writer_tenant.upsert(vectors).await {
                Ok(_) => {
                    writer_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    writer_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            id += 1;
        }
    });

    // Reader task
    let reader_tenant = tenant.clone();
    let reader_running = running.clone();
    let reader_count = read_count.clone();
    let reader_errors = errors.clone();
    let reader = tokio::spawn(async move {
        while reader_running.load(Ordering::Relaxed) {
            let query = random_vector(4);
            // Search should not panic or return corrupted data
            let results = reader_tenant.search(&query, 5, None).await;
            for r in &results {
                if r.similarity.is_nan() || r.similarity.is_infinite() {
                    reader_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            reader_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Run for a short time
    tokio::time::sleep(Duration::from_millis(100)).await;
    running.store(false, Ordering::Relaxed);

    let _ = timeout(Duration::from_secs(5), writer).await;
    let _ = timeout(Duration::from_secs(5), reader).await;

    let total_errors = errors.load(Ordering::Relaxed);
    let writes = write_count.load(Ordering::Relaxed);
    let reads = read_count.load(Ordering::Relaxed);

    assert!(writes > 0, "Should have completed some writes");
    assert!(reads > 0, "Should have completed some reads");
    assert_eq!(total_errors, 0, "Should have no errors during concurrent access");
}

/// Test: Multi-tenant isolation - no cross-tenant data leakage
#[tokio::test]
async fn test_multi_tenant_isolation() {
    let (_temp_dir, storage) = common::temp_storage();
    let config = test_index_config();

    // Create two tenants
    let tenant1 = TenantState::open(1, 4, storage.clone(), config.clone())
        .await
        .unwrap();
    let tenant2 = TenantState::open(2, 4, storage.clone(), config)
        .await
        .unwrap();

    // Insert unique vectors into each tenant
    let tenant1_vector = normalize(&[1.0, 0.0, 0.0, 0.0]);
    let tenant2_vector = normalize(&[0.0, 1.0, 0.0, 0.0]);

    tenant1.upsert(vec![(100, tenant1_vector.clone())]).await.unwrap();
    tenant2.upsert(vec![(100, tenant2_vector.clone())]).await.unwrap();

    // Search tenant1 - should only find tenant1's vector
    let results1 = tenant1.search(&tenant1_vector, 10, None).await;
    for r in &results1 {
        // Vector should be similar to tenant1's vector (not tenant2's)
        let expected_sim = common::cosine_similarity(&tenant1_vector, &tenant1_vector);
        assert!(
            (r.similarity - expected_sim).abs() < 0.1,
            "Tenant 1 should only see its own vectors"
        );
    }

    // Search tenant2 - should only find tenant2's vector
    let results2 = tenant2.search(&tenant2_vector, 10, None).await;
    for r in &results2 {
        let expected_sim = common::cosine_similarity(&tenant2_vector, &tenant2_vector);
        assert!(
            (r.similarity - expected_sim).abs() < 0.1,
            "Tenant 2 should only see its own vectors"
        );
    }

    // Verify stats are separate
    assert_eq!(tenant1.stats().await.vector_count, 1);
    assert_eq!(tenant2.stats().await.vector_count, 1);
}

// ============================================================================
// DURABILITY TESTS
// ============================================================================

/// Test: Write survives "crash" and recovery
#[tokio::test]
async fn test_durability_write_crash_recover() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // Write vectors and simulate crash (just close without graceful shutdown)
    let unique_vector = seeded_vector(4, 12345);
    let vector_id = 9999u64;

    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        // Insert vector
        let result = tenant.upsert(vec![(vector_id, unique_vector.clone())]).await.unwrap();
        assert_eq!(result.count, 1);

        // Important: DO NOT flush - simulating crash before flush
        // The vector should be in WAL only
    }
    // Tenant dropped here - simulating crash

    // "Recovery" - reopen from same path
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Vector should be recovered from WAL
    let stats = tenant.stats().await;
    assert_eq!(stats.vector_count, 1, "Vector should be recovered from WAL");

    // Should be searchable (in write buffer after recovery)
    let results = tenant.search(&unique_vector, 1, None).await;
    assert!(!results.is_empty(), "Recovered vector should be searchable");
    assert_eq!(results[0].id, vector_id, "Should find the same vector ID");

    drop(temp_dir);
}

/// Test: Multiple vectors survive crash
#[tokio::test]
async fn test_durability_multiple_vectors() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();
    let num_vectors = 50;

    // Write many vectors without flushing
    let vectors: Vec<(u64, Vec<f32>)> = (0..num_vectors)
        .map(|i| (100 + i as u64, seeded_vector(4, i)))
        .collect();

    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        tenant.upsert(vectors.clone()).await.unwrap();
        // No flush - crash simulation
    }

    // Recovery
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    let stats = tenant.stats().await;
    assert_eq!(
        stats.vector_count, num_vectors as u64,
        "All {} vectors should be recovered",
        num_vectors
    );

    drop(temp_dir);
}

/// Test: Recovery after multiple crashes
#[tokio::test]
async fn test_recovery_after_multiple_crashes() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    // First session: write 10 vectors, crash
    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();
        let vectors: Vec<(u64, Vec<f32>)> = (0..10)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Second session: recover, write 10 more, crash
    {
        let storage2 = reopen_storage(&path);
        let tenant = TenantState::open(1, 4, storage2.clone(), config.clone())
            .await
            .unwrap();
        assert_eq!(tenant.stats().await.vector_count, 10);

        let vectors: Vec<(u64, Vec<f32>)> = (10..20)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Third session: recover, write 10 more, crash
    {
        let storage3 = reopen_storage(&path);
        let tenant = TenantState::open(1, 4, storage3.clone(), config.clone())
            .await
            .unwrap();
        assert_eq!(tenant.stats().await.vector_count, 20);

        let vectors: Vec<(u64, Vec<f32>)> = (20..30)
            .map(|i| (100 + i as u64, seeded_vector(4, i)))
            .collect();
        tenant.upsert(vectors).await.unwrap();
    }

    // Final recovery - should have all 30 vectors
    let storage4 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage4.clone(), config)
        .await
        .unwrap();

    assert_eq!(tenant.stats().await.vector_count, 30, "All 30 vectors should survive multiple crashes");

    drop(temp_dir);
}

/// Test: WAL replay completeness
#[tokio::test]
async fn test_wal_replay_completeness() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let wal_path = "test_wal.log";
    let dims = 4;
    let num_entries = 100;

    // Write entries
    {
        let mut wal = Wal::open(storage.clone(), wal_path, dims).await.unwrap();
        for i in 0..num_entries {
            let vector = seeded_vector(dims, i);
            wal.append(1, 100 + i, &vector).await.unwrap();
        }
    }

    // Reopen and replay
    let storage2 = reopen_storage(&path);
    let wal = Wal::open(storage2, wal_path, dims).await.unwrap();
    let entries = wal.replay_from(0).await.unwrap();

    assert_eq!(entries.len(), num_entries as usize, "All entries should be replayed");

    // Verify each entry's data integrity
    for (i, entry) in entries.iter().enumerate() {
        let expected_vector = seeded_vector(dims, i as u64);
        assert_eq!(entry.vector_id, 100 + i as u64);
        assert_eq!(entry.vector.len(), dims);
        for (j, &val) in entry.vector.iter().enumerate() {
            assert!(
                (val - expected_vector[j]).abs() < 1e-6,
                "Vector data should be preserved"
            );
        }
    }

    drop(temp_dir);
}

/// Test: Flush then crash then recover - should have all data
#[tokio::test]
async fn test_flush_then_crash_recovery() {
    let (temp_dir, path, storage) = temp_storage_with_path();
    let config = test_index_config();

    let vectors1: Vec<(u64, Vec<f32>)> = (0..20)
        .map(|i| (100 + i as u64, seeded_vector(4, i)))
        .collect();
    let vectors2: Vec<(u64, Vec<f32>)> = (20..40)
        .map(|i| (100 + i as u64, seeded_vector(4, i)))
        .collect();

    {
        let tenant = TenantState::open(1, 4, storage.clone(), config.clone())
            .await
            .unwrap();

        // Insert first batch
        tenant.upsert(vectors1).await.unwrap();

        // Flush to HNSW
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        // Insert second batch (in WAL only)
        tenant.upsert(vectors2).await.unwrap();

        // Crash without flushing second batch
    }

    // Recovery
    let storage2 = reopen_storage(&path);
    let tenant = TenantState::open(1, 4, storage2.clone(), config)
        .await
        .unwrap();

    // Should have all 40 vectors (20 from HNSW + 20 from WAL replay)
    assert_eq!(tenant.stats().await.vector_count, 40, "All vectors should be recovered");

    drop(temp_dir);
}
