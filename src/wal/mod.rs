//! Write-Ahead Log
//!
//! Provides durability guarantees:
//! - Entries are written atomically
//! - fsync ensures persistence before acknowledgment
//! - CRC32 detects corruption during replay

pub mod entry;
pub mod reader;

use crate::storage::{BlockStorage, StorageResult};
use entry::{WalEntry, WalError, WAL_HEADER_SIZE};
use tokio::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Write-Ahead Log
pub struct Wal {
    storage: Arc<dyn BlockStorage>,
    path: String,
    dims: usize,
    next_sequence: AtomicU64,
    entry_size: usize,
    write_lock: Mutex<()>,
}

impl Wal {
    /// Open or create a WAL
    pub async fn open(
        storage: Arc<dyn BlockStorage>,
        path: impl Into<String>,
        dims: usize,
    ) -> Result<Self, WalError> {
        let path = path.into();
        let entry_size = WAL_HEADER_SIZE + dims * 4;

        // Determine next sequence by scanning existing entries
        let next_sequence = if storage.exists(&path).await? {
            let entries = reader::replay_wal(&*storage, &path, dims, 0).await?;
            entries.last().map(|e| e.sequence + 1).unwrap_or(1)
        } else {
            // Create empty file
            storage.write(&path, &[]).await?;
            1
        };

        Ok(Self {
            storage,
            path,
            dims,
            next_sequence: AtomicU64::new(next_sequence),
            entry_size,
            write_lock: Mutex::new(()),
        })
    }

    /// Append a single entry to the WAL
    /// Returns the sequence number assigned to this entry
    ///
    /// GUARANTEE: When this returns Ok, the data is durable on disk
    pub async fn append(
        &self,
        tenant_id: u64,
        vector_id: u64,
        vector: &[f32],
    ) -> Result<u64, WalError> {
        // Acquire write lock (single writer)
        let _guard = self.write_lock.lock().await;

        let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);

        let entry = WalEntry {
            sequence,
            tenant_id,
            vector_id,
            vector: vector.to_vec(),
        };

        let data = entry.serialize(self.dims);

        // Append to file
        self.storage.append(&self.path, &data).await?;

        // CRITICAL: fsync to ensure durability
        self.storage.sync(&self.path).await?;

        Ok(sequence)
    }

    /// Append multiple entries atomically
    /// All entries are written and fsynced together
    pub async fn append_batch(
        &self,
        entries: Vec<(u64, u64, Vec<f32>)>, // (tenant_id, vector_id, vector)
    ) -> Result<u64, WalError> {
        if entries.is_empty() {
            return Ok(self.current_sequence());
        }

        let _guard = self.write_lock.lock().await;

        let start_sequence = self
            .next_sequence
            .fetch_add(entries.len() as u64, Ordering::SeqCst);

        // Serialize all entries
        let mut buffer = Vec::with_capacity(entries.len() * self.entry_size);

        for (i, (tenant_id, vector_id, vector)) in entries.into_iter().enumerate() {
            let entry = WalEntry {
                sequence: start_sequence + i as u64,
                tenant_id,
                vector_id,
                vector,
            };
            buffer.extend(entry.serialize(self.dims));
        }

        // Write all at once
        self.storage.append(&self.path, &buffer).await?;

        // Single fsync for entire batch
        self.storage.sync(&self.path).await?;

        Ok(self.current_sequence())
    }

    /// Get current sequence number (last written)
    pub fn current_sequence(&self) -> u64 {
        let next = self.next_sequence.load(Ordering::SeqCst);
        if next == 0 {
            0
        } else {
            next - 1
        }
    }

    /// Replay all entries from the WAL
    pub async fn replay_all(&self) -> Result<Vec<WalEntry>, WalError> {
        reader::replay_wal(&*self.storage, &self.path, self.dims, 0).await
    }

    /// Replay entries after a given sequence number
    pub async fn replay_from(&self, after_sequence: u64) -> Result<Vec<WalEntry>, WalError> {
        reader::replay_wal(&*self.storage, &self.path, self.dims, after_sequence).await
    }

    /// Get WAL file path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get WAL file size
    pub async fn size(&self) -> Result<u64, WalError> {
        Ok(self.storage.size(&self.path).await?)
    }

    /// Get dimensions
    pub fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{MockBlockStorage, MockStorageConfig};

    #[tokio::test]
    async fn test_wal_append_and_replay() {
        let storage =
            Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let wal = Wal::open(storage, "wal.log", 4).await.unwrap();

        // Append some entries
        let seq1 = wal.append(1, 100, &[1.0, 2.0, 3.0, 4.0]).await.unwrap();
        let seq2 = wal.append(1, 101, &[5.0, 6.0, 7.0, 8.0]).await.unwrap();

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);

        // Replay and verify
        let entries = wal.replay_all().await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].vector_id, 100);
        assert_eq!(entries[1].vector_id, 101);
    }

    #[tokio::test]
    async fn test_wal_batch() {
        let storage =
            Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let wal = Wal::open(storage, "wal.log", 4).await.unwrap();

        let batch = vec![
            (1, 100, vec![1.0, 2.0, 3.0, 4.0]),
            (1, 101, vec![5.0, 6.0, 7.0, 8.0]),
            (1, 102, vec![9.0, 10.0, 11.0, 12.0]),
        ];

        let last_seq = wal.append_batch(batch).await.unwrap();
        assert_eq!(last_seq, 3);

        let entries = wal.replay_all().await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn test_wal_recovery() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write some entries
        {
            let storage =
                Arc::new(MockBlockStorage::new(&path, MockStorageConfig::fast()).unwrap());
            let wal = Wal::open(storage, "wal.log", 4).await.unwrap();
            wal.append(1, 100, &[1.0, 2.0, 3.0, 4.0]).await.unwrap();
            wal.append(1, 101, &[5.0, 6.0, 7.0, 8.0]).await.unwrap();
        }

        // Reopen and verify recovery
        {
            let storage =
                Arc::new(MockBlockStorage::new(&path, MockStorageConfig::fast()).unwrap());
            let wal = Wal::open(storage, "wal.log", 4).await.unwrap();

            assert_eq!(wal.current_sequence(), 2);

            let entries = wal.replay_all().await.unwrap();
            assert_eq!(entries.len(), 2);
        }
    }
}
