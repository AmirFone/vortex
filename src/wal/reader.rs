//! WAL replay logic

use super::entry::{WalEntry, WalError, WAL_HEADER_SIZE};
use crate::storage::BlockStorage;

/// Replay WAL entries from storage
pub async fn replay_wal(
    storage: &dyn BlockStorage,
    path: &str,
    dims: usize,
    after_sequence: u64,
) -> Result<Vec<WalEntry>, WalError> {
    if !storage.exists(path).await? {
        return Ok(Vec::new());
    }

    let data = storage.read(path).await?;
    replay_from_bytes(&data, dims, after_sequence)
}

/// Replay WAL entries from byte buffer
pub fn replay_from_bytes(
    data: &[u8],
    dims: usize,
    after_sequence: u64,
) -> Result<Vec<WalEntry>, WalError> {
    let entry_size = WAL_HEADER_SIZE + dims * 4;
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + entry_size <= data.len() {
        // Try to parse entry
        match WalEntry::deserialize(&data[offset..], dims) {
            Ok(entry) => {
                if entry.sequence > after_sequence {
                    entries.push(entry);
                }
                offset += entry_size;
            }
            Err(WalError::InvalidMagic) => {
                // Reached end of valid entries (might be zeros or garbage)
                break;
            }
            Err(WalError::ChecksumMismatch) => {
                // Corruption detected, stop here
                tracing::warn!(
                    offset,
                    "WAL corruption detected at offset {}, stopping replay",
                    offset
                );
                break;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_multiple_entries() {
        let entries = vec![
            WalEntry {
                sequence: 1,
                tenant_id: 1,
                vector_id: 100,
                vector: vec![1.0, 2.0, 3.0, 4.0],
            },
            WalEntry {
                sequence: 2,
                tenant_id: 1,
                vector_id: 101,
                vector: vec![5.0, 6.0, 7.0, 8.0],
            },
            WalEntry {
                sequence: 3,
                tenant_id: 1,
                vector_id: 102,
                vector: vec![9.0, 10.0, 11.0, 12.0],
            },
        ];

        // Serialize all
        let mut buffer = Vec::new();
        for entry in &entries {
            buffer.extend(entry.serialize(4));
        }

        // Replay all
        let replayed = replay_from_bytes(&buffer, 4, 0).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].vector_id, 100);
        assert_eq!(replayed[1].vector_id, 101);
        assert_eq!(replayed[2].vector_id, 102);
    }

    #[test]
    fn test_replay_from_sequence() {
        let entries = vec![
            WalEntry {
                sequence: 1,
                tenant_id: 1,
                vector_id: 100,
                vector: vec![1.0, 2.0, 3.0, 4.0],
            },
            WalEntry {
                sequence: 2,
                tenant_id: 1,
                vector_id: 101,
                vector: vec![5.0, 6.0, 7.0, 8.0],
            },
            WalEntry {
                sequence: 3,
                tenant_id: 1,
                vector_id: 102,
                vector: vec![9.0, 10.0, 11.0, 12.0],
            },
        ];

        let mut buffer = Vec::new();
        for entry in &entries {
            buffer.extend(entry.serialize(4));
        }

        // Replay after sequence 1
        let replayed = replay_from_bytes(&buffer, 4, 1).unwrap();
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].sequence, 2);
        assert_eq!(replayed[1].sequence, 3);
    }

    #[test]
    fn test_replay_handles_trailing_garbage() {
        let entry = WalEntry {
            sequence: 1,
            tenant_id: 1,
            vector_id: 100,
            vector: vec![1.0, 2.0, 3.0, 4.0],
        };

        let mut buffer = entry.serialize(4);
        // Add some garbage that doesn't look like a valid entry
        buffer.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);

        let replayed = replay_from_bytes(&buffer, 4, 0).unwrap();
        assert_eq!(replayed.len(), 1);
    }
}
