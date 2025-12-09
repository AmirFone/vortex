//! WAL entry format
//!
//! Binary format (all little-endian):
//! ┌────────┬────────┬──────────┬──────────┬──────────┬────────────────────┐
//! │ Magic  │ CRC32  │ SeqNo    │ TenantID │ VectorID │ Vector Data        │
//! │ 4 bytes│ 4 bytes│ 8 bytes  │ 8 bytes  │ 8 bytes  │ dims × 4 bytes     │
//! └────────┴────────┴──────────┴──────────┴──────────┴────────────────────┘
//!
//! Total header: 32 bytes
//! Total entry: 32 + (dims × 4) bytes

use crate::storage::StorageError;

pub const WAL_MAGIC: u32 = 0x57414C31; // "WAL1"
pub const WAL_HEADER_SIZE: usize = 32;

/// Parsed WAL entry
#[derive(Clone, Debug)]
pub struct WalEntry {
    pub sequence: u64,
    pub tenant_id: u64,
    pub vector_id: u64,
    pub vector: Vec<f32>,
}

impl WalEntry {
    /// Serialize entry to bytes
    pub fn serialize(&self, dims: usize) -> Vec<u8> {
        let vector_bytes_len = dims * 4;
        let total_len = WAL_HEADER_SIZE + vector_bytes_len;
        let mut buffer = vec![0u8; total_len];

        // Write magic
        buffer[0..4].copy_from_slice(&WAL_MAGIC.to_le_bytes());

        // Skip CRC for now (bytes 4-8), will fill after

        // Write sequence
        buffer[8..16].copy_from_slice(&self.sequence.to_le_bytes());

        // Write tenant_id
        buffer[16..24].copy_from_slice(&self.tenant_id.to_le_bytes());

        // Write vector_id
        buffer[24..32].copy_from_slice(&self.vector_id.to_le_bytes());

        // Copy vector data
        for (i, &val) in self.vector.iter().take(dims).enumerate() {
            let offset = WAL_HEADER_SIZE + i * 4;
            buffer[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Calculate CRC over everything after the CRC field (bytes 8..end)
        let crc = crc32fast::hash(&buffer[8..]);
        buffer[4..8].copy_from_slice(&crc.to_le_bytes());

        buffer
    }

    /// Deserialize entry from bytes
    pub fn deserialize(data: &[u8], dims: usize) -> Result<Self, WalError> {
        let expected_len = WAL_HEADER_SIZE + dims * 4;
        if data.len() < expected_len {
            return Err(WalError::TruncatedEntry);
        }

        // Helper function for safe byte conversion
        fn to_u32(data: &[u8], start: usize) -> Result<u32, WalError> {
            data.get(start..start + 4)
                .ok_or(WalError::TruncatedEntry)?
                .try_into()
                .map(u32::from_le_bytes)
                .map_err(|_| WalError::TruncatedEntry)
        }

        fn to_u64(data: &[u8], start: usize) -> Result<u64, WalError> {
            data.get(start..start + 8)
                .ok_or(WalError::TruncatedEntry)?
                .try_into()
                .map(u64::from_le_bytes)
                .map_err(|_| WalError::TruncatedEntry)
        }

        // Parse magic
        let magic = to_u32(data, 0)?;
        if magic != WAL_MAGIC {
            return Err(WalError::InvalidMagic);
        }

        // Parse and validate CRC
        let stored_crc = to_u32(data, 4)?;
        let expected_crc = crc32fast::hash(&data[8..expected_len]);
        if stored_crc != expected_crc {
            return Err(WalError::ChecksumMismatch);
        }

        // Parse header fields
        let sequence = to_u64(data, 8)?;
        let tenant_id = to_u64(data, 16)?;
        let vector_id = to_u64(data, 24)?;

        // Parse vector with safe conversion
        let vector: Result<Vec<f32>, WalError> = data[WAL_HEADER_SIZE..expected_len]
            .chunks_exact(4)
            .map(|chunk| {
                chunk.try_into()
                    .map(f32::from_le_bytes)
                    .map_err(|_| WalError::TruncatedEntry)
            })
            .collect();

        Ok(Self {
            sequence,
            tenant_id,
            vector_id,
            vector: vector?,
        })
    }

    /// Get the total size of a serialized entry
    pub fn entry_size(dims: usize) -> usize {
        WAL_HEADER_SIZE + dims * 4
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("Invalid WAL magic number")]
    InvalidMagic,

    #[error("CRC checksum mismatch")]
    ChecksumMismatch,

    #[error("Truncated entry")]
    TruncatedEntry,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let entry = WalEntry {
            sequence: 42,
            tenant_id: 123,
            vector_id: 456,
            vector: vec![1.0, 2.0, 3.0, 4.0],
        };

        let serialized = entry.serialize(4);
        let deserialized = WalEntry::deserialize(&serialized, 4).unwrap();

        assert_eq!(deserialized.sequence, 42);
        assert_eq!(deserialized.tenant_id, 123);
        assert_eq!(deserialized.vector_id, 456);
        assert_eq!(deserialized.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_corruption_detection() {
        let entry = WalEntry {
            sequence: 1,
            tenant_id: 1,
            vector_id: 1,
            vector: vec![1.0, 2.0, 3.0, 4.0],
        };

        let mut serialized = entry.serialize(4);

        // Corrupt a byte
        serialized[20] ^= 0xFF;

        let result = WalEntry::deserialize(&serialized, 4);
        assert!(matches!(result, Err(WalError::ChecksumMismatch)));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = vec![0u8; 48];
        // Wrong magic
        data[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());

        let result = WalEntry::deserialize(&data, 4);
        assert!(matches!(result, Err(WalError::InvalidMagic)));
    }

    #[test]
    fn test_truncated() {
        let entry = WalEntry {
            sequence: 1,
            tenant_id: 1,
            vector_id: 1,
            vector: vec![1.0, 2.0, 3.0, 4.0],
        };

        let serialized = entry.serialize(4);
        // Truncate
        let truncated = &serialized[..30];

        let result = WalEntry::deserialize(truncated, 4);
        assert!(matches!(result, Err(WalError::TruncatedEntry)));
    }
}
