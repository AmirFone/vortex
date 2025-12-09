//! Vector store file format definitions

use crate::storage::StorageError;

pub const VECTOR_STORE_MAGIC: u32 = 0x56454353; // "VECS"
pub const HEADER_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VectorStoreHeader {
    pub magic: u32,
    pub version: u32,
    pub dims: u32,
    pub count: u64,
    pub _reserved: [u8; 44],
}

impl VectorStoreHeader {
    pub fn new(dims: usize) -> Self {
        Self {
            magic: VECTOR_STORE_MAGIC,
            version: 1,
            dims: dims as u32,
            count: 0,
            _reserved: [0; 44],
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.dims.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.count.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StorageError> {
        if bytes.len() < HEADER_SIZE {
            return Err(StorageError::Backend("Header too short".into()));
        }

        // Safe slice conversion - we've already verified length above
        let magic = u32::from_le_bytes(
            bytes.get(0..4)
                .ok_or_else(|| StorageError::Backend("Invalid magic bytes".into()))?
                .try_into()
                .map_err(|_| StorageError::Backend("Invalid magic byte conversion".into()))?
        );
        let version = u32::from_le_bytes(
            bytes.get(4..8)
                .ok_or_else(|| StorageError::Backend("Invalid version bytes".into()))?
                .try_into()
                .map_err(|_| StorageError::Backend("Invalid version byte conversion".into()))?
        );
        let dims = u32::from_le_bytes(
            bytes.get(8..12)
                .ok_or_else(|| StorageError::Backend("Invalid dims bytes".into()))?
                .try_into()
                .map_err(|_| StorageError::Backend("Invalid dims byte conversion".into()))?
        );
        let count = u64::from_le_bytes(
            bytes.get(12..20)
                .ok_or_else(|| StorageError::Backend("Invalid count bytes".into()))?
                .try_into()
                .map_err(|_| StorageError::Backend("Invalid count byte conversion".into()))?
        );

        Ok(Self {
            magic,
            version,
            dims,
            count,
            _reserved: [0; 44],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = VectorStoreHeader {
            magic: VECTOR_STORE_MAGIC,
            version: 1,
            dims: 384,
            count: 1000,
            _reserved: [0; 44],
        };

        let bytes = header.to_bytes();
        let parsed = VectorStoreHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, VECTOR_STORE_MAGIC);
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.dims, 384);
        assert_eq!(parsed.count, 1000);
    }
}
