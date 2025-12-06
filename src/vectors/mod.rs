//! Contiguous vector storage with memory mapping
//!
//! File format:
//! ┌──────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                            │
//! │   magic: u32      = 0x56454353 ("VECS")                      │
//! │   version: u32    = 1                                        │
//! │   dims: u32       = vector dimensions                        │
//! │   count: u64      = number of vectors                        │
//! │   _reserved: [u8; 44]                                        │
//! ├──────────────────────────────────────────────────────────────┤
//! │ Vector 0: [f32; dims]                                        │
//! │ Vector 1: [f32; dims]                                        │
//! │ Vector 2: [f32; dims]                                        │
//! │ ...                                                          │
//! └──────────────────────────────────────────────────────────────┘

pub mod format;

use crate::storage::{BlockStorage, StorageError, StorageResult};
use format::{VectorStoreHeader, HEADER_SIZE, VECTOR_STORE_MAGIC};
use memmap2::Mmap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Memory-mapped vector storage
pub struct VectorStore {
    storage: Arc<dyn BlockStorage>,
    path: String,
    dims: usize,
    entry_size: usize,
    count: AtomicU64,
    mmap: RwLock<Option<Mmap>>,
}

impl VectorStore {
    /// Open or create a vector store
    pub async fn open(
        storage: Arc<dyn BlockStorage>,
        path: impl Into<String>,
        dims: usize,
    ) -> StorageResult<Self> {
        let path = path.into();
        let entry_size = dims * 4;

        let count = if storage.exists(&path).await? {
            // Read existing header
            let header_bytes = storage.read_range(&path, 0, HEADER_SIZE).await?;
            let header = VectorStoreHeader::from_bytes(&header_bytes)?;

            if header.magic != VECTOR_STORE_MAGIC {
                return Err(StorageError::Backend("Invalid vector store magic".into()));
            }

            if header.dims as usize != dims {
                return Err(StorageError::Backend(format!(
                    "Dimension mismatch: file has {}, expected {}",
                    header.dims, dims
                )));
            }

            header.count
        } else {
            // Create new file with header
            let header = VectorStoreHeader::new(dims);
            storage.write(&path, &header.to_bytes()).await?;
            storage.sync(&path).await?;
            0
        };

        // Create mmap
        let mmap = storage.mmap(&path)?;

        Ok(Self {
            storage,
            path,
            dims,
            entry_size,
            count: AtomicU64::new(count),
            mmap: RwLock::new(mmap),
        })
    }

    /// Append a single vector, returns its index
    pub async fn append(&self, vector: &[f32]) -> StorageResult<u32> {
        debug_assert_eq!(vector.len(), self.dims);

        let index = self.count.fetch_add(1, Ordering::SeqCst) as u32;
        let offset = HEADER_SIZE + (index as usize) * self.entry_size;

        // Serialize vector
        let mut data = Vec::with_capacity(self.entry_size);
        for &val in vector {
            data.extend_from_slice(&val.to_le_bytes());
        }

        // Read current file content
        let mut file_data = if self.storage.exists(&self.path).await? {
            self.storage.read(&self.path).await?
        } else {
            Vec::new()
        };

        // Ensure file is large enough
        let new_size = offset + self.entry_size;
        if file_data.len() < new_size {
            file_data.resize(new_size, 0);
        }

        // Write vector data
        file_data[offset..offset + self.entry_size].copy_from_slice(&data);

        // Update header with new count
        let header = VectorStoreHeader {
            magic: VECTOR_STORE_MAGIC,
            version: 1,
            dims: self.dims as u32,
            count: self.count.load(Ordering::SeqCst),
            _reserved: [0; 44],
        };
        file_data[..HEADER_SIZE].copy_from_slice(&header.to_bytes());

        self.storage.write(&self.path, &file_data).await?;
        self.storage.sync(&self.path).await?;

        // Refresh mmap
        *self.mmap.write() = self.storage.mmap(&self.path)?;

        Ok(index)
    }

    /// Append multiple vectors, returns starting index
    /// Optimized: Uses append-only writes instead of read-modify-write
    pub async fn append_batch(&self, vectors: &[Vec<f32>]) -> StorageResult<u32> {
        if vectors.is_empty() {
            return Ok(self.count.load(Ordering::Acquire) as u32);
        }

        let start_index = self.count.fetch_add(vectors.len() as u64, Ordering::AcqRel) as u32;
        let start_offset = HEADER_SIZE + (start_index as usize) * self.entry_size;
        let total_data_size = vectors.len() * self.entry_size;

        // Serialize all vectors
        let mut vector_data = Vec::with_capacity(total_data_size);
        for vector in vectors {
            debug_assert_eq!(vector.len(), self.dims);
            for &val in vector {
                vector_data.extend_from_slice(&val.to_le_bytes());
            }
        }

        // Append-only write: write vectors at offset (no full file read)
        self.storage
            .write_at(&self.path, start_offset, &vector_data)
            .await?;

        // Update header with new count (small write at offset 0)
        let header = VectorStoreHeader {
            magic: VECTOR_STORE_MAGIC,
            version: 1,
            dims: self.dims as u32,
            count: self.count.load(Ordering::Acquire),
            _reserved: [0; 44],
        };
        self.storage
            .write_at(&self.path, 0, &header.to_bytes())
            .await?;

        // Single fsync for entire batch
        self.storage.sync(&self.path).await?;

        // Refresh mmap
        *self.mmap.write() = self.storage.mmap(&self.path)?;

        Ok(start_index)
    }

    /// Get a vector by index (returns a copy)
    pub fn get(&self, index: u32) -> Option<Vec<f32>> {
        let count = self.count.load(Ordering::SeqCst);
        if index as u64 >= count {
            return None;
        }

        let mmap = self.mmap.read();
        let mmap = mmap.as_ref()?;

        let offset = HEADER_SIZE + (index as usize) * self.entry_size;
        if offset + self.entry_size > mmap.len() {
            return None;
        }

        let data = &mmap[offset..offset + self.entry_size];
        let vector: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Some(vector)
    }

    /// Get a slice reference to a vector (zero-copy, requires mmap lock held)
    /// Returns None if mmap is not available or index out of bounds
    pub fn get_slice<'a>(&'a self, index: u32, mmap: &'a Mmap) -> Option<&'a [f32]> {
        let count = self.count.load(Ordering::SeqCst);
        if index as u64 >= count {
            return None;
        }

        let offset = HEADER_SIZE + (index as usize) * self.entry_size;
        if offset + self.entry_size > mmap.len() {
            return None;
        }

        // SAFETY: We've verified bounds and the mmap data is valid f32s
        Some(unsafe {
            std::slice::from_raw_parts(mmap[offset..].as_ptr() as *const f32, self.dims)
        })
    }

    /// Get mmap handle for batch operations
    pub fn mmap(&self) -> parking_lot::RwLockReadGuard<'_, Option<Mmap>> {
        self.mmap.read()
    }

    /// Get number of vectors
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::SeqCst)
    }

    /// Get dimensions
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Sync to disk
    pub async fn sync(&self) -> StorageResult<()> {
        self.storage.sync(&self.path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{MockBlockStorage, MockStorageConfig};

    #[tokio::test]
    async fn test_vector_store_append_and_get() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let store = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        let idx = store.append(&[1.0, 2.0, 3.0, 4.0]).await.unwrap();
        assert_eq!(idx, 0);

        let vec = store.get(0).unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_vector_store_batch() {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let store = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        let start_idx = store.append_batch(&vectors).await.unwrap();
        assert_eq!(start_idx, 0);
        assert_eq!(store.count(), 3);

        assert_eq!(store.get(0).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.get(1).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(store.get(2).unwrap(), vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[tokio::test]
    async fn test_vector_store_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write vectors
        {
            let storage = Arc::new(MockBlockStorage::new(&path, MockStorageConfig::fast()).unwrap());
            let store = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();
            store.append(&[1.0, 2.0, 3.0, 4.0]).await.unwrap();
            store.append(&[5.0, 6.0, 7.0, 8.0]).await.unwrap();
        }

        // Reopen and verify
        {
            let storage = Arc::new(MockBlockStorage::new(&path, MockStorageConfig::fast()).unwrap());
            let store = VectorStore::open(storage, "vectors.bin", 4).await.unwrap();
            assert_eq!(store.count(), 2);
            assert_eq!(store.get(0).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(store.get(1).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        }
    }
}
