//! Storage abstraction layer
//!
//! Provides traits for block storage (EBS-like) and object storage (S3-like)
//! with mock implementations for local development and real AWS implementations
//! for production.

pub mod mock;
pub mod types;

#[cfg(feature = "aws-storage")]
pub mod aws;

use async_trait::async_trait;
use bytes::Bytes;
use std::path::Path;

pub use types::*;

/// Error type for storage operations
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Object not found: {key}")]
    NotFound { key: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Storage backend error: {0}")]
    Backend(String),
}

pub type StorageResult<T> = Result<T, StorageError>;

/// Block storage trait (EBS-like)
///
/// Simulates a persistent block device with:
/// - Random read/write access
/// - fsync for durability
/// - File-based operations
#[async_trait]
pub trait BlockStorage: Send + Sync + 'static {
    /// Write data to a file path (relative to storage root)
    /// Does NOT guarantee durability until sync() is called
    async fn write(&self, path: &str, data: &[u8]) -> StorageResult<()>;

    /// Write data at a specific offset (for append-only writes)
    /// Creates/extends file if needed. Does NOT guarantee durability until sync() is called
    async fn write_at(&self, path: &str, offset: usize, data: &[u8]) -> StorageResult<()>;

    /// Append data to a file, returns the offset where data was written
    async fn append(&self, path: &str, data: &[u8]) -> StorageResult<u64>;

    /// Read entire file
    async fn read(&self, path: &str) -> StorageResult<Vec<u8>>;

    /// Read range of bytes from file
    async fn read_range(&self, path: &str, offset: u64, length: usize) -> StorageResult<Vec<u8>>;

    /// Check if file exists
    async fn exists(&self, path: &str) -> StorageResult<bool>;

    /// Get file size
    async fn size(&self, path: &str) -> StorageResult<u64>;

    /// Sync file to durable storage (fsync)
    /// CRITICAL: This is the durability guarantee
    async fn sync(&self, path: &str) -> StorageResult<()>;

    /// Delete file
    async fn delete(&self, path: &str) -> StorageResult<()>;

    /// List files in directory
    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>>;

    /// Create directory
    async fn create_dir(&self, path: &str) -> StorageResult<()>;

    /// Get memory-mapped view of file (for zero-copy reads)
    fn mmap(&self, path: &str) -> StorageResult<Option<memmap2::Mmap>>;

    /// Get the root path (for diagnostics)
    fn root_path(&self) -> &Path;
}

/// Object storage trait (S3-like)
///
/// Simulates an object store with:
/// - PUT/GET/DELETE operations
/// - No random access (whole-object only)
#[async_trait]
pub trait ObjectStorage: Send + Sync + 'static {
    /// Put object
    async fn put(&self, key: &str, data: Bytes) -> StorageResult<()>;

    /// Get object
    async fn get(&self, key: &str) -> StorageResult<Bytes>;

    /// Check if object exists
    async fn exists(&self, key: &str) -> StorageResult<bool>;

    /// Delete object
    async fn delete(&self, key: &str) -> StorageResult<()>;

    /// List objects with prefix
    async fn list(&self, prefix: &str) -> StorageResult<Vec<ObjectInfo>>;
}

/// Combined storage backend
pub struct StorageBackend {
    pub block: Box<dyn BlockStorage>,
    pub object: Box<dyn ObjectStorage>,
}

impl StorageBackend {
    pub fn new(block: impl BlockStorage, object: impl ObjectStorage) -> Self {
        Self {
            block: Box::new(block),
            object: Box::new(object),
        }
    }
}
