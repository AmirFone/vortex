//! Unified error types for Vortex
//!
//! This module provides a centralized error hierarchy that all components
//! can use, enabling consistent error handling across the codebase.

use crate::storage::StorageError;
use crate::wal::entry::WalError;

/// Main error type for Vortex operations
#[derive(Debug, thiserror::Error)]
pub enum VortexError {
    /// Storage layer errors (block storage, object storage)
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    /// Write-ahead log errors
    #[error("WAL error: {0}")]
    Wal(#[from] WalError),

    /// Index operation errors (HNSW, DiskANN)
    #[error("Index error: {0}")]
    Index(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Vector dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Tenant not found
    #[error("Tenant not found: {tenant_id}")]
    TenantNotFound { tenant_id: u64 },

    /// Vector not found
    #[error("Vector not found: {vector_id}")]
    VectorNotFound { vector_id: u64 },

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Deserialization errors
    #[error("Deserialization error: {0}")]
    Deserialization(String),
}

/// Result type alias for Vortex operations
pub type Result<T> = std::result::Result<T, VortexError>;

impl VortexError {
    /// Create an index error
    pub fn index(msg: impl Into<String>) -> Self {
        Self::Index(msg.into())
    }

    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a deserialization error
    pub fn deserialization(msg: impl Into<String>) -> Self {
        Self::Deserialization(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VortexError::dimension_mismatch(384, 128);
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected 384, got 128"
        );
    }

    #[test]
    fn test_error_constructors() {
        let err = VortexError::index("HNSW build failed");
        assert!(matches!(err, VortexError::Index(_)));

        let err = VortexError::config("Invalid M parameter");
        assert!(matches!(err, VortexError::Config(_)));
    }
}
