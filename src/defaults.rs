//! Centralized default values and constants
//!
//! This module consolidates all magic numbers and default values used throughout
//! the codebase, making them easy to find, modify, and document.

// ============================================================================
// Vector Dimensions
// ============================================================================

/// Default vector dimensionality (common for sentence embeddings like all-MiniLM-L6-v2)
pub const DEFAULT_DIMENSIONS: usize = 384;

// ============================================================================
// HNSW Index Parameters
// ============================================================================

/// Default number of bidirectional links per node (M parameter)
/// Higher values improve recall but increase memory and build time
pub const DEFAULT_HNSW_M: usize = 16;

/// Maximum connections at layer 0 (typically 2*M)
pub const DEFAULT_HNSW_M_MAX0: usize = 32;

/// Default ef value during construction
/// Higher values improve index quality but slow down builds
pub const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;

/// Default ef value during search
/// Higher values improve recall but slow down queries
pub const DEFAULT_HNSW_EF_SEARCH: usize = 100;

/// Multiplier for level generation probability (1/ln(M))
pub const DEFAULT_HNSW_ML: f64 = 0.36; // approximately 1/ln(16)

// ============================================================================
// DiskANN Index Parameters
// ============================================================================

/// Default maximum out-degree for DiskANN graph
pub const DEFAULT_DISKANN_MAX_DEGREE: usize = 64;

/// Default alpha parameter for pruning (controls graph density)
pub const DEFAULT_DISKANN_ALPHA: f32 = 1.2;

/// Default search beam width
pub const DEFAULT_DISKANN_SEARCH_BEAM: usize = 64;

// ============================================================================
// WAL (Write-Ahead Log) Constants
// ============================================================================

/// WAL file magic number ("WAL1" in little-endian)
pub const WAL_MAGIC: u32 = 0x57414C31;

/// WAL header size in bytes
pub const WAL_HEADER_SIZE: usize = 32;

// ============================================================================
// Vector Store Constants
// ============================================================================

/// Vector store file magic number ("VECS" in little-endian)
pub const VECTOR_STORE_MAGIC: u32 = 0x56454353;

/// Vector store header size in bytes
pub const VECTOR_STORE_HEADER_SIZE: usize = 64;

// ============================================================================
// Engine Configuration
// ============================================================================

/// Default flush interval in seconds
pub const DEFAULT_FLUSH_INTERVAL_SECS: u64 = 30;

/// Default write buffer capacity (number of vectors before forcing flush)
pub const DEFAULT_WRITE_BUFFER_CAPACITY: usize = 10_000;

// ============================================================================
// Server Configuration
// ============================================================================

/// Default server host
pub const DEFAULT_HOST: &str = "0.0.0.0";

/// Default server port
pub const DEFAULT_PORT: u16 = 3000;

// ============================================================================
// Search Parameters
// ============================================================================

/// Default number of results to return (k)
pub const DEFAULT_TOP_K: usize = 10;

// ============================================================================
// Mock Storage Configuration
// ============================================================================

/// Default latency for mock storage operations (milliseconds)
pub const DEFAULT_MOCK_LATENCY_MS: u64 = 0;

/// Default latency jitter for mock storage (milliseconds)
pub const DEFAULT_MOCK_JITTER_MS: u64 = 0;

/// Default failure rate for mock storage (0.0 to 1.0)
pub const DEFAULT_MOCK_FAILURE_RATE: f64 = 0.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_are_valid() {
        // Verify HNSW parameters make sense
        assert!(DEFAULT_HNSW_M > 0);
        assert!(DEFAULT_HNSW_M_MAX0 >= DEFAULT_HNSW_M);
        assert!(DEFAULT_HNSW_EF_CONSTRUCTION > DEFAULT_HNSW_M);
        assert!(DEFAULT_HNSW_EF_SEARCH > 0);

        // Verify dimensions are reasonable
        assert!(DEFAULT_DIMENSIONS > 0);
        assert!(DEFAULT_DIMENSIONS <= 4096);

        // Verify DiskANN parameters
        assert!(DEFAULT_DISKANN_MAX_DEGREE > 0);
        assert!(DEFAULT_DISKANN_ALPHA > 1.0);

        // Verify flush interval is reasonable
        assert!(DEFAULT_FLUSH_INTERVAL_SECS > 0);
    }

    #[test]
    fn test_magic_numbers_are_ascii() {
        // WAL_MAGIC = 0x57414C31 ("WAL1" in big-endian ASCII)
        // When stored as little-endian, the bytes are reversed
        let wal_bytes = WAL_MAGIC.to_le_bytes();
        assert_eq!(&wal_bytes, b"1LAW"); // Little-endian order

        // VECTOR_STORE_MAGIC = 0x56454353 ("VECS" in big-endian ASCII)
        let vecs_bytes = VECTOR_STORE_MAGIC.to_le_bytes();
        assert_eq!(&vecs_bytes, b"SCEV"); // Little-endian order

        // Verify the actual magic values are correct
        assert_eq!(WAL_MAGIC, 0x57414C31);
        assert_eq!(VECTOR_STORE_MAGIC, 0x56454353);
    }
}
