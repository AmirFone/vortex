//! ANN Index trait abstraction
//!
//! Provides a common interface for different ANN (Approximate Nearest Neighbor)
//! index implementations, allowing seamless switching between HNSW and DiskANN.

use async_trait::async_trait;
use std::fmt::Debug;

use crate::storage::{BlockStorage, StorageResult};
use crate::vectors::VectorStore;

/// Common interface for all ANN index implementations
///
/// This trait abstracts over different index backends (HNSW, DiskANN, etc.)
/// allowing the tenant system to work with any index type.
#[async_trait]
pub trait AnnIndex: Send + Sync + Debug + 'static {
    /// Insert a vector into the index
    ///
    /// # Arguments
    /// * `vector_index` - The index of the vector in the VectorStore
    /// * `vectors` - Reference to the VectorStore containing the actual vector data
    fn insert(&self, vector_index: u32, vectors: &VectorStore);

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search expansion factor (index-specific, may be ignored by some implementations)
    /// * `vectors` - Reference to the VectorStore containing the actual vector data
    ///
    /// # Returns
    /// Vector of (vector_index, distance) pairs, sorted by distance ascending
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
    ) -> Vec<(u32, f32)>;

    /// Save the index to persistent storage
    ///
    /// # Arguments
    /// * `storage` - The block storage backend
    /// * `path` - Path prefix for the index files
    async fn save(&self, storage: &dyn BlockStorage, path: &str) -> StorageResult<()>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rebuild the index from a list of vector indices
    ///
    /// This is used during recovery or when the index needs to be reconstructed.
    /// For incremental indexes (like HNSW), this may just insert each vector.
    /// For batch indexes (like some DiskANN variants), this may trigger a full rebuild.
    ///
    /// # Arguments
    /// * `indices` - Vector indices to include in the rebuilt index
    /// * `vectors` - Reference to the VectorStore containing the actual vector data
    async fn rebuild(&self, indices: &[u32], vectors: &VectorStore) -> StorageResult<()>;

    /// Check if this index supports efficient incremental inserts
    ///
    /// Returns true if inserts are O(log n) or better.
    /// Returns false if inserts trigger full index rebuilds.
    fn supports_incremental_insert(&self) -> bool;

    /// Get the name of this index type (for logging/debugging)
    fn index_type_name(&self) -> &'static str;
}

/// Extension trait for AnnIndex that provides utility methods
pub trait AnnIndexExt: AnnIndex {
    /// Batch insert multiple vectors
    fn insert_batch(&self, indices: &[u32], vectors: &VectorStore) {
        for &idx in indices {
            self.insert(idx, vectors);
        }
    }
}

// Blanket implementation
impl<T: AnnIndex> AnnIndexExt for T {}
