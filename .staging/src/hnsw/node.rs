//! HNSW node representation

use serde::{Deserialize, Serialize};

/// A node in the HNSW graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswNode {
    /// Index into the external VectorStore
    pub vector_index: u32,
    /// Maximum level this node exists at
    pub level: usize,
    /// Neighbors at each level (level 0 is index 0)
    pub neighbors: Vec<Vec<u32>>,
}

impl HnswNode {
    pub fn new(vector_index: u32, level: usize) -> Self {
        Self {
            vector_index,
            level,
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
        }
    }

    /// Get neighbors at a specific level
    pub fn neighbors_at(&self, level: usize) -> &[u32] {
        if level < self.neighbors.len() {
            &self.neighbors[level]
        } else {
            &[]
        }
    }

    /// Add a neighbor at a specific level
    pub fn add_neighbor(&mut self, level: usize, neighbor: u32) {
        if level < self.neighbors.len() {
            if !self.neighbors[level].contains(&neighbor) {
                self.neighbors[level].push(neighbor);
            }
        }
    }

    /// Set neighbors at a specific level
    pub fn set_neighbors(&mut self, level: usize, neighbors: Vec<u32>) {
        if level < self.neighbors.len() {
            self.neighbors[level] = neighbors;
        }
    }
}
