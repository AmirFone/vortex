//! HNSW search algorithms

use super::node::HnswNode;
use super::HnswIndex;
use crate::vectors::VectorStore;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Candidate for search (min-heap by distance)
#[derive(Clone, Copy)]
pub struct Candidate {
    pub node_id: u32,
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse for min-heap (smaller distance = higher priority)
        other
            .distance
            .partial_cmp(&self.distance)
            .or(Some(Ordering::Equal))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (larger distance = higher priority, used for result pruning)
#[derive(Clone, Copy)]
pub struct MaxCandidate {
    pub node_id: u32,
    pub distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance
            .partial_cmp(&other.distance)
            .or(Some(Ordering::Equal))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Search for k nearest neighbors
pub fn search_knn(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
    ef: usize,
    vectors: &VectorStore,
) -> Vec<(u32, f32)> {
    let entry = index.entry_point();
    if entry == u32::MAX {
        return Vec::new();
    }

    let nodes = index.nodes().read();
    if nodes.is_empty() {
        return Vec::new();
    }

    let max_level = index.max_level();

    let mut current = entry;

    // Phase 1: Greedy search from top level to level 1
    for level in (1..=max_level).rev() {
        current = greedy_search_layer(&nodes, query, current, level, vectors);
    }

    // Phase 2: Search layer 0 with ef candidates
    let candidates = search_layer(&nodes, query, current, ef.max(k), 0, vectors);

    // Convert to output format (vector_index, similarity)
    candidates
        .into_iter()
        .take(k)
        .map(|c| {
            let node = &nodes[c.node_id as usize];
            // Convert distance to similarity (1 - distance for cosine)
            (node.vector_index, 1.0 - c.distance)
        })
        .collect()
}

/// Greedy search to find single nearest node at a level
pub fn greedy_search_layer(
    nodes: &[HnswNode],
    query: &[f32],
    entry: u32,
    level: usize,
    vectors: &VectorStore,
) -> u32 {
    let mut current = entry;
    let mut current_dist = distance_to_node(nodes, query, current, vectors);

    loop {
        let node = &nodes[current as usize];
        if level >= node.neighbors.len() {
            break;
        }

        let mut changed = false;
        for &neighbor_id in &node.neighbors[level] {
            if neighbor_id as usize >= nodes.len() {
                continue;
            }
            let dist = distance_to_node(nodes, query, neighbor_id, vectors);
            if dist < current_dist {
                current = neighbor_id;
                current_dist = dist;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    current
}

/// Search layer with ef candidates
pub fn search_layer(
    nodes: &[HnswNode],
    query: &[f32],
    entry: u32,
    ef: usize,
    level: usize,
    vectors: &VectorStore,
) -> Vec<Candidate> {
    let entry_dist = distance_to_node(nodes, query, entry, vectors);

    let mut visited = HashSet::new();
    visited.insert(entry);

    // Min-heap for candidates to explore
    let mut candidates = BinaryHeap::new();
    candidates.push(Candidate {
        node_id: entry,
        distance: entry_dist,
    });

    // Max-heap for results (worst at top for easy pruning)
    let mut results = BinaryHeap::new();
    results.push(MaxCandidate {
        node_id: entry,
        distance: entry_dist,
    });

    while let Some(current) = candidates.pop() {
        // Stop if current is farther than worst result
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if current.distance > worst.distance {
                    break;
                }
            }
        }

        let node = &nodes[current.node_id as usize];
        if level >= node.neighbors.len() {
            continue;
        }

        for &neighbor_id in &node.neighbors[level] {
            if visited.contains(&neighbor_id) {
                continue;
            }
            if neighbor_id as usize >= nodes.len() {
                continue;
            }
            visited.insert(neighbor_id);

            let dist = distance_to_node(nodes, query, neighbor_id, vectors);

            let should_add = results.len() < ef
                || results
                    .peek()
                    .map(|worst| dist < worst.distance)
                    .unwrap_or(true);

            if should_add {
                candidates.push(Candidate {
                    node_id: neighbor_id,
                    distance: dist,
                });
                results.push(MaxCandidate {
                    node_id: neighbor_id,
                    distance: dist,
                });

                while results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    // Convert max-heap results to sorted vec
    let mut result_vec: Vec<Candidate> = results
        .into_iter()
        .map(|mc| Candidate {
            node_id: mc.node_id,
            distance: mc.distance,
        })
        .collect();

    result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result_vec
}

/// Compute distance from query to a node
#[inline]
pub fn distance_to_node(nodes: &[HnswNode], query: &[f32], node_id: u32, vectors: &VectorStore) -> f32 {
    let node = &nodes[node_id as usize];
    if let Some(vector) = vectors.get(node.vector_index) {
        cosine_distance(query, &vector)
    } else {
        f32::MAX
    }
}

/// Cosine distance (1 - dot product for normalized vectors)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Cosine similarity (dot product for normalized vectors)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate {
            node_id: 0,
            distance: 0.5,
        });
        heap.push(Candidate {
            node_id: 1,
            distance: 0.2,
        });
        heap.push(Candidate {
            node_id: 2,
            distance: 0.8,
        });

        // Min-heap: smallest distance first
        assert_eq!(heap.pop().unwrap().node_id, 1);
        assert_eq!(heap.pop().unwrap().node_id, 0);
        assert_eq!(heap.pop().unwrap().node_id, 2);
    }
}
