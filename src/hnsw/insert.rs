//! HNSW insert algorithms

use super::node::HnswNode;
use super::search::{greedy_search_layer, search_layer, Candidate};
use super::HnswIndex;
use crate::vectors::VectorStore;
use rand::Rng;
use std::cmp::Ordering;

/// Insert a node into the HNSW index
pub fn insert_node(index: &HnswIndex, vector_index: u32, vectors: &VectorStore) {
    let config = index.config();

    // Generate random level for this node
    let level = random_level(config.ml);

    // Create the new node
    let node = HnswNode::new(vector_index, level);

    // Get write lock
    let mut nodes = index.nodes().write();
    let new_node_id = nodes.len() as u32;

    // Handle first node
    if nodes.is_empty() {
        nodes.push(node);
        index.set_entry_point(0);
        index.set_max_level(level);
        return;
    }

    let entry_point = index.entry_point();
    let current_max_level = index.max_level();

    // Find entry point at the node's level
    let mut current = entry_point;

    // Traverse from top to the node's level + 1 (greedy search)
    for l in (level + 1..=current_max_level).rev() {
        current = greedy_search_layer(&nodes, &get_vector(vectors, vector_index), current, l, vectors);
    }

    // Add the node before we start connecting it
    nodes.push(node);

    // Insert at each level from level down to 0
    for l in (0..=level.min(current_max_level)).rev() {
        // Find ef_construction nearest neighbors at this level
        let candidates = search_layer(
            &nodes,
            &get_vector(vectors, vector_index),
            current,
            config.ef_construction,
            l,
            vectors,
        );

        // Select neighbors using diversity heuristic
        let max_neighbors = if l == 0 { config.m_max0 } else { config.m };
        let selected = select_neighbors_heuristic(&candidates, max_neighbors, &nodes, vectors);

        // Set neighbors for new node
        nodes[new_node_id as usize].set_neighbors(l, selected.clone());

        // Create bidirectional edges
        for &neighbor_id in &selected {
            if neighbor_id as usize >= nodes.len() {
                continue;
            }

            // Add new_node_id to neighbor's neighbors
            nodes[neighbor_id as usize].add_neighbor(l, new_node_id);

            // Prune if necessary
            let max_n = if l == 0 { config.m_max0 } else { config.m };
            if nodes[neighbor_id as usize].neighbors.len() > l
                && nodes[neighbor_id as usize].neighbors[l].len() > max_n
            {
                // Collect data needed for pruning
                let neighbor_vec_idx = nodes[neighbor_id as usize].vector_index;
                let neighbor_vec = get_vector(vectors, neighbor_vec_idx);
                let current_neighbors: Vec<u32> =
                    nodes[neighbor_id as usize].neighbors[l].clone();

                let mut neighbor_candidates: Vec<(u32, f32)> = current_neighbors
                    .into_iter()
                    .filter_map(|n_id| {
                        if n_id as usize >= nodes.len() {
                            return None;
                        }
                        let n_vec = get_vector(vectors, nodes[n_id as usize].vector_index);
                        let dist = cosine_distance(&neighbor_vec, &n_vec);
                        Some((n_id, dist))
                    })
                    .collect();

                neighbor_candidates
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                let pruned: Vec<u32> = neighbor_candidates
                    .into_iter()
                    .take(max_n)
                    .map(|(id, _)| id)
                    .collect();

                nodes[neighbor_id as usize].neighbors[l] = pruned;
            }
        }

        // Use closest neighbor as entry point for next level
        if !candidates.is_empty() {
            current = candidates[0].node_id;
        }
    }

    // Update entry point if new node has higher level
    if level > current_max_level {
        index.set_entry_point(new_node_id);
        index.set_max_level(level);
    }
}

/// Generate random level using exponential distribution
fn random_level(ml: f64) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    (-r.ln() * ml).floor() as usize
}

/// Select neighbors - simple approach (closest M) which works well for high-dimensional vectors
/// The diversity heuristic can hurt recall in high dimensions where vectors are roughly orthogonal
/// Candidates must be sorted by distance to the query vector
fn select_neighbors_heuristic(
    candidates: &[Candidate],
    max_count: usize,
    _nodes: &[super::node::HnswNode],
    _vectors: &VectorStore,
) -> Vec<u32> {
    if candidates.is_empty() || max_count == 0 {
        return Vec::new();
    }

    // Simple selection: take closest M neighbors (candidates are already sorted)
    let mut result = Vec::with_capacity(max_count);
    for candidate in candidates.iter() {
        if result.len() >= max_count {
            break;
        }
        // Skip duplicates
        if !result.contains(&candidate.node_id) {
            result.push(candidate.node_id);
        }
    }

    result
}

/// Get vector from store (helper)
fn get_vector(vectors: &VectorStore, index: u32) -> Vec<f32> {
    vectors.get(index).unwrap_or_default()
}

/// Cosine distance (1 - dot product for normalized vectors)
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_level_distribution() {
        let ml = 1.0 / (16.0_f64).ln();
        let mut level_counts = [0usize; 10];

        for _ in 0..10000 {
            let level = random_level(ml);
            if level < 10 {
                level_counts[level] += 1;
            }
        }

        // Most nodes should be at level 0
        assert!(level_counts[0] > level_counts[1]);
        assert!(level_counts[1] > level_counts[2]);
    }
}
