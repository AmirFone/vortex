//! HNSW index persistence

use super::node::HnswNode;
use super::{HnswConfig, HnswIndex};
use crate::storage::{BlockStorage, StorageResult};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

/// Serializable HNSW state
#[derive(Serialize, Deserialize)]
struct HnswState {
    entry_point: u32,
    max_level: usize,
    nodes: Vec<HnswNode>,
}

/// Save index to storage
pub async fn save_index(
    index: &HnswIndex,
    storage: &dyn BlockStorage,
    path: &str,
) -> StorageResult<()> {
    // Serialize in a separate scope to drop the lock before await
    let data = {
        let nodes = index.nodes().read();

        let state = HnswState {
            entry_point: index.entry_point(),
            max_level: index.max_level(),
            nodes: nodes.clone(),
        };

        bincode::serialize(&state)
            .map_err(|e| crate::storage::StorageError::Serialization(e.to_string()))?
    };

    storage.write(path, &data).await?;
    storage.sync(path).await?;

    Ok(())
}

/// Load index from storage
pub async fn load_index(
    storage: &dyn BlockStorage,
    path: &str,
    config: HnswConfig,
) -> StorageResult<HnswIndex> {
    if !storage.exists(path).await? {
        return Ok(HnswIndex::new(config));
    }

    let data = storage.read(path).await?;

    let state: HnswState = bincode::deserialize(&data)
        .map_err(|e| crate::storage::StorageError::Serialization(e.to_string()))?;

    let index = HnswIndex::new(config);

    // Set state
    index.entry_point.store(state.entry_point, Ordering::SeqCst);
    index.max_level.store(state.max_level, Ordering::SeqCst);
    *index.nodes.write() = state.nodes;

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::{MockBlockStorage, MockStorageConfig};

    #[tokio::test]
    async fn test_save_load_empty() {
        let storage = MockBlockStorage::temp(MockStorageConfig::fast()).unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(config.clone());

        save_index(&index, &storage, "test.hnsw").await.unwrap();

        let loaded = load_index(&storage, "test.hnsw", config).await.unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[tokio::test]
    async fn test_save_load_with_nodes() {
        let storage = MockBlockStorage::temp(MockStorageConfig::fast()).unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(config.clone());

        // Add some nodes manually
        {
            let mut nodes = index.nodes().write();
            nodes.push(HnswNode::new(0, 1));
            nodes.push(HnswNode::new(1, 0));
            nodes.push(HnswNode::new(2, 0));

            // Add some connections
            nodes[0].add_neighbor(0, 1);
            nodes[0].add_neighbor(0, 2);
            nodes[1].add_neighbor(0, 0);
            nodes[2].add_neighbor(0, 0);
        }
        index.set_entry_point(0);
        index.set_max_level(1);

        save_index(&index, &storage, "test.hnsw").await.unwrap();

        let loaded = load_index(&storage, "test.hnsw", config).await.unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.entry_point(), 0);
        assert_eq!(loaded.max_level(), 1);

        let nodes = loaded.nodes().read();
        assert_eq!(nodes[0].neighbors[0].len(), 2);
    }
}
