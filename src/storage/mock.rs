//! Mock storage implementation for local development
//!
//! Features:
//! - Uses local filesystem
//! - Simulates EBS latency (configurable)
//! - Simulates S3 latency
//! - Full API compatibility with real AWS

use super::*;
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for mock storage behavior
#[derive(Debug, Clone)]
pub struct MockStorageConfig {
    /// Simulated EBS fsync latency
    pub ebs_fsync_latency: Duration,
    /// Simulated EBS read latency
    pub ebs_read_latency: Duration,
    /// Simulated S3 PUT latency
    pub s3_put_latency: Duration,
    /// Simulated S3 GET latency
    pub s3_get_latency: Duration,
    /// Random latency variance (0.0 - 1.0)
    pub latency_variance: f64,
}

impl Default for MockStorageConfig {
    fn default() -> Self {
        Self {
            // Realistic EBS gp3 latencies
            ebs_fsync_latency: Duration::from_millis(3),
            ebs_read_latency: Duration::from_micros(100),
            // Realistic S3 latencies
            s3_put_latency: Duration::from_millis(50),
            s3_get_latency: Duration::from_millis(20),
            latency_variance: 0.2,
        }
    }
}

impl MockStorageConfig {
    /// Config for fast tests (no artificial latency)
    pub fn fast() -> Self {
        Self {
            ebs_fsync_latency: Duration::ZERO,
            ebs_read_latency: Duration::ZERO,
            s3_put_latency: Duration::ZERO,
            s3_get_latency: Duration::ZERO,
            latency_variance: 0.0,
        }
    }

    /// Config for realistic simulation
    pub fn realistic() -> Self {
        Self::default()
    }
}

/// Mock EBS (block storage)
pub struct MockBlockStorage {
    root: PathBuf,
    config: MockStorageConfig,
}

impl MockBlockStorage {
    pub fn new(root: impl Into<PathBuf>, config: MockStorageConfig) -> std::io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;

        Ok(Self { root, config })
    }

    /// Create with temp directory (for tests)
    pub fn temp(config: MockStorageConfig) -> std::io::Result<Self> {
        let temp_dir = tempfile::tempdir()?.into_path();
        Self::new(temp_dir, config)
    }

    fn full_path(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }

    async fn simulate_latency(&self, base: Duration) {
        // Always yield to the runtime to allow other tasks to progress
        // This is critical for cooperative multitasking in single-threaded runtimes
        tokio::task::yield_now().await;

        if base.is_zero() {
            return;
        }

        let variance = self.config.latency_variance;
        let jitter = if variance > 0.0 {
            let factor = 1.0 + (rand::random::<f64>() * 2.0 - 1.0) * variance;
            base.mul_f64(factor)
        } else {
            base
        };

        sleep(jitter).await;
    }
}

#[async_trait]
impl BlockStorage for MockBlockStorage {
    async fn write(&self, path: &str, data: &[u8]) -> StorageResult<()> {
        // Yield to runtime for cooperative scheduling
        tokio::task::yield_now().await;
        let full_path = self.full_path(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(&full_path)?;
        file.write_all(data)?;

        Ok(())
    }

    async fn write_at(&self, path: &str, offset: usize, data: &[u8]) -> StorageResult<()> {
        // Yield to runtime for cooperative scheduling
        tokio::task::yield_now().await;
        let full_path = self.full_path(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Open file for writing, create if doesn't exist
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(&full_path)?;

        // Extend file if needed
        let file_len = file.metadata()?.len() as usize;
        if offset + data.len() > file_len {
            file.set_len((offset + data.len()) as u64)?;
        }

        // Seek to offset and write
        file.seek(SeekFrom::Start(offset as u64))?;
        file.write_all(data)?;

        Ok(())
    }

    async fn append(&self, path: &str, data: &[u8]) -> StorageResult<u64> {
        // Yield to runtime for cooperative scheduling
        tokio::task::yield_now().await;
        let full_path = self.full_path(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&full_path)?;

        let offset = file.seek(SeekFrom::End(0))?;
        file.write_all(data)?;

        Ok(offset)
    }

    async fn read(&self, path: &str) -> StorageResult<Vec<u8>> {
        self.simulate_latency(self.config.ebs_read_latency).await;

        let full_path = self.full_path(path);
        if !full_path.exists() {
            return Err(StorageError::NotFound {
                key: path.to_string(),
            });
        }

        let data = fs::read(&full_path)?;
        Ok(data)
    }

    async fn read_range(&self, path: &str, offset: u64, length: usize) -> StorageResult<Vec<u8>> {
        self.simulate_latency(self.config.ebs_read_latency).await;

        let full_path = self.full_path(path);
        let mut file = File::open(&full_path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0u8; length];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    async fn exists(&self, path: &str) -> StorageResult<bool> {
        // Yield to runtime for cooperative scheduling
        tokio::task::yield_now().await;
        Ok(self.full_path(path).exists())
    }

    async fn size(&self, path: &str) -> StorageResult<u64> {
        let full_path = self.full_path(path);
        if !full_path.exists() {
            return Err(StorageError::NotFound {
                key: path.to_string(),
            });
        }
        let metadata = fs::metadata(&full_path)?;
        Ok(metadata.len())
    }

    async fn sync(&self, path: &str) -> StorageResult<()> {
        // Simulate fsync latency - THIS IS THE CRITICAL DURABILITY OPERATION
        self.simulate_latency(self.config.ebs_fsync_latency).await;

        let full_path = self.full_path(path);
        if full_path.exists() {
            let file = File::open(&full_path)?;
            file.sync_all()?;
        }

        Ok(())
    }

    async fn delete(&self, path: &str) -> StorageResult<()> {
        let full_path = self.full_path(path);
        if full_path.exists() {
            fs::remove_file(&full_path)?;
        }
        Ok(())
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>> {
        let full_path = self.full_path(prefix);
        let mut results = Vec::new();

        if full_path.is_dir() {
            for entry in fs::read_dir(&full_path)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().to_string();
                results.push(format!("{}/{}", prefix.trim_end_matches('/'), name));
            }
        }

        Ok(results)
    }

    async fn create_dir(&self, path: &str) -> StorageResult<()> {
        fs::create_dir_all(self.full_path(path))?;
        Ok(())
    }

    fn mmap(&self, path: &str) -> StorageResult<Option<memmap2::Mmap>> {
        let full_path = self.full_path(path);
        if !full_path.exists() {
            return Ok(None);
        }
        let file = File::open(&full_path)?;
        if file.metadata()?.len() == 0 {
            return Ok(None);
        }
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(Some(mmap))
    }

    fn root_path(&self) -> &Path {
        &self.root
    }
}

/// Mock S3 (object storage)
pub struct MockObjectStorage {
    root: PathBuf,
    config: MockStorageConfig,
    // Track write timestamps for eventual consistency simulation
    _write_times: RwLock<HashMap<String, std::time::Instant>>,
}

impl MockObjectStorage {
    pub fn new(root: impl Into<PathBuf>, config: MockStorageConfig) -> std::io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;

        Ok(Self {
            root,
            config,
            _write_times: RwLock::new(HashMap::new()),
        })
    }

    pub fn temp(config: MockStorageConfig) -> std::io::Result<Self> {
        let temp_dir = tempfile::tempdir()?.into_path();
        Self::new(temp_dir, config)
    }

    fn full_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    async fn simulate_latency(&self, base: Duration) {
        if base.is_zero() {
            return;
        }

        let variance = self.config.latency_variance;
        let jitter = if variance > 0.0 {
            let factor = 1.0 + (rand::random::<f64>() * 2.0 - 1.0) * variance;
            base.mul_f64(factor)
        } else {
            base
        };

        sleep(jitter).await;
    }
}

#[async_trait]
impl ObjectStorage for MockObjectStorage {
    async fn put(&self, key: &str, data: Bytes) -> StorageResult<()> {
        self.simulate_latency(self.config.s3_put_latency).await;

        let full_path = self.full_path(key);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&full_path, &data)?;

        self._write_times
            .write()
            .insert(key.to_string(), std::time::Instant::now());

        Ok(())
    }

    async fn get(&self, key: &str) -> StorageResult<Bytes> {
        self.simulate_latency(self.config.s3_get_latency).await;

        let full_path = self.full_path(key);
        if !full_path.exists() {
            return Err(StorageError::NotFound {
                key: key.to_string(),
            });
        }

        let data = Bytes::from(fs::read(&full_path)?);
        Ok(data)
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        Ok(self.full_path(key).exists())
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let full_path = self.full_path(key);
        if full_path.exists() {
            fs::remove_file(&full_path)?;
        }
        self._write_times.write().remove(key);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<ObjectInfo>> {
        let mut objects = Vec::new();

        fn visit_dir(
            base: &Path,
            current: &Path,
            prefix: &str,
            objects: &mut Vec<ObjectInfo>,
        ) -> std::io::Result<()> {
            if current.is_dir() {
                for entry in fs::read_dir(current)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        visit_dir(base, &path, prefix, objects)?;
                    } else {
                        let key = path
                            .strip_prefix(base)
                            .unwrap()
                            .to_string_lossy()
                            .to_string();

                        if key.starts_with(prefix) {
                            let metadata = fs::metadata(&path)?;
                            objects.push(ObjectInfo {
                                key,
                                size: metadata.len(),
                                last_modified: metadata
                                    .modified()
                                    .map(chrono::DateTime::from)
                                    .unwrap_or_else(|_| chrono::Utc::now()),
                            });
                        }
                    }
                }
            }
            Ok(())
        }

        visit_dir(&self.root, &self.root, prefix, &mut objects)?;
        objects.sort_by(|a, b| a.key.cmp(&b.key));

        Ok(objects)
    }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/// Create mock storage backend for testing/development
pub fn create_mock_storage(
    ebs_root: impl Into<PathBuf>,
    s3_root: impl Into<PathBuf>,
    config: MockStorageConfig,
) -> std::io::Result<StorageBackend> {
    let block = MockBlockStorage::new(ebs_root, config.clone())?;
    let object = MockObjectStorage::new(s3_root, config)?;

    Ok(StorageBackend::new(block, object))
}

/// Create mock storage with temp directories (for tests)
pub fn create_temp_storage(config: MockStorageConfig) -> std::io::Result<StorageBackend> {
    let block = MockBlockStorage::temp(config.clone())?;
    let object = MockObjectStorage::temp(config)?;

    Ok(StorageBackend::new(block, object))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_block_storage() {
        let storage = MockBlockStorage::temp(MockStorageConfig::fast()).unwrap();

        // Write and read
        storage.write("test.txt", b"hello").await.unwrap();
        storage.sync("test.txt").await.unwrap();

        let data = storage.read("test.txt").await.unwrap();
        assert_eq!(data, b"hello");
    }

    #[tokio::test]
    async fn test_mock_block_storage_append() {
        let storage = MockBlockStorage::temp(MockStorageConfig::fast()).unwrap();

        // Append multiple times
        let offset1 = storage.append("test.txt", b"hello").await.unwrap();
        assert_eq!(offset1, 0);

        let offset2 = storage.append("test.txt", b"world").await.unwrap();
        assert_eq!(offset2, 5);

        storage.sync("test.txt").await.unwrap();

        let data = storage.read("test.txt").await.unwrap();
        assert_eq!(data, b"helloworld");
    }

    #[tokio::test]
    async fn test_mock_object_storage() {
        let storage = MockObjectStorage::temp(MockStorageConfig::fast()).unwrap();

        // Put and get
        storage
            .put("test/object.bin", Bytes::from("data"))
            .await
            .unwrap();

        let data = storage.get("test/object.bin").await.unwrap();
        assert_eq!(&data[..], b"data");
    }

    #[tokio::test]
    async fn test_mmap() {
        let storage = MockBlockStorage::temp(MockStorageConfig::fast()).unwrap();

        storage.write("test.bin", b"mmap test data").await.unwrap();
        storage.sync("test.bin").await.unwrap();

        let mmap = storage.mmap("test.bin").unwrap();
        assert!(mmap.is_some());
        assert_eq!(&mmap.unwrap()[..], b"mmap test data");
    }
}
