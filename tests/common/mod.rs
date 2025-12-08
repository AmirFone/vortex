//! Shared test utilities for VectorDB testing
//!
//! This module provides:
//! - Random vector generation with normalization
//! - Failing storage wrapper for error injection
//! - Test timing utilities
//! - Common assertions

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use rand::Rng;
use tempfile::TempDir;

use vortex::storage::mock::{MockBlockStorage, MockStorageConfig};
use vortex::storage::{BlockStorage, StorageError, StorageResult};
use vortex::index::AnnIndexConfig;

/// Create a default HNSW index config for testing
pub fn test_index_config() -> AnnIndexConfig {
    AnnIndexConfig::default()
}

/// Helper to create io::Error for injection
fn io_error(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg)
}

/// Generate a random normalized vector
pub fn random_vector(dims: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize(&v)
}

/// Generate a deterministic vector based on seed
pub fn seeded_vector(dims: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize(&v)
}

/// Normalize a vector to unit length
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Cosine similarity (dot product for normalized vectors)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Create a temporary storage for testing
pub fn temp_storage() -> (TempDir, Arc<MockBlockStorage>) {
    let temp_dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(
        MockBlockStorage::new(temp_dir.path(), MockStorageConfig::fast()).unwrap(),
    );
    (temp_dir, storage)
}

/// Create a temporary storage with path for reopening
pub fn temp_storage_with_path() -> (TempDir, PathBuf, Arc<MockBlockStorage>) {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().to_path_buf();
    let storage = Arc::new(
        MockBlockStorage::new(&path, MockStorageConfig::fast()).unwrap(),
    );
    (temp_dir, path, storage)
}

/// Reopen storage at the same path
pub fn reopen_storage(path: &PathBuf) -> Arc<MockBlockStorage> {
    Arc::new(MockBlockStorage::new(path, MockStorageConfig::fast()).unwrap())
}

/// Failure injection mode for FailingStorage
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FailureMode {
    /// No failures
    None,
    /// Fail all operations
    FailAll,
    /// Fail after N operations
    FailAfterN(u64),
    /// Fail only write operations
    FailWrites,
    /// Fail only sync operations
    FailSync,
    /// Fail only append operations
    FailAppend,
    /// Simulate partial write (truncate data)
    PartialWrite(usize),
}

/// Storage wrapper that can inject failures for testing
pub struct FailingStorage {
    inner: Arc<dyn BlockStorage>,
    failure_mode: Arc<std::sync::RwLock<FailureMode>>,
    operation_count: AtomicU64,
    enabled: AtomicBool,
}

impl FailingStorage {
    pub fn new(inner: Arc<dyn BlockStorage>) -> Self {
        Self {
            inner,
            failure_mode: Arc::new(std::sync::RwLock::new(FailureMode::None)),
            operation_count: AtomicU64::new(0),
            enabled: AtomicBool::new(false),
        }
    }

    /// Set the failure mode
    pub fn set_failure_mode(&self, mode: FailureMode) {
        *self.failure_mode.write().unwrap() = mode;
        self.enabled.store(mode != FailureMode::None, Ordering::SeqCst);
        self.operation_count.store(0, Ordering::SeqCst);
    }

    /// Disable all failures
    pub fn disable_failures(&self) {
        self.set_failure_mode(FailureMode::None);
    }

    /// Check if should fail this operation
    fn should_fail(&self, op_type: &str) -> bool {
        if !self.enabled.load(Ordering::SeqCst) {
            return false;
        }

        let count = self.operation_count.fetch_add(1, Ordering::SeqCst);
        let mode = *self.failure_mode.read().unwrap();

        match mode {
            FailureMode::None => false,
            FailureMode::FailAll => true,
            FailureMode::FailAfterN(n) => count >= n,
            FailureMode::FailWrites => op_type == "write",
            FailureMode::FailSync => op_type == "sync",
            FailureMode::FailAppend => op_type == "append",
            FailureMode::PartialWrite(_) => false, // Handled specially
        }
    }

    /// Get current operation count
    pub fn operation_count(&self) -> u64 {
        self.operation_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl BlockStorage for FailingStorage {
    async fn write(&self, path: &str, data: &[u8]) -> StorageResult<()> {
        if self.should_fail("write") {
            return Err(StorageError::Io(io_error("Injected write failure")));
        }

        // Handle partial write
        let mode = *self.failure_mode.read().unwrap();
        if let FailureMode::PartialWrite(max_bytes) = mode {
            if data.len() > max_bytes {
                let truncated = &data[..max_bytes];
                return self.inner.write(path, truncated).await;
            }
        }

        self.inner.write(path, data).await
    }

    async fn write_at(&self, path: &str, offset: usize, data: &[u8]) -> StorageResult<()> {
        if self.should_fail("write") {
            return Err(StorageError::Io(io_error("Injected write_at failure")));
        }

        // Handle partial write
        let mode = *self.failure_mode.read().unwrap();
        if let FailureMode::PartialWrite(max_bytes) = mode {
            if data.len() > max_bytes {
                let truncated = &data[..max_bytes];
                return self.inner.write_at(path, offset, truncated).await;
            }
        }

        self.inner.write_at(path, offset, data).await
    }

    async fn append(&self, path: &str, data: &[u8]) -> StorageResult<u64> {
        if self.should_fail("append") {
            return Err(StorageError::Io(io_error("Injected append failure")));
        }

        // Handle partial write
        let mode = *self.failure_mode.read().unwrap();
        if let FailureMode::PartialWrite(max_bytes) = mode {
            if data.len() > max_bytes {
                let truncated = &data[..max_bytes];
                return self.inner.append(path, truncated).await;
            }
        }

        self.inner.append(path, data).await
    }

    async fn read(&self, path: &str) -> StorageResult<Vec<u8>> {
        if self.should_fail("read") {
            return Err(StorageError::Io(io_error("Injected read failure")));
        }
        self.inner.read(path).await
    }

    async fn read_range(&self, path: &str, offset: u64, len: usize) -> StorageResult<Vec<u8>> {
        if self.should_fail("read_range") {
            return Err(StorageError::Io(io_error("Injected read_range failure")));
        }
        self.inner.read_range(path, offset, len).await
    }

    async fn sync(&self, path: &str) -> StorageResult<()> {
        if self.should_fail("sync") {
            return Err(StorageError::Io(io_error("Injected sync failure")));
        }
        self.inner.sync(path).await
    }

    async fn size(&self, path: &str) -> StorageResult<u64> {
        self.inner.size(path).await
    }

    async fn exists(&self, path: &str) -> StorageResult<bool> {
        self.inner.exists(path).await
    }

    async fn delete(&self, path: &str) -> StorageResult<()> {
        self.inner.delete(path).await
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn create_dir(&self, path: &str) -> StorageResult<()> {
        self.inner.create_dir(path).await
    }

    fn mmap(&self, path: &str) -> StorageResult<Option<memmap2::Mmap>> {
        self.inner.mmap(path)
    }

    fn root_path(&self) -> &Path {
        self.inner.root_path()
    }
}

/// Timer for measuring operation latencies
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn elapsed_us(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000_000.0
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile from sorted values
pub fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_values.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

// ============================================================================
// BENCHMARKING UTILITIES
// ============================================================================

use std::time::Duration;

/// Latency histogram with percentile breakdown
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    pub min: Duration,
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub count: usize,
}

impl LatencyHistogram {
    /// Create histogram from a list of latencies
    pub fn from_latencies(mut latencies: Vec<Duration>) -> Self {
        if latencies.is_empty() {
            return Self {
                min: Duration::ZERO,
                p50: Duration::ZERO,
                p90: Duration::ZERO,
                p95: Duration::ZERO,
                p99: Duration::ZERO,
                p999: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
                count: 0,
            };
        }

        latencies.sort();
        let count = latencies.len();
        let total: Duration = latencies.iter().sum();
        let mean = total / count as u32;

        Self {
            min: latencies[0],
            p50: latencies[count * 50 / 100],
            p90: latencies[count * 90 / 100],
            p95: latencies[count * 95 / 100],
            p99: latencies[count * 99 / 100],
            p999: latencies[(count * 999 / 1000).min(count - 1)],
            max: latencies[count - 1],
            mean,
            count,
        }
    }

    /// Print a summary of the histogram
    pub fn print_summary(&self, label: &str) {
        println!("=== {} Latency Summary ({} samples) ===", label, self.count);
        println!("  min:  {:?}", self.min);
        println!("  p50:  {:?}", self.p50);
        println!("  p90:  {:?}", self.p90);
        println!("  p95:  {:?}", self.p95);
        println!("  p99:  {:?}", self.p99);
        println!("  p999: {:?}", self.p999);
        println!("  max:  {:?}", self.max);
        println!("  mean: {:?}", self.mean);
    }
}

/// Throughput measurement result
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    pub ops_per_second: f64,
    pub vectors_per_second: f64,
    pub bytes_per_second: f64,
    pub total_ops: usize,
    pub total_duration: Duration,
}

impl ThroughputResult {
    pub fn new(ops: usize, vectors: usize, bytes: usize, duration: Duration) -> Self {
        let secs = duration.as_secs_f64();
        Self {
            ops_per_second: ops as f64 / secs,
            vectors_per_second: vectors as f64 / secs,
            bytes_per_second: bytes as f64 / secs,
            total_ops: ops,
            total_duration: duration,
        }
    }

    pub fn print_summary(&self, label: &str) {
        println!("=== {} Throughput Summary ===", label);
        println!("  Operations/sec: {:.2}", self.ops_per_second);
        println!("  Vectors/sec:    {:.2}", self.vectors_per_second);
        println!("  Bytes/sec:      {:.2} ({:.2} MB/s)",
            self.bytes_per_second,
            self.bytes_per_second / 1_000_000.0
        );
        println!("  Total ops:      {}", self.total_ops);
        println!("  Total time:     {:?}", self.total_duration);
    }
}

/// Measure latency of a single operation
pub fn measure_latency<F, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    (start.elapsed(), result)
}

/// Measure latency of an async operation
pub async fn measure_latency_async<F, Fut, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    (start.elapsed(), result)
}

/// Collect latencies from multiple iterations
pub async fn collect_latencies_async<F, Fut, R>(iterations: usize, mut f: F) -> Vec<Duration>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let mut latencies = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let (duration, _) = measure_latency_async(|| f(i)).await;
        latencies.push(duration);
    }
    latencies
}

/// Assert latency is under a threshold
pub fn assert_latency_under(actual: Duration, limit: Duration, label: &str) {
    assert!(
        actual <= limit,
        "{}: latency {:?} exceeds limit {:?}",
        label,
        actual,
        limit
    );
}

/// Assert throughput meets minimum requirement
pub fn assert_throughput_above(result: &ThroughputResult, min_ops_per_sec: f64, label: &str) {
    assert!(
        result.ops_per_second >= min_ops_per_sec,
        "{}: throughput {:.2} ops/s below minimum {:.2} ops/s",
        label,
        result.ops_per_second,
        min_ops_per_sec
    );
}

/// Generate batch of random vectors for testing
pub fn generate_random_vectors(count: usize, dims: usize) -> Vec<(u64, Vec<f32>)> {
    (0..count)
        .map(|i| (i as u64, random_vector(dims)))
        .collect()
}

/// Generate batch of seeded vectors for reproducible testing
pub fn generate_seeded_vectors(count: usize, dims: usize, base_seed: u64) -> Vec<(u64, Vec<f32>)> {
    (0..count)
        .map(|i| (i as u64, seeded_vector(dims, base_seed + i as u64)))
        .collect()
}

/// Calculate recall@k between search results and ground truth
pub fn calculate_recall(results: &[u64], ground_truth: &[u64], k: usize) -> f64 {
    let results_set: std::collections::HashSet<_> = results.iter().take(k).collect();
    let truth_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();

    let intersection = results_set.intersection(&truth_set).count();
    intersection as f64 / k.min(truth_set.len()) as f64
}

/// Brute force k-nearest neighbor search for ground truth
pub fn brute_force_knn(query: &[f32], vectors: &[(u64, Vec<f32>)], k: usize) -> Vec<u64> {
    let mut distances: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, cosine_similarity(query, v)))
        .collect();

    // Sort by similarity descending
    distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    distances.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Corrupt bytes at specified offset in a byte vector
pub fn corrupt_bytes(data: &mut [u8], offset: usize, corruption: &[u8]) {
    for (i, &byte) in corruption.iter().enumerate() {
        if offset + i < data.len() {
            data[offset + i] = byte;
        }
    }
}

/// Truncate file at specified offset
pub async fn truncate_file(storage: &dyn BlockStorage, path: &str, new_size: usize) -> StorageResult<()> {
    let data = storage.read(path).await?;
    let truncated = &data[..new_size.min(data.len())];
    storage.write(path, truncated).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_seeded_vector_deterministic() {
        let v1 = seeded_vector(10, 42);
        let v2 = seeded_vector(10, 42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_cosine_similarity() {
        let v = normalize(&[1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }
}
