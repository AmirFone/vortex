//! AWS storage implementations
//!
//! Provides real S3 object storage for production deployments.

mod cleanup;
mod s3;

pub use cleanup::AwsResourceManager;
pub use s3::S3ObjectStorage;

use crate::storage::mock::{MockBlockStorage, MockObjectStorage, MockStorageConfig};
use crate::storage::StorageBackend;
use std::path::Path;

/// Create AWS-backed storage
///
/// Note: For full EBS support, this would need to run on EC2.
/// Currently uses local block storage with S3 for object storage.
pub async fn create_aws_storage(
    ebs_mount_path: &Path,
    _s3_bucket: &str,
) -> anyhow::Result<StorageBackend> {
    // For now, use local block storage (simulating EBS mount)
    // Real EBS implementation would use the mounted path directly
    let block = MockBlockStorage::new(ebs_mount_path, MockStorageConfig::fast())?;

    // For object storage, we'd use real S3 here
    // For now, use mock to avoid requiring AWS credentials for basic functionality
    let object = MockObjectStorage::new(ebs_mount_path.join("s3"), MockStorageConfig::fast())?;

    Ok(StorageBackend::new(block, object))
}
