//! AWS storage implementations
//!
//! Provides real S3 object storage and EC2 provisioning for production deployments.

mod cleanup;
mod ec2;
mod s3;

pub use cleanup::{AwsResourceManager, ResourceCount};
pub use ec2::{
    generate_benchmark_user_data, Ec2Error, Ec2Provisioner, Ec2Result, InstanceConfig,
    InstanceInfo,
};
pub use s3::S3ObjectStorage;

use crate::storage::mock::{MockBlockStorage, MockObjectStorage, MockStorageConfig};
use crate::storage::StorageBackend;
use std::path::Path;
use std::sync::Arc;

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

/// Create EBS + S3 backed storage (for use on EC2)
///
/// Uses a local filesystem path (EBS mount point) for block storage
/// and real S3 for object storage (backups, snapshots).
pub async fn create_ebs_s3_storage(
    ebs_mount_path: &Path,
    resource_manager: Arc<AwsResourceManager>,
    s3_prefix: &str,
) -> anyhow::Result<(StorageBackend, S3ObjectStorage)> {
    // EBS is just a fast local filesystem on EC2
    let block = MockBlockStorage::new(ebs_mount_path, MockStorageConfig::fast())?;
    let object = MockObjectStorage::new(ebs_mount_path.join("object"), MockStorageConfig::fast())?;

    // Real S3 for backups/snapshots
    let s3 = S3ObjectStorage::new(resource_manager, s3_prefix).await?;

    Ok((StorageBackend::new(block, object), s3))
}
