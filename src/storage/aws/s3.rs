//! S3 Object Storage Implementation
//!
//! Implements the ObjectStorage trait for real AWS S3.

use super::AwsResourceManager;
use crate::storage::{ObjectInfo, ObjectStorage, StorageError, StorageResult};
use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{BucketLocationConstraint, CreateBucketConfiguration};
use bytes::Bytes;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

/// S3-backed object storage
pub struct S3ObjectStorage {
    resource_manager: Arc<AwsResourceManager>,
    bucket: String,
    prefix: String,
    owns_bucket: bool,
}

impl S3ObjectStorage {
    /// Create a new S3 object storage with an auto-generated bucket name
    pub async fn new(
        resource_manager: Arc<AwsResourceManager>,
        prefix: &str,
    ) -> StorageResult<Self> {
        let bucket = format!("vortex-sim-{}", Uuid::new_v4().to_string()[..8].to_lowercase());
        Self::with_bucket(resource_manager, &bucket, prefix, true).await
    }

    /// Create S3 object storage with a specific bucket name
    pub async fn with_bucket(
        resource_manager: Arc<AwsResourceManager>,
        bucket: &str,
        prefix: &str,
        create_bucket: bool,
    ) -> StorageResult<Self> {
        let client = resource_manager.s3_client();
        let region = resource_manager.region();

        let owns_bucket = if create_bucket {
            // Check if bucket exists
            let exists = client
                .head_bucket()
                .bucket(bucket)
                .send()
                .await
                .is_ok();

            if !exists {
                // Create the bucket
                // Note: us-east-1 is special - don't specify LocationConstraint for it
                let request = client.create_bucket().bucket(bucket);

                let request = if region == "us-east-1" {
                    request
                } else {
                    let constraint = BucketLocationConstraint::from(region);
                    let cfg = CreateBucketConfiguration::builder()
                        .location_constraint(constraint)
                        .build();
                    request.create_bucket_configuration(cfg)
                };

                request
                    .send()
                    .await
                    .map_err(|e| StorageError::Backend(format!("Failed to create bucket: {}", e)))?;

                info!(bucket = %bucket, region = %region, "Created S3 bucket");
                resource_manager.register_bucket(bucket.to_string());
                true
            } else {
                info!(bucket = %bucket, "Using existing S3 bucket");
                false
            }
        } else {
            false
        };

        Ok(Self {
            resource_manager,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            owns_bucket,
        })
    }

    /// Get the full S3 key for a given path
    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// Get bucket name
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Get prefix
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

#[async_trait]
impl ObjectStorage for S3ObjectStorage {
    async fn put(&self, key: &str, data: Bytes) -> StorageResult<()> {
        let full_key = self.full_key(key);
        let client = self.resource_manager.s3_client();

        client
            .put_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .body(ByteStream::from(data.to_vec()))
            .send()
            .await
            .map_err(|e| StorageError::Backend(format!("S3 put failed: {}", e)))?;

        // Register for cleanup
        self.resource_manager
            .register_object(self.bucket.clone(), full_key.clone());

        debug!(bucket = %self.bucket, key = %full_key, "Put object to S3");
        Ok(())
    }

    async fn get(&self, key: &str) -> StorageResult<Bytes> {
        let full_key = self.full_key(key);
        let client = self.resource_manager.s3_client();

        let response = client
            .get_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| {
                if e.to_string().contains("NoSuchKey") {
                    StorageError::NotFound {
                        key: full_key.clone(),
                    }
                } else {
                    StorageError::Backend(format!("S3 get failed: {}", e))
                }
            })?;

        let data = response
            .body
            .collect()
            .await
            .map_err(|e| StorageError::Backend(format!("Failed to read S3 body: {}", e)))?;

        debug!(bucket = %self.bucket, key = %full_key, "Got object from S3");
        Ok(Bytes::from(data.into_bytes().to_vec()))
    }

    async fn exists(&self, key: &str) -> StorageResult<bool> {
        let full_key = self.full_key(key);
        let client = self.resource_manager.s3_client();

        let result = client
            .head_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await;

        Ok(result.is_ok())
    }

    async fn delete(&self, key: &str) -> StorageResult<()> {
        let full_key = self.full_key(key);
        let client = self.resource_manager.s3_client();

        client
            .delete_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| StorageError::Backend(format!("S3 delete failed: {}", e)))?;

        debug!(bucket = %self.bucket, key = %full_key, "Deleted object from S3");
        Ok(())
    }

    async fn list(&self, prefix: &str) -> StorageResult<Vec<ObjectInfo>> {
        let full_prefix = self.full_key(prefix);
        let client = self.resource_manager.s3_client();

        let mut objects = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&full_prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .map_err(|e| StorageError::Backend(format!("S3 list failed: {}", e)))?;

            if let Some(contents) = response.contents {
                for object in contents {
                    if let (Some(key), Some(size)) = (object.key, object.size) {
                        // Strip the prefix to return relative keys
                        let relative_key = if !self.prefix.is_empty() && key.starts_with(&self.prefix) {
                            key[self.prefix.len()..].trim_start_matches('/').to_string()
                        } else {
                            key
                        };

                        objects.push(ObjectInfo {
                            key: relative_key,
                            size: size as u64,
                            last_modified: object
                                .last_modified
                                .map(|dt| {
                                    chrono::DateTime::from_timestamp(dt.secs(), dt.subsec_nanos())
                                        .unwrap_or_else(chrono::Utc::now)
                                })
                                .unwrap_or_else(chrono::Utc::now),
                        });
                    }
                }
            }

            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }

        debug!(bucket = %self.bucket, prefix = %full_prefix, count = objects.len(), "Listed objects from S3");
        Ok(objects)
    }
}

impl Drop for S3ObjectStorage {
    fn drop(&mut self) {
        if self.owns_bucket {
            debug!(bucket = %self.bucket, "S3ObjectStorage dropped (bucket cleanup handled by ResourceManager)");
        }
    }
}
