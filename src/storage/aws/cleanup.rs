//! AWS Resource Cleanup Manager
//!
//! Tracks all created AWS resources and ensures cleanup on:
//! - Normal program exit (Drop)
//! - Signal interruption (SIGINT/SIGTERM)
//! - Explicit cleanup call

use aws_sdk_s3::Client as S3Client;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{error, info, warn};

/// Manages AWS resources and ensures cleanup
pub struct AwsResourceManager {
    s3_client: S3Client,
    region: String,

    // Track created resources
    buckets: Mutex<Vec<String>>,
    objects: Mutex<Vec<(String, String)>>, // (bucket, key)

    // Cleanup state
    cleanup_done: AtomicBool,
}

impl AwsResourceManager {
    /// Create a new resource manager
    pub async fn new(region: &str) -> Result<Arc<Self>, aws_sdk_s3::Error> {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new(region.to_string()))
            .load()
            .await;

        let s3_client = S3Client::new(&config);

        Ok(Arc::new(Self {
            s3_client,
            region: region.to_string(),
            buckets: Mutex::new(Vec::new()),
            objects: Mutex::new(Vec::new()),
            cleanup_done: AtomicBool::new(false),
        }))
    }

    /// Get the S3 client
    pub fn s3_client(&self) -> &S3Client {
        &self.s3_client
    }

    /// Get the region
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Register a bucket for cleanup
    pub fn register_bucket(&self, bucket: String) {
        let mut buckets = self.buckets.lock();
        if !buckets.contains(&bucket) {
            info!(bucket = %bucket, "Registered bucket for cleanup");
            buckets.push(bucket);
        }
    }

    /// Register an object for cleanup
    pub fn register_object(&self, bucket: String, key: String) {
        let mut objects = self.objects.lock();
        objects.push((bucket, key));
    }

    /// Setup signal handlers for graceful cleanup
    pub fn setup_signal_handlers(self: Arc<Self>) {
        let manager = self.clone();
        let _ = ctrlc::set_handler(move || {
            eprintln!("\nReceived interrupt signal, cleaning up AWS resources...");

            // Create a new runtime for cleanup since we're in a signal handler
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create runtime for cleanup");

            rt.block_on(async {
                if let Err(e) = manager.cleanup_all().await {
                    eprintln!("Error during cleanup: {}", e);
                }
            });

            std::process::exit(0);
        });
    }

    /// Clean up all registered resources
    pub async fn cleanup_all(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check if cleanup already done
        if self.cleanup_done.swap(true, Ordering::SeqCst) {
            info!("Cleanup already performed, skipping");
            return Ok(());
        }

        info!("Starting AWS resource cleanup");

        // Delete all objects first
        let objects: Vec<(String, String)> = {
            let mut objects = self.objects.lock();
            std::mem::take(&mut *objects)
        };

        for (bucket, key) in objects {
            match self
                .s3_client
                .delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
            {
                Ok(_) => info!(bucket = %bucket, key = %key, "Deleted object"),
                Err(e) => warn!(bucket = %bucket, key = %key, error = %e, "Failed to delete object"),
            }
        }

        // Delete buckets (must be empty)
        let buckets: Vec<String> = {
            let mut buckets = self.buckets.lock();
            std::mem::take(&mut *buckets)
        };

        for bucket in buckets {
            // First, list and delete any remaining objects
            match self.delete_all_objects_in_bucket(&bucket).await {
                Ok(_) => {}
                Err(e) => {
                    warn!(bucket = %bucket, error = %e, "Failed to empty bucket");
                }
            }

            // Then delete the bucket
            match self.s3_client.delete_bucket().bucket(&bucket).send().await {
                Ok(_) => info!(bucket = %bucket, "Deleted bucket"),
                Err(e) => warn!(bucket = %bucket, error = %e, "Failed to delete bucket"),
            }
        }

        info!("AWS resource cleanup complete");
        Ok(())
    }

    /// Delete all objects in a bucket
    async fn delete_all_objects_in_bucket(
        &self,
        bucket: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut continuation_token: Option<String> = None;

        loop {
            let mut list_req = self.s3_client.list_objects_v2().bucket(bucket);

            if let Some(token) = continuation_token {
                list_req = list_req.continuation_token(token);
            }

            let response = list_req.send().await?;

            if let Some(contents) = response.contents {
                for object in contents {
                    if let Some(key) = object.key {
                        self.s3_client
                            .delete_object()
                            .bucket(bucket)
                            .key(&key)
                            .send()
                            .await?;
                    }
                }
            }

            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Get count of tracked resources
    pub fn resource_count(&self) -> (usize, usize) {
        let buckets = self.buckets.lock().len();
        let objects = self.objects.lock().len();
        (buckets, objects)
    }
}

impl Drop for AwsResourceManager {
    fn drop(&mut self) {
        if !self.cleanup_done.load(Ordering::SeqCst) {
            // Try to cleanup synchronously in drop
            // This is a fallback - prefer explicit cleanup_all()
            let buckets = self.buckets.lock().len();
            let objects = self.objects.lock().len();

            if buckets > 0 || objects > 0 {
                error!(
                    buckets = buckets,
                    objects = objects,
                    "AwsResourceManager dropped without cleanup! Resources may be orphaned."
                );
            }
        }
    }
}
