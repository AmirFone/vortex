//! AWS Resource Cleanup Manager
//!
//! Tracks all created AWS resources and ensures cleanup on:
//! - Normal program exit (Drop)
//! - Signal interruption (SIGINT/SIGTERM)
//! - Explicit cleanup call

use aws_sdk_ec2::Client as Ec2Client;
use aws_sdk_s3::Client as S3Client;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{error, info, warn};

/// Manages AWS resources and ensures cleanup
pub struct AwsResourceManager {
    s3_client: S3Client,
    ec2_client: Ec2Client,
    region: String,

    // Track created S3 resources
    buckets: Mutex<Vec<String>>,
    objects: Mutex<Vec<(String, String)>>, // (bucket, key)

    // Track created EC2 resources
    instances: Mutex<Vec<String>>,
    key_pairs: Mutex<Vec<String>>,
    security_groups: Mutex<Vec<String>>,

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
        let ec2_client = Ec2Client::new(&config);

        Ok(Arc::new(Self {
            s3_client,
            ec2_client,
            region: region.to_string(),
            buckets: Mutex::new(Vec::new()),
            objects: Mutex::new(Vec::new()),
            instances: Mutex::new(Vec::new()),
            key_pairs: Mutex::new(Vec::new()),
            security_groups: Mutex::new(Vec::new()),
            cleanup_done: AtomicBool::new(false),
        }))
    }

    /// Get the S3 client
    pub fn s3_client(&self) -> &S3Client {
        &self.s3_client
    }

    /// Get the EC2 client
    pub fn ec2_client(&self) -> &Ec2Client {
        &self.ec2_client
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

    /// Register an EC2 instance for cleanup
    pub fn register_instance(&self, instance_id: String) {
        let mut instances = self.instances.lock();
        if !instances.contains(&instance_id) {
            info!(instance_id = %instance_id, "Registered instance for cleanup");
            instances.push(instance_id);
        }
    }

    /// Register a key pair for cleanup
    pub fn register_key_pair(&self, key_name: String) {
        let mut key_pairs = self.key_pairs.lock();
        if !key_pairs.contains(&key_name) {
            info!(key_name = %key_name, "Registered key pair for cleanup");
            key_pairs.push(key_name);
        }
    }

    /// Register a security group for cleanup
    pub fn register_security_group(&self, sg_id: String) {
        let mut security_groups = self.security_groups.lock();
        if !security_groups.contains(&sg_id) {
            info!(sg_id = %sg_id, "Registered security group for cleanup");
            security_groups.push(sg_id);
        }
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

        // ========== EC2 CLEANUP (instances first) ==========

        // Terminate EC2 instances
        let instances: Vec<String> = {
            let mut instances = self.instances.lock();
            std::mem::take(&mut *instances)
        };

        if !instances.is_empty() {
            info!(count = instances.len(), "Terminating EC2 instances");
            match self
                .ec2_client
                .terminate_instances()
                .set_instance_ids(Some(instances.clone()))
                .send()
                .await
            {
                Ok(_) => {
                    info!("Terminated instances: {:?}", instances);
                    // Wait for instances to terminate
                    self.wait_for_instances_terminated(&instances).await;
                }
                Err(e) => warn!(error = %e, "Failed to terminate instances"),
            }
        }

        // Delete key pairs
        let key_pairs: Vec<String> = {
            let mut key_pairs = self.key_pairs.lock();
            std::mem::take(&mut *key_pairs)
        };

        for key_name in key_pairs {
            match self
                .ec2_client
                .delete_key_pair()
                .key_name(&key_name)
                .send()
                .await
            {
                Ok(_) => info!(key_name = %key_name, "Deleted key pair"),
                Err(e) => warn!(key_name = %key_name, error = %e, "Failed to delete key pair"),
            }
        }

        // Delete security groups (must wait for instances to terminate first)
        let security_groups: Vec<String> = {
            let mut security_groups = self.security_groups.lock();
            std::mem::take(&mut *security_groups)
        };

        for sg_id in security_groups {
            // Retry a few times since SG deletion can fail if instance is still terminating
            for attempt in 0..5 {
                match self
                    .ec2_client
                    .delete_security_group()
                    .group_id(&sg_id)
                    .send()
                    .await
                {
                    Ok(_) => {
                        info!(sg_id = %sg_id, "Deleted security group");
                        break;
                    }
                    Err(e) => {
                        if attempt < 4 {
                            warn!(sg_id = %sg_id, attempt = attempt, "Retrying security group deletion");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        } else {
                            warn!(sg_id = %sg_id, error = %e, "Failed to delete security group");
                        }
                    }
                }
            }
        }

        // ========== S3 CLEANUP ==========

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

    /// Wait for instances to reach terminated state
    async fn wait_for_instances_terminated(&self, instance_ids: &[String]) {
        info!("Waiting for instances to terminate...");
        for _ in 0..60 {
            // Max 5 minutes
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            match self
                .ec2_client
                .describe_instances()
                .set_instance_ids(Some(instance_ids.to_vec()))
                .send()
                .await
            {
                Ok(response) => {
                    let all_terminated = response
                        .reservations()
                        .iter()
                        .flat_map(|r| r.instances())
                        .all(|i| {
                            i.state()
                                .map(|s| s.name())
                                .flatten()
                                .map(|n| n.as_str() == "terminated")
                                .unwrap_or(false)
                        });

                    if all_terminated {
                        info!("All instances terminated");
                        return;
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Error checking instance state");
                }
            }
        }
        warn!("Timeout waiting for instances to terminate");
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

    /// Get count of tracked resources (S3: buckets, objects; EC2: instances, key_pairs, security_groups)
    pub fn resource_count(&self) -> ResourceCount {
        ResourceCount {
            buckets: self.buckets.lock().len(),
            objects: self.objects.lock().len(),
            instances: self.instances.lock().len(),
            key_pairs: self.key_pairs.lock().len(),
            security_groups: self.security_groups.lock().len(),
        }
    }
}

/// Resource counts for tracking
#[derive(Debug, Clone)]
pub struct ResourceCount {
    pub buckets: usize,
    pub objects: usize,
    pub instances: usize,
    pub key_pairs: usize,
    pub security_groups: usize,
}

impl ResourceCount {
    pub fn total(&self) -> usize {
        self.buckets + self.objects + self.instances + self.key_pairs + self.security_groups
    }
}

impl Drop for AwsResourceManager {
    fn drop(&mut self) {
        if !self.cleanup_done.load(Ordering::SeqCst) {
            // Try to cleanup synchronously in drop
            // This is a fallback - prefer explicit cleanup_all()
            let count = ResourceCount {
                buckets: self.buckets.lock().len(),
                objects: self.objects.lock().len(),
                instances: self.instances.lock().len(),
                key_pairs: self.key_pairs.lock().len(),
                security_groups: self.security_groups.lock().len(),
            };

            if count.total() > 0 {
                error!(
                    buckets = count.buckets,
                    objects = count.objects,
                    instances = count.instances,
                    key_pairs = count.key_pairs,
                    security_groups = count.security_groups,
                    "AwsResourceManager dropped without cleanup! Resources may be orphaned."
                );
            }
        }
    }
}
