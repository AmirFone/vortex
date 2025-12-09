//! EC2 Provisioning Module
//!
//! Handles creation and management of EC2 instances for cloud benchmarking.

use super::AwsResourceManager;
use aws_sdk_ec2::types::{
    BlockDeviceMapping, EbsBlockDevice, InstanceType, ResourceType, Tag, TagSpecification,
    VolumeType,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Result type for EC2 operations
pub type Ec2Result<T> = Result<T, Ec2Error>;

/// EC2 error types
#[derive(Debug, thiserror::Error)]
pub enum Ec2Error {
    #[error("AWS SDK error: {0}")]
    Sdk(String),
    #[error("Instance not found: {0}")]
    InstanceNotFound(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Configuration for launching an EC2 instance
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    pub instance_type: String,
    pub ami_id: String,
    pub key_name: String,
    pub security_group_id: String,
    pub user_data: String,
    pub ebs_size_gb: i32,
    pub instance_name: String,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self {
            instance_type: "c6i.xlarge".to_string(),
            ami_id: String::new(), // Must be set
            key_name: String::new(),
            security_group_id: String::new(),
            user_data: String::new(),
            ebs_size_gb: 50,
            instance_name: "vortex-benchmark".to_string(),
        }
    }
}

/// Information about a launched instance
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub instance_id: String,
    pub public_ip: Option<String>,
    pub private_ip: Option<String>,
    pub state: String,
}

/// EC2 Provisioner for creating and managing instances
pub struct Ec2Provisioner {
    resource_manager: Arc<AwsResourceManager>,
}

impl Ec2Provisioner {
    /// Create a new EC2 provisioner
    pub fn new(resource_manager: Arc<AwsResourceManager>) -> Self {
        Self { resource_manager }
    }

    /// Get the Amazon Linux 2023 AMI ID for the current region
    pub async fn get_amazon_linux_ami(&self) -> Ec2Result<String> {
        let client = self.resource_manager.ec2_client();

        let response = client
            .describe_images()
            .owners("amazon")
            .filters(
                aws_sdk_ec2::types::Filter::builder()
                    .name("name")
                    .values("al2023-ami-2023.*-x86_64")
                    .build(),
            )
            .filters(
                aws_sdk_ec2::types::Filter::builder()
                    .name("state")
                    .values("available")
                    .build(),
            )
            .filters(
                aws_sdk_ec2::types::Filter::builder()
                    .name("architecture")
                    .values("x86_64")
                    .build(),
            )
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(e.to_string()))?;

        // Get the most recent AMI
        let images = response.images();
        let ami = images
            .iter()
            .filter(|img| img.image_id().is_some())
            .max_by(|a, b| {
                a.creation_date()
                    .unwrap_or_default()
                    .cmp(&b.creation_date().unwrap_or_default())
            })
            .and_then(|img| img.image_id())
            .ok_or_else(|| Ec2Error::Sdk("No Amazon Linux 2023 AMI found".to_string()))?;

        info!(ami_id = %ami, "Found Amazon Linux 2023 AMI");
        Ok(ami.to_string())
    }

    /// Create an EC2 key pair for SSH access
    pub async fn create_key_pair(&self, name: &str) -> Ec2Result<String> {
        let client = self.resource_manager.ec2_client();

        // Delete existing key pair if it exists
        let _ = client.delete_key_pair().key_name(name).send().await;

        let response = client
            .create_key_pair()
            .key_name(name)
            .key_type(aws_sdk_ec2::types::KeyType::Rsa)
            .key_format(aws_sdk_ec2::types::KeyFormat::Pem)
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to create key pair: {}", e)))?;

        let private_key = response
            .key_material()
            .ok_or_else(|| Ec2Error::Sdk("No key material returned".to_string()))?
            .to_string();

        self.resource_manager.register_key_pair(name.to_string());
        info!(key_name = %name, "Created EC2 key pair");

        Ok(private_key)
    }

    /// Create a security group for the benchmark instance
    pub async fn create_security_group(&self, name: &str, description: &str) -> Ec2Result<String> {
        let client = self.resource_manager.ec2_client();

        // Delete existing security group if it exists (by name)
        let existing = client
            .describe_security_groups()
            .group_names(name)
            .send()
            .await;

        if let Ok(resp) = existing {
            for sg in resp.security_groups() {
                if let Some(sg_id) = sg.group_id() {
                    let _ = client
                        .delete_security_group()
                        .group_id(sg_id)
                        .send()
                        .await;
                }
            }
        }

        // Create new security group
        let create_response = client
            .create_security_group()
            .group_name(name)
            .description(description)
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to create security group: {}", e)))?;

        let sg_id = create_response
            .group_id()
            .ok_or_else(|| Ec2Error::Sdk("No security group ID returned".to_string()))?
            .to_string();

        // Add SSH ingress rule (port 22 from anywhere - for simplicity)
        // In production, you'd want to restrict this to specific IPs
        client
            .authorize_security_group_ingress()
            .group_id(&sg_id)
            .ip_protocol("tcp")
            .from_port(22)
            .to_port(22)
            .cidr_ip("0.0.0.0/0")
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to add SSH rule: {}", e)))?;

        self.resource_manager.register_security_group(sg_id.clone());
        info!(sg_id = %sg_id, name = %name, "Created security group with SSH access");

        Ok(sg_id)
    }

    /// Launch an EC2 instance with the specified configuration
    pub async fn launch_instance(&self, config: InstanceConfig) -> Ec2Result<InstanceInfo> {
        let client = self.resource_manager.ec2_client();

        // Parse instance type
        let instance_type = InstanceType::from(config.instance_type.as_str());

        // Create EBS block device mapping for root volume
        let ebs = EbsBlockDevice::builder()
            .volume_size(config.ebs_size_gb)
            .volume_type(VolumeType::Gp3)
            .delete_on_termination(true)
            .build();

        let block_device = BlockDeviceMapping::builder()
            .device_name("/dev/xvda")
            .ebs(ebs)
            .build();

        // Create tags for the instance
        let tags = TagSpecification::builder()
            .resource_type(ResourceType::Instance)
            .tags(
                Tag::builder()
                    .key("Name")
                    .value(&config.instance_name)
                    .build(),
            )
            .tags(
                Tag::builder()
                    .key("vortex-benchmark")
                    .value("true")
                    .build(),
            )
            .build();

        // Encode user data as base64
        let user_data_b64 = base64_encode(&config.user_data);

        // Launch instance
        let run_response = client
            .run_instances()
            .image_id(&config.ami_id)
            .instance_type(instance_type)
            .key_name(&config.key_name)
            .security_group_ids(&config.security_group_id)
            .user_data(&user_data_b64)
            .block_device_mappings(block_device)
            .tag_specifications(tags)
            .min_count(1)
            .max_count(1)
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to launch instance: {}", e)))?;

        let instance = run_response
            .instances()
            .first()
            .ok_or_else(|| Ec2Error::Sdk("No instance returned".to_string()))?;

        let instance_id = instance
            .instance_id()
            .ok_or_else(|| Ec2Error::Sdk("No instance ID returned".to_string()))?
            .to_string();

        self.resource_manager
            .register_instance(instance_id.clone());
        info!(instance_id = %instance_id, "Launched EC2 instance");

        Ok(InstanceInfo {
            instance_id,
            public_ip: None,
            private_ip: instance.private_ip_address().map(|s| s.to_string()),
            state: "pending".to_string(),
        })
    }

    /// Wait for an instance to reach the running state and get its public IP
    pub async fn wait_for_running(&self, instance_id: &str) -> Ec2Result<InstanceInfo> {
        let client = self.resource_manager.ec2_client();
        let timeout = Duration::from_secs(300); // 5 minutes
        let start = std::time::Instant::now();

        info!(instance_id = %instance_id, "Waiting for instance to be running...");

        // Initial delay to let EC2 register the instance
        tokio::time::sleep(Duration::from_secs(5)).await;

        let mut consecutive_errors = 0;

        loop {
            if start.elapsed() > timeout {
                return Err(Ec2Error::Timeout(format!(
                    "Instance {} did not reach running state within {:?}",
                    instance_id, timeout
                )));
            }

            let response = match client
                .describe_instances()
                .instance_ids(instance_id)
                .send()
                .await
            {
                Ok(resp) => {
                    consecutive_errors = 0;
                    resp
                }
                Err(e) => {
                    consecutive_errors += 1;
                    warn!(
                        instance_id = %instance_id,
                        error = %e,
                        attempt = consecutive_errors,
                        "Error describing instance, will retry..."
                    );

                    // Allow up to 5 consecutive errors before failing
                    if consecutive_errors >= 5 {
                        return Err(Ec2Error::Sdk(format!(
                            "Failed to describe instance after {} attempts: {}",
                            consecutive_errors, e
                        )));
                    }

                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
            };

            let instance = response
                .reservations()
                .first()
                .and_then(|r| r.instances().first());

            // Handle case where instance isn't visible yet
            let instance = match instance {
                Some(i) => i,
                None => {
                    debug!(instance_id = %instance_id, "Instance not yet visible, retrying...");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
            };

            let state = instance
                .state()
                .and_then(|s| s.name())
                .map(|n| n.as_str())
                .unwrap_or("unknown");

            debug!(instance_id = %instance_id, state = %state, "Instance state");

            if state == "running" {
                let info = InstanceInfo {
                    instance_id: instance_id.to_string(),
                    public_ip: instance.public_ip_address().map(|s| s.to_string()),
                    private_ip: instance.private_ip_address().map(|s| s.to_string()),
                    state: state.to_string(),
                };

                info!(
                    instance_id = %instance_id,
                    public_ip = ?info.public_ip,
                    "Instance is running"
                );

                return Ok(info);
            }

            if state == "terminated" || state == "shutting-down" {
                return Err(Ec2Error::Sdk(format!(
                    "Instance {} entered {} state",
                    instance_id, state
                )));
            }

            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }

    /// Wait for user data script to complete by polling S3 for a completion marker
    pub async fn wait_for_completion(
        &self,
        bucket: &str,
        marker_key: &str,
        timeout: Duration,
    ) -> Ec2Result<()> {
        let s3_client = self.resource_manager.s3_client();
        let start = std::time::Instant::now();

        info!(bucket = %bucket, marker_key = %marker_key, "Waiting for benchmark completion...");

        loop {
            if start.elapsed() > timeout {
                return Err(Ec2Error::Timeout(format!(
                    "Benchmark did not complete within {:?}",
                    timeout
                )));
            }

            // Check if completion marker exists in S3
            let result = s3_client
                .head_object()
                .bucket(bucket)
                .key(marker_key)
                .send()
                .await;

            if result.is_ok() {
                info!("Benchmark completion marker found");
                return Ok(());
            }

            // Print progress every 30 seconds
            let elapsed = start.elapsed();
            if elapsed.as_secs() % 30 == 0 && elapsed.as_secs() > 0 {
                let pct = (elapsed.as_secs() as f64 / timeout.as_secs() as f64 * 100.0) as u32;
                info!(
                    elapsed = ?elapsed,
                    progress = %format!("{}%", pct.min(99)),
                    "Still waiting for benchmark..."
                );
            }

            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    }

    /// Get console output from an instance (useful for debugging)
    pub async fn get_console_output(&self, instance_id: &str) -> Ec2Result<String> {
        let client = self.resource_manager.ec2_client();

        let response = client
            .get_console_output()
            .instance_id(instance_id)
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to get console output: {}", e)))?;

        let output = response
            .output()
            .map(|s| {
                // Console output is base64 encoded
                base64_decode(s).unwrap_or_else(|_| s.to_string())
            })
            .unwrap_or_default();

        Ok(output)
    }

    /// Get the default VPC ID
    pub async fn get_default_vpc(&self) -> Ec2Result<String> {
        let client = self.resource_manager.ec2_client();

        let response = client
            .describe_vpcs()
            .filters(
                aws_sdk_ec2::types::Filter::builder()
                    .name("is-default")
                    .values("true")
                    .build(),
            )
            .send()
            .await
            .map_err(|e| Ec2Error::Sdk(format!("Failed to describe VPCs: {}", e)))?;

        let vpc_id = response
            .vpcs()
            .first()
            .and_then(|v| v.vpc_id())
            .ok_or_else(|| Ec2Error::Sdk("No default VPC found".to_string()))?;

        Ok(vpc_id.to_string())
    }
}

/// Base64 encode a string
fn base64_encode(s: &str) -> String {
    use std::io::Write;
    let mut encoder = base64_writer(Vec::new());
    encoder.write_all(s.as_bytes()).unwrap();
    String::from_utf8(encoder.into_inner()).unwrap()
}

fn base64_writer(writer: Vec<u8>) -> Base64Writer {
    Base64Writer { inner: writer }
}

struct Base64Writer {
    inner: Vec<u8>,
}

impl std::io::Write for Base64Writer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        let mut i = 0;
        while i + 2 < buf.len() {
            let n = ((buf[i] as u32) << 16) | ((buf[i + 1] as u32) << 8) | (buf[i + 2] as u32);
            self.inner.push(ALPHABET[((n >> 18) & 0x3F) as usize]);
            self.inner.push(ALPHABET[((n >> 12) & 0x3F) as usize]);
            self.inner.push(ALPHABET[((n >> 6) & 0x3F) as usize]);
            self.inner.push(ALPHABET[(n & 0x3F) as usize]);
            i += 3;
        }

        // Handle remaining bytes
        let remaining = buf.len() - i;
        if remaining == 1 {
            let n = (buf[i] as u32) << 16;
            self.inner.push(ALPHABET[((n >> 18) & 0x3F) as usize]);
            self.inner.push(ALPHABET[((n >> 12) & 0x3F) as usize]);
            self.inner.push(b'=');
            self.inner.push(b'=');
        } else if remaining == 2 {
            let n = ((buf[i] as u32) << 16) | ((buf[i + 1] as u32) << 8);
            self.inner.push(ALPHABET[((n >> 18) & 0x3F) as usize]);
            self.inner.push(ALPHABET[((n >> 12) & 0x3F) as usize]);
            self.inner.push(ALPHABET[((n >> 6) & 0x3F) as usize]);
            self.inner.push(b'=');
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Base64Writer {
    fn into_inner(self) -> Vec<u8> {
        self.inner
    }
}

/// Base64 decode a string
fn base64_decode(s: &str) -> Result<String, &'static str> {
    const DECODE: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1,
        -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1,
        -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    ];

    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut result = Vec::new();

    let mut i = 0;
    while i + 3 < bytes.len() {
        let a = DECODE[bytes[i] as usize] as u32;
        let b = DECODE[bytes[i + 1] as usize] as u32;
        let c = if bytes[i + 2] == b'=' {
            0
        } else {
            DECODE[bytes[i + 2] as usize] as u32
        };
        let d = if bytes[i + 3] == b'=' {
            0
        } else {
            DECODE[bytes[i + 3] as usize] as u32
        };

        let n = (a << 18) | (b << 12) | (c << 6) | d;

        result.push((n >> 16) as u8);
        if bytes[i + 2] != b'=' {
            result.push((n >> 8) as u8);
        }
        if bytes[i + 3] != b'=' {
            result.push(n as u8);
        }

        i += 4;
    }

    String::from_utf8(result).map_err(|_| "Invalid UTF-8")
}

/// Generate user data script for running Vortex benchmark on EC2
pub fn generate_benchmark_user_data(
    s3_bucket: &str,
    vectors: usize,
    aws_access_key: &str,
    aws_secret_key: &str,
    region: &str,
    index_type: &str,
) -> String {
    // Determine cargo features based on index type
    let features = if index_type == "diskann" {
        "aws-storage,diskann-index"
    } else {
        "aws-storage"
    };

    format!(
        r#"#!/bin/bash
set -e
exec > >(tee /var/log/vortex-benchmark.log) 2>&1

echo "=========================================="
echo "Vortex Cloud Benchmark - Starting"
echo "Index Type: {index_type}"
echo "=========================================="

# Set HOME explicitly (cloud-init doesn't always set it)
export HOME=/root

# Export AWS credentials
export AWS_ACCESS_KEY_ID="{aws_access_key}"
export AWS_SECRET_ACCESS_KEY="{aws_secret_key}"
export AWS_DEFAULT_REGION="{region}"

# Set index type
export INDEX_TYPE="{index_type}"

# Install dependencies
echo "Installing dependencies..."
yum update -y
yum install -y git gcc gcc-c++ make openssl-devel

# Install Rust
echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source /root/.cargo/env

# Clone Vortex
echo "Cloning Vortex repository..."
cd /opt
git clone https://github.com/AmirFone/vortex.git
cd vortex

# Build with appropriate features
echo "Building Vortex with features: {features}..."
cargo build --release --features {features}

# Run benchmark
echo "Running benchmark with {vectors} vectors using $INDEX_TYPE index..."
./target/release/simulate \
    --vectors {vectors} \
    --storage mock \
    --verbose \
    2>&1 | tee /tmp/benchmark_results.txt

echo "=========================================="
echo "Benchmark Complete - Uploading Results"
echo "=========================================="

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    yum install -y awscli
fi

# Upload results to S3
aws s3 cp /tmp/benchmark_results.txt s3://{s3_bucket}/results/benchmark.txt
aws s3 cp /var/log/vortex-benchmark.log s3://{s3_bucket}/results/bootstrap.log

# Create completion marker
echo "COMPLETE" | aws s3 cp - s3://{s3_bucket}/results/COMPLETE

echo "=========================================="
echo "All done! Results uploaded to S3."
echo "=========================================="
"#,
        aws_access_key = aws_access_key,
        aws_secret_key = aws_secret_key,
        region = region,
        s3_bucket = s3_bucket,
        vectors = vectors,
        index_type = index_type,
        features = features,
    )
}
