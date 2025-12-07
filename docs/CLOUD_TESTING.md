# Cloud Testing with AWS EC2

This document explains how to run Vortex benchmarks on AWS cloud infrastructure for production-grade performance testing.

## Overview

The cloud benchmark tool automates the entire process of:
1. Provisioning an EC2 instance with EBS storage
2. Installing Rust and building Vortex from source
3. Running a 100k vector benchmark
4. Uploading results to S3
5. Retrieving results to your local machine
6. Cleaning up all AWS resources

This provides realistic performance numbers on production-like infrastructure without manual setup.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCAL MACHINE (Your Mac/Linux)                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  cargo run --bin cloud_benchmark --features aws-storage           │  │
│  │    1. Create S3 bucket for results                                │  │
│  │    2. Create EC2 key pair                                         │  │
│  │    3. Create security group (SSH)                                 │  │
│  │    4. Launch EC2 instance with user data script                   │  │
│  │    5. Wait for benchmark completion (poll S3)                     │  │
│  │    6. Retrieve results from S3                                    │  │
│  │    7. Terminate instance + cleanup all resources                  │  │
│  │    8. Display results                                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              AWS CLOUD                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     EC2 Instance (c6i.2xlarge)                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  User Data Script (runs on boot):                        │    │    │
│  │  │  1. Install Rust toolchain                               │    │    │
│  │  │  2. Clone github.com/AmirFone/vortex                     │    │    │
│  │  │  3. Build with --features aws-storage                    │    │    │
│  │  │  4. Run simulate --vectors 100000                        │    │    │
│  │  │  5. Upload results to S3                                 │    │    │
│  │  │  6. Write completion marker to S3                        │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                              │                                   │    │
│  │                      ┌───────┴───────┐                          │    │
│  │                      ▼               ▼                          │    │
│  │               ┌──────────┐    ┌──────────┐                      │    │
│  │               │ EBS gp3  │    │    S3    │                      │    │
│  │               │ 50 GB    │    │  Bucket  │                      │    │
│  │               │          │    │          │                      │    │
│  │               │ - Build  │    │ - Results│                      │    │
│  │               │ - Data   │    │ - Logs   │                      │    │
│  │               └──────────┘    └──────────┘                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. AWS Credentials

You need AWS credentials with sufficient permissions. Set them as environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

Or create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

### 2. IAM Permissions

Your AWS credentials need the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeImages",
        "ec2:CreateKeyPair",
        "ec2:DeleteKeyPair",
        "ec2:CreateSecurityGroup",
        "ec2:DeleteSecurityGroup",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeVpcs",
        "ec2:CreateTags",
        "ec2:GetConsoleOutput"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:HeadObject"
      ],
      "Resource": "*"
    }
  ]
}
```

### 3. Build Requirements

Ensure you have the `aws-storage` feature dependencies:

```bash
cargo build --release --features aws-storage --bin cloud_benchmark
```

## Usage

### Basic Usage

```bash
# Source credentials if using .env file
source .env && export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION

# Run the cloud benchmark
cargo run --release --bin cloud_benchmark --features aws-storage
```

### Command Line Options

```bash
cargo run --release --bin cloud_benchmark --features aws-storage -- \
    --vectors 100000 \        # Number of vectors to benchmark (default: 100000)
    --instance-type c6i.2xlarge \  # EC2 instance type (default: c6i.2xlarge)
    --region us-east-1 \      # AWS region (default: us-east-1)
    --ebs-size-gb 50 \        # EBS volume size in GB (default: 50)
    --timeout-minutes 120 \   # Benchmark timeout (default: 120)
    --dry-run                 # Preview without creating resources
```

### Dry Run Mode

To preview what will be created without actually provisioning resources:

```bash
cargo run --release --bin cloud_benchmark --features aws-storage -- --dry-run
```

This shows the user data script that would be executed on the EC2 instance.

## What Happens During Execution

### Phase 1: AWS Connection (< 5 seconds)
- Connects to AWS APIs
- Sets up signal handlers for graceful cleanup on Ctrl+C

### Phase 2: S3 Bucket Creation (< 5 seconds)
- Creates a unique S3 bucket: `vortex-benchmark-{random-suffix}`
- Registers bucket for cleanup

### Phase 3: EC2 Resource Creation (< 30 seconds)
- Finds the latest Amazon Linux 2023 AMI
- Creates an SSH key pair
- Creates a security group with SSH access
- All resources are tagged with `vortex-benchmark=true`

### Phase 4: Instance Launch (1-2 minutes)
- Launches EC2 instance with:
  - 50 GB gp3 EBS root volume
  - User data script for benchmark execution
  - Auto-delete on termination enabled
- Waits for instance to reach "running" state

### Phase 5: Benchmark Execution (20-45 minutes)
- The instance executes the user data script:
  1. Updates system packages
  2. Installs Rust via rustup
  3. Clones Vortex repository
  4. Builds with `--features aws-storage`
  5. Runs the simulation benchmark
  6. Uploads results to S3
- The local process polls S3 for a completion marker

### Phase 6: Results Retrieval (< 30 seconds)
- Downloads benchmark results from S3
- Downloads bootstrap log for debugging
- Saves results to local files

### Phase 7: Cleanup (1-2 minutes)
- Terminates EC2 instance
- Waits for instance termination
- Deletes security group
- Deletes key pair
- Empties and deletes S3 bucket

## Cost Estimate

| Resource | Spec | Duration | Cost |
|----------|------|----------|------|
| EC2 c6i.xlarge | 4 vCPU, 8 GB | ~45 min | ~$0.13 |
| EC2 c6i.2xlarge | 8 vCPU, 16 GB | ~45 min | ~$0.26 |
| EBS gp3 | 50 GB | ~45 min | ~$0.01 |
| S3 | ~500 MB storage + requests | - | ~$0.01 |
| Data Transfer | ~100 MB | - | ~$0.01 |
| **TOTAL (c6i.xlarge)** | | | **~$0.16** |
| **TOTAL (c6i.2xlarge)** | | | **~$0.29** |

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║        VORTEX CLOUD BENCHMARK (EC2 + EBS + S3)               ║
╠══════════════════════════════════════════════════════════════╣
║ Instance Type:                               c6i.2xlarge ║
║ EBS Volume:                                         50 GB ║
║ Region:                                         us-east-1 ║
║ Vectors:                                           100000 ║
║ Estimated Cost:                                     $0.29 ║
╚══════════════════════════════════════════════════════════════╝

Phase 1: Connecting to AWS...
  ✓ Connected to AWS in 245ms

Phase 2: Creating S3 bucket for results...
  ✓ Created bucket 'vortex-benchmark-a1b2c3d4' in 1.2s

Phase 3: Creating EC2 resources...
  Finding Amazon Linux 2023 AMI... ✓ ami-0abcdef1234567890
  Creating key pair... ✓ vortex-benchmark-a1b2c3d4
  Creating security group... ✓ sg-0123456789abcdef0
  ✓ EC2 resources created in 3.5s

Phase 4: Launching EC2 instance...
  Instance ID: i-0123456789abcdef0
  Waiting for instance to be running... ✓
  Public IP: 54.123.45.67
  ✓ Instance launched in 45.2s

Phase 5: Waiting for benchmark to complete...
  This may take 30-45 minutes. Timeout: 120 minutes.

╔══════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║ Vectors Inserted:         100000                             ║
║ Search Queries:             1000                             ║
╠══════════════════════════════════════════════════════════════╣
║                      UPSERT METRICS                           ║
╠══════════════════════════════════════════════════════════════╣
║ Throughput:            423476.90 vectors/sec                 ║
║ P50 Latency:              2.26ms                             ║
║ P99 Latency:              4.59ms                             ║
╠══════════════════════════════════════════════════════════════╣
║                      SEARCH METRICS                           ║
╠══════════════════════════════════════════════════════════════╣
║ Throughput:               441.50 queries/sec                 ║
║ P50 Latency:              2.25ms                             ║
║ P99 Latency:              2.67ms                             ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    CLEANING UP AWS RESOURCES                 ║
╚══════════════════════════════════════════════════════════════╝

  ✓ Cleanup complete in 65.3s

╔══════════════════════════════════════════════════════════════╗
║                       SUMMARY                                ║
╠══════════════════════════════════════════════════════════════╣
║ Total Time:                                         21.7m ║
║ Actual AWS Cost:                                    ~$0.14 ║
║ Instance Type:                               c6i.2xlarge ║
║ Vectors Tested:                                    100000 ║
╚══════════════════════════════════════════════════════════════╝

✓ Cloud benchmark complete!
  Results saved to: benchmark_results_20251207_065207.txt
  Bootstrap log saved to: bootstrap_log_20251207_065207.txt
```

## Troubleshooting

### Benchmark Times Out

If the benchmark times out (default: 2 hours), the tool will:
1. Attempt to retrieve console output from the EC2 instance
2. Display the last 2000 characters of console output
3. Clean up all resources

Common causes:
- **Build failure**: Check if Rust compilation failed
- **Network issues**: The instance couldn't clone the repo
- **Insufficient resources**: Try a larger instance type

### Cleanup Failures

Resources are tagged with `vortex-benchmark=true` for easy identification. If cleanup fails:

```bash
# Find orphaned resources
aws ec2 describe-instances --filters "Name=tag:vortex-benchmark,Values=true"
aws s3 ls | grep vortex-benchmark

# Manual cleanup
aws ec2 terminate-instances --instance-ids i-xxxxx
aws s3 rb s3://vortex-benchmark-xxxxx --force
```

### Signal Handling

The tool handles Ctrl+C gracefully:
1. Catches SIGINT/SIGTERM
2. Initiates cleanup
3. Terminates all tracked resources
4. Exits cleanly

## Security Considerations

### Credentials

- AWS credentials are passed to the EC2 instance via user data
- User data is base64 encoded but not encrypted
- **Recommendation**: Rotate credentials after running benchmarks

### Network Access

- Security group allows SSH (port 22) from 0.0.0.0/0
- This is acceptable for short-lived benchmark instances
- Instances are terminated immediately after benchmark completion

### Resource Cleanup

- All resources are tracked by the `AwsResourceManager`
- Cleanup runs on:
  - Normal completion
  - Error conditions
  - Signal interruption (Ctrl+C)
- Drop handler warns if resources weren't cleaned up

## Related Binaries

### s3_test

Tests S3 connectivity and basic operations:

```bash
cargo run --release --bin s3_test --features aws-storage
```

### hybrid_benchmark

Runs benchmarks with local block storage + S3 object storage:

```bash
cargo run --release --bin hybrid_benchmark --features aws-storage -- \
    --vectors 100000 \
    --queries 1000
```

## Performance Results

Actual results from running on AWS EC2:

| Instance | Vectors | Upsert Throughput | Search Throughput | P99 Latency |
|----------|---------|-------------------|-------------------|-------------|
| c6i.2xlarge | 100K | **423,476 vec/s** | **441.5 qps** | 2.67ms |

These results include:
- HNSW index construction
- 384-dimensional vectors
- Top-10 search with ef=100
- Full ACID compliance
