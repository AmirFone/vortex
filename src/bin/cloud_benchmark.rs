//! Cloud Benchmark - EC2 + EBS + S3
//!
//! This binary orchestrates a complete cloud benchmark:
//! 1. Provisions an EC2 instance with EBS storage
//! 2. Clones the Vortex repo and builds it
//! 3. Runs the benchmark with 100k vectors
//! 4. Uploads results to S3
//! 5. Retrieves results to local machine
//! 6. Terminates all AWS resources
//!
//! Usage:
//!   cargo run --release --bin cloud_benchmark --features aws-storage
//!
//! Options:
//!   --vectors 100000       Number of vectors to benchmark
//!   --instance-type c6i.2xlarge   EC2 instance type
//!   --region us-east-1     AWS region
//!   --ebs-size-gb 50       EBS volume size

use clap::Parser;
use std::time::{Duration, Instant};
use vortex::storage::aws::{
    generate_benchmark_user_data, AwsResourceManager, Ec2Provisioner, InstanceConfig,
    S3ObjectStorage,
};
use vortex::storage::ObjectStorage;

#[derive(Parser)]
#[command(name = "cloud-benchmark")]
#[command(about = "Run Vortex benchmark on EC2 with EBS + S3")]
struct Args {
    /// Number of vectors to benchmark
    #[arg(long, default_value = "100000")]
    vectors: usize,

    /// EC2 instance type (c6i.xlarge = 4 vCPU, c6i.2xlarge = 8 vCPU)
    #[arg(long, default_value = "c6i.2xlarge")]
    instance_type: String,

    /// AWS region
    #[arg(long, default_value = "us-east-1")]
    region: String,

    /// EBS volume size in GB
    #[arg(long, default_value = "50")]
    ebs_size_gb: i32,

    /// Benchmark timeout in minutes
    #[arg(long, default_value = "120")]
    timeout_minutes: u64,

    /// Dry run - don't actually create AWS resources
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Get AWS credentials from environment
    let aws_access_key = std::env::var("AWS_ACCESS_KEY_ID")
        .expect("AWS_ACCESS_KEY_ID environment variable required");
    let aws_secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
        .expect("AWS_SECRET_ACCESS_KEY environment variable required");

    // Calculate cost estimate
    let hourly_rate = match args.instance_type.as_str() {
        "c6i.xlarge" => 0.17,
        "c6i.2xlarge" => 0.34,
        "c6i.4xlarge" => 0.68,
        "c6i.8xlarge" => 1.36,
        "c6i.12xlarge" => 2.04,
        "c6i.16xlarge" => 2.72,
        _ => 0.34,
    };
    let estimated_hours = 0.75; // ~45 minutes
    let estimated_cost = hourly_rate * estimated_hours + 0.02; // + EBS + S3

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        VORTEX CLOUD BENCHMARK (EC2 + EBS + S3)               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Instance Type:        {:>38} ║",
        args.instance_type
    );
    println!("║ EBS Volume:           {:>32} GB ║", args.ebs_size_gb);
    println!("║ Region:               {:>38} ║", args.region);
    println!("║ Vectors:              {:>38} ║", args.vectors);
    println!(
        "║ Estimated Cost:       {:>37} ║",
        format!("${:.2}", estimated_cost)
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Security notice
    println!("Security Notes:");
    println!("  - SSH port 22 open to 0.0.0.0/0 (acceptable for short-lived benchmark)");
    println!("  - Credentials passed via user data (rotated after benchmark recommended)");
    println!("  - All resources will be cleaned up after completion\n");

    if args.dry_run {
        println!("DRY RUN MODE - No AWS resources will be created\n");
        print_user_data_preview(&args, &aws_access_key, &aws_secret_key);
        return Ok(());
    }

    let total_start = Instant::now();

    // ========== Phase 1: Initialize AWS ==========
    println!("Phase 1: Connecting to AWS...");
    let phase_start = Instant::now();

    let manager = AwsResourceManager::new(&args.region).await?;
    manager.clone().setup_signal_handlers();

    println!("  ✓ Connected to AWS in {:?}\n", phase_start.elapsed());

    // Run the benchmark with guaranteed cleanup on any error
    let result = run_benchmark(&args, &manager, &aws_access_key, &aws_secret_key, total_start).await;

    // Always cleanup, regardless of success or failure
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    CLEANING UP AWS RESOURCES                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let cleanup_start = Instant::now();
    if let Err(e) = manager.cleanup_all().await {
        eprintln!("  Warning: Cleanup error: {}", e);
    }
    println!("  ✓ Cleanup complete in {:?}\n", cleanup_start.elapsed());

    // Return the original result
    result
}

async fn run_benchmark(
    args: &Args,
    manager: &std::sync::Arc<AwsResourceManager>,
    aws_access_key: &str,
    aws_secret_key: &str,
    total_start: Instant,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ========== Phase 2: Create S3 Bucket ==========
    println!("Phase 2: Creating S3 bucket for results...");
    let phase_start = Instant::now();

    // Use empty prefix - user data uploads directly to /results/
    let s3 = S3ObjectStorage::new(manager.clone(), "").await?;
    let bucket = s3.bucket().to_string();

    println!(
        "  ✓ Created bucket '{}' in {:?}\n",
        bucket,
        phase_start.elapsed()
    );

    // ========== Phase 3: Create EC2 Resources ==========
    println!("Phase 3: Creating EC2 resources...");
    let phase_start = Instant::now();

    let provisioner = Ec2Provisioner::new(manager.clone());

    // Get AMI
    print!("  Finding Amazon Linux 2023 AMI... ");
    let ami_id = provisioner.get_amazon_linux_ami().await?;
    println!("✓ {}", ami_id);

    // Create key pair
    print!("  Creating key pair... ");
    let key_name = format!("vortex-benchmark-{}", &bucket[bucket.len() - 8..]);
    let _private_key = provisioner.create_key_pair(&key_name).await?;
    println!("✓ {}", key_name);

    // Create security group
    print!("  Creating security group... ");
    let sg_name = format!("vortex-benchmark-sg-{}", &bucket[bucket.len() - 8..]);
    let sg_id = provisioner
        .create_security_group(&sg_name, "Vortex benchmark security group")
        .await?;
    println!("✓ {}", sg_id);

    println!("  ✓ EC2 resources created in {:?}\n", phase_start.elapsed());

    // ========== Phase 4: Launch Instance ==========
    println!("Phase 4: Launching EC2 instance...");
    let phase_start = Instant::now();

    // Generate user data script
    let user_data = generate_benchmark_user_data(
        &bucket,
        args.vectors,
        aws_access_key,
        aws_secret_key,
        &args.region,
    );

    let config = InstanceConfig {
        instance_type: args.instance_type.clone(),
        ami_id,
        key_name,
        security_group_id: sg_id,
        user_data,
        ebs_size_gb: args.ebs_size_gb,
        instance_name: "vortex-benchmark".to_string(),
    };

    let instance = provisioner.launch_instance(config).await?;
    println!("  Instance ID: {}", instance.instance_id);

    // Wait for running state
    print!("  Waiting for instance to be running... ");
    let instance_info = provisioner.wait_for_running(&instance.instance_id).await?;
    println!("✓");

    if let Some(ip) = &instance_info.public_ip {
        println!("  Public IP: {}", ip);
    }

    println!("  ✓ Instance launched in {:?}\n", phase_start.elapsed());

    // ========== Phase 5: Wait for Benchmark ==========
    println!("Phase 5: Waiting for benchmark to complete...");
    println!(
        "  This may take 30-45 minutes. Timeout: {} minutes.",
        args.timeout_minutes
    );
    println!("  Progress is being logged to the instance.\n");

    let phase_start = Instant::now();
    let timeout = Duration::from_secs(args.timeout_minutes * 60);

    // Poll for completion - note: user data script uploads to /results/COMPLETE (no prefix)
    let completion_result = provisioner
        .wait_for_completion(&bucket, "results/COMPLETE", timeout)
        .await;

    match completion_result {
        Ok(_) => {
            println!(
                "  ✓ Benchmark completed in {:?}\n",
                phase_start.elapsed()
            );
        }
        Err(e) => {
            println!("  ✗ Benchmark failed or timed out: {}\n", e);

            // Try to get console output for debugging
            println!("  Attempting to retrieve console output...");
            match provisioner
                .get_console_output(&instance.instance_id)
                .await
            {
                Ok(output) => {
                    println!("  Console output (last 2000 chars):");
                    let output_end = if output.len() > 2000 {
                        &output[output.len() - 2000..]
                    } else {
                        &output
                    };
                    println!("{}", output_end);
                }
                Err(e) => println!("  Could not get console output: {}", e),
            }

            // Error will be returned; main() will cleanup
            return Err(e.into());
        }
    }

    // ========== Phase 6: Retrieve Results ==========
    println!("Phase 6: Retrieving benchmark results...");
    let phase_start = Instant::now();

    // Get benchmark results (paths match what user data script uploads)
    let results = match s3.get("results/benchmark.txt").await {
        Ok(data) => String::from_utf8_lossy(&data).to_string(),
        Err(e) => {
            println!("  Warning: Could not retrieve benchmark.txt: {}", e);
            "Results not available".to_string()
        }
    };

    // Get bootstrap log
    let bootstrap_log = match s3.get("results/bootstrap.log").await {
        Ok(data) => String::from_utf8_lossy(&data).to_string(),
        Err(e) => {
            println!("  Warning: Could not retrieve bootstrap.log: {}", e);
            String::new()
        }
    };

    println!("  ✓ Results retrieved in {:?}\n", phase_start.elapsed());

    // ========== Phase 7: Display Results ==========
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse and display results
    display_benchmark_results(&results);

    // ========== Summary ==========
    let total_time = total_start.elapsed();
    let hourly_rate = match args.instance_type.as_str() {
        "c6i.xlarge" => 0.17,
        "c6i.2xlarge" => 0.34,
        "c6i.4xlarge" => 0.68,
        "c6i.8xlarge" => 1.36,
        "c6i.12xlarge" => 2.04,
        "c6i.16xlarge" => 2.72,
        _ => 0.34,
    };
    let actual_cost = (total_time.as_secs_f64() / 3600.0) * hourly_rate + 0.02;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                       SUMMARY                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Total Time:           {:>38} ║",
        format!("{:.1?}", total_time)
    );
    println!(
        "║ Actual AWS Cost:      {:>37} ║",
        format!("~${:.3}", actual_cost)
    );
    println!(
        "║ Instance Type:        {:>38} ║",
        args.instance_type
    );
    println!("║ Vectors Tested:       {:>38} ║", args.vectors);
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("✓ Cloud benchmark complete!");

    // Save results to local file
    let results_file = format!(
        "benchmark_results_{}.txt",
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::write(&results_file, &results)?;
    println!("  Results saved to: {}", results_file);

    if !bootstrap_log.is_empty() {
        let log_file = format!(
            "bootstrap_log_{}.txt",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );
        std::fs::write(&log_file, &bootstrap_log)?;
        println!("  Bootstrap log saved to: {}", log_file);
    }

    Ok(())
}

fn print_user_data_preview(args: &Args, access_key: &str, _secret_key: &str) {
    let user_data = generate_benchmark_user_data(
        "EXAMPLE-BUCKET",
        args.vectors,
        &access_key[..8], // Only show prefix
        "***REDACTED***",
        &args.region,
    );

    println!("User Data Script Preview:");
    println!("─────────────────────────────────────────────────────────────────");
    for line in user_data.lines().take(30) {
        println!("{}", line);
    }
    println!("... (truncated)");
    println!("─────────────────────────────────────────────────────────────────");
}

fn display_benchmark_results(results: &str) {
    // Try to find and display key metrics from the results
    let mut found_results = false;

    for line in results.lines() {
        // Look for result lines
        if line.contains("Throughput")
            || line.contains("Latency")
            || line.contains("P50")
            || line.contains("P99")
            || line.contains("vectors/sec")
            || line.contains("queries/sec")
            || line.starts_with("║")
            || line.starts_with("╔")
            || line.starts_with("╚")
            || line.starts_with("╠")
        {
            println!("{}", line);
            found_results = true;
        }
    }

    if !found_results {
        // Just print the raw results
        println!("{}", results);
    }
}
