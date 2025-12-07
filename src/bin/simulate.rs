//! Vortex Simulation CLI
//!
//! Run realistic benchmarks on mock or real AWS infrastructure.
//!
//! Usage:
//!   cargo run --bin simulate -- --vectors 100000 --storage mock
//!   cargo run --bin simulate --features aws-storage -- --vectors 100000 --storage s3

use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vortex::simulation::{SimulationConfig, SimulationRunner, StorageType};

#[derive(Parser)]
#[command(name = "vortex-simulate")]
#[command(about = "Run Vortex benchmarks on mock or AWS infrastructure")]
#[command(version)]
struct Args {
    /// Number of vectors to insert
    #[arg(long, default_value = "100000")]
    vectors: usize,

    /// Vector dimensions
    #[arg(long, default_value = "384")]
    dimensions: usize,

    /// Batch size for upserts
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Number of search queries to run
    #[arg(long, default_value = "1000")]
    search_queries: usize,

    /// Top-k results to retrieve
    #[arg(long, default_value = "10")]
    k: usize,

    /// Search ef parameter (higher = more accurate, slower)
    #[arg(long, default_value = "100")]
    ef: usize,

    /// Storage backend: mock, s3
    #[arg(long, default_value = "mock")]
    storage: String,

    /// AWS region (only used with S3 storage)
    #[arg(long, default_value = "us-east-1")]
    region: String,

    /// Enable verbose logging
    #[arg(long, short)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    // Setup logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| format!("vortex={},simulate={}", log_level, log_level)),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse storage type
    let storage_type: StorageType = match args.storage.parse() {
        Ok(st) => st,
        Err(e) => {
            eprintln!("Error: {}", e);
            #[cfg(not(feature = "aws-storage"))]
            eprintln!("Note: AWS storage requires the 'aws-storage' feature. Use: cargo run --bin simulate --features aws-storage");
            std::process::exit(1);
        }
    };

    // Build config
    let config = SimulationConfig {
        vector_count: args.vectors,
        dimensions: args.dimensions,
        batch_size: args.batch_size,
        search_queries: args.search_queries,
        k: args.k,
        ef_search: args.ef,
        storage_type,
        aws_region: args.region,
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              VORTEX VECTOR DATABASE SIMULATION               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Vectors:              {:>10}                             ║", config.vector_count);
    println!("║ Dimensions:           {:>10}                             ║", config.dimensions);
    println!("║ Batch Size:           {:>10}                             ║", config.batch_size);
    println!("║ Search Queries:       {:>10}                             ║", config.search_queries);
    println!("║ Top-K:                {:>10}                             ║", config.k);
    println!("║ EF Search:            {:>10}                             ║", config.ef_search);
    println!("║ Storage:              {:>10?}                            ║", config.storage_type);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Run simulation
    let mut runner = SimulationRunner::new(config);
    let results = runner.run().await?;

    // Print results
    results.print_summary();

    Ok(())
}
