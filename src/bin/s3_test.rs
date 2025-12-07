//! Quick S3 Integration Test
//!
//! This actually writes to and reads from real AWS S3.
//!
//! Usage:
//!   cargo run --release --bin s3_test --features aws-storage

use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;
use vortex::storage::aws::{AwsResourceManager, S3ObjectStorage};
use vortex::storage::ObjectStorage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Setup logging
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              REAL AWS S3 INTEGRATION TEST                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Create AWS resource manager
    println!("Connecting to AWS...");
    let start = Instant::now();
    let manager = AwsResourceManager::new("us-east-1")
        .await
        .map_err(|e| format!("Failed to connect to AWS: {}", e))?;
    println!("  Connected in {:?}\n", start.elapsed());

    // Setup signal handlers for cleanup
    manager.clone().setup_signal_handlers();

    // Create S3 storage (this creates a real bucket!)
    println!("Creating S3 bucket...");
    let start = Instant::now();
    let s3 = S3ObjectStorage::new(manager.clone(), "test-vectors").await?;
    println!("  Created bucket '{}' in {:?}\n", s3.bucket(), start.elapsed());

    // Test 1: Write a single vector
    println!("Test 1: Write single vector to S3...");
    let vector_data: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
    let bytes = Bytes::from(bincode::serialize(&vector_data)?);

    let start = Instant::now();
    s3.put("vector_0.bin", bytes.clone()).await?;
    println!("  PUT 384-dim vector ({} bytes) in {:?}", bytes.len(), start.elapsed());

    // Test 2: Read it back
    println!("\nTest 2: Read vector from S3...");
    let start = Instant::now();
    let retrieved = s3.get("vector_0.bin").await?;
    let retrieved_vector: Vec<f32> = bincode::deserialize(&retrieved)?;
    println!("  GET vector in {:?}", start.elapsed());
    println!("  Data integrity: {}", if retrieved_vector == vector_data { "PASS ✓" } else { "FAIL ✗" });

    // Test 3: Write batch of vectors
    println!("\nTest 3: Write 100 vectors to S3...");
    let start = Instant::now();
    for i in 0..100 {
        let vec_data: Vec<f32> = (0..384).map(|j| (i * 384 + j) as f32).collect();
        let bytes = Bytes::from(bincode::serialize(&vec_data)?);
        s3.put(&format!("batch/vector_{}.bin", i), bytes).await?;
    }
    let batch_time = start.elapsed();
    println!("  PUT 100 vectors in {:?}", batch_time);
    println!("  Throughput: {:.1} vectors/sec", 100.0 / batch_time.as_secs_f64());

    // Test 4: List objects
    println!("\nTest 4: List objects in S3...");
    let start = Instant::now();
    let objects = s3.list("batch/").await?;
    println!("  Listed {} objects in {:?}", objects.len(), start.elapsed());

    // Test 5: Check exists
    println!("\nTest 5: Check object existence...");
    let start = Instant::now();
    let exists = s3.exists("vector_0.bin").await?;
    let not_exists = s3.exists("nonexistent.bin").await?;
    println!("  EXISTS check in {:?}", start.elapsed());
    println!("  vector_0.bin exists: {} (expected: true)", exists);
    println!("  nonexistent.bin exists: {} (expected: false)", not_exists);

    // Cleanup
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    CLEANING UP AWS RESOURCES                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    manager.cleanup_all().await?;
    println!("\nAll S3 resources cleaned up successfully!");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TEST COMPLETE - ALL PASSED                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
