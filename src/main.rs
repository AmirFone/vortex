//! VectorDB Server Entry Point

use vectordb::{Config, VectorEngine, api};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,vectordb=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting VectorDB server...");

    // Load config
    let config = Config::from_env()?;
    tracing::info!("Loaded config: {:?}", config);

    // Create storage backend
    let storage = config.storage.create_backend().await?;
    tracing::info!("Storage backend initialized");

    // Create engine
    let engine = VectorEngine::new(config.engine.clone(), storage).await?;
    tracing::info!("Vector engine initialized");

    // Start API server
    api::serve(engine, config.api).await?;

    Ok(())
}
