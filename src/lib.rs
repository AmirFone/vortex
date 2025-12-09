//! # Vortex Vector Database
//!
//! A high-performance, ACID-compliant vector database with pluggable index backends.
//!
//! ## Architecture
//!
//! ```text
//! HTTP API (Axum)
//!     │
//!     ▼
//! VectorEngine (tenant lifecycle)
//!     │
//!     ▼
//! TenantState (isolated per-tenant)
//!     ├── WAL (durability)
//!     ├── VectorStore (mmap storage)
//!     ├── AnnIndex (HNSW or DiskANN)
//!     └── WriteBuffer (unflushed vectors)
//! ```
//!
//! ## Features
//!
//! - **ACID Compliance**: Write-ahead log with CRC32 checksums ensures durability
//! - **HNSW Indexing**: Fast approximate nearest neighbor search
//! - **DiskANN Support**: Memory-efficient indexing for large datasets
//! - **Multi-tenant**: Isolated storage per tenant with no cross-tenant leakage
//! - **Mock Storage**: Local development with temp files
//! - **AWS Storage**: Production deployment with S3 and EBS
//!
//! ## Quick Start
//!
//! ```ignore
//! use vortex::{Config, VectorEngine};
//!
//! let config = Config::from_env();
//! let engine = VectorEngine::new(config, storage_backend).await?;
//! engine.upsert(tenant_id, vector_id, &embedding).await?;
//! let results = engine.search(tenant_id, &query, k).await?;
//! ```

pub mod defaults;
pub mod error;
pub mod storage;
pub mod wal;
pub mod vectors;
pub mod hnsw;
pub mod index;
pub mod tenant;
pub mod engine;
pub mod api;
pub mod config;
pub mod simulation;

pub use config::Config;
pub use defaults::*;
pub use error::{Result, VortexError};
pub use engine::VectorEngine;
