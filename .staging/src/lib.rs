//! VectorDB - ACID-Compliant Vector Database
//!
//! A high-performance vector database with:
//! - ACID compliance via Write-Ahead Log
//! - HNSW indexing for fast similarity search
//! - Mock storage for development, real AWS for production

pub mod storage;
pub mod wal;
pub mod vectors;
pub mod hnsw;
pub mod tenant;
pub mod engine;
pub mod api;
pub mod config;

pub use config::Config;
pub use engine::VectorEngine;
