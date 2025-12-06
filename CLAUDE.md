# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vortex is a high-performance, ACID-compliant vector database written in Rust with HNSW (Hierarchical Navigable Small World) indexing. It provides a RESTful API via Axum for vector similarity search with multi-tenant isolation.

## Build & Development Commands

```bash
# Build
cargo build --release

# Run the server
cargo run --release

# Run all tests
cargo test

# Run specific test suite
cargo test --test acid_tests
cargo test --test stress_tests
cargo test --test concurrency_tests

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
cargo bench --bench performance_benchmarks
cargo bench --bench search_benchmark
```

## Architecture

### Data Flow
```
HTTP API (Axum) → VectorEngine → TenantState → [WAL + VectorStore + HNSW Index]
```

### Core Components

- **VectorEngine** (`src/engine/mod.rs`): Top-level orchestrator. Manages tenant lifecycle via `DashMap<u64, Arc<TenantState>>`. Spawns background flush task.

- **TenantState** (`src/tenant/mod.rs`): Per-tenant isolated state containing:
  - WAL for durability
  - VectorStore (mmap-backed)
  - HNSW index for ANN search
  - Write buffer for recent vectors not yet in HNSW
  - ID mappings (`id_map`: vector_id→array_index, `reverse_id_map`: array_index→vector_id)

- **HNSW Index** (`src/hnsw/`): Approximate nearest neighbor search. Key files:
  - `insert.rs`: Graph construction algorithm
  - `search.rs`: Query algorithm
  - `persistence.rs`: Index serialization

- **WAL** (`src/wal/`): Write-ahead log with CRC32 checksums for ACID durability. Supports batch writes and replay.

- **VectorStore** (`src/vectors/`): Memory-mapped file storage for vector data with zero-copy reads.

- **Storage** (`src/storage/`): Abstraction layer with mock (temp file) and optional S3 backends. Feature flag `aws-storage` enables S3.

### Key Design Patterns

1. **Write Path**: Upsert → WAL append (durability) → VectorStore append → write_buffer → (background) flush to HNSW

2. **Search Path**: Query HNSW + brute-force write_buffer → merge results by similarity

3. **Recovery**: On startup, replay WAL entries after `last_flushed_seq` to rebuild write_buffer

4. **Concurrency**: `parking_lot::RwLock` for tenant state, `DashMap` for tenant registry

## Configuration

Environment variables: `HOST`, `PORT`, `EBS_PATH`, `S3_PATH`, `DIMENSIONS`, `FLUSH_INTERVAL_SECS`

HNSW parameters: `M` (connections per node), `ef_construction`, `ef_search`

## Feature Flags

- `mock-storage` (default): Uses temp file storage
- `aws-storage`: Enables AWS S3 backend
