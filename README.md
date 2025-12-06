# Vortex

A high-performance, ACID-compliant vector database written in Rust with HNSW indexing.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **HNSW Indexing** - Hierarchical Navigable Small World graphs for efficient approximate nearest neighbor search
- **ACID Compliance** - Write-ahead logging with CRC32 checksums ensures durability and crash recovery
- **Multi-Tenant Architecture** - Isolated storage and indexing per tenant with no cross-tenant data leakage
- **Memory-Mapped Storage** - Zero-copy vector reads via mmap for optimal memory utilization
- **High Throughput** - Optimized write buffer with batch operations for sustained insert performance
- **RESTful API** - Clean HTTP endpoints via Axum for easy integration
- **Configurable Search** - Tune recall vs latency with adjustable `ef` parameter

## Performance

Benchmarks run on 384-dimensional normalized vectors:

| Metric | Value |
|--------|-------|
| Insert Throughput | **8,338 vectors/sec** |
| Search p50 Latency | **< 10ms** |
| Search p99 Latency | **< 50ms** |
| Recall@10 (ef=200) | **100%** |
| Concurrent Writers (2 threads) | **6,793 vectors/sec** |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AmirFone/vortex.git
cd vortex

# Build in release mode
cargo build --release

# Run the server
cargo run --release
```

### Basic Usage

**Insert vectors:**
```bash
curl -X POST http://localhost:3000/tenants/1/upsert \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": 1, "values": [0.1, 0.2, 0.3, ...]},
      {"id": 2, "values": [0.4, 0.5, 0.6, ...]}
    ]
  }'
```

**Search for similar vectors:**
```bash
curl -X POST http://localhost:3000/tenants/1/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "k": 10,
    "ef": 200
  }'
```

**Check health:**
```bash
curl http://localhost:3000/health
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/tenants/:id/upsert` | Insert or update vectors |
| `POST` | `/tenants/:id/search` | Search for nearest neighbors |
| `GET` | `/tenants/:id/stats` | Get tenant statistics |
| `POST` | `/tenants/:id/flush` | Force flush write buffer to HNSW index |

### Upsert Request

```json
{
  "vectors": [
    {
      "id": 1,
      "values": [0.1, 0.2, 0.3]
    }
  ]
}
```

### Search Request

```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "ef": 200
}
```

### Search Response

```json
{
  "results": [
    {"id": 42, "score": 0.95},
    {"id": 17, "score": 0.89}
  ],
  "latency_ms": 5
}
```

## Architecture

```
+-------------------------------------------------------------+
|                        HTTP API (Axum)                       |
+-------------------------------------------------------------+
|                         Engine                               |
|  +----------------------------------------------------------+
|  |                    Tenant Manager                         |
|  |  +-----------+  +-----------+  +-----------+             |
|  |  | Tenant 1  |  | Tenant 2  |  | Tenant N  |             |
|  |  +-----------+  +-----------+  +-----------+             |
|  +----------------------------------------------------------+
+-------------------------------------------------------------+
|                      Per-Tenant State                        |
|  +-----------+  +-----------+  +-------------------+        |
|  |Write Buffer| -> |HNSW Index|  |Vector Store (mmap)|       |
|  +-----------+  +-----------+  +-------------------+        |
+-------------------------------------------------------------+
|                    Durability Layer                          |
|  +----------------------------------------------------------+
|  |              Write-Ahead Log (WAL)                        |
|  |              CRC32 Checksums | fsync                      |
|  +----------------------------------------------------------+
+-------------------------------------------------------------+
|                      Storage Backend                         |
|  +-------------------+  +---------------------------+       |
|  |   Mock Storage    |  |    AWS S3 (optional)      |       |
|  +-------------------+  +---------------------------+       |
+-------------------------------------------------------------+
```

### Components

- **Engine**: Top-level orchestration, manages tenant lifecycle
- **Tenant State**: Isolated per-tenant data with write buffer, HNSW index, and vector storage
- **Write Buffer**: In-memory buffer for recent writes, periodically flushed to HNSW
- **HNSW Index**: Hierarchical Navigable Small World graph for approximate nearest neighbor search
- **Vector Store**: Memory-mapped file storage for vector data
- **WAL**: Write-ahead log for durability with CRC32 integrity checks

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3000` | Server port |
| `EBS_PATH` | `/tmp/vectordb/ebs` | Local storage path |
| `S3_PATH` | `/tmp/vectordb/s3` | S3-compatible storage path |
| `DIMENSIONS` | `384` | Vector dimensionality |
| `FLUSH_INTERVAL_SECS` | `30` | Auto-flush interval |

## HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per node per layer |
| `ef_construction` | 200 | Size of dynamic candidate list during construction |
| `ef` | 100 | Size of dynamic candidate list during search |

Tuning guidelines:
- **Higher M**: Better recall, more memory, slower inserts
- **Higher ef_construction**: Better graph quality, slower builds
- **Higher ef (search)**: Better recall, higher latency

## Testing

The test suite includes 112 tests covering:

- **ACID Compliance** - Atomicity, consistency, isolation, durability
- **Concurrency** - Multi-threaded read/write operations
- **Crash Recovery** - WAL replay and data integrity after failures
- **Stress Testing** - High load and edge case scenarios
- **Correctness** - Search recall and data integrity validation

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test suite
cargo test --test acid_tests
cargo test --test stress_tests
cargo test --test concurrency_tests

# Run release mode tests (faster)
cargo test --release
```

## Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench performance_benchmarks
cargo bench --bench search_benchmark
```

Benchmark groups:
- `upsert_batch` - Insert throughput at various batch sizes
- `search_ef` - Search latency with different ef values
- `search_k` - Search latency with different k values
- `search_index_size` - Search latency vs index size
- `flush` - Flush performance
- `cosine_distance` - Distance computation overhead
- `write_buffer_search` - Buffer search performance
- `mixed_read_write` - Combined workload

## Project Structure

```
src/
├── api/          # HTTP endpoints (Axum handlers)
├── engine/       # Top-level orchestration
├── hnsw/         # HNSW index implementation
│   ├── insert.rs # Insert algorithm
│   ├── search.rs # Search algorithm
│   └── persistence.rs # Index serialization
├── storage/      # Storage backends
│   ├── mock/     # In-memory/temp file storage
│   └── s3/       # AWS S3 backend (optional)
├── tenant/       # Per-tenant state management
├── vectors/      # Vector storage (mmap)
├── wal/          # Write-ahead log
├── config.rs     # Configuration
└── lib.rs        # Library root

tests/
├── acid_tests.rs           # ACID compliance
├── concurrency_tests.rs    # Multi-threaded operations
├── crash_recovery_tests.rs # Failure recovery
├── stress_tests.rs         # Load testing
└── ...

benches/
├── performance_benchmarks.rs # Criterion benchmarks
└── search_benchmark.rs       # End-to-end search benchmarks
```

## License

MIT License - see [LICENSE](LICENSE) for details.
