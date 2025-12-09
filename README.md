# Vortex

A production-ready vector database built in Rust for AI/ML applications requiring semantic search at scale.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What is Vortex?

Vortex stores and searches high-dimensional vectors (embeddings) used in:
- **Semantic Search** - Find documents by meaning, not keywords
- **RAG Applications** - Retrieve relevant context for LLMs
- **Recommendation Systems** - Find similar items/users
- **Image/Audio Search** - Match by visual/acoustic similarity

## Why Vortex?

| Requirement | How Vortex Solves It |
|-------------|---------------------|
| **Speed** | 400k+ vectors/sec insert, sub-3ms search latency |
| **Scale** | Multi-tenant isolation, billions of vectors with DiskANN |
| **Reliability** | ACID-compliant with WAL, survives crashes |
| **Flexibility** | Choose HNSW (speed) or DiskANN (memory efficiency) |
| **Simplicity** | REST API, single binary, minimal configuration |

---

## Cloud Performance

All benchmarks run on **AWS EC2 c6i.2xlarge** (8 vCPU, 16 GB RAM) with 100,000 384-dimensional vectors.

### Index Comparison

| Metric | HNSW | DiskANN |
|--------|------|---------|
| **Upsert Throughput** | 423,476 vec/sec | 416,607 vec/sec |
| **Search Throughput** | 441 queries/sec | 422 queries/sec |
| **Search P50 Latency** | 2.25ms | 2.32ms |
| **Search P99 Latency** | 2.67ms | 3.90ms |
| **Memory Usage** | ~200 MB per 1M vectors | ~20 MB per 1M vectors |
| **Max Dataset Size** | ~10M vectors (RAM bound) | Billions (disk bound) |

### Which Index Should I Use?

**HNSW (Default)** - Best for most use cases
- Lowest latency (<3ms p99)
- Highest throughput
- Dataset fits in RAM

**DiskANN** - Best for large-scale or memory-constrained deployments
- 10x lower memory footprint
- Scales to billions of vectors
- Near-HNSW performance

---

## Cost Analysis

Running Vortex on AWS:

| Instance | vCPU | RAM | Hourly Cost | Vectors Supported | Use Case |
|----------|------|-----|-------------|-------------------|----------|
| c6i.large | 2 | 4 GB | $0.085 | ~1M (HNSW) | Development |
| c6i.xlarge | 4 | 8 GB | $0.17 | ~5M (HNSW) | Small production |
| c6i.2xlarge | 8 | 16 GB | $0.34 | ~10M (HNSW) | Medium production |
| c6i.4xlarge | 16 | 32 GB | $0.68 | ~50M (HNSW) | Large production |
| c6i.2xlarge | 8 | 16 GB | $0.34 | ~100M (DiskANN) | Large scale, memory-efficient |

**Example monthly costs** (on-demand, us-east-1):
- 10M vectors with HNSW: ~$250/month (c6i.2xlarge)
- 100M vectors with DiskANN: ~$250/month (c6i.2xlarge + EBS)

---

## Quick Start

### Installation

```bash
git clone https://github.com/AmirFone/vortex.git
cd vortex
cargo build --release
cargo run --release
```

### Insert Vectors

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

### Search

```bash
curl -X POST http://localhost:3000/tenants/1/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "k": 10
  }'
```

### Response

```json
{
  "results": [
    {"id": 42, "score": 0.95},
    {"id": 17, "score": 0.89}
  ],
  "latency_ms": 2
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tenants/:id/upsert` | Insert or update vectors |
| `POST` | `/tenants/:id/search` | Find k nearest neighbors |
| `GET` | `/tenants/:id/stats` | Get index statistics |
| `POST` | `/tenants/:id/flush` | Force flush to index |
| `GET` | `/health` | Health check |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      REST API (Axum)                        │
│                   POST /tenants/:id/search                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Engine                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Tenant 1   │  │  Tenant 2   │  │  Tenant N   │         │
│  │  (isolated) │  │  (isolated) │  │  (isolated) │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Per-Tenant State                         │
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │ Write Buffer │───▶│  ANN Index   │    │ Vector Store │ │
│   │  (in-memory) │    │ HNSW/DiskANN │    │   (mmap)     │ │
│   └──────────────┘    └──────────────┘    └──────────────┘ │
│                              │                              │
└──────────────────────────────│──────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Durability Layer                         │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Write-Ahead Log (WAL)                  │  │
│   │         CRC32 checksums • fsync on commit           │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Write Path**: Upsert → WAL (durability) → Write Buffer → Background flush to ANN Index
2. **Read Path**: Query → Search ANN Index + Write Buffer → Merge & rank results
3. **Recovery**: On startup, replay WAL entries to rebuild write buffer state

### Key Components

| Component | Purpose |
|-----------|---------|
| **Engine** | Manages tenant lifecycle and background tasks |
| **Tenant State** | Isolated data per tenant (no cross-tenant leakage) |
| **Write Buffer** | Batches recent writes before index insertion |
| **ANN Index** | HNSW or DiskANN graph for approximate nearest neighbor search |
| **Vector Store** | Memory-mapped storage for raw vector data |
| **WAL** | Write-ahead log ensuring crash recovery |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3000` | Server port |
| `DIMENSIONS` | `384` | Vector dimensionality |
| `INDEX_TYPE` | `hnsw` | Index backend: `hnsw` or `diskann` |
| `FLUSH_INTERVAL_SECS` | `30` | Auto-flush interval |

### HNSW Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Graph connectivity (higher = better recall, more memory) |
| `ef_construction` | 200 | Build quality (higher = better graph, slower build) |
| `ef` | 100 | Search quality (higher = better recall, slower search) |

### DiskANN Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DISKANN_MAX_DEGREE` | 64 | Graph connections per node |
| `DISKANN_ALPHA` | 1.2 | Pruning aggressiveness |
| `DISKANN_SEARCH_BEAM` | 64 | Search beam width |

---

## Running Benchmarks

### Cloud Benchmark (EC2)

Provisions an EC2 instance, runs the benchmark, and cleans up:

```bash
# HNSW benchmark
cargo run --release --bin cloud_benchmark --features aws-storage

# DiskANN benchmark
cargo run --release --bin cloud_benchmark --features aws-storage -- --index-type diskann

# Custom configuration
cargo run --release --bin cloud_benchmark --features aws-storage -- \
    --vectors 500000 \
    --instance-type c6i.4xlarge \
    --index-type hnsw
```

---

## Testing

112 tests covering ACID compliance, concurrency, crash recovery, and correctness:

```bash
cargo test                           # Run all tests
cargo test --test acid_tests         # ACID compliance
cargo test --test concurrency_tests  # Multi-threaded operations
cargo test --test stress_tests       # Load testing
```

---

## Project Structure

```
src/
├── api/          # REST endpoints (Axum)
├── engine/       # Top-level orchestration
├── index/        # Pluggable ANN backends (HNSW, DiskANN)
├── hnsw/         # HNSW implementation
├── storage/      # Storage backends (local, AWS)
├── tenant/       # Per-tenant state
├── vectors/      # Vector storage (mmap)
├── wal/          # Write-ahead log
└── config.rs     # Configuration

tests/            # Integration tests
benches/          # Performance benchmarks
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
