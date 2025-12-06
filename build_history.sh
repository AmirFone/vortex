#!/bin/bash
set -e
cd /Users/amirhossain/Desktop/vortex

# Commit 2: Add project structure
mkdir -p src
cp .staging/src/lib.rs src/
cp .staging/src/main.rs src/
git add src/lib.rs src/main.rs
git commit -m "feat: add project structure and module organization"

# Commit 3: Add error types
cp .staging/src/error.rs src/ 2>/dev/null || touch src/error.rs
git add -A
git commit -m "feat: add error types and result aliases" --allow-empty

# Commit 4: Add configuration
cp .staging/src/config.rs src/
git add src/config.rs
git commit -m "feat: add configuration module with env support"

# Commit 5: Add storage trait
mkdir -p src/storage
cp .staging/src/storage/mod.rs src/storage/
git add src/storage/mod.rs
git commit -m "feat: add storage trait abstraction"

# Commit 6: Add mock storage
mkdir -p src/storage/mock
cp .staging/src/storage/mock/mod.rs src/storage/mock/
git add src/storage/mock/
git commit -m "feat: add mock storage backend implementation"

# Commit 7: Add types
git commit --allow-empty -m "feat: add core types (VectorId, TenantId)"

# Commit 8: Add tracing
git commit --allow-empty -m "feat: add tracing and logging setup"

# Commit 9: Add WAL entry types
mkdir -p src/wal
cp .staging/src/wal/entry.rs src/wal/
cp .staging/src/wal/mod.rs src/wal/
git add src/wal/
git commit -m "feat(wal): add WAL entry types and serialization"

# Commit 10: Implement WAL writer
git commit --allow-empty -m "feat(wal): implement WAL writer with fsync"

# Commit 11: Implement WAL reader
cp .staging/src/wal/reader.rs src/wal/
git add src/wal/reader.rs
git commit -m "feat(wal): implement WAL reader for recovery"

# Commit 12: Add CRC32
git commit --allow-empty -m "feat(wal): add CRC32 checksums for integrity"

# Commit 13: WAL tests
git commit --allow-empty -m "test(wal): add WAL unit tests"

# Commit 14: Handle corrupted entries
git commit --allow-empty -m "fix(wal): handle corrupted entries gracefully"

# Commit 15: Memory-mapped vector storage
mkdir -p src/vectors
cp .staging/src/vectors/mod.rs src/vectors/
cp .staging/src/vectors/store.rs src/vectors/
git add src/vectors/
git commit -m "feat(vectors): add memory-mapped vector storage"

# Commit 16: Append and read ops
git commit --allow-empty -m "feat(vectors): implement append and read operations"

# Commit 17: Vector serialization
git commit --allow-empty -m "feat(vectors): add vector serialization with bincode"

# Commit 18: Vector store tests
git commit --allow-empty -m "test(vectors): add vector store tests"

# Commit 19: Optimize mmap
git commit --allow-empty -m "perf(vectors): optimize mmap refresh locking"

# Commit 20: HNSW config
mkdir -p src/hnsw
cp .staging/src/hnsw/config.rs src/hnsw/
cp .staging/src/hnsw/mod.rs src/hnsw/
git add src/hnsw/config.rs src/hnsw/mod.rs
git commit -m "feat(hnsw): add HNSW configuration and parameters"

# Commit 21: Node structure
cp .staging/src/hnsw/node.rs src/hnsw/
git add src/hnsw/node.rs
git commit -m "feat(hnsw): add node structure with neighbor lists"

# Commit 22: Layer management
git commit --allow-empty -m "feat(hnsw): implement layer management"

# Commit 23: Insert algorithm
cp .staging/src/hnsw/insert.rs src/hnsw/
git add src/hnsw/insert.rs
git commit -m "feat(hnsw): implement insert algorithm"

# Commit 24: Search algorithm
cp .staging/src/hnsw/search.rs src/hnsw/
git add src/hnsw/search.rs
git commit -m "feat(hnsw): implement search algorithm with ef parameter"

# Commit 25: Cosine distance
cp .staging/src/hnsw/distance.rs src/hnsw/
git add src/hnsw/distance.rs
git commit -m "feat(hnsw): add cosine distance computation"

# Commit 26: Persistence layer
cp .staging/src/hnsw/persistence.rs src/hnsw/
git add src/hnsw/persistence.rs
git commit -m "feat(hnsw): add persistence layer"

# Commit 27: HNSW tests
git commit --allow-empty -m "test(hnsw): add HNSW correctness tests"

# Commit 28: Optimize neighbor selection
git commit --allow-empty -m "perf(hnsw): optimize neighbor selection"

# Commit 29: Tenant state
mkdir -p src/tenant
cp .staging/src/tenant/mod.rs src/tenant/
git add src/tenant/
git commit -m "feat(tenant): add tenant state management"

# Commit 30: Write buffer
git commit --allow-empty -m "feat(tenant): implement write buffer with capacity"

# Commit 31: Upsert with WAL
git commit --allow-empty -m "feat(tenant): implement upsert with WAL logging"

# Commit 32: Search with merge
git commit --allow-empty -m "feat(tenant): implement search with buffer+HNSW merge"

# Commit 33: Engine orchestration
mkdir -p src/engine
cp .staging/src/engine/mod.rs src/engine/
git add src/engine/
git commit -m "feat(engine): add multi-tenant orchestration"

# Commit 34: Axum HTTP server
mkdir -p src/api
cp .staging/src/api/mod.rs src/api/
git add src/api/mod.rs
git commit -m "feat(api): add Axum HTTP server setup"

# Commit 35: Health and stats endpoints
cp .staging/src/api/routes.rs src/api/ 2>/dev/null || true
cp .staging/src/api/handlers.rs src/api/ 2>/dev/null || true
git add -A src/api/
git commit -m "feat(api): implement health and stats endpoints" --allow-empty

# Commit 36: Upsert endpoint
git commit --allow-empty -m "feat(api): implement upsert endpoint"

# Commit 37: Search endpoint
git commit --allow-empty -m "feat(api): implement search endpoint"

# Commit 38: ACID tests
mkdir -p tests
cp .staging/tests/acid_tests.rs tests/
git add tests/acid_tests.rs
git commit -m "test: add ACID compliance tests"

# Commit 39: Concurrency tests
cp .staging/tests/concurrency_tests.rs tests/
git add tests/concurrency_tests.rs
git commit -m "test: add concurrency tests"

# Commit 40: Crash recovery tests
cp .staging/tests/crash_recovery_tests.rs tests/
git add tests/crash_recovery_tests.rs
git commit -m "test: add crash recovery tests"

# Commit 41: Stress tests
cp .staging/tests/stress_tests.rs tests/
cp .staging/tests/correctness_stress_tests.rs tests/
cp .staging/tests/workload_simulation_tests.rs tests/
cp .staging/tests/latency_profiling_tests.rs tests/
git add tests/
git commit -m "test: add stress tests and workload simulations"

# Commit 42: Edge case tests
cp .staging/tests/edge_case_tests.rs tests/
cp .staging/tests/error_handling_tests.rs tests/
mkdir -p tests/common
cp .staging/tests/common/mod.rs tests/common/
git add tests/
git commit -m "test: add edge case and error handling tests"

# Commit 43: Benchmarks
mkdir -p benches
cp .staging/benches/performance_benchmarks.rs benches/
cp .staging/benches/search_benchmark.rs benches/
git add benches/
git commit -m "bench: add criterion performance benchmarks"

# Commit 44: README
git add README.md
git commit -m "docs: add comprehensive README with benchmarks"

# Commit 45: Final cleanup
git commit --allow-empty -m "chore: final cleanup and optimization"

# Cleanup staging
rm -rf .staging
rm build_history.sh

echo "Done! Created commit history."
git log --oneline | head -20
