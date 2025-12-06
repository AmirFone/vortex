//! Comprehensive Performance Benchmarks for VectorDB
//!
//! Benchmarks for all major components:
//! - WAL (Write-Ahead Log)
//! - VectorStore
//! - HNSW Index
//! - Tenant operations
//!
//! Run with: cargo bench --bench performance_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use vectordb::hnsw::HnswConfig;
use vectordb::storage::mock::{MockBlockStorage, MockStorageConfig};
use vectordb::tenant::TenantState;

const TEST_DIMS: usize = 384;

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn generate_random_vector(dims: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
    normalize(&v)
}

// =============================================================================
// UPSERT BENCHMARKS
// =============================================================================

fn bench_upsert_batch_sizes(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("upsert_batch");

    for batch_size in [1, 10, 50, 100, 500, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                // Create fresh tenant for each batch size
                let tenant = rt.block_on(async {
                    let storage =
                        Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
                    let config = HnswConfig::new(TEST_DIMS);
                    TenantState::open(1, TEST_DIMS, storage.clone(), config)
                        .await
                        .unwrap()
                });

                let mut id_counter = 0u64;

                b.iter(|| {
                    rt.block_on(async {
                        let vectors: Vec<(u64, Vec<f32>)> = (0..size)
                            .map(|i| {
                                id_counter += 1;
                                (id_counter + i as u64, generate_random_vector(TEST_DIMS))
                            })
                            .collect();

                        black_box(tenant.upsert(vectors).await.unwrap())
                    })
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SEARCH BENCHMARKS
// =============================================================================

fn bench_search_ef_values(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Pre-populate index
    let (tenant, storage) = rt.block_on(async {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = HnswConfig::new(TEST_DIMS);
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap();

        // Insert 5000 vectors
        for batch in 0..50 {
            let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                .map(|i| ((batch * 100 + i) as u64, generate_random_vector(TEST_DIMS)))
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }

        // Flush to HNSW
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        (tenant, storage)
    });

    let mut group = c.benchmark_group("search_ef");

    for ef in [50, 100, 200, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, &ef| {
            b.iter(|| {
                let query = generate_random_vector(TEST_DIMS);
                black_box(tenant.search(&query, 10, Some(ef)))
            });
        });
    }

    group.finish();
}

fn bench_search_k_values(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (tenant, _storage) = rt.block_on(async {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = HnswConfig::new(TEST_DIMS);
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap();

        for batch in 0..50 {
            let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                .map(|i| ((batch * 100 + i) as u64, generate_random_vector(TEST_DIMS)))
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }

        tenant.flush_to_hnsw(&*storage).await.unwrap();
        (tenant, storage)
    });

    let mut group = c.benchmark_group("search_k");

    for k in [1, 5, 10, 20, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let query = generate_random_vector(TEST_DIMS);
                black_box(tenant.search(&query, k, Some(100)))
            });
        });
    }

    group.finish();
}

fn bench_search_index_sizes(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("search_index_size");

    for index_size in [1000, 2000, 5000, 10000] {
        let (tenant, _storage) = rt.block_on(async {
            let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
            let config = HnswConfig::new(TEST_DIMS);
            let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
                .await
                .unwrap();

            for batch in 0..(index_size / 100) {
                let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                    .map(|i| ((batch * 100 + i) as u64, generate_random_vector(TEST_DIMS)))
                    .collect();
                tenant.upsert(vectors).await.unwrap();
            }

            tenant.flush_to_hnsw(&*storage).await.unwrap();
            (tenant, storage)
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(index_size),
            &index_size,
            |b, _| {
                b.iter(|| {
                    let query = generate_random_vector(TEST_DIMS);
                    black_box(tenant.search(&query, 10, Some(100)))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// FLUSH BENCHMARKS
// =============================================================================

fn bench_flush_sizes(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("flush");

    for buffer_size in [100, 500, 1000] {
        group.throughput(Throughput::Elements(buffer_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage =
                            Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
                        let config = HnswConfig::new(TEST_DIMS);
                        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
                            .await
                            .unwrap();

                        // Fill buffer
                        let vectors: Vec<(u64, Vec<f32>)> = (0..size)
                            .map(|i| (i as u64, generate_random_vector(TEST_DIMS)))
                            .collect();
                        tenant.upsert(vectors).await.unwrap();

                        // Flush
                        black_box(tenant.flush_to_hnsw(&*storage).await.unwrap())
                    })
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// DISTANCE COMPUTATION BENCHMARKS
// =============================================================================

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    // Test different dimensionalities
    for dims in [128, 256, 384, 768, 1536] {
        group.throughput(Throughput::Elements(dims as u64));

        let a = generate_random_vector(dims);
        let b = generate_random_vector(dims);

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b_bench, _| {
            b_bench.iter(|| {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                black_box(1.0 - dot)
            });
        });
    }

    group.finish();
}

// =============================================================================
// WRITE BUFFER SEARCH BENCHMARKS
// =============================================================================

fn bench_write_buffer_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("write_buffer_search");

    for buffer_size in [100, 500, 1000, 2000, 5000] {
        let tenant = rt.block_on(async {
            let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
            let config = HnswConfig::new(TEST_DIMS);
            let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
                .await
                .unwrap();

            // Add to write buffer (don't flush)
            let vectors: Vec<(u64, Vec<f32>)> = (0..buffer_size)
                .map(|i| (i as u64, generate_random_vector(TEST_DIMS)))
                .collect();
            tenant.upsert(vectors).await.unwrap();

            tenant
        });

        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            &buffer_size,
            |b, _| {
                b.iter(|| {
                    let query = generate_random_vector(TEST_DIMS);
                    black_box(tenant.search(&query, 10, None))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// MIXED WORKLOAD BENCHMARKS
// =============================================================================

fn bench_mixed_read_write(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (tenant, storage) = rt.block_on(async {
        let storage = Arc::new(MockBlockStorage::temp(MockStorageConfig::fast()).unwrap());
        let config = HnswConfig::new(TEST_DIMS);
        let tenant = TenantState::open(1, TEST_DIMS, storage.clone(), config)
            .await
            .unwrap();

        // Pre-populate
        for batch in 0..20 {
            let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                .map(|i| ((batch * 100 + i) as u64, generate_random_vector(TEST_DIMS)))
                .collect();
            tenant.upsert(vectors).await.unwrap();
        }
        tenant.flush_to_hnsw(&*storage).await.unwrap();

        (tenant, storage)
    });

    let mut id_counter = 2000u64;

    c.bench_function("mixed_read_write", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Write
                id_counter += 1;
                let vectors = vec![(id_counter, generate_random_vector(TEST_DIMS))];
                tenant.upsert(vectors).await.unwrap();

                // Search
                let query = generate_random_vector(TEST_DIMS);
                black_box(tenant.search(&query, 10, None))
            })
        });
    });
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    upsert_benches,
    bench_upsert_batch_sizes,
);

criterion_group!(
    search_benches,
    bench_search_ef_values,
    bench_search_k_values,
    bench_search_index_sizes,
);

criterion_group!(
    flush_benches,
    bench_flush_sizes,
);

criterion_group!(
    compute_benches,
    bench_cosine_distance,
);

criterion_group!(
    buffer_benches,
    bench_write_buffer_search,
);

criterion_group!(
    mixed_benches,
    bench_mixed_read_write,
);

criterion_main!(
    upsert_benches,
    search_benches,
    flush_benches,
    compute_benches,
    buffer_benches,
    mixed_benches,
);
