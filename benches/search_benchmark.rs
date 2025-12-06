//! Performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vectordb::storage::mock::{create_temp_storage, MockStorageConfig};
use vectordb::{Config, VectorEngine};

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

fn bench_write_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Setup
    let (engine, _) = rt.block_on(async {
        let storage = create_temp_storage(MockStorageConfig::fast()).unwrap();
        let config = vectordb::config::EngineConfig::default();
        let engine = VectorEngine::new(config, storage).await.unwrap();
        (engine, ())
    });

    let mut id_counter = 0u64;

    c.bench_function("write_single_vector", |b| {
        b.iter(|| {
            rt.block_on(async {
                id_counter += 1;
                let vector = generate_random_vector(384);
                engine
                    .upsert(1, vec![(id_counter, vector)])
                    .await
                    .unwrap();
            })
        })
    });
}

fn bench_write_batch_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let (engine, _) = rt.block_on(async {
        let storage = create_temp_storage(MockStorageConfig::fast()).unwrap();
        let config = vectordb::config::EngineConfig::default();
        let engine = VectorEngine::new(config, storage).await.unwrap();
        (engine, ())
    });

    let mut batch_counter = 0u64;

    c.bench_function("write_batch_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start_id = batch_counter * 100;
                batch_counter += 1;
                let vectors: Vec<(u64, Vec<f32>)> = (0..100)
                    .map(|i| (start_id + i, generate_random_vector(384)))
                    .collect();
                engine.upsert(1, vectors).await.unwrap();
            })
        })
    });
}

fn bench_search_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let engine = rt.block_on(async {
        let storage = create_temp_storage(MockStorageConfig::fast()).unwrap();
        let config = vectordb::config::EngineConfig::default();
        let engine = VectorEngine::new(config, storage).await.unwrap();

        // Pre-populate with vectors
        let vectors: Vec<(u64, Vec<f32>)> = (0..1000)
            .map(|i| (i as u64, generate_random_vector(384)))
            .collect();
        engine.upsert(1, vectors).await.unwrap();

        // Flush to HNSW
        engine.flush_all().await.unwrap();

        engine
    });

    c.bench_function("search_k10", |b| {
        b.iter(|| {
            rt.block_on(async {
                let query = generate_random_vector(384);
                let results = engine.search(1, black_box(query), 10, None).await.unwrap();
                black_box(results);
            })
        })
    });
}

criterion_group!(
    benches,
    bench_write_latency,
    bench_write_batch_latency,
    bench_search_latency
);
criterion_main!(benches);
