//! HTTP API
//!
//! REST endpoints:
//! - POST /v1/tenants/{id}/vectors - Upsert vectors
//! - POST /v1/tenants/{id}/search - Search k-NN
//! - GET /v1/tenants/{id}/stats - Get stats
//! - GET /health - Health check

use crate::config::ApiConfig;
use crate::engine::VectorEngine;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// API state
pub struct ApiState {
    pub engine: Arc<VectorEngine>,
}

/// Serve the API
pub async fn serve(engine: Arc<VectorEngine>, config: ApiConfig) -> anyhow::Result<()> {
    let state = Arc::new(ApiState { engine });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/tenants/:tenant_id/vectors", post(upsert_vectors))
        .route("/v1/tenants/:tenant_id/search", post(search))
        .route("/v1/tenants/:tenant_id/stats", get(get_stats))
        .route("/v1/admin/flush", post(flush_all))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// HANDLERS
// ============================================================================

/// Health check
async fn health(
    State(_state): State<Arc<ApiState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

/// Upsert vectors
#[axum::debug_handler]
async fn upsert_vectors(
    Path(tenant_id): Path<u64>,
    State(state): State<Arc<ApiState>>,
    Json(request): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, ApiError> {
    let start = Instant::now();

    // Validate input
    if request.vectors.is_empty() {
        return Err(ApiError::BadRequest("No vectors provided".into()));
    }

    // Convert to internal format
    let vectors: Vec<(u64, Vec<f32>)> = request
        .vectors
        .into_iter()
        .map(|v| (v.id, v.vector))
        .collect();

    // Upsert
    let result = state.engine.upsert(tenant_id, vectors).await?;

    let elapsed = start.elapsed();

    Ok(Json(UpsertResponse {
        count: result.count,
        sequence: result.sequence,
        latency_ms: elapsed.as_secs_f64() * 1000.0,
    }))
}

/// Search for similar vectors
async fn search(
    Path(tenant_id): Path<u64>,
    State(state): State<Arc<ApiState>>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let start = Instant::now();

    // Validate input
    if request.vector.is_empty() {
        return Err(ApiError::BadRequest("No query vector provided".into()));
    }

    let k = request.k.unwrap_or(10);
    let ef = request.ef;

    // Search
    let results = state.engine.search(tenant_id, request.vector, k, ef).await?;

    let elapsed = start.elapsed();

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(|r| SearchResultItem {
                id: r.id,
                similarity: r.similarity,
            })
            .collect(),
        latency_ms: elapsed.as_secs_f64() * 1000.0,
    }))
}

/// Get tenant stats
#[axum::debug_handler]
async fn get_stats(
    Path(tenant_id): Path<u64>,
    State(state): State<Arc<ApiState>>,
) -> Result<Json<StatsResponse>, ApiError> {
    let stats = state.engine.stats(tenant_id).await?;

    Ok(Json(StatsResponse {
        tenant_id: stats.tenant_id,
        vector_count: stats.vector_count,
        index_nodes: stats.index_nodes,
        write_buffer_size: stats.write_buffer_size,
        wal_sequence: stats.wal_sequence,
    }))
}

/// Flush all tenants
async fn flush_all(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<FlushResponse>, ApiError> {
    let count = state.engine.flush_all().await?;
    Ok(Json(FlushResponse { flushed: count }))
}

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
struct UpsertRequest {
    vectors: Vec<VectorItem>,
}

#[derive(Deserialize)]
struct VectorItem {
    id: u64,
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct UpsertResponse {
    count: usize,
    sequence: u64,
    latency_ms: f64,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: Option<usize>,
    ef: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
    latency_ms: f64,
}

#[derive(Serialize)]
struct SearchResultItem {
    id: u64,
    similarity: f32,
}

#[derive(Serialize)]
struct StatsResponse {
    tenant_id: u64,
    vector_count: u64,
    index_nodes: u64,
    write_buffer_size: usize,
    wal_sequence: u64,
}

#[derive(Serialize)]
struct FlushResponse {
    flushed: usize,
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    Internal(String),
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::Internal(e.to_string())
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = serde_json::json!({
            "error": message
        });

        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::storage::mock::{create_temp_storage, MockStorageConfig};

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[tokio::test]
    async fn test_api_integration() {
        // Create engine with 4 dimensions for test
        let storage = create_temp_storage(MockStorageConfig::fast()).unwrap();
        let mut config = EngineConfig::default();
        config.default_dims = 4;
        let engine = VectorEngine::new(config, storage).await.unwrap();

        // Upsert directly
        let vectors: Vec<(u64, Vec<f32>)> = (0..5)
            .map(|i| {
                let v = normalize(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                (100 + i as u64, v)
            })
            .collect();

        let result = engine.upsert(1, vectors).await.unwrap();
        assert_eq!(result.count, 5);

        // Search directly
        let query = normalize(&[1.0, 2.0, 3.0, 4.0]);
        let results = engine.search(1, query, 3, None).await.unwrap();
        assert!(!results.is_empty());
    }
}
