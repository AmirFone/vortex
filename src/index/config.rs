//! Index configuration types
//!
//! Provides configuration structures for different index backends.

use crate::hnsw::HnswConfig;

/// Configuration for ANN index backends
#[derive(Clone, Debug)]
pub enum AnnIndexConfig {
    /// HNSW (Hierarchical Navigable Small World) - in-memory graph
    Hnsw(HnswParams),
    /// DiskANN - disk-based Vamana graph
    #[cfg(feature = "diskann-index")]
    DiskAnn(DiskAnnParams),
}

impl Default for AnnIndexConfig {
    fn default() -> Self {
        AnnIndexConfig::Hnsw(HnswParams::default())
    }
}

impl AnnIndexConfig {
    /// Create HNSW config with default parameters
    pub fn hnsw() -> Self {
        AnnIndexConfig::Hnsw(HnswParams::default())
    }

    /// Create HNSW config with custom M parameter
    pub fn hnsw_with_m(m: usize) -> Self {
        AnnIndexConfig::Hnsw(HnswParams::with_m(m))
    }

    /// Create DiskANN config with default parameters
    #[cfg(feature = "diskann-index")]
    pub fn diskann(dims: usize) -> Self {
        AnnIndexConfig::DiskAnn(DiskAnnParams::new(dims))
    }

    /// Get the index type name
    pub fn index_type_name(&self) -> &'static str {
        match self {
            AnnIndexConfig::Hnsw(_) => "HNSW",
            #[cfg(feature = "diskann-index")]
            AnnIndexConfig::DiskAnn(_) => "DiskANN",
        }
    }
}

/// HNSW index parameters
#[derive(Clone, Debug)]
pub struct HnswParams {
    /// Max connections per node (M parameter)
    pub m: usize,
    /// Max connections at layer 0 (usually 2*M)
    pub m_max0: usize,
    /// Search width during construction
    pub ef_construction: usize,
    /// Level multiplier (1/ln(M))
    pub ml: f64,
    /// Default search ef (can be overridden per query)
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 500,
            ml: 1.0 / (m as f64).ln(),
            ef_search: 200,
        }
    }
}

impl HnswParams {
    /// Create with custom M parameter
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 500,
            ml: 1.0 / (m as f64).ln(),
            ef_search: 200,
        }
    }

    /// Convert to HnswConfig (for internal use)
    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ml: self.ml,
            ef_search: self.ef_search,
        }
    }
}

impl From<HnswParams> for HnswConfig {
    fn from(params: HnswParams) -> Self {
        params.to_hnsw_config()
    }
}

impl From<HnswConfig> for HnswParams {
    fn from(config: HnswConfig) -> Self {
        Self {
            m: config.m,
            m_max0: config.m_max0,
            ef_construction: config.ef_construction,
            ml: config.ml,
            ef_search: config.ef_search,
        }
    }
}

/// DiskANN index parameters
#[cfg(feature = "diskann-index")]
#[derive(Clone, Debug)]
pub struct DiskAnnParams {
    /// Vector dimensions
    pub dims: usize,
    /// Maximum out-degree of the graph (R parameter)
    pub max_degree: usize,
    /// Alpha parameter for pruning (typically 1.0-1.5)
    pub alpha: f32,
    /// Beam width during index construction (L_build)
    pub build_beam_width: usize,
    /// Beam width during search (L_search)
    pub search_beam_width: usize,
    /// Whether to use PQ (Product Quantization) for compressed search
    pub use_pq: bool,
    /// Number of PQ subspaces (if using PQ)
    pub pq_subspaces: usize,
}

#[cfg(feature = "diskann-index")]
impl DiskAnnParams {
    /// Create with required dimensions
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            max_degree: 64,
            alpha: 1.2,
            build_beam_width: 128,
            search_beam_width: 64,
            use_pq: false,
            pq_subspaces: 0,
        }
    }

    /// Set max degree
    pub fn with_max_degree(mut self, max_degree: usize) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set alpha
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set search beam width
    pub fn with_search_beam_width(mut self, width: usize) -> Self {
        self.search_beam_width = width;
        self
    }
}

#[cfg(feature = "diskann-index")]
impl Default for DiskAnnParams {
    fn default() -> Self {
        Self::new(384) // Default to common embedding dimension
    }
}
