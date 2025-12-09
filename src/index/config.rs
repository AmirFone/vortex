//! Index configuration types
//!
//! Provides configuration structures for different index backends.
//!
//! This module defines `HnswParams` as the single source of truth for HNSW
//! configuration. The `HnswConfig` type in `hnsw/mod.rs` is a type alias
//! that points to `HnswParams`.

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

    /// Create with custom parameters (builder pattern)
    pub fn builder() -> HnswParamsBuilder {
        HnswParamsBuilder::default()
    }
}

/// Builder for HnswParams with fluent API
#[derive(Clone, Debug, Default)]
pub struct HnswParamsBuilder {
    m: Option<usize>,
    m_max0: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
}

impl HnswParamsBuilder {
    /// Set M parameter (max connections per node)
    pub fn m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Set max connections at layer 0 (default: 2*M)
    pub fn m_max0(mut self, m_max0: usize) -> Self {
        self.m_max0 = Some(m_max0);
        self
    }

    /// Set ef_construction (build quality)
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = Some(ef);
        self
    }

    /// Set ef_search (search quality)
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Build the HnswParams
    pub fn build(self) -> HnswParams {
        let m = self.m.unwrap_or(16);
        HnswParams {
            m,
            m_max0: self.m_max0.unwrap_or(m * 2),
            ef_construction: self.ef_construction.unwrap_or(500),
            ml: 1.0 / (m as f64).ln(),
            ef_search: self.ef_search.unwrap_or(200),
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
