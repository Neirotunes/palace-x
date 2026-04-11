use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique node identifier in the hierarchical memory system
#[derive(
    Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize,
)]
pub struct NodeId(pub u64);

/// Metadata associated with a memory fragment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaData {
    /// Timestamp when the fragment was created (unix timestamp in seconds)
    pub timestamp: u64,

    /// Source of the fragment (e.g., "user_input", "internal_reasoning")
    pub source: String,

    /// Searchable tags for categorization
    pub tags: Vec<String>,

    /// Additional flexible metadata
    pub extra: HashMap<String, String>,
}

impl MetaData {
    /// Create a new metadata entry with minimal fields
    pub fn new(timestamp: u64, source: impl Into<String>) -> Self {
        Self {
            timestamp,
            source: source.into(),
            tags: Vec::new(),
            extra: HashMap::new(),
        }
    }
}

/// A search result fragment from the memory index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    /// The node identifier
    pub node_id: NodeId,

    /// Relevance score (typically 0.0-1.0)
    pub score: f32,

    /// Associated metadata
    pub metadata: MetaData,

    /// Optional embedding vector
    pub vector: Option<Vec<f32>>,
}

/// Configuration for search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum number of results to return
    pub limit: usize,

    /// Enable two-stage reranking (coarse + topological refinement)
    pub enable_reranking: bool,

    /// Weight for cosine similarity in scoring (0.0-1.0)
    pub alpha: f32,

    /// Weight for topological distance in scoring (0.0-1.0)
    pub beta: f32,

    /// Number of candidates to consider in Stage 2 (reranking)
    pub rerank_k: usize,
}

impl SearchConfig {
    /// Create a default search configuration
    pub fn default_with_limit(limit: usize) -> Self {
        Self {
            limit,
            enable_reranking: true,
            alpha: 0.7,
            beta: 0.3,
            rerank_k: limit * 2,
        }
    }

    /// Validate that weights sum appropriately
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err("alpha must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.beta) {
            return Err("beta must be between 0.0 and 1.0".to_string());
        }
        if self.limit == 0 {
            return Err("limit must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Vector compression methods for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression, store full float32 vectors
    Binary,

    /// Ternary quantization: -1, 0, +1 values
    Ternary,

    /// Bit-plane encoding with configurable mantissa bits
    BitPlane { mantissa_bits: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_ordering() {
        let id1 = NodeId(1);
        let id2 = NodeId(2);
        assert!(id1 < id2);
    }

    #[test]
    fn test_search_config_validation() {
        let valid = SearchConfig {
            limit: 10,
            enable_reranking: true,
            alpha: 0.5,
            beta: 0.5,
            rerank_k: 20,
        };
        assert!(valid.validate().is_ok());

        let invalid_alpha = SearchConfig {
            limit: 10,
            enable_reranking: true,
            alpha: 1.5,
            beta: 0.5,
            rerank_k: 20,
        };
        assert!(invalid_alpha.validate().is_err());
    }

    #[test]
    fn test_metadata_creation() {
        let meta = MetaData::new(1234567890, "test_source");
        assert_eq!(meta.timestamp, 1234567890);
        assert_eq!(meta.source, "test_source");
        assert!(meta.tags.is_empty());
    }
}
