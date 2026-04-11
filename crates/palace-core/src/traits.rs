// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::error::MemoryError;
use crate::types::{Fragment, MetaData, NodeId, SearchConfig};

/// The core trait for hierarchical memory operations in Palace-X
///
/// Implementations are expected to:
/// - Support O(log N) amortized ingestion via NSW (Navigable Small World) graphs
/// - Perform two-stage retrieval: coarse NSW search → topological reranking
/// - Provide GDPR-compliant deletion with async graph rebalancing
pub trait MemoryProvider: Send + Sync {
    /// Ingest a vector with metadata into the memory index.
    ///
    /// # Arguments
    /// * `vector` - The embedding vector to store
    /// * `meta` - Associated metadata
    ///
    /// # Returns
    /// The newly assigned `NodeId` or an error
    ///
    /// # Complexity
    /// O(log N) amortized time
    async fn ingest(&self, vector: Vec<f32>, meta: MetaData) -> Result<NodeId, MemoryError>;

    /// Retrieve fragments matching a query vector using two-stage search.
    ///
    /// Stage 1: Coarse search using NSW graph traversal
    /// Stage 2: Topological reranking (if enabled) to refine candidates
    ///
    /// # Arguments
    /// * `query` - The query embedding vector
    /// * `config` - Search configuration (limit, weights, reranking parameters)
    ///
    /// # Returns
    /// A vector of `Fragment`s sorted by combined score (higher is better)
    async fn retrieve(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<Fragment>, MemoryError>;

    /// Remove nodes and rebalance the graph asynchronously.
    ///
    /// This operation is designed to be GDPR-compliant by ensuring
    /// complete removal and graph consistency after deletion.
    ///
    /// # Arguments
    /// * `nodes` - IDs of nodes to delete
    ///
    /// # Returns
    /// The number of nodes successfully deleted
    async fn vacuum(&self, nodes: &[NodeId]) -> Result<u64, MemoryError>;

    /// Get the total number of nodes in the index
    async fn len(&self) -> usize;

    /// Check if the index is empty
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}
