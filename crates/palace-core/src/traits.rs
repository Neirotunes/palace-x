// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::error::MemoryError;
use crate::types::{Fragment, MetaData, NodeId, SearchConfig};

/// The core trait for hierarchical memory operations in Palace-X
///
/// Implementations are expected to:
/// - Support O(log N) amortized ingestion via HNSW (Hierarchical Navigable Small World) graphs
/// - Perform two-stage retrieval: coarse HNSW search → topological reranking
/// - Provide GDPR-compliant deletion with async graph rebalancing
///
/// # Object Safety Note
/// This trait uses `async fn` syntax (stabilized in Rust 1.75). Using `dyn MemoryProvider`
/// requires boxing the futures or using the `async_trait` macro. For `impl MemoryProvider`
/// (static dispatch) no boxing is required.
#[allow(async_fn_in_trait)]
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
    /// Stage 1: Coarse search using HNSW graph traversal (O(ef · log N))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MemoryError;
    use crate::types::{Fragment, MetaData, NodeId, SearchConfig};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Minimal in-memory provider for trait contract tests.
    struct MockProvider {
        count: AtomicU64,
    }

    impl MockProvider {
        fn new() -> Self {
            Self { count: AtomicU64::new(0) }
        }
    }

    impl MemoryProvider for MockProvider {
        async fn ingest(&self, vector: Vec<f32>, _meta: MetaData) -> Result<NodeId, MemoryError> {
            if vector.is_empty() {
                return Err(MemoryError::DimensionMismatch { expected: 1, got: 0 });
            }
            let id = self.count.fetch_add(1, Ordering::SeqCst);
            Ok(NodeId(id))
        }

        async fn retrieve(
            &self,
            _query: &[f32],
            config: &SearchConfig,
        ) -> Result<Vec<Fragment>, MemoryError> {
            config.validate().map_err(MemoryError::StorageError)?;
            Ok(Vec::new())
        }

        async fn vacuum(&self, nodes: &[NodeId]) -> Result<u64, MemoryError> {
            Ok(nodes.len() as u64)
        }

        async fn len(&self) -> usize {
            self.count.load(Ordering::SeqCst) as usize
        }
    }

    #[tokio::test]
    async fn test_is_empty_default_impl() {
        let provider = MockProvider::new();
        // Default is_empty() is implemented in terms of len()
        assert!(provider.is_empty().await, "freshly-created provider must be empty");

        provider.ingest(vec![1.0], MetaData::new(0, "t")).await.unwrap();
        assert!(!provider.is_empty().await, "after ingest, provider must be non-empty");
    }

    #[tokio::test]
    async fn test_ingest_returns_sequential_ids() {
        let provider = MockProvider::new();
        let id0 = provider.ingest(vec![1.0], MetaData::new(0, "a")).await.unwrap();
        let id1 = provider.ingest(vec![2.0], MetaData::new(0, "b")).await.unwrap();
        assert_eq!(id0.0, 0, "first id must be 0");
        assert_eq!(id1.0, 1, "second id must be 1");
    }

    #[tokio::test]
    async fn test_ingest_dimension_error_is_propagated() {
        let provider = MockProvider::new();
        let result = provider.ingest(vec![], MetaData::new(0, "bad")).await;
        assert!(result.is_err(), "empty vector must produce an error");
        assert!(matches!(result.unwrap_err(), MemoryError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn test_retrieve_validates_config() {
        let provider = MockProvider::new();
        let bad_config = SearchConfig {
            limit: 0, // invalid: must be > 0
            enable_reranking: false,
            alpha: 0.7,
            beta: 0.3,
            rerank_k: 10,
        };
        let result = provider.retrieve(&[1.0], &bad_config).await;
        assert!(result.is_err(), "limit=0 must produce a validation error");
    }

    #[tokio::test]
    async fn test_vacuum_reports_count() {
        let provider = MockProvider::new();
        let deleted = provider
            .vacuum(&[NodeId(0), NodeId(1), NodeId(2)])
            .await
            .unwrap();
        assert_eq!(deleted, 3, "vacuum should report each node as deleted");
    }

    #[tokio::test]
    async fn test_len_reflects_ingests() {
        let provider = MockProvider::new();
        assert_eq!(provider.len().await, 0);
        for i in 0..5u64 {
            provider.ingest(vec![i as f32], MetaData::new(i, "x")).await.unwrap();
        }
        assert_eq!(provider.len().await, 5);
    }
}
