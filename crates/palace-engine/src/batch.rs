// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! Batch operations for efficient concurrent ingestion
//!
//! Provides helpers for ingesting multiple vectors concurrently.

use crate::engine::PalaceEngine;
use palace_core::{MemoryError, MetaData, NodeId};
use tracing::debug;

impl PalaceEngine {
    /// Ingest multiple vectors concurrently
    ///
    /// Note: The engine processes commands sequentially internally,
    /// so this pipelined approach provides better throughput for many small
    /// ingestion requests by overlapping send operations.
    ///
    /// # Arguments
    /// * `items` - Vector of (embedding, metadata) tuples
    ///
    /// # Returns
    /// Vector of results in the same order as input items
    pub async fn ingest_batch(
        &self,
        items: Vec<(Vec<f32>, MetaData)>,
    ) -> Vec<Result<NodeId, MemoryError>> {
        let len = items.len();
        debug!("Starting batch ingest of {} items", len);

        let futures: Vec<_> = items
            .into_iter()
            .map(|(vector, metadata)| self.ingest(vector, metadata))
            .collect();

        let results = futures::future::join_all(futures).await;

        debug!("Batch ingest completed: {} items processed", len);
        results
    }

    /// Check if any item in the batch failed
    pub fn batch_has_errors(results: &[Result<NodeId, MemoryError>]) -> bool {
        results.iter().any(Result::is_err)
    }

    /// Extract error count from batch results
    pub fn batch_error_count(results: &[Result<NodeId, MemoryError>]) -> usize {
        results.iter().filter(|r| r.is_err()).count()
    }

    /// Extract successful node IDs from batch results
    pub fn batch_success_ids(results: &[Result<NodeId, MemoryError>]) -> Vec<NodeId> {
        results
            .iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use palace_core::SearchConfig;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_ingest() {
        let engine = PalaceEngine::start(8);

        let items = vec![
            (
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                MetaData::new(1, "batch1"),
            ),
            (
                vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                MetaData::new(2, "batch2"),
            ),
            (
                vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                MetaData::new(3, "batch3"),
            ),
        ];

        let results = engine.ingest_batch(items).await;

        assert_eq!(results.len(), 3);
        assert!(!PalaceEngine::batch_has_errors(&results));

        let ids = PalaceEngine::batch_success_ids(&results);
        assert_eq!(ids.len(), 3);

        let _ = engine.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_with_search() {
        let engine = PalaceEngine::start(4);

        let items = vec![
            (vec![1.0, 0.0, 0.0, 0.0], MetaData::new(1, "item1")),
            (vec![0.0, 1.0, 0.0, 0.0], MetaData::new(2, "item2")),
            (vec![0.0, 0.0, 1.0, 0.0], MetaData::new(3, "item3")),
        ];

        let results = engine.ingest_batch(items).await;
        assert!(!PalaceEngine::batch_has_errors(&results));

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let config = SearchConfig::default_with_limit(2);
        let search_results = engine.search(query, config).await;

        assert!(search_results.is_ok());
        if let Ok(fragments) = search_results {
            assert!(!fragments.is_empty());
        }

        let _ = engine.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_batch_ingestion() {
        let engine = std::sync::Arc::new(PalaceEngine::start(4));

        let mut handles = vec![];

        for i in 0..4 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let items = vec![
                    (vec![0.1, 0.2, 0.3, 0.4], MetaData::new(i * 10 + 1, "task")),
                    (vec![0.2, 0.3, 0.4, 0.5], MetaData::new(i * 10 + 2, "task")),
                ];
                engine_clone.ingest_batch(items).await
            });
            handles.push(handle);
        }

        let mut total_ingested = 0;
        for handle in handles {
            let results = handle.await.unwrap();
            total_ingested += results.len();
        }

        assert_eq!(total_ingested, 8);

        // Arc::into_inner will be called when all clones drop
        let engine = std::sync::Arc::try_unwrap(engine)
            .unwrap_or_else(|_| panic!("All Arc clones should have been dropped"));
        let _ = engine.shutdown().await;
    }
}
