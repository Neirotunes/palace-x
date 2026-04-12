// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Palace Graph: HNSW index with UMA-native cache-aware extensions
//!
//! This crate implements Hierarchical Navigable Small World (HNSW) graph search
//! with α-RNG neighbor selection (Vamana-style) and UMA-optimized hot/cold tier
//! layout for Apple Silicon's unified memory architecture.
//!
//! ## Modules
//! - `hnsw` — Core HNSW index with concurrent DashMap + ArcSwap snapshot
//! - `uma_hnsw` — Cache-aware layer layout + speculative ARM64 prefetch
//! - `nsw` — Legacy flat NSW (deprecated, kept for compatibility)
//! - `node` — Vector node types and distance functions
//! - `heuristic` — Hub-Highway heuristics

pub mod heuristic;
pub mod hnsw;
pub mod node;
pub mod nsw;
pub mod uma_hnsw;

pub use hnsw::{HnswDistanceMetric, HnswIndex, HnswNode};
pub use node::{GraphNode, MetaData};
pub use nsw::{DistanceMetric, NswIndex};
pub use palace_core::NodeId;
pub use uma_hnsw::{CacheTier, HnswPrefetcher, HotTierStore};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion_and_search() {
        let index = NswIndex::new(128, 32, 200);

        // Insert a single vector
        let vector = vec![0.1; 128];
        let metadata = MetaData {
            label: "test".to_string(),
        };
        let id = index.insert(vector, metadata);
        assert_eq!(id, NodeId(0));
    }
}
