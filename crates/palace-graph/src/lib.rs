// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Palace Graph: Navigable Small World (NSW) index with Hub-Highway optimization
//!
//! This crate implements a flat NSW approach from the paper "Down with the Hierarchy"
//! (arXiv:2412.01940) with Hub-Highway optimization for the Palace-X project.

pub mod heuristic;
pub mod node;
pub mod nsw;

pub use node::{GraphNode, MetaData};
pub use nsw::{DistanceMetric, NswIndex};
pub use palace_core::NodeId;

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
