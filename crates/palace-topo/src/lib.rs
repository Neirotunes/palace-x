// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! palace-topo: Topological reranking module for the Palace-X project
//!
//! Implements the normalized Betti number β₁ metric for semantic structure-aware retrieval.
//! This module provides:
//! - Ego-graph construction (2-hop neighborhoods around node pairs)
//! - Topological invariants (β₀ connected components, β₁ first Betti number)
//! - Palace-X d_total distance metric combining cosine similarity with structural connectivity
//! - Topological reranking for Stage 2 candidate refinement

pub mod betti;
pub mod ego_cache;
pub mod ego_graph;
pub mod metric;
pub mod persistence;
pub mod reranker;

pub use betti::{beta_0, beta_1, euler_characteristic};
pub use ego_graph::EgoGraph;
pub use metric::d_total;
pub use persistence::PersistenceDiagram;
pub use reranker::TopologicalReranker;

/// Re-export NodeId from palace-core for convenience
pub use palace_core::NodeId;
