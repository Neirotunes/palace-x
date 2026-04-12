// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! # palace-storage
//!
//! Concrete in-memory MemoryProvider implementation for the Palace-X project.
//!
//! This crate combines:
//! - **HNSW Index** (palace-graph): Hierarchical Navigable Small World — 99.8% R@10
//! - **Topological Reranking** (palace-topo): Structure-aware candidate refinement
//! - **Bit-Plane Storage** (palace-bitplane): Precision-proportional vector retrieval
//!
//! The result is a two-stage retrieval pipeline:
//! 1. **Stage 1 (Coarse)**: HNSW hierarchical beam search with L2 distance
//! 2. **Stage 2 (Fine)**: Topological reranking with ego-graphs and d_total metric
//!
//! This enables fast, high-quality approximate search over hierarchical memory in autonomous agents.
//!
//! ## v0.2 Changes
//! - Replaced flat NSW (~1% recall) with HNSW (99.8% R@10)
//! - Replaced raw UMA arena pointers with safe `Vec<f32>` storage (use-after-free fix)
//! - Auto-publish HNSW snapshot every 1000 inserts

pub mod memory_palace;
pub mod stats;

pub use memory_palace::MemoryPalace;
pub use stats::PalaceStats;
