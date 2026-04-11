//! # palace-storage
//!
//! Concrete in-memory MemoryProvider implementation for the Palace-X project.
//!
//! This crate combines:
//! - **NSW Index** (palace-graph): Navigable Small World graph with Hub-Highway optimization
//! - **Topological Reranking** (palace-topo): Structure-aware candidate refinement
//! - **Bit-Plane Storage** (palace-bitplane): Precision-proportional vector retrieval
//!
//! The result is a two-stage retrieval pipeline:
//! 1. **Stage 1 (Coarse)**: Binary Hamming search via NSW
//! 2. **Stage 2 (Fine)**: Topological reranking with ego-graphs and d_total metric
//!
//! This enables fast, high-quality approximate search over hierarchical memory in autonomous agents.

pub mod memory_palace;
pub mod stats;

pub use memory_palace::MemoryPalace;
pub use stats::PalaceStats;
