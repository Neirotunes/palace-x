// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # palace-core
//!
//! Foundation types for the Palace-X project: a high-performance hierarchical memory
//! system for autonomous AI agents.
//!
//! ## Core Components
//!
//! - **Types** (`types`): Core data structures including `NodeId`, `Fragment`, `MetaData`,
//!   `SearchConfig`, and `CompressionMethod`
//! - **Traits** (`traits`): The `MemoryProvider` trait defining the memory interface
//! - **Errors** (`error`): Comprehensive error types for memory operations

pub mod error;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use error::MemoryError;
pub use traits::MemoryProvider;
pub use types::{CompressionMethod, Fragment, MetaData, NodeId, SearchConfig};
