// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::NodeId;
use thiserror::Error;

/// Errors that can occur during memory operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Error)]
pub enum MemoryError {
    /// Index has reached capacity
    #[error("Index capacity exceeded: {capacity}")]
    IndexFull { capacity: usize },

    /// Node not found in the index
    #[error("Node not found: {id:?}")]
    NodeNotFound { id: NodeId },

    /// Vector dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Storage layer error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Compression operation failed
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// Graph structure corrupted
    #[error("Graph corrupted: {0}")]
    GraphCorrupted(String),

    /// Graceful degradation: system fell back to a fallback mode
    #[error("Degraded mode: {reason} (fallback: {fallback_used})")]
    DegradedMode {
        reason: String,
        fallback_used: String,
    },
}
