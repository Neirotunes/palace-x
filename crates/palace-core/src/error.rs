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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = MemoryError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let msg = format!("{}", err);
        assert!(
            msg.contains("128"),
            "error message should include expected dims"
        );
        assert!(
            msg.contains("64"),
            "error message should include actual dims"
        );
    }

    #[test]
    fn test_node_not_found_display() {
        let err = MemoryError::NodeNotFound { id: NodeId(42) };
        let msg = format!("{}", err);
        assert!(msg.contains("42"), "error message should include node id");
    }

    #[test]
    fn test_storage_error_display() {
        let err = MemoryError::StorageError("disk full".into());
        assert!(format!("{}", err).contains("disk full"));
    }

    #[test]
    fn test_index_full_display() {
        let err = MemoryError::IndexFull {
            capacity: 1_000_000,
        };
        assert!(format!("{}", err).contains("1000000"));
    }

    #[test]
    fn test_degraded_mode_display() {
        let err = MemoryError::DegradedMode {
            reason: "thermal throttle".into(),
            fallback_used: "reduced-ef search".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("thermal throttle"));
        assert!(msg.contains("reduced-ef search"));
    }

    #[test]
    fn test_graph_corrupted_display() {
        let err = MemoryError::GraphCorrupted("dangling pointer".into());
        assert!(format!("{}", err).contains("dangling pointer"));
    }

    #[test]
    fn test_error_is_clone() {
        let err = MemoryError::StorageError("test".into());
        let cloned = err.clone();
        assert_eq!(format!("{}", err), format!("{}", cloned));
    }
}
