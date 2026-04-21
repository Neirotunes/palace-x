// Copyright (c) 2026 M.Diach
// Proprietary — All Rights Reserved

//! Actor message types and command routing
//!
//! Defines the command protocol for communicating with the PalaceEngine actor.

use palace_core::{Fragment, MemoryError, MetaData, NodeId, SearchConfig};
use palace_storage::PalaceStats;
use tokio::sync::oneshot;

/// Commands that can be sent to the PalaceEngine
#[derive(Debug)]
pub enum Command {
    /// Ingest a vector with metadata
    Ingest {
        vector: Vec<f32>,
        metadata: MetaData,
        reply: oneshot::Sender<Result<NodeId, MemoryError>>,
    },

    /// Search for similar vectors
    Search {
        query: Vec<f32>,
        config: SearchConfig,
        reply: oneshot::Sender<Result<Vec<Fragment>, MemoryError>>,
    },

    /// Remove nodes from the index
    Vacuum {
        nodes: Vec<NodeId>,
        reply: oneshot::Sender<Result<u64, MemoryError>>,
    },

    /// Get statistics about the engine state
    Stats {
        reply: oneshot::Sender<Result<PalaceStats, MemoryError>>,
    },

    /// Gracefully shut down the engine
    Shutdown,
}
