// Copyright (c) 2026 M.Diach
// Proprietary — All Rights Reserved

//! Core engine implementation
//!
//! PalaceEngine wraps MemoryPalace in an async actor pattern for safe concurrent access.
//! Commands are processed sequentially through a single task, ensuring thread-safety without locks.

use crate::actor::Command;
use palace_core::{Fragment, MemoryError, MemoryProvider, MetaData, NodeId, SearchConfig};
use palace_storage::{MemoryPalace, PalaceStats};
use tokio::sync::mpsc;
use tracing::{debug, info};

/// PalaceEngine — async actor wrapping MemoryPalace
///
/// Processes commands via mpsc channel for safe concurrent access.
/// The engine runs a single async task that serializes all operations.
pub struct PalaceEngine {
    sender: mpsc::Sender<Command>,
    handle: tokio::task::JoinHandle<()>,
}

impl PalaceEngine {
    /// Create and start the engine with default configuration
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    pub fn start(dimensions: usize) -> Self {
        Self::start_with_config(dimensions, 16, 200, 0.7, 0.3, 1024)
    }

    /// Create and start the engine with custom configuration
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `max_neighbors` - Maximum number of neighbors in NSW graph
    /// * `ef_construction` - Construction parameter for NSW
    /// * `alpha` - Weight for cosine similarity in scoring (0.0-1.0)
    /// * `beta` - Weight for topological distance in scoring (0.0-1.0)
    /// * `channel_size` - Size of the command channel buffer
    pub fn start_with_config(
        dimensions: usize,
        max_neighbors: usize,
        ef_construction: usize,
        alpha: f32,
        beta: f32,
        channel_size: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel(channel_size);

        // Create the MemoryPalace with specified configuration
        let palace = MemoryPalace::new(dimensions);

        // Spawn the actor task
        let handle = tokio::spawn(Self::run_loop(palace, rx, max_neighbors, ef_construction));

        info!(
            dimensions = dimensions,
            max_neighbors = max_neighbors,
            ef_construction = ef_construction,
            alpha = alpha,
            beta = beta,
            "PalaceEngine started"
        );

        Self { sender: tx, handle }
    }

    /// Main event loop processing commands
    async fn run_loop(
        palace: MemoryPalace,
        mut rx: mpsc::Receiver<Command>,
        _max_neighbors: usize,
        _ef_construction: usize,
    ) {
        debug!("PalaceEngine actor task started");

        #[cfg(target_arch = "aarch64")]
        let prefetcher = palace_optimizer::SpeculativePrefetcher::new();

        // Silicon-Native Optimization: Pin this core actor task to Firestorm P-cores
        #[cfg(target_arch = "aarch64")]
        {
            use palace_optimizer::{pin_current_thread, set_thread_qos, CoreType, QosClass};
            if let Err(e) = pin_current_thread(CoreType::Performance) {
                tracing::warn!("Failed to pin PalaceEngine to P-core: error {}", e);
            } else {
                info!("PalaceEngine pinned to Firestorm P-core successfully");
            }
            set_thread_qos(QosClass::UserInteractive);
        }

        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::Ingest {
                    vector,
                    metadata,
                    reply,
                } => {
                    debug!("Processing Ingest command");
                    let result = palace.ingest(vector, metadata).await;
                    let _ = reply.send(result);
                }
                Command::Search {
                    query,
                    config,
                    reply,
                } => {
                    debug!("Processing Search command with limit={}", config.limit);
                    let result = palace.retrieve(&query, &config).await;

                    // Silicon-Native Optimization: Speculative Prefetching
                    #[cfg(target_arch = "aarch64")]
                    if let Ok(ref fragments) = result {
                        if let Some(top_fragment) = fragments.first() {
                            let node_id = top_fragment.node_id.0;

                            // If we have a prediction for what the user might search next,
                            // prefetch that vector's memory into L1/L2 cache now.
                            if let Some(predicted_id) = prefetcher.predict_next(node_id) {
                                if let Some(ptr) =
                                    palace.get_vector_ptr(palace_core::NodeId(predicted_id))
                                {
                                    prefetcher.prefetch_predicted(predicted_id, ptr as *const u8);
                                }
                            }

                            // Learn the pattern: current node was matched
                            prefetcher.record_access(node_id);
                        }
                    }

                    let _ = reply.send(result);
                }
                Command::Vacuum { nodes, reply } => {
                    debug!("Processing Vacuum command for {} nodes", nodes.len());
                    let result = palace.vacuum(&nodes).await;
                    let _ = reply.send(result);
                }
                Command::Stats { reply } => {
                    debug!("Processing Stats command");
                    let stats = palace.stats();
                    let _ = reply.send(Ok(stats));
                }
                Command::Shutdown => {
                    info!("Received shutdown command");
                    break;
                }
            }
        }

        debug!("PalaceEngine actor task shutting down");
    }

    /// Ingest a vector with metadata into the palace
    ///
    /// # Arguments
    /// * `vector` - The embedding vector
    /// * `metadata` - Associated metadata
    pub async fn ingest(
        &self,
        vector: Vec<f32>,
        metadata: MetaData,
    ) -> Result<NodeId, MemoryError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(Command::Ingest {
                vector,
                metadata,
                reply: tx,
            })
            .await
            .map_err(|_| MemoryError::StorageError("Engine shutdown".into()))?;
        rx.await
            .map_err(|_| MemoryError::StorageError("Engine dropped".into()))?
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `config` - Search configuration
    pub async fn search(
        &self,
        query: Vec<f32>,
        config: SearchConfig,
    ) -> Result<Vec<Fragment>, MemoryError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(Command::Search {
                query,
                config,
                reply: tx,
            })
            .await
            .map_err(|_| MemoryError::StorageError("Engine shutdown".into()))?;
        rx.await
            .map_err(|_| MemoryError::StorageError("Engine dropped".into()))?
    }

    /// Remove nodes from the palace
    ///
    /// # Arguments
    /// * `nodes` - Node IDs to remove
    pub async fn vacuum(&self, nodes: Vec<NodeId>) -> Result<u64, MemoryError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(Command::Vacuum { nodes, reply: tx })
            .await
            .map_err(|_| MemoryError::StorageError("Engine shutdown".into()))?;
        rx.await
            .map_err(|_| MemoryError::StorageError("Engine dropped".into()))?
    }

    /// Get statistics about the engine state
    pub async fn stats(&self) -> Result<PalaceStats, MemoryError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(Command::Stats { reply: tx })
            .await
            .map_err(|_| MemoryError::StorageError("Engine shutdown".into()))?;
        rx.await
            .map_err(|_| MemoryError::StorageError("Engine dropped".into()))?
    }

    /// Gracefully shut down the engine
    pub async fn shutdown(self) -> Result<(), MemoryError> {
        let _ = self.sender.send(Command::Shutdown).await;
        self.handle
            .await
            .map_err(|e| MemoryError::StorageError(format!("Join error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = PalaceEngine::start(128);
        assert!(engine.shutdown().await.is_ok());
    }

    #[tokio::test]
    async fn test_basic_ingest() {
        let engine = PalaceEngine::start(8);
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let metadata = MetaData::new(1234567890, "test");

        let result = engine.ingest(vector, metadata).await;
        assert!(result.is_ok());

        let _ = engine.shutdown().await;
    }

    #[tokio::test]
    async fn test_stats() {
        let engine = PalaceEngine::start(8);
        let stats = engine.stats().await;
        assert!(stats.is_ok());

        let _ = engine.shutdown().await;
    }
}
