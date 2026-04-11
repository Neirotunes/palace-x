//! # palace-engine
//!
//! Async actor pipeline orchestrator for the Palace-X project.
//!
//! Wraps `MemoryPalace` in an async actor system for safe concurrent ingestion, search, and maintenance.
//!
//! ## Architecture
//!
//! - **Actor Model**: Single async task processes all commands sequentially via mpsc channel
//! - **Zero-Copy**: Vectors passed directly without cloning
//! - **Graceful Shutdown**: Coordinated teardown with `shutdown()` method
//! - **Batch Operations**: Pipelined concurrent ingestion for throughput
//!
//! ## Example
//!
//! ```ignore
//! use palace_engine::PalaceEngine;
//! use palace_core::{MetaData, SearchConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Start the engine with 768-dim vectors
//!     let engine = PalaceEngine::start(768);
//!
//!     // Ingest an embedding
//!     let vector = vec![0.1; 768];
//!     let metadata = MetaData::new(1234567890, "user_input");
//!     let node_id = engine.ingest(vector, metadata).await?;
//!
//!     // Search for similar vectors
//!     let query = vec![0.1; 768];
//!     let config = SearchConfig::default_with_limit(10);
//!     let results = engine.search(query, config).await?;
//!
//!     println!("Found {} similar fragments", results.len());
//!
//!     // Get statistics
//!     let stats = engine.stats().await?;
//!     println!("Engine stats: {:?}", stats);
//!
//!     // Graceful shutdown
//!     engine.shutdown().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod actor;
pub mod batch;
pub mod engine;

// Re-export main API
pub use engine::PalaceEngine;

// Re-export commonly used types from palace-core
pub use palace_core::{CompressionMethod, Fragment, MemoryError, MetaData, NodeId, SearchConfig};

// Re-export storage types
pub use palace_storage::PalaceStats;
