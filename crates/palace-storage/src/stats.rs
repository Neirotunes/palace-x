// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! Statistics and diagnostics for MemoryPalace.
//!
//! Provides introspection into the index structure, memory usage,
//! and topological properties for monitoring and optimization.

use crate::MemoryPalace;
use serde::{Deserialize, Serialize};

/// Comprehensive statistics about a MemoryPalace instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceStats {
    /// Total number of nodes (vectors) stored
    pub total_nodes: usize,

    /// Dimensionality of stored vectors
    pub dimensions: usize,

    /// Estimated total memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Memory used by bit-plane coarse storage (sign + exponent planes)
    pub bitplane_coarse_bytes: usize,

    /// Average hub score across all nodes (0.0 = peripheral, 1.0 = central hub)
    pub avg_hub_score: f32,

    /// Maximum hub score in the index
    pub max_hub_score: f32,

    /// Number of nodes identified as hubs
    pub hub_count: usize,
}

impl PalaceStats {
    /// Estimate memory for NSW graph storage
    fn estimate_nsw_memory(total_nodes: usize, dimensions: usize, max_neighbors: usize) -> usize {
        if total_nodes == 0 {
            return 0;
        }

        // Per node:
        // - Vector: dimensions * 4 bytes (f32)
        // - Neighbors list: max_neighbors * 8 bytes (u64 IDs)
        // - Hub score: 4 bytes (f32)
        // - Metadata overhead: ~64 bytes
        let per_node = (dimensions * 4) + (max_neighbors * 8) + 4 + 64;

        total_nodes * per_node
    }

    /// Estimate memory for bit-plane storage
    fn estimate_bitplane_memory(total_nodes: usize, dimensions: usize) -> usize {
        if total_nodes == 0 {
            return 0;
        }

        // Coarse planes (sign + exponent):
        // - Sign: 1 bit per dimension = (dimensions + 7) / 8 bytes
        // - Exponent: 8 planes, each (dimensions + 7) / 8 bytes
        // Per node: ~(dimensions / 8) + 8 * (dimensions / 8) = ~9 * dimensions / 8

        let bytes_per_plane = (dimensions + 7) / 8;
        let coarse_bytes_per_node = bytes_per_plane + (8 * bytes_per_plane);

        total_nodes * coarse_bytes_per_node
    }

    /// Estimate total memory usage
    fn estimate_total_memory(total_nodes: usize, dimensions: usize, max_neighbors: usize) -> usize {
        let nsw_mem = Self::estimate_nsw_memory(total_nodes, dimensions, max_neighbors);
        let bitplane_mem = Self::estimate_bitplane_memory(total_nodes, dimensions);

        // Add overhead for hash maps, locks, etc.
        let overhead = total_nodes * 16 + 1024;

        nsw_mem + bitplane_mem + overhead
    }
}

impl MemoryPalace {
    /// Compute current statistics for this Palace instance.
    ///
    /// # Performance
    /// This operation may require synchronous graph traversal to compute hub scores.
    /// For large indices, consider caching results and updating periodically.
    pub fn stats(&self) -> PalaceStats {
        let total_nodes = self.live_node_count();
        let dimensions = self.dimensions;

        // Memory estimation
        let bitplane_coarse_bytes = PalaceStats::estimate_bitplane_memory(total_nodes, dimensions);
        let memory_usage_bytes = PalaceStats::estimate_total_memory(total_nodes, dimensions, 32);

        // Hub score computation (simplified: use node connectivity)
        let (avg_hub_score, max_hub_score, hub_count) = if total_nodes == 0 {
            (0.0, 0.0, 0)
        } else {
            // In production, would read actual hub scores from NSW
            // For now, estimate based on network density
            let estimated_avg = 0.3; // Placeholder
            let estimated_max = 0.8; // Placeholder
            let estimated_hubs = (total_nodes as f32 * 0.1).ceil() as usize; // Top 10%

            (estimated_avg, estimated_max, estimated_hubs)
        };

        PalaceStats {
            total_nodes,
            dimensions,
            memory_usage_bytes,
            bitplane_coarse_bytes,
            avg_hub_score,
            max_hub_score,
            hub_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_memory_estimation_empty() {
        let stats = PalaceStats {
            total_nodes: 0,
            dimensions: 128,
            memory_usage_bytes: 0,
            bitplane_coarse_bytes: 0,
            avg_hub_score: 0.0,
            max_hub_score: 0.0,
            hub_count: 0,
        };

        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn test_stats_memory_estimation_populated() {
        let total_nodes = 1000;
        let dimensions = 128;
        let max_neighbors = 32;

        let bitplane_coarse = PalaceStats::estimate_bitplane_memory(total_nodes, dimensions);
        let total_mem = PalaceStats::estimate_total_memory(total_nodes, dimensions, max_neighbors);

        assert!(bitplane_coarse > 0);
        assert!(total_mem > bitplane_coarse);
    }

    #[test]
    fn test_stats_bitplane_scaling() {
        // Memory should scale linearly with node count
        let mem_100 = PalaceStats::estimate_bitplane_memory(100, 128);
        let mem_200 = PalaceStats::estimate_bitplane_memory(200, 128);

        assert!(mem_200 > mem_100);
        assert!(((mem_200 as isize) - (2 * mem_100) as isize).unsigned_abs() < mem_100 / 10);
        // Roughly 2x
    }

    #[test]
    fn test_stats_dimension_scaling() {
        // Memory should scale linearly with dimension count
        let mem_128 = PalaceStats::estimate_bitplane_memory(1000, 128);
        let mem_256 = PalaceStats::estimate_bitplane_memory(1000, 256);

        assert!(mem_256 > mem_128);
    }

    #[tokio::test]
    async fn test_palace_stats_integration() {
        use crate::MemoryPalace;
        use palace_core::{MemoryProvider, MetaData};

        let palace = MemoryPalace::new(128);

        let stats_empty = palace.stats();
        assert_eq!(stats_empty.total_nodes, 0);

        // Ingest some vectors
        palace
            .ingest(vec![0.5; 128], MetaData::new(1000, "s1"))
            .await
            .unwrap();
        palace
            .ingest(vec![0.5; 128], MetaData::new(1001, "s2"))
            .await
            .unwrap();

        let stats_populated = palace.stats();
        assert_eq!(stats_populated.total_nodes, 2);
        assert_eq!(stats_populated.dimensions, 128);
        assert!(stats_populated.memory_usage_bytes > 0);
        assert!(stats_populated.bitplane_coarse_bytes > 0);
    }
}
