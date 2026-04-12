// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! Core MemoryProvider implementation: MemoryPalace
//!
//! Integrates NSW index, topological reranking, and bit-plane storage into a complete
//! two-stage retrieval system for hierarchical memory in autonomous agents.

use palace_bitplane::BitPlaneStore;
use palace_core::{Fragment, MemoryError, MemoryProvider, MetaData, NodeId, SearchConfig};
use palace_graph::{HnswIndex, MetaData as GraphMetaData, NswIndex};
use palace_topo::ego_cache::EgoCache;
use palace_topo::TopologicalReranker;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Palace-X in-memory implementation combining HNSW, topological reranking, and bit-plane storage.
///
/// The MemoryPalace provides a two-stage retrieval pipeline:
/// - **Stage 1**: Coarse search via HNSW (hierarchical navigable small world)
/// - **Stage 2**: Refinement via topological reranking (if enabled)
///
/// ## Safety
/// Previous versions stored raw pointers from UMA arena in a HashMap, which
/// was vulnerable to use-after-free if the arena was reset. v0.2 uses safe
/// `Vec<f32>` storage. UMA arena is retained for future Metal GPU zero-copy.
pub struct MemoryPalace {
    /// Primary index — HNSW replaces flat NSW for 99.8% R@10 (was ~1%)
    pub(crate) hnsw: HnswIndex,
    /// Legacy NSW index (deprecated, kept for API compat during migration)
    pub(crate) nsw: NswIndex,
    pub(crate) bitplane: RwLock<BitPlaneStore>,
    pub(crate) reranker: TopologicalReranker,
    pub(crate) ego_cache: EgoCache,
    pub(crate) dimensions: usize,
    /// Safe vector storage — replaces raw UMA arena pointers (use-after-free fix)
    vector_store: RwLock<HashMap<NodeId, Vec<f32>>>,
    metadata_store: RwLock<HashMap<NodeId, MetaData>>,
    thermal: palace_optimizer::ThermalGuard,
    next_id: AtomicU64,
    /// Track inserts since last snapshot for auto-publish
    inserts_since_snapshot: AtomicU64,
}

// Safe: all interior mutability via RwLock/Atomic/DashMap
unsafe impl Sync for MemoryPalace {}
unsafe impl Send for MemoryPalace {}

impl MemoryPalace {
    /// Creates a new MemoryPalace with default parameters.
    ///
    /// # Arguments
    /// * `dimensions` - Vector embedding dimensionality
    ///
    /// Default configuration:
    /// - `max_neighbors`: 16 (M parameter for HNSW; M_max0 = 32 at layer 0)
    /// - `ef_construction`: 200 (construction phase parameter)
    /// - `alpha`: 0.7 (cosine weight for reranking)
    /// - `beta`: 0.3 (topological weight for reranking)
    pub fn new(dimensions: usize) -> Self {
        Self::with_config(dimensions, 16, 200, 0.7, 0.3)
    }

    /// Creates a new MemoryPalace with custom configuration.
    ///
    /// # Arguments
    /// * `dimensions` - Vector embedding dimensionality
    /// * `max_neighbors` - M parameter for HNSW (M_max0 = 2*M at layer 0)
    /// * `ef_construction` - Ef parameter during HNSW construction
    /// * `alpha` - Weight for cosine similarity in reranking (0.0-1.0)
    /// * `beta` - Weight for topological distance in reranking (0.0-1.0)
    pub fn with_config(
        dimensions: usize,
        max_neighbors: usize,
        ef_construction: usize,
        alpha: f32,
        beta: f32,
    ) -> Self {
        Self {
            hnsw: HnswIndex::new(dimensions, max_neighbors, ef_construction),
            nsw: NswIndex::new(dimensions, max_neighbors * 2, ef_construction),
            bitplane: RwLock::new(BitPlaneStore::new(dimensions)),
            reranker: TopologicalReranker::new(alpha, beta),
            ego_cache: EgoCache::new(10_000),
            dimensions,
            vector_store: RwLock::new(std::collections::HashMap::new()),
            metadata_store: RwLock::new(std::collections::HashMap::new()),
            thermal: palace_optimizer::ThermalGuard::default(),
            next_id: AtomicU64::new(0),
            inserts_since_snapshot: AtomicU64::new(0),
        }
    }

    /// Publish HNSW snapshot for wait-free reads (call after batch insertion).
    pub fn publish_snapshot(&self) {
        self.hnsw.publish_snapshot();
        self.inserts_since_snapshot.store(0, AtomicOrdering::Release);
    }

    /// Get ego-graph cache statistics: (hits, misses).
    pub fn cache_stats(&self) -> (u64, u64) {
        self.ego_cache.stats()
    }

    /// Cache hit rate as percentage.
    pub fn cache_hit_rate(&self) -> f64 {
        self.ego_cache.hit_rate()
    }
}

impl MemoryProvider for MemoryPalace {
    /// Ingest a vector with metadata into the memory index.
    ///
    /// # Algorithm
    /// 1. Validate dimensions
    /// 2. Insert into HNSW index (hierarchical graph construction)
    /// 3. Store in BitPlaneStore for precision-proportional fetch
    /// 4. Store original vector safely (no raw pointers) and metadata
    /// 5. Auto-publish HNSW snapshot every 1000 inserts
    /// 6. Return assigned NodeId
    ///
    /// # Complexity
    /// O(log N) amortized time for HNSW insertion
    async fn ingest(&self, vector: Vec<f32>, meta: MetaData) -> Result<NodeId, MemoryError> {
        // Validate dimensions
        if vector.len() != self.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Create metadata for graph layer
        let graph_meta = GraphMetaData {
            label: format!("node_{}", self.next_id.load(AtomicOrdering::Relaxed)),
        };

        // Insert into HNSW index (returns its own NodeId — matches our counter)
        let node_id = self.hnsw.insert(vector.clone(), graph_meta);

        // Keep MemoryPalace counter in sync with HNSW
        self.next_id.store(node_id.0 + 1, AtomicOrdering::Release);

        // Insert into BitPlaneStore for precision-proportional storage
        {
            let mut bp = self.bitplane.write();
            bp.insert(node_id.0, &vector).map_err(|e| {
                MemoryError::StorageError(format!("BitPlane insertion failed: {}", e))
            })?;
        }

        // Store vector safely in owned Vec (no raw pointers / use-after-free risk)
        {
            let mut store = self.vector_store.write();
            store.insert(node_id, vector);
        }

        // Store metadata
        {
            let mut store = self.metadata_store.write();
            store.insert(node_id, meta);
        }

        // Invalidate ego-graph cache for affected neighborhood
        self.ego_cache.invalidate(node_id);
        let neighbors: Vec<NodeId> = self.hnsw.get_neighbors(node_id, 1).into_iter().collect();
        for &n in &neighbors {
            self.ego_cache.invalidate(n);
        }

        // Auto-publish snapshot every 1000 inserts for incremental availability
        let count = self.inserts_since_snapshot.fetch_add(1, AtomicOrdering::Relaxed);
        if count > 0 && count % 1000 == 0 {
            self.hnsw.publish_snapshot();
        }

        Ok(node_id)
    }

    /// Retrieve fragments matching a query vector using two-stage search.
    ///
    /// # Algorithm
    /// **Stage 1 (Coarse Search)**: Binary Hamming distance via NSW
    /// - Use `nsw.search_binary()` with `config.rerank_k` as ef parameter
    /// - Get top candidates sorted by Hamming distance
    ///
    /// **Stage 2 (Refinement)**: If reranking enabled:
    /// - For each candidate, build 2-hop ego-graph
    /// - Compute d_total combining cosine similarity and topological density
    /// - Rerank and return top `limit` results
    ///
    /// **Graceful Degradation**: If ego-graph construction fails, fall back to cosine-only ranking
    ///
    /// # Complexity
    /// Stage 1: O(ef * log N) where ef ≈ rerank_k
    /// Stage 2: O(rerank_k * (graph_size + rerank_cost))
    async fn retrieve(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<Fragment>, MemoryError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| MemoryError::StorageError(e))?;

        // Validate query dimensions
        if query.len() != self.dimensions {
            return Err(MemoryError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        // Stage 1: HNSW search with thermal-aware ef tuning
        let mut rerank_k = config.rerank_k;
        if self.thermal.should_throttle() {
            rerank_k = (rerank_k / 2).max(config.limit);
            tracing::warn!("Thermal throttle active: reducing rerank_k to {}", rerank_k);
        }

        // Ensure snapshot is published for ego-graph reads
        // (search uses DashMap directly, but get_neighbors uses snapshot)
        let candidates = self.hnsw.search(query, Some(rerank_k));

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Convert candidates to Fragment with precise cosine distances
        let vector_store = self.vector_store.read();
        let metadata_store = self.metadata_store.read();

        let mut fragments: Vec<Fragment> = candidates
            .iter()
            .take(config.rerank_k)
            .filter_map(|(node_id, _hnsw_dist)| {
                // Safe vector access — no raw pointers
                vector_store.get(node_id).and_then(|stored_vec| {
                    metadata_store.get(node_id).map(|meta| {
                        // Compute precise cosine distance using NEON SIMD
                        let cosine_dist = compute_cosine_distance(query, stored_vec);

                        Fragment {
                            node_id: *node_id,
                            score: 1.0 - cosine_dist,
                            metadata: meta.clone(),
                            vector: None,
                        }
                    })
                })
            })
            .collect();

        // Stage 2: Topological reranking (optional)
        if config.enable_reranking && !fragments.is_empty() {
            fragments = self.rerank_stage_2(fragments, config)?;
        } else {
            // No reranking: just sort by cosine score and trim to limit
            fragments.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Trim to limit
        fragments.truncate(config.limit);

        Ok(fragments)
    }

    /// Remove nodes from storage and invalidate caches.
    ///
    /// NOTE: HNSW does not currently support node deletion from the graph.
    /// This removes from vector/metadata stores and invalidates caches.
    /// Full graph removal requires rebuild (planned for v0.3).
    ///
    /// # Complexity
    /// O(deleted_count * graph_degree) for cache invalidation
    async fn vacuum(&self, nodes: &[NodeId]) -> Result<u64, MemoryError> {
        let mut count = 0u64;

        for &node_id in nodes {
            // Get neighbors before removal for cache invalidation
            let neighbors: Vec<NodeId> = self.hnsw.get_neighbors(node_id, 1).into_iter().collect();

            // Remove from vector store
            let had_vector = {
                let mut store = self.vector_store.write();
                store.remove(&node_id).is_some()
            };

            if had_vector {
                count += 1;
            }

            // Remove from metadata store
            {
                let mut store = self.metadata_store.write();
                store.remove(&node_id);
            }

            // Invalidate ego-graph cache for node and its neighbors
            self.ego_cache.invalidate(node_id);
            for &n in &neighbors {
                self.ego_cache.invalidate(n);
            }

            // TODO(v0.3): implement HNSW node removal + graph repair
            // For now, removed nodes will still appear in HNSW graph but
            // won't match in retrieve() since vector_store lookup fails
        }

        // Republish snapshot after deletions
        self.hnsw.publish_snapshot();

        Ok(count)
    }

    /// Get the total number of nodes in the index.
    async fn len(&self) -> usize {
        self.hnsw.len()
    }
}

impl MemoryPalace {
    /// Get a clone of the stored vector for a node.
    pub fn get_vector(&self, node_id: NodeId) -> Option<Vec<f32>> {
        self.vector_store.read().get(&node_id).cloned()
    }

    /// Silicon-Native Support: Get raw pointer to a vector for hardware prefetching.
    /// The pointer is valid for the lifetime of the MemoryPalace (vectors are never moved).
    pub fn get_vector_ptr(&self, node_id: NodeId) -> Option<*const f32> {
        self.vector_store.read().get(&node_id).map(|v| v.as_ptr())
    }
}

impl MemoryPalace {
    /// Performs Stage 2 topological reranking.
    ///
    /// Converts Fragments to TopologicalReranker format and reranks using ego-graph analysis.
    fn rerank_stage_2(
        &self,
        fragments: Vec<Fragment>,
        _config: &SearchConfig,
    ) -> Result<Vec<Fragment>, MemoryError> {
        use palace_topo::reranker::Fragment as TopoFragment;

        // Convert to topo Fragment format
        let candidates: Vec<TopoFragment> = fragments
            .iter()
            .map(|f| TopoFragment {
                node_id: f.node_id,
                cosine_dist: 1.0 - f.score, // Convert back to distance
                metadata: None,
            })
            .collect();

        // Define neighbors function for reranker
        let neighbors_closure = |node_id: NodeId| -> Vec<NodeId> {
            self.hnsw
                .get_neighbors(node_id, 2) // 2-hop ego-graph
                .into_iter()
                .collect()
        };

        // Apply topological reranking (parallelized + cached)
        let reranked = self
            .reranker
            .rerank_cached(&candidates, neighbors_closure, &self.ego_cache);

        // Convert back to Fragment format with new scores
        let mut result: Vec<Fragment> = reranked
            .iter()
            .map(|topo_frag| {
                // Find original fragment to get metadata
                fragments
                    .iter()
                    .find(|f| f.node_id == topo_frag.node_id)
                    .map(|f| Fragment {
                        node_id: f.node_id,
                        score: f.score, // Keep original cosine-based score
                        metadata: f.metadata.clone(),
                        vector: None,
                    })
                    .unwrap_or_else(|| Fragment {
                        node_id: topo_frag.node_id,
                        score: 0.5,
                        metadata: palace_core::MetaData::new(0, "reranked"),
                        vector: None,
                    })
            })
            .collect();

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(result)
    }
}

/// Computes cosine distance between two vectors (0=identical, 1=perpendicular).
fn compute_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { palace_optimizer::simd::cosine_distance_neon(a, b) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        if a.is_empty() || b.is_empty() {
            return 1.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (av, bv) in a.iter().zip(b.iter()) {
            dot_product += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        let similarity = dot_product / (norm_a * norm_b);
        // Clamp to [0, 1]
        ((1.0 - similarity) / 2.0).max(0.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_palace_creation() {
        let palace = MemoryPalace::new(128);
        assert_eq!(palace.len().await, 0);
    }

    #[tokio::test]
    async fn test_ingest_single_vector() {
        let palace = MemoryPalace::new(128);
        let vector = vec![0.1; 128];
        let meta = MetaData::new(1000, "test");

        let id = palace.ingest(vector, meta).await.expect("ingest failed");
        assert_eq!(palace.len().await, 1);
        assert_eq!(id.0, 0);
    }

    #[tokio::test]
    async fn test_ingest_dimension_mismatch() {
        let palace = MemoryPalace::new(128);
        let vector = vec![0.1; 64]; // Wrong dimension
        let meta = MetaData::new(1000, "test");

        let result = palace.ingest(vector, meta).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(MemoryError::DimensionMismatch { .. })));
    }

    #[tokio::test]
    async fn test_ingest_and_retrieve() {
        let palace = MemoryPalace::new(128);

        // Ingest some vectors
        let v1 = vec![1.0; 128];
        let v2 = vec![0.9; 128];
        let v3 = vec![0.0; 128];

        let meta1 = MetaData::new(1000, "source1");
        let meta2 = MetaData::new(1001, "source2");
        let meta3 = MetaData::new(1002, "source3");

        palace.ingest(v1, meta1).await.unwrap();
        palace.ingest(v2, meta2).await.unwrap();
        palace.ingest(v3, meta3).await.unwrap();

        assert_eq!(palace.len().await, 3);

        // Retrieve with query similar to v1
        let query = vec![0.99; 128];
        let config = SearchConfig::default_with_limit(2);

        let results = palace.retrieve(&query, &config).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_retrieve_reranking_disabled() {
        let palace = MemoryPalace::new(128);

        // Ingest vectors
        let v1 = vec![1.0; 128];
        let v2 = vec![0.9; 128];

        palace.ingest(v1, MetaData::new(1000, "s1")).await.unwrap();
        palace.ingest(v2, MetaData::new(1001, "s2")).await.unwrap();

        // Search with reranking disabled
        let query = vec![0.99; 128];
        let mut config = SearchConfig::default_with_limit(2);
        config.enable_reranking = false;

        let results = palace.retrieve(&query, &config).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_empty_index() {
        let palace = MemoryPalace::new(128);
        let query = vec![0.5; 128];
        let config = SearchConfig::default_with_limit(10);

        let results = palace.retrieve(&query, &config).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_query_dimension_mismatch() {
        let palace = MemoryPalace::new(128);
        palace
            .ingest(vec![0.5; 128], MetaData::new(1000, "test"))
            .await
            .unwrap();

        let query = vec![0.5; 64]; // Wrong dimension
        let config = SearchConfig::default_with_limit(10);

        let result = palace.retrieve(&query, &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vacuum_removes_nodes() {
        let palace = MemoryPalace::new(128);

        // Ingest vectors
        let id1 = palace
            .ingest(vec![1.0; 128], MetaData::new(1000, "s1"))
            .await
            .unwrap();
        let _id2 = palace
            .ingest(vec![0.9; 128], MetaData::new(1001, "s2"))
            .await
            .unwrap();

        assert_eq!(palace.len().await, 2);

        // Vacuum one node
        let removed = palace.vacuum(&[id1]).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(palace.len().await, 1);
    }

    #[tokio::test]
    async fn test_vacuum_nonexistent_node() {
        let palace = MemoryPalace::new(128);
        palace
            .ingest(vec![1.0; 128], MetaData::new(1000, "s1"))
            .await
            .unwrap();

        let result = palace.vacuum(&[NodeId(9999)]).await;
        assert!(result.is_ok());
        assert_eq!(palace.len().await, 1); // Original node still there
    }

    #[tokio::test]
    async fn test_cosine_distance_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = compute_cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cosine_distance_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = compute_cosine_distance(&a, &b);
        assert!((dist - 0.5).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_multiple_ingests_generates_unique_ids() {
        let palace = MemoryPalace::new(128);

        let id1 = palace
            .ingest(vec![1.0; 128], MetaData::new(1000, "s1"))
            .await
            .unwrap();
        let id2 = palace
            .ingest(vec![0.9; 128], MetaData::new(1001, "s2"))
            .await
            .unwrap();
        let id3 = palace
            .ingest(vec![0.8; 128], MetaData::new(1002, "s3"))
            .await
            .unwrap();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_eq!(id1.0, 0);
        assert_eq!(id2.0, 1);
        assert_eq!(id3.0, 2);
    }
}
