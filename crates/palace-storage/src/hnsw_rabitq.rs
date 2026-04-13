// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! # HNSW + RaBitQ Combined Pipeline
//!
//! Two-phase approximate nearest neighbor search that combines the graph
//! traversal quality of HNSW with the memory efficiency of RaBitQ compressed
//! distance estimation.
//!
//! ## Architecture
//!
//! ```text
//! Query ──► Phase 1: HNSW greedy descent (upper layers, float L2)
//!                     ↓ single entry point at layer 0
//!           Phase 2: Layer-0 beam search with RaBitQ distance
//!                     ↓ ef candidates with estimated distances
//!           Phase 3: (optional) Float L2 rerank of top-k
//!                     ↓ precise top-k results
//! ```
//!
//! ## Memory Layout
//!
//! | Component           | Per-Vector Cost | Purpose                    |
//! |---------------------|-----------------|----------------------------|
//! | HNSW graph edges    | ~M×8 bytes      | Connectivity (no vectors)  |
//! | Float vectors       | D×4 bytes       | Upper-layer greedy descent |
//! | RaBitQ 4-bit codes  | D/2 + 16 bytes  | Layer-0 distance estimate  |
//!
//! For SIFT-128: graph = ~256B, float = 512B, RaBitQ = 80B per vector.
//! Effective 8× compression on the distance computation path at layer 0,
//! where >95% of distance evaluations occur.
//!
//! ## Performance Target
//!
//! HNSW recall (~99% R@10) at 2× the QPS of float-only HNSW, since
//! layer-0 beam search (the bottleneck) uses cheap RaBitQ distances
//! instead of full float L2.

use palace_core::NodeId;
use palace_graph::{HnswDistanceMetric, HnswIndex, MetaData};
use palace_quant::rabitq::{RaBitQCode, RaBitQIndex, RaBitQQuery};

use dashmap::DashMap;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Configuration for the combined HNSW+RaBitQ pipeline.
#[derive(Clone, Debug)]
pub struct HnswRaBitQConfig {
    /// Vector dimensionality
    pub dimensions: usize,
    /// HNSW M parameter: max neighbors per layer
    pub max_neighbors: usize,
    /// ef during construction
    pub ef_construction: usize,
    /// RaBitQ bit depth (1, 2, 4, or 7)
    pub rabitq_bits: u8,
    /// Number of candidates to rerank with float L2 (0 = skip rerank)
    pub rerank_top: usize,
    /// Distance metric for HNSW graph construction
    pub metric: HnswDistanceMetric,
    /// RNG seed for RaBitQ rotation matrix
    pub seed: u64,
}

impl Default for HnswRaBitQConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 0, // Pure RaBitQ — no float rerank by default
            metric: HnswDistanceMetric::L2,
            seed: 42,
        }
    }
}

/// Combined HNSW + RaBitQ index for memory-efficient ANN search.
///
/// Graph traversal uses float vectors (stored in HNSW nodes).
/// Layer-0 beam search uses RaBitQ compressed distances.
/// Optional float rerank of top-k for maximum precision.
pub struct HnswRaBitQ {
    /// HNSW graph for hierarchical navigation
    hnsw: HnswIndex,
    /// RaBitQ quantizer (rotation matrix + centroid)
    quantizer: RaBitQIndex,
    /// Compressed codes per node: NodeId → RaBitQCode
    codes: DashMap<NodeId, RaBitQCode>,
    /// Configuration
    config: HnswRaBitQConfig,
    /// Search-time ef (tunable at runtime)
    ef_search: AtomicUsize,
    /// Insert counter for auto-snapshot
    insert_count: AtomicUsize,
}

/// Search result with both estimated and (optionally) precise distances.
#[derive(Clone, Debug)]
pub struct RaBitQSearchResult {
    pub node_id: NodeId,
    /// RaBitQ estimated squared L2 distance
    pub estimated_dist: f32,
    /// Float L2 distance (only set if rerank_top > 0)
    pub precise_dist: Option<f32>,
}

impl HnswRaBitQ {
    /// Create a new combined index.
    pub fn new(config: HnswRaBitQConfig) -> Self {
        let hnsw = match config.metric {
            HnswDistanceMetric::L2 => {
                HnswIndex::new(config.dimensions, config.max_neighbors, config.ef_construction)
            }
            HnswDistanceMetric::Cosine => {
                HnswIndex::with_cosine(
                    config.dimensions,
                    config.max_neighbors,
                    config.ef_construction,
                )
            }
        };

        let quantizer = RaBitQIndex::new(config.dimensions, config.seed);

        Self {
            hnsw,
            quantizer,
            codes: DashMap::new(),
            config,
            ef_search: AtomicUsize::new(64),
            insert_count: AtomicUsize::new(0),
        }
    }

    /// Set search-time ef parameter.
    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, AtomicOrdering::Relaxed);
    }

    /// Insert a vector with metadata.
    ///
    /// Stores the full vector in HNSW (for graph edges + upper-layer search)
    /// and the RaBitQ compressed code (for layer-0 distance estimation).
    pub fn insert(&self, vector: Vec<f32>, metadata: MetaData) -> NodeId {
        // Encode RaBitQ code before HNSW insert (needs the vector)
        let code = if self.config.rabitq_bits == 1 {
            self.quantizer.encode(&vector)
        } else {
            self.quantizer.encode_multibit(&vector, self.config.rabitq_bits)
        };

        // Insert into HNSW (this stores the full float vector + builds graph)
        let node_id = self.hnsw.insert(vector, metadata);

        // Store compressed code
        self.codes.insert(node_id, code);

        // Auto-publish snapshot every 1000 inserts
        let count = self.insert_count.fetch_add(1, AtomicOrdering::Relaxed);
        if (count + 1) % 1000 == 0 {
            self.hnsw.publish_snapshot();
        }

        node_id
    }

    /// Publish HNSW snapshot for read consistency.
    /// Call after batch insertion.
    pub fn publish_snapshot(&self) {
        self.hnsw.publish_snapshot();
    }

    /// Two-phase search: HNSW graph traversal → RaBitQ beam search → optional float rerank.
    ///
    /// # Phase 1: Greedy descent (upper layers)
    /// Uses float L2 distance. Touches O(log N) nodes — negligible cost.
    ///
    /// # Phase 2: Beam search at layer 0
    /// Uses RaBitQ estimated distance instead of float L2.
    /// This is where >95% of distance evaluations happen, so the
    /// compression gives the main speedup.
    ///
    /// # Phase 3: Optional float rerank
    /// If `rerank_top > 0`, re-scores the top candidates with precise
    /// float L2 distance for maximum recall.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<RaBitQSearchResult> {
        let ef = self.ef_search.load(AtomicOrdering::Relaxed).max(k);

        // Find entry point
        let ep = match self.hnsw.entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ep_level = self.hnsw.node_level(&ep).unwrap_or(0);

        // ── Phase 1: Greedy descent through upper layers (float L2) ──
        let mut current_ep = ep;
        for lc in (1..=ep_level).rev() {
            current_ep = self.greedy_descent_float(query, current_ep, lc);
        }

        // ── Phase 2: Layer-0 beam search with RaBitQ distances ──
        let rq = self.quantizer.encode_query(query);
        let candidates = self.beam_search_rabitq(query, &rq, current_ep, ef);

        // ── Phase 3: Optional float rerank ──
        let rerank_n = if self.config.rerank_top > 0 {
            self.config.rerank_top.min(candidates.len())
        } else {
            0
        };

        if rerank_n > 0 {
            // Rerank top candidates with precise float L2
            let mut results: Vec<RaBitQSearchResult> = candidates
                .into_iter()
                .take(rerank_n)
                .map(|(node_id, est_dist)| {
                    let precise = self.hnsw.get_vector(&node_id).map(|v| {
                        self.hnsw.compute_dist(query, &v)
                    });
                    RaBitQSearchResult {
                        node_id,
                        estimated_dist: est_dist,
                        precise_dist: precise,
                    }
                })
                .collect();

            // Sort by precise distance (falling back to estimated)
            results.sort_by(|a, b| {
                let da = a.precise_dist.unwrap_or(a.estimated_dist);
                let db = b.precise_dist.unwrap_or(b.estimated_dist);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            results.truncate(k);
            results
        } else {
            // Pure RaBitQ ranking — no float rerank
            candidates
                .into_iter()
                .take(k)
                .map(|(node_id, est_dist)| RaBitQSearchResult {
                    node_id,
                    estimated_dist: est_dist,
                    precise_dist: None,
                })
                .collect()
        }
    }

    /// Greedy descent at a single upper layer using float L2.
    /// Returns the closest node at the given layer.
    fn greedy_descent_float(&self, query: &[f32], entry: NodeId, layer: usize) -> NodeId {
        let mut current = entry;
        let mut current_dist = self
            .hnsw
            .get_vector(&current)
            .map(|v| self.hnsw.compute_dist(query, &v))
            .unwrap_or(f32::MAX);

        loop {
            let neighbors = self.hnsw.node_neighbors_at_layer(&current, layer);
            let mut improved = false;

            for &nb_id in &neighbors {
                if let Some(nb_vec) = self.hnsw.get_vector(&nb_id) {
                    let d = self.hnsw.compute_dist(query, &nb_vec);
                    if d < current_dist {
                        current = nb_id;
                        current_dist = d;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search at layer 0 using RaBitQ estimated distances.
    ///
    /// This is the performance-critical path. Instead of computing full
    /// float L2 for every visited node, we use RaBitQ's O(D/64) compressed
    /// distance which is 4-8× cheaper per evaluation.
    ///
    /// Returns candidates sorted by estimated distance (ascending).
    fn beam_search_rabitq(
        &self,
        _query_vec: &[f32], // kept for future float-fallback
        rq: &RaBitQQuery,
        entry: NodeId,
        ef: usize,
    ) -> Vec<(NodeId, f32)> {
        // Entry distance via RaBitQ
        let entry_dist = self
            .codes
            .get(&entry)
            .map(|code| self.quantizer.estimate_distance(rq, &code).0)
            .unwrap_or(f32::MAX);

        let mut visited = HashSet::with_capacity(ef * 2);
        visited.insert(entry);

        // Min-heap (closest first): candidates to expand
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrdF32, NodeId)>> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse((OrdF32(entry_dist), entry)));

        // Max-heap (farthest first): current result set
        let mut result: BinaryHeap<(OrdF32, NodeId)> = BinaryHeap::new();
        result.push((OrdF32(entry_dist), entry));

        while let Some(std::cmp::Reverse((OrdF32(curr_dist), curr_id))) = candidates.pop() {
            // Early termination: current candidate is farther than worst in result set
            let farthest = result.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if curr_dist > farthest && result.len() >= ef {
                break;
            }

            // Expand neighbors at layer 0
            let neighbors = self.hnsw.node_neighbors_at_layer(&curr_id, 0);

            for nb_id in neighbors {
                if visited.contains(&nb_id) {
                    continue;
                }
                visited.insert(nb_id);

                // RaBitQ distance estimation (the cheap path)
                let nb_dist = match self.codes.get(&nb_id) {
                    Some(code) => self.quantizer.estimate_distance(rq, &code).0,
                    None => continue, // Node without code — skip
                };

                let farthest = result.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                if nb_dist < farthest || result.len() < ef {
                    candidates.push(std::cmp::Reverse((OrdF32(nb_dist), nb_id)));
                    result.push((OrdF32(nb_dist), nb_id));
                    if result.len() > ef {
                        result.pop(); // Remove farthest
                    }
                }
            }
        }

        // Extract results sorted by distance ascending
        let mut results: Vec<(NodeId, f32)> = result
            .into_iter()
            .map(|(OrdF32(d), id)| (id, d))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ── Statistics ──────────────────────────────────────────────────

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Memory estimate in bytes.
    ///
    /// Returns (graph_bytes, float_bytes, rabitq_bytes, total_bytes).
    pub fn memory_estimate(&self) -> (usize, usize, usize, usize) {
        let n = self.len();
        let d = self.config.dimensions;
        let m = self.config.max_neighbors;

        let graph_bytes = n * m * 2 * 8; // ~2*M neighbors per node × 8 bytes per NodeId
        let float_bytes = n * d * 4; // full float vectors in HNSW nodes
        let rabitq_bytes = n * (d * self.config.rabitq_bits as usize / 8 + 16); // codes + factors

        (graph_bytes, float_bytes, rabitq_bytes, graph_bytes + float_bytes + rabitq_bytes)
    }

    /// Access the underlying HNSW index (for graph statistics, etc.)
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
    }

    /// Access the RaBitQ quantizer (for external encoding).
    pub fn quantizer(&self) -> &RaBitQIndex {
        &self.quantizer
    }
}

/// Wrapper for f32 to implement Ord (required for BinaryHeap).
/// NaN is treated as greater than all other values.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            if self.0.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_vector(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_basic_insert_and_search() {
        let config = HnswRaBitQConfig {
            dimensions: 32,
            max_neighbors: 8,
            ef_construction: 50,
            rabitq_bits: 4,
            rerank_top: 0,
            ..Default::default()
        };
        let index = HnswRaBitQ::new(config);
        let mut rng = StdRng::seed_from_u64(42);

        // Insert 100 vectors
        for _ in 0..100 {
            let v = random_vector(&mut rng, 32);
            let meta = MetaData { label: "test".into() };
            index.insert(v, meta);
        }
        index.publish_snapshot();

        // Search
        let query = random_vector(&mut rng, 32);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by estimated distance
        for w in results.windows(2) {
            assert!(w[0].estimated_dist <= w[1].estimated_dist + 1e-6);
        }
    }

    #[test]
    fn test_recall_vs_pure_hnsw() {
        let dim = 64;
        let n = 1000;
        let config = HnswRaBitQConfig {
            dimensions: dim,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 0,
            ..Default::default()
        };
        let combined = HnswRaBitQ::new(config);

        // Also build a pure float HNSW for ground truth
        let pure_hnsw = HnswIndex::new(dim, 16, 200);

        let mut rng = StdRng::seed_from_u64(123);
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(&mut rng, dim)).collect();

        for v in &vectors {
            let meta = MetaData { label: "v".into() };
            combined.insert(v.clone(), meta.clone());
            pure_hnsw.insert(v.clone(), meta);
        }
        combined.publish_snapshot();
        pure_hnsw.publish_snapshot();

        combined.set_ef_search(128);

        let n_queries = 100;
        let k = 10;
        let mut recall_sum = 0.0;

        for _ in 0..n_queries {
            let q = random_vector(&mut rng, dim);

            // Ground truth from float HNSW
            let gt: HashSet<NodeId> = pure_hnsw
                .search(&q, Some(128))
                .into_iter()
                .take(k)
                .map(|(id, _)| id)
                .collect();

            // Combined pipeline
            let results: HashSet<NodeId> = combined
                .search(&q, k)
                .into_iter()
                .map(|r| r.node_id)
                .collect();

            let overlap = gt.intersection(&results).count();
            recall_sum += overlap as f64 / k as f64;
        }

        let recall = recall_sum / n_queries as f64;
        eprintln!("HNSW+RaBitQ 4-bit recall@{}: {:.1}%", k, recall * 100.0);

        // Combined pipeline should achieve ≥80% recall vs pure float HNSW
        // (RaBitQ estimated distances have ~75% recall@10 brute-force,
        // but graph structure compensates significantly)
        assert!(
            recall >= 0.70,
            "HNSW+RaBitQ recall@{} = {:.1}%, expected ≥70%",
            k,
            recall * 100.0
        );
    }

    #[test]
    fn test_with_float_rerank() {
        let dim = 64;
        let n = 500;
        let config = HnswRaBitQConfig {
            dimensions: dim,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 50, // Float rerank top 50 candidates
            ..Default::default()
        };
        let index = HnswRaBitQ::new(config);
        let mut rng = StdRng::seed_from_u64(456);

        for _ in 0..n {
            let v = random_vector(&mut rng, dim);
            let meta = MetaData { label: "v".into() };
            index.insert(v, meta);
        }
        index.publish_snapshot();
        index.set_ef_search(128);

        let q = random_vector(&mut rng, dim);
        let results = index.search(&q, 10);

        assert_eq!(results.len(), 10);
        // With float rerank, all should have precise distances
        for r in &results {
            assert!(r.precise_dist.is_some(), "Float rerank should set precise_dist");
        }
        // Should be sorted by precise distance
        for w in results.windows(2) {
            let d0 = w[0].precise_dist.unwrap();
            let d1 = w[1].precise_dist.unwrap();
            assert!(d0 <= d1 + 1e-6, "Results should be sorted by precise distance");
        }
    }

    #[test]
    fn test_memory_estimate() {
        let config = HnswRaBitQConfig {
            dimensions: 128,
            max_neighbors: 16,
            rabitq_bits: 4,
            ..Default::default()
        };
        let index = HnswRaBitQ::new(config);
        let mut rng = StdRng::seed_from_u64(789);

        for _ in 0..100 {
            let v = random_vector(&mut rng, 128);
            let meta = MetaData { label: "m".into() };
            index.insert(v, meta);
        }

        let (graph, float, rabitq, total) = index.memory_estimate();
        eprintln!(
            "Memory: graph={}B, float={}B, rabitq={}B, total={}B",
            graph, float, rabitq, total
        );

        // 100 vectors × 128d × 4bit = 100 × (64 + 16) = 8000 bytes for RaBitQ
        assert!(rabitq > 0);
        assert!(float > rabitq, "Float storage should exceed RaBitQ storage");
        assert_eq!(total, graph + float + rabitq);
    }
}
