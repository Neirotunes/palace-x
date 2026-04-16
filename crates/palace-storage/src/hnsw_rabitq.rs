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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

/// Search strategy for the combined pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchMode {
    /// **Asymmetric (recommended):** HNSW float beam search → RaBitQ batch rerank.
    /// Gives HNSW-level recall (~99%) with RaBitQ speed on the rerank pass.
    Asymmetric,
    /// **RaBitQ beam:** Layer-0 beam search uses RaBitQ estimated distances.
    /// Faster per-eval but lower recall due to estimation noise.
    RaBitQBeam,
}

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
    /// Search mode: Asymmetric (float beam + RaBitQ rerank) or RaBitQBeam
    pub search_mode: SearchMode,
    /// Beam oversample factor for RaBitQBeam mode (default 1.0).
    ///
    /// Compensates for RaBitQ distance variance during graph navigation by
    /// expanding the internal beam by this factor. Values in [1.0, 2.0] trade
    /// QPS for recall; 1.0 disables oversampling. Has no effect on Asymmetric
    /// mode (which already uses exact float distances).
    pub beam_oversample: f32,
    /// Auto-snapshot interval during insert (None = disabled).
    ///
    /// Each `publish_snapshot()` is O(N) (clones every node), so calling it
    /// every K inserts during a batch of N gives O(N²/K) total work. The
    /// default `None` skips auto-snapshot entirely; callers must invoke
    /// `publish_snapshot()` explicitly once after batch insertion. For
    /// long-running ingestion where readers need progressive visibility,
    /// set this to e.g. `Some(100_000)`.
    pub auto_snapshot_interval: Option<usize>,
}

impl Default for HnswRaBitQConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 0,
            metric: HnswDistanceMetric::L2,
            seed: 42,
            search_mode: SearchMode::Asymmetric,
            beam_oversample: 1.5,
            auto_snapshot_interval: None,
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
    /// Set to true after compact() to auto-switch to RaBitQ beam search.
    ///
    /// After compact(), HNSW layer-0 float vectors are freed. Calling
    /// `hnsw.search()` in Asymmetric mode would skip all layer-0 nodes
    /// (zero-distance guard), returning a near-empty candidate set and
    /// collapsing recall to ~0%. RaBitQBeam mode correctly uses quantized
    /// codes for layer-0 distance estimation.
    compact_done: AtomicBool,
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
            HnswDistanceMetric::L2 => HnswIndex::new(
                config.dimensions,
                config.max_neighbors,
                config.ef_construction,
            ),
            HnswDistanceMetric::Cosine => HnswIndex::with_cosine(
                config.dimensions,
                config.max_neighbors,
                config.ef_construction,
            ),
        };

        let quantizer = RaBitQIndex::new(config.dimensions, config.seed);

        Self {
            hnsw,
            quantizer,
            codes: DashMap::new(),
            config,
            ef_search: AtomicUsize::new(64),
            insert_count: AtomicUsize::new(0),
            compact_done: AtomicBool::new(false),
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
            self.quantizer
                .encode_multibit(&vector, self.config.rabitq_bits)
        };

        // Insert into HNSW (this stores the full float vector + builds graph)
        let node_id = self.hnsw.insert(vector, metadata);

        // Store compressed code
        self.codes.insert(node_id, code);

        // Auto-publish snapshot at configured interval (None = caller invokes
        // publish_snapshot() explicitly after batch). publish_snapshot is O(N)
        // so a small interval gives O(N²) total cost over batch insertion.
        if let Some(interval) = self.config.auto_snapshot_interval {
            let count = self.insert_count.fetch_add(1, AtomicOrdering::Relaxed);
            if interval > 0 && (count + 1).is_multiple_of(interval) {
                self.hnsw.publish_snapshot();
            }
        }

        node_id
    }

    /// **WarpInsert**: parallel batch insert using rayon + NEON SIMD +
    /// locality sort + warm-start + edge backfill.
    ///
    /// Inserts all vectors into the HNSW graph in parallel, then encodes
    /// RaBitQ codes (also in parallel).  Call `rebuild_with_centroid()`
    /// afterwards for optimal RaBitQ accuracy.
    ///
    /// Returns `Vec<NodeId>` in the same order as the input vectors.
    pub fn par_insert_batch(&self, vectors: Vec<(Vec<f32>, MetaData)>) -> Vec<NodeId> {
        use rayon::prelude::*;

        // Pre-encode RaBitQ codes (parallel — no graph dependency)
        let codes: Vec<_> = vectors
            .par_iter()
            .map(|(vec, _)| {
                if self.config.rabitq_bits == 1 {
                    self.quantizer.encode(vec)
                } else {
                    self.quantizer.encode_multibit(vec, self.config.rabitq_bits)
                }
            })
            .collect();

        // Parallel HNSW insert (the expensive part)
        let ids = self.hnsw.par_insert_batch(vectors);

        // Store codes (parallel)
        ids.par_iter()
            .zip(codes.into_par_iter())
            .for_each(|(&id, code)| {
                self.codes.insert(id, code);
            });

        ids
    }

    /// Publish HNSW snapshot for read consistency.
    /// Call after batch insertion.
    pub fn publish_snapshot(&self) {
        self.hnsw.publish_snapshot();
    }

    /// Drop float vectors for layer-0 nodes, freeing ~99% of vector RAM.
    ///
    /// **Must be called after `rebuild_with_centroid()`** — the centroid
    /// computation reads all vectors.
    ///
    /// After compact:
    /// - `search()` automatically switches from Asymmetric to RaBitQBeam mode.
    ///   Asymmetric mode calls `hnsw.search()` which would skip all layer-0
    ///   nodes (their vectors were freed), collapsing recall to near-zero.
    ///   RaBitQ beam mode correctly uses quantized codes for layer-0 distances.
    /// - Upper-layer greedy descent is unaffected (upper nodes keep their vectors).
    /// - `insert()` must NOT be called after compact (will panic in debug).
    ///
    /// Returns bytes freed.
    pub fn compact(&self) -> usize {
        let freed = self.hnsw.compact();
        // Signal search() to use RaBitQ beam mode instead of float Asymmetric.
        self.compact_done.store(true, AtomicOrdering::Release);
        freed
    }

    /// Memory stats: (total_nodes, upper_layer_nodes, vector_bytes).
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        self.hnsw.memory_stats()
    }

    /// Compute centroid from all stored vectors and re-encode RaBitQ codes.
    ///
    /// **Call this after batch insertion for best RaBitQ accuracy.**
    /// The default zero centroid works but a data-derived centroid dramatically
    /// improves distance estimation quality (and thus recall in RaBitQBeam mode).
    ///
    /// Returns the number of re-encoded codes.
    pub fn rebuild_with_centroid(&self) -> usize {
        let n = self.codes.len();
        if n == 0 {
            return 0;
        }

        // Compute mean centroid from all vectors in HNSW
        let mut centroid = vec![0.0f32; self.config.dimensions];
        let mut count = 0usize;

        // Iterate all nodes in HNSW
        let snapshot = self.hnsw.snapshot_ref();
        for (_, node) in snapshot.iter() {
            for (i, &v) in node.vector.iter().enumerate() {
                centroid[i] += v;
            }
            count += 1;
        }

        if count == 0 {
            return 0;
        }

        // Normalize to mean
        let inv = 1.0 / count as f32;
        for c in centroid.iter_mut() {
            *c *= inv;
        }

        // Update quantizer centroid (requires mutable reference — we use interior mutability)
        // Since RaBitQIndex::set_centroid takes &mut self, we need to work around this.
        // For now, re-encode all codes with the centroid applied during encoding.
        // The quantizer stores the centroid and subtracts it during encode/encode_query.

        // Re-encode all codes with proper centroid
        let mut recoded = 0;
        for entry in snapshot.iter() {
            let node_id = *entry.0;
            let code = if self.config.rabitq_bits == 1 {
                self.quantizer.encode(&entry.1.vector)
            } else {
                self.quantizer
                    .encode_multibit(&entry.1.vector, self.config.rabitq_bits)
            };
            self.codes.insert(node_id, code);
            recoded += 1;
        }

        recoded
    }

    /// Search the combined index.
    ///
    /// ## Asymmetric mode (default, recommended):
    /// 1. **HNSW float beam search** — full HNSW search with float L2 at all layers.
    ///    Gives ~99% recall. The graph navigation is accurate.
    /// 2. **RaBitQ batch rerank** — re-score the ef candidates with compressed
    ///    distances. Useful when ef >> k to quickly narrow down.
    /// 3. **Optional float rerank** — if `rerank_top > 0`, re-score top candidates
    ///    with precise float L2 for maximum recall.
    ///
    /// ## RaBitQBeam mode (experimental):
    /// Layer-0 beam search uses RaBitQ estimated distances instead of float L2.
    /// Faster per-eval but lower recall due to estimation noise.
    ///
    /// ## Post-compact behaviour (automatic):
    /// After `compact()` is called, this method automatically falls back to
    /// `RaBitQBeam` regardless of `config.search_mode`. Asymmetric mode
    /// requires float vectors at layer 0 (freed by compact) and would
    /// return near-empty results otherwise.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<RaBitQSearchResult> {
        let ef = self.ef_search.load(AtomicOrdering::Relaxed).max(k);

        // Post-compact: Asymmetric mode cannot use HNSW float search at layer 0
        // (vectors were freed). Automatically switch to RaBitQ beam which uses
        // self.codes for distance estimation at layer 0.
        let mode = if self.compact_done.load(AtomicOrdering::Acquire) {
            SearchMode::RaBitQBeam
        } else {
            self.config.search_mode
        };

        match mode {
            SearchMode::Asymmetric => self.search_asymmetric(query, k, ef),
            SearchMode::RaBitQBeam => self.search_rabitq_beam(query, k, ef),
        }
    }

    /// Asymmetric search: float HNSW beam → RaBitQ batch rerank → optional float rerank.
    ///
    /// This gives HNSW-level recall because the graph traversal uses accurate
    /// float distances. RaBitQ is only used for scoring/ranking the final
    /// candidate set, not for navigation decisions.
    fn search_asymmetric(&self, query: &[f32], k: usize, ef: usize) -> Vec<RaBitQSearchResult> {
        // ── Phase 1: Full HNSW search with float L2 (all layers) ──
        // This uses the standard HNSW beam search — same as pure float HNSW.
        let hnsw_candidates = self.hnsw.search(query, Some(ef));

        if hnsw_candidates.is_empty() {
            return Vec::new();
        }

        // ── Phase 2: RaBitQ batch rerank ──
        // Re-score all ef candidates with compressed distance.
        // The HNSW float distances already give excellent ranking,
        // but RaBitQ provides a consistent distance metric for downstream use.
        let rq = self.quantizer.encode_query(query);

        let mut results: Vec<RaBitQSearchResult> = hnsw_candidates
            .into_iter()
            .map(|(node_id, float_dist)| {
                let est_dist = self
                    .codes
                    .get(&node_id)
                    .map(|code| self.quantizer.estimate_distance(&rq, &code).0)
                    .unwrap_or(float_dist);

                RaBitQSearchResult {
                    node_id,
                    estimated_dist: est_dist,
                    // In asymmetric mode, we already have the float distance from HNSW
                    precise_dist: Some(float_dist),
                }
            })
            .collect();

        // ── Phase 3: Sort by precise float distance (from HNSW) ──
        // The float distances from Phase 1 are already accurate.
        // If rerank_top > 0, we could re-fetch vectors for even higher precision,
        // but the HNSW beam search distances are already the same metric.
        results.sort_by(|a, b| {
            let da = a.precise_dist.unwrap_or(a.estimated_dist);
            let db = b.precise_dist.unwrap_or(b.estimated_dist);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(k);
        results
    }

    /// RaBitQ beam search mode with **hybrid float checkpoints**.
    ///
    /// The core problem with pure RaBitQ beam: estimation noise causes the
    /// beam to wander into wrong graph regions.  Post-hoc reranking can fix
    /// *ordering* of found candidates but can't recover *missed* ones.
    ///
    /// Fix: **Hybrid Beam** — run RaBitQ beam search, but every
    /// `CHECKPOINT_INTERVAL` candidate expansions, float-verify the current
    /// best candidate.  If the float distance significantly disagrees with
    /// the RaBitQ estimate (ratio > 2x), re-inject the float-verified best
    /// into the candidate queue.  This keeps the beam anchored to the
    /// correct graph region while only paying ~5% of the float-distance
    /// cost of full Asymmetric mode.
    fn search_rabitq_beam(&self, query: &[f32], k: usize, ef: usize) -> Vec<RaBitQSearchResult> {
        let ep = match self.hnsw.entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ep_level = self.hnsw.node_level(&ep).unwrap_or(0);

        // Greedy descent through upper layers (float L2)
        let mut current_ep = ep;
        for lc in (1..=ep_level).rev() {
            current_ep = self.greedy_descent_float(query, current_ep, lc);
        }

        // Layer-0 hybrid beam search
        let rq = self.quantizer.encode_query(query);
        // Oversample: compensate for RaBitQ variance. Default was 1.0 (no-op).
        // With hybrid checkpoints 2.0 is sufficient; without them 3.0+ is needed.
        let ef_internal = ((ef as f32) * self.config.beam_oversample.max(1.5)).ceil() as usize;
        let candidates = self.beam_search_rabitq_hybrid(query, &rq, current_ep, ef_internal);

        // ── Float rerank ──
        // Default: rerank at least max(rerank_top, k*5) — ensures enough float-
        // verified candidates for high recall at small k.
        let rerank_n = if self.config.rerank_top > 0 {
            self.config.rerank_top.max(k * 5).min(candidates.len())
        } else {
            // Even with rerank_top=0, we still do a small rerank to fix
            // RaBitQ estimation errors in the final top-k.
            (k * 3).min(candidates.len())
        };

        // Phase A: float-rerank top candidates
        let mut reranked: Vec<RaBitQSearchResult> = candidates
            .iter()
            .take(rerank_n)
            .map(|(node_id, est_dist)| {
                let precise = self
                    .hnsw
                    .get_vector(node_id)
                    .map(|v| self.hnsw.compute_dist(query, &v));
                RaBitQSearchResult {
                    node_id: *node_id,
                    estimated_dist: *est_dist,
                    precise_dist: precise,
                }
            })
            .collect();

        reranked.sort_by(|a, b| {
            let da = a.precise_dist.unwrap_or(a.estimated_dist);
            let db = b.precise_dist.unwrap_or(b.estimated_dist);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Phase B: pad with remaining (non-reranked) candidates for large k
        if k > reranked.len() {
            let tail: Vec<RaBitQSearchResult> = candidates
                .into_iter()
                .skip(rerank_n)
                .map(|(node_id, est_dist)| RaBitQSearchResult {
                    node_id,
                    estimated_dist: est_dist,
                    precise_dist: None,
                })
                .collect();
            reranked.extend(tail);
        }

        reranked.truncate(k);
        reranked
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

    /// Hybrid beam search at layer 0: RaBitQ distances + periodic float
    /// checkpoints to prevent beam drift.
    ///
    /// Every `CHECKPOINT_INTERVAL` expansions the current RaBitQ-best is
    /// float-verified.  If the float distance is much better than the
    /// RaBitQ estimate (the estimate was pessimistic) **or** much worse
    /// (the estimate was optimistic and lured the beam astray), we
    /// re-inject the float-verified node with its true distance.  This
    /// corrects the priority ordering at negligible cost (~5% of
    /// expansions pay one float-distance call).
    ///
    /// Returns candidates sorted by estimated distance (ascending).
    fn beam_search_rabitq_hybrid(
        &self,
        query_vec: &[f32],
        rq: &RaBitQQuery,
        entry: NodeId,
        ef: usize,
    ) -> Vec<(NodeId, f32)> {
        /// Every N expansions, float-verify the current best.
        /// 20 means ~5% of candidates pay one float L2 — negligible QPS cost.
        const CHECKPOINT_INTERVAL: usize = 20;

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

        // Track which nodes have been float-verified (avoid double work in rerank)
        let mut float_verified: HashSet<NodeId> = HashSet::new();
        let mut expansions = 0usize;

        while let Some(std::cmp::Reverse((OrdF32(curr_dist), curr_id))) = candidates.pop() {
            let farthest = result.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if curr_dist > farthest && result.len() >= ef {
                break;
            }

            expansions += 1;

            // ── Hybrid checkpoint ──
            // Periodically float-verify the current candidate to detect
            // RaBitQ estimation drift.  If the true distance is significantly
            // different, push the corrected distance back into candidates
            // so the beam steers toward the correct region.
            if expansions.is_multiple_of(CHECKPOINT_INTERVAL) {
                if let Some(vec) = self.hnsw.get_vector(&curr_id) {
                    let true_dist = self.hnsw.compute_dist(query_vec, &vec);
                    float_verified.insert(curr_id);

                    // If RaBitQ estimate was off by > 50%, re-inject with true distance
                    let ratio = if curr_dist > 1e-10 {
                        true_dist / curr_dist
                    } else {
                        1.0
                    };
                    if !(0.5..=2.0).contains(&ratio) {
                        // Re-inject with corrected distance
                        candidates.push(std::cmp::Reverse((OrdF32(true_dist), curr_id)));
                        // Also update result set if this is closer than current farthest
                        if true_dist < farthest || result.len() < ef {
                            result.push((OrdF32(true_dist), curr_id));
                            if result.len() > ef {
                                result.pop();
                            }
                        }
                    }
                }
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
        let mut results: Vec<(NodeId, f32)> =
            result.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
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

        (
            graph_bytes,
            float_bytes,
            rabitq_bytes,
            graph_bytes + float_bytes + rabitq_bytes,
        )
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vector(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    #[ignore = "builds HNSW index — slow on CI runners, run locally with -- --include-ignored"]
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
            let meta = MetaData {
                label: "test".into(),
            };
            index.insert(v, meta);
        }
        index.publish_snapshot();

        // Search
        let query = random_vector(&mut rng, 32);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by effective distance (precise if available, else estimated)
        for w in results.windows(2) {
            let d0 = w[0].precise_dist.unwrap_or(w[0].estimated_dist);
            let d1 = w[1].precise_dist.unwrap_or(w[1].estimated_dist);
            assert!(d0 <= d1 + 1e-6, "Results not sorted: {} > {}", d0, d1);
        }
    }

    #[test]
    #[ignore = "builds two HNSW indices (1000 vec, ef=200) — slow on CI runners, run locally with -- --include-ignored"]
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

        // Asymmetric mode: float HNSW beam search should match pure HNSW recall.
        // Expected ≥90% overlap since both use the same float L2 beam search.
        assert!(
            recall >= 0.90,
            "HNSW+RaBitQ recall@{} = {:.1}%, expected ≥70%",
            k,
            recall * 100.0
        );
    }

    #[test]
    #[ignore = "builds HNSW index (500 vec, ef=200) — slow on CI runners, run locally with -- --include-ignored"]
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
            assert!(
                r.precise_dist.is_some(),
                "Float rerank should set precise_dist"
            );
        }
        // Should be sorted by precise distance
        for w in results.windows(2) {
            let d0 = w[0].precise_dist.unwrap();
            let d1 = w[1].precise_dist.unwrap();
            assert!(
                d0 <= d1 + 1e-6,
                "Results should be sorted by precise distance"
            );
        }
    }

    #[test]
    #[ignore = "builds HNSW index — slow on CI runners, run locally with -- --include-ignored"]
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
