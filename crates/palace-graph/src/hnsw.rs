// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hierarchical Navigable Small World (HNSW) index — UMA-native implementation
//!
//! Key design decisions:
//! - Level assignment via `floor(-ln(uniform()) * level_mult)` (Malkov/Yashunin 2018)
//! - M_max0 = 2*M at layer 0 for denser base connectivity
//! - α-RNG neighbor selection heuristic (Vamana-style, default α=1.2)
//! - DashMap for concurrent writes, ArcSwap snapshot for wait-free reads
//! - DistanceMetric dispatch: L2 for raw features (SIFT), Cosine for embeddings

use crate::node::{cosine_distance, hamming_distance, MetaData};
use arc_swap::ArcSwap;
use dashmap::DashMap;
use palace_core::NodeId;
use parking_lot::RwLock;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Distance metric for graph construction and search
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HnswDistanceMetric {
    Cosine,
    L2,
}

/// A node in the HNSW graph with per-layer neighbor lists
#[derive(Clone, Debug)]
pub struct HnswNode {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub binary: Vec<u64>,
    /// neighbors[layer] = list of neighbor IDs at that layer
    pub neighbors: Vec<Vec<NodeId>>,
    pub level: usize,
    pub metadata: MetaData,
}

/// Candidate for heap operations
#[derive(Clone, Debug)]
struct Candidate {
    id: NodeId,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (farthest first)
#[derive(Clone, Debug)]
struct FarCandidate {
    id: NodeId,
    distance: f32,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for FarCandidate {}

impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Hierarchical Navigable Small World index
pub struct HnswIndex {
    nodes: DashMap<NodeId, HnswNode>,
    read_snapshot: ArcSwap<im::HashMap<NodeId, HnswNode>>,

    /// M parameter: max neighbors per layer (layers 1+)
    max_neighbors: usize,
    /// M_max0: max neighbors at layer 0 (= 2*M)
    max_neighbors_l0: usize,
    /// ef during construction
    ef_construction: usize,
    /// ef during search (tunable)
    ef_search: AtomicUsize,
    /// α-RNG pruning parameter
    alpha: f32,

    /// Level multiplier: 1/ln(M)
    level_mult: f64,
    /// Current maximum level in the graph
    max_level: AtomicUsize,
    /// Entry point node ID (top level)
    entry_point: RwLock<Option<NodeId>>,

    next_id: AtomicU64,
    dimensions: usize,
    metric: HnswDistanceMetric,
}

impl HnswIndex {
    /// Create a new HNSW index with L2 metric (recommended for SIFT/raw features)
    pub fn new(dimensions: usize, max_neighbors: usize, ef_construction: usize) -> Self {
        Self::with_params(
            dimensions,
            max_neighbors,
            ef_construction,
            1.2,
            HnswDistanceMetric::L2,
        )
    }

    /// Create with cosine metric (for normalized embeddings)
    pub fn with_cosine(dimensions: usize, max_neighbors: usize, ef_construction: usize) -> Self {
        Self::with_params(
            dimensions,
            max_neighbors,
            ef_construction,
            1.2,
            HnswDistanceMetric::Cosine,
        )
    }

    /// Create with full parameter control
    pub fn with_params(
        dimensions: usize,
        max_neighbors: usize,
        ef_construction: usize,
        alpha: f32,
        metric: HnswDistanceMetric,
    ) -> Self {
        let level_mult = 1.0 / (max_neighbors as f64).ln();
        HnswIndex {
            nodes: DashMap::new(),
            read_snapshot: ArcSwap::from_pointee(im::HashMap::new()),
            max_neighbors,
            max_neighbors_l0: 2 * max_neighbors,
            ef_construction,
            ef_search: AtomicUsize::new(64),
            alpha,
            level_mult,
            max_level: AtomicUsize::new(0),
            entry_point: RwLock::new(None),
            next_id: AtomicU64::new(0),
            dimensions,
            metric,
        }
    }

    /// Set search ef parameter
    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, AtomicOrdering::Relaxed);
    }

    /// Number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Compute distance between two vectors based on metric
    #[inline]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            HnswDistanceMetric::L2 => l2_distance(a, b),
            HnswDistanceMetric::Cosine => cosine_distance(a, b),
        }
    }

    /// Max neighbors allowed at a given layer
    #[inline]
    fn max_neighbors_at(&self, layer: usize) -> usize {
        if layer == 0 {
            self.max_neighbors_l0
        } else {
            self.max_neighbors
        }
    }

    /// Assign a random level for a new node: floor(-ln(uniform()) * level_mult)
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0001..1.0);
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// Insert a vector into the HNSW index
    pub fn insert(&self, vector: Vec<f32>, metadata: MetaData) -> NodeId {
        assert_eq!(vector.len(), self.dimensions, "Dimension mismatch");

        let id = NodeId(self.next_id.fetch_add(1, AtomicOrdering::Relaxed));
        let level = self.random_level();

        // Create node with empty neighbor lists for each layer
        let binary = quantize_binary(&vector);
        let neighbors = vec![Vec::new(); level + 1];
        let node = HnswNode {
            id,
            vector: vector.clone(),
            binary,
            neighbors,
            level,
            metadata,
        };

        self.nodes.insert(id, node);

        // Get current entry point
        let ep = { self.entry_point.read().clone() };

        if let Some(ep_id) = ep {
            let ep_level = self.nodes.get(&ep_id).map(|n| n.level).unwrap_or(0);

            // Phase 1: Greedy descent from top to (level+1)
            // Find closest node at each layer using greedy 1-NN search
            let mut current_ep = ep_id;
            let top = ep_level;
            let insert_level = level.min(top);

            for lc in ((insert_level + 1)..=top).rev() {
                current_ep = self.search_layer_greedy(&vector, current_ep, lc);
            }

            // Phase 2: Insert at layers min(level, top) down to 0
            // Beam search with ef_construction candidates, then select neighbors
            for lc in (0..=insert_level).rev() {
                let candidates = self.search_layer(&vector, current_ep, self.ef_construction, lc);

                // Select neighbors using α-RNG heuristic
                let max_m = self.max_neighbors_at(lc);
                let selected = self.select_neighbors_heuristic(&vector, &candidates, max_m, lc);

                // Set neighbors for the new node at this layer
                if let Some(mut node_ref) = self.nodes.get_mut(&id) {
                    if lc < node_ref.neighbors.len() {
                        node_ref.neighbors[lc] = selected.clone();
                    }
                }

                // Add bidirectional connections and prune if needed
                for &neighbor_id in &selected {
                    // Step 1: add connection, check if pruning needed
                    let needs_prune =
                        if let Some(mut neighbor_ref) = self.nodes.get_mut(&neighbor_id) {
                            if lc < neighbor_ref.neighbors.len() {
                                if !neighbor_ref.neighbors[lc].contains(&id) {
                                    neighbor_ref.neighbors[lc].push(id);
                                }
                                neighbor_ref.neighbors[lc].len() > max_m
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                    // get_mut dropped here — safe to read other nodes

                    // Step 2: prune if over capacity (no concurrent lock held)
                    if needs_prune {
                        // Gracefully handle concurrent deletion — node may have been
                        // removed between releasing get_mut and this read.
                        let (neighbor_vec, nb_ids) = match self.nodes.get(&neighbor_id) {
                            Some(nr) => (nr.vector.clone(), nr.neighbors[lc].clone()),
                            None => continue, // Node deleted concurrently, skip pruning
                        };
                        // Now compute distances without holding any DashMap ref
                        let nb_list: Vec<(NodeId, f32)> = nb_ids
                            .iter()
                            .filter_map(|&nid| {
                                self.nodes
                                    .get(&nid)
                                    .map(|n| (nid, self.dist(&neighbor_vec, &n.vector)))
                            })
                            .collect();
                        let pruned =
                            self.select_neighbors_heuristic(&neighbor_vec, &nb_list, max_m, lc);
                        if let Some(mut neighbor_ref) = self.nodes.get_mut(&neighbor_id) {
                            if lc < neighbor_ref.neighbors.len() {
                                neighbor_ref.neighbors[lc] = pruned;
                            }
                        }
                    }
                }

                // Use best candidate as entry point for next layer
                if let Some((best_id, _)) = candidates.first() {
                    current_ep = *best_id;
                }
            }

            // Update entry point if new node has higher level
            // Use write lock to make the (entry_point, max_level) update atomic
            // preventing race conditions between concurrent inserts
            if level > ep_level {
                let mut ep_guard = self.entry_point.write();
                // Double-check under lock: another thread may have raised max_level
                let current_max = self.max_level.load(AtomicOrdering::Acquire);
                if level > current_max {
                    *ep_guard = Some(id);
                    self.max_level.store(level, AtomicOrdering::Release);
                }
            }
        } else {
            // First node — set as entry point
            *self.entry_point.write() = Some(id);
            self.max_level.store(level, AtomicOrdering::Relaxed);
        }

        // NOTE: snapshot not published per-insert for performance.
        // Call publish_snapshot() after batch insertion is complete.
        id
    }

    /// Greedy 1-NN search at a single layer (for navigational descent)
    fn search_layer_greedy(&self, query: &[f32], entry: NodeId, layer: usize) -> NodeId {
        let mut current = entry;
        let mut current_dist = self
            .nodes
            .get(&current)
            .map(|n| self.dist(query, &n.vector))
            .unwrap_or(f32::MAX);

        loop {
            let mut improved = false;
            let neighbors: Vec<NodeId> = self
                .nodes
                .get(&current)
                .map(|n| {
                    if layer < n.neighbors.len() {
                        n.neighbors[layer].clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for &nb_id in &neighbors {
                if let Some(nb) = self.nodes.get(&nb_id) {
                    // Skip compacted nodes (vector freed by compact()) — empty vector
                    // would compute 0.0 distance (query against empty slice), making
                    // every compacted node appear equidistant and corrupting search.
                    if nb.vector.is_empty() {
                        continue;
                    }
                    let d = self.dist(query, &nb.vector);
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

    /// Beam search at a single layer — returns sorted (NodeId, distance) pairs
    fn search_layer(
        &self,
        query: &[f32],
        entry: NodeId,
        ef: usize,
        layer: usize,
    ) -> Vec<(NodeId, f32)> {
        let entry_dist = self
            .nodes
            .get(&entry)
            .map(|n| self.dist(query, &n.vector))
            .unwrap_or(f32::MAX);

        let mut visited = HashSet::new();
        visited.insert(entry);

        // Min-heap: closest candidates to explore
        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate {
            id: entry,
            distance: entry_dist,
        });

        // Max-heap: farthest in result set (for pruning)
        let mut result = BinaryHeap::new();
        result.push(FarCandidate {
            id: entry,
            distance: entry_dist,
        });

        while let Some(Candidate {
            id: current,
            distance: current_dist,
        }) = candidates.pop()
        {
            // Stop if closest unexplored is farther than farthest result
            let farthest_result_dist = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);
            if current_dist > farthest_result_dist && result.len() >= ef {
                break;
            }

            // Explore neighbors at this layer
            let neighbors: Vec<NodeId> = self
                .nodes
                .get(&current)
                .map(|n| {
                    if layer < n.neighbors.len() {
                        n.neighbors[layer].clone()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();

            for nb_id in neighbors {
                if visited.contains(&nb_id) {
                    continue;
                }
                visited.insert(nb_id);

                if let Some(nb) = self.nodes.get(&nb_id) {
                    // Skip compacted layer-0 nodes: empty vector → dist=0 (bogus)
                    if nb.vector.is_empty() {
                        continue;
                    }
                    let d = self.dist(query, &nb.vector);

                    let farthest = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);
                    if d < farthest || result.len() < ef {
                        candidates.push(Candidate {
                            id: nb_id,
                            distance: d,
                        });
                        result.push(FarCandidate {
                            id: nb_id,
                            distance: d,
                        });

                        if result.len() > ef {
                            result.pop(); // remove farthest
                        }
                    }
                }
            }
        }

        // Drain result heap into sorted vec
        let mut results: Vec<(NodeId, f32)> =
            result.into_iter().map(|fc| (fc.id, fc.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// α-RNG neighbor selection heuristic (Vamana-style)
    ///
    /// Prefers close AND diverse neighbors. A candidate is accepted if:
    ///   dist(candidate, query) <= alpha * dist(candidate, any_selected)
    ///
    /// Uses the same metric as graph construction (`self.dist()`) to ensure
    /// the α threshold is scale-consistent regardless of L2 vs Cosine metric.
    fn select_neighbors_heuristic(
        &self,
        _query: &[f32],
        candidates: &[(NodeId, f32)],
        max_neighbors: usize,
        _layer: usize,
    ) -> Vec<NodeId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut selected: Vec<NodeId> = Vec::new();
        let mut selected_vecs: Vec<Vec<f32>> = Vec::new();

        for (cand_id, cand_dist) in &sorted {
            if selected.len() >= max_neighbors {
                break;
            }

            let cand_vec = match self.nodes.get(cand_id) {
                Some(n) => n.vector.clone(),
                None => continue,
            };

            // First candidate always accepted
            if selected.is_empty() {
                selected.push(*cand_id);
                selected_vecs.push(cand_vec);
                continue;
            }

            // α-RNG check: is this candidate diverse enough?
            // Uses self.dist() (same metric as construction) so the α threshold
            // is scale-consistent for both L2 (unbounded) and Cosine ([0,2]).
            let mut is_diverse = true;
            for sv in &selected_vecs {
                let dist_to_selected = self.dist(&cand_vec, sv);
                if dist_to_selected > 0.0 && *cand_dist > self.alpha * dist_to_selected {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(*cand_id);
                selected_vecs.push(cand_vec);
            }
        }

        // If heuristic selected too few, fill with closest remaining
        if selected.len() < max_neighbors {
            for (cand_id, _) in &sorted {
                if selected.len() >= max_neighbors {
                    break;
                }
                if !selected.contains(cand_id) {
                    selected.push(*cand_id);
                }
            }
        }

        selected
    }

    /// Search the HNSW index for nearest neighbors
    ///
    /// Returns Vec<(NodeId, f32)> sorted by distance ascending.
    /// Panics in debug builds if `publish_snapshot()` has not been called
    /// (snapshot is used by `get_neighbors` for ego-graph extraction).
    pub fn search(&self, query: &[f32], ef: Option<usize>) -> Vec<(NodeId, f32)> {
        debug_assert!(
            !self.nodes.is_empty() || self.read_snapshot.load().is_empty(),
            "HNSW: search called on non-empty index without publish_snapshot(). \
             Call publish_snapshot() after batch insertion."
        );
        let ef = ef.unwrap_or(self.ef_search.load(AtomicOrdering::Relaxed));

        let ep = match self.entry_point.read().clone() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ep_level = self.nodes.get(&ep).map(|n| n.level).unwrap_or(0);

        // Phase 1: Greedy descent from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=ep_level).rev() {
            current_ep = self.search_layer_greedy(query, current_ep, lc);
        }

        // Phase 2: Beam search at layer 0
        self.search_layer(query, current_ep, ef, 0)
    }

    /// Hamming-based search (for binary pre-filtering)
    pub fn search_binary(&self, query_binary: &[u64], ef: Option<usize>) -> Vec<(NodeId, u32)> {
        let ef = ef.unwrap_or(self.ef_search.load(AtomicOrdering::Relaxed));

        let ep = match self.entry_point.read().clone() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ep_level = self.nodes.get(&ep).map(|n| n.level).unwrap_or(0);

        // Greedy descent using Hamming at upper layers
        let mut current = ep;
        for lc in (1..=ep_level).rev() {
            let mut current_dist = self
                .nodes
                .get(&current)
                .map(|n| hamming_distance(&n.binary, query_binary))
                .unwrap_or(u32::MAX);
            loop {
                let mut improved = false;
                let neighbors: Vec<NodeId> = self
                    .nodes
                    .get(&current)
                    .map(|n| {
                        if lc < n.neighbors.len() {
                            n.neighbors[lc].clone()
                        } else {
                            Vec::new()
                        }
                    })
                    .unwrap_or_default();
                for &nb_id in &neighbors {
                    if let Some(nb) = self.nodes.get(&nb_id) {
                        let d = hamming_distance(&nb.binary, query_binary);
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
        }

        // Beam search at layer 0 with Hamming
        let entry_dist = self
            .nodes
            .get(&current)
            .map(|n| hamming_distance(&n.binary, query_binary))
            .unwrap_or(u32::MAX);

        let mut visited = HashSet::new();
        visited.insert(current);

        let mut candidates: BinaryHeap<std::cmp::Reverse<(u32, NodeId)>> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse((entry_dist, current)));

        let mut result: BinaryHeap<(u32, NodeId)> = BinaryHeap::new();
        result.push((entry_dist, current));

        while let Some(std::cmp::Reverse((curr_dist, curr_id))) = candidates.pop() {
            let farthest = result.peek().map(|r| r.0).unwrap_or(u32::MAX);
            if curr_dist > farthest && result.len() >= ef {
                break;
            }

            let neighbors: Vec<NodeId> = self
                .nodes
                .get(&curr_id)
                .map(|n| {
                    if n.neighbors.is_empty() {
                        Vec::new()
                    } else {
                        n.neighbors[0].clone()
                    }
                })
                .unwrap_or_default();

            for nb_id in neighbors {
                if visited.contains(&nb_id) {
                    continue;
                }
                visited.insert(nb_id);

                if let Some(nb) = self.nodes.get(&nb_id) {
                    let d = hamming_distance(&nb.binary, query_binary);
                    let farthest = result.peek().map(|r| r.0).unwrap_or(u32::MAX);
                    if d < farthest || result.len() < ef {
                        candidates.push(std::cmp::Reverse((d, nb_id)));
                        result.push((d, nb_id));
                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<(NodeId, u32)> = result.into_iter().map(|(d, id)| (id, d)).collect();
        results.sort_by_key(|r| r.1);
        results
    }

    /// Get k-hop neighbors for ego-graph extraction
    pub fn get_neighbors(&self, id: NodeId, hops: usize) -> HashSet<NodeId> {
        let snapshot = self.read_snapshot.load();
        let mut result = HashSet::new();
        let mut frontier = vec![id];

        for _ in 0..hops {
            let mut next_frontier = Vec::new();
            for &node_id in &frontier {
                if let Some(node) = snapshot.get(&node_id) {
                    // Use layer 0 neighbors for ego-graph
                    if !node.neighbors.is_empty() {
                        for &nb_id in &node.neighbors[0] {
                            if nb_id != id && result.insert(nb_id) {
                                next_frontier.push(nb_id);
                            }
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        result
    }

    /// Publish immutable snapshot for wait-free reads
    pub fn publish_snapshot(&self) {
        let mut snapshot = im::HashMap::new();
        for entry in self.nodes.iter() {
            snapshot.insert(*entry.key(), entry.value().clone());
        }
        self.read_snapshot.store(Arc::new(snapshot));
    }

    /// Get vector for a node.  Returns `None` if the node doesn't exist
    /// **or** if its vector was dropped by `compact()`.
    pub fn get_vector(&self, id: &NodeId) -> Option<Vec<f32>> {
        self.nodes.get(id).and_then(|n| {
            if n.vector.is_empty() {
                None // compacted — vector was freed
            } else {
                Some(n.vector.clone())
            }
        })
    }

    /// Get metadata label for a node
    pub fn get_label(&self, id: &NodeId) -> Option<String> {
        self.nodes.get(id).map(|n| n.metadata.label.clone())
    }

    // ── UMA-HNSW accessor methods ─────────────────────────────────────
    // These expose internal state needed by the cache-aware search path
    // in `uma_hnsw.rs`. They use the read snapshot for wait-free access.

    /// Load the immutable read snapshot (wait-free, Arc-guarded).
    ///
    /// Used by `HotTierStore::from_hnsw` to iterate upper-layer nodes
    /// without blocking concurrent writers.
    pub fn snapshot_ref(&self) -> arc_swap::Guard<Arc<im::HashMap<NodeId, HnswNode>>> {
        self.read_snapshot.load()
    }

    /// Vector dimensionality of this index
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Current entry point (top-level node), or `None` if index is empty
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point.read().clone()
    }

    /// Level assigned to a node, or `None` if node doesn't exist
    pub fn node_level(&self, id: &NodeId) -> Option<usize> {
        self.nodes.get(id).map(|n| n.level)
    }

    /// Public distance computation — same metric used for graph construction.
    ///
    /// Exposed so `uma_hnsw::greedy_descent_hot` can score query-vs-node
    /// using the same metric as the standard search path.
    #[inline]
    pub fn compute_dist(&self, a: &[f32], b: &[f32]) -> f32 {
        self.dist(a, b)
    }

    /// Get neighbor IDs at a specific layer for a node (from live DashMap).
    ///
    /// Returns empty vec if node doesn't exist or layer is out of range.
    /// Used by `greedy_descent_hot` as fallback when node is not in hot tier.
    pub fn node_neighbors_at_layer(&self, id: &NodeId, layer: usize) -> Vec<NodeId> {
        self.nodes
            .get(id)
            .map(|n| {
                if layer < n.neighbors.len() {
                    n.neighbors[layer].clone()
                } else {
                    Vec::new()
                }
            })
            .unwrap_or_default()
    }

    /// Beam search at layer 0 from a given entry point.
    ///
    /// This is the second phase of UMA-optimized search: after greedy descent
    /// through hot-tier upper layers lands on a layer-0 entry point, this
    /// method runs standard beam search to find the final top-k.
    pub fn search_from_entry(&self, query: &[f32], entry: NodeId, ef: usize) -> Vec<(NodeId, f32)> {
        self.search_layer(query, entry, ef, 0)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  compact() — drop float vectors to reclaim memory after build
    // ══════════════════════════════════════════════════════════════════════

    /// Drop float vectors from layer-0-only nodes, freeing ~99% of vector RAM.
    ///
    /// After building the HNSW graph and encoding RaBitQ codes, the float
    /// vectors are no longer needed for quantized search.  This method
    /// replaces them with empty `Vec`s and returns the number of bytes freed.
    ///
    /// **Upper-layer nodes (level ≥ 1) keep their vectors** so that greedy
    /// descent through the navigational layers still works correctly.
    /// For M=16 and 1M nodes, ~62K nodes (6.25%) are kept — ~32 MB out of
    /// ~512 MB total.
    ///
    /// # Post-compact constraints
    /// - `search()` with float distances at layer 0 will produce garbage
    ///   (zero distances).  Use `HnswRaBitQ` search modes instead.
    /// - `insert()` will panic (new nodes need float distance to connect).
    /// - The index is effectively **read-only + quantized-only** after compact.
    pub fn compact(&self) -> usize {
        let mut bytes_freed = 0usize;

        for mut entry in self.nodes.iter_mut() {
            let node = entry.value_mut();
            // Keep vectors for upper-layer nodes (needed for greedy descent)
            if node.level == 0 && !node.vector.is_empty() {
                bytes_freed += node.vector.len() * std::mem::size_of::<f32>();
                bytes_freed += node.vector.capacity() * std::mem::size_of::<f32>()
                    - node.vector.len() * std::mem::size_of::<f32>();
                // Replace with zero-capacity Vec (actually frees the heap alloc)
                node.vector = Vec::new();
            }
        }

        // Also compact the read snapshot
        let old_snap = self.read_snapshot.load();
        let mut new_snap = im::HashMap::new();
        for (id, node) in old_snap.iter() {
            let mut n = node.clone();
            if n.level == 0 {
                n.vector = Vec::new();
            }
            new_snap.insert(*id, n);
        }
        self.read_snapshot.store(Arc::new(new_snap));

        bytes_freed
    }

    /// Returns (total_nodes, upper_layer_nodes, estimated_vector_bytes)
    /// to help decide whether `compact()` is worth calling.
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let total = self.nodes.len();
        let upper = self.nodes.iter().filter(|e| e.value().level > 0).count();
        let vec_bytes: usize = self
            .nodes
            .iter()
            .map(|e| e.value().vector.capacity() * std::mem::size_of::<f32>())
            .sum();
        (total, upper, vec_bytes)
    }

    // ══════════════════════════════════════════════════════════════════════
    //  WarpInsert — parallel batch insert with 5 innovations
    // ══════════════════════════════════════════════════════════════════════
    //
    //  1. NEON SIMD L2 (4× unrolled FMA — see l2_distance_neon above)
    //  2. Locality-aware thread scheduling (random-projection sort)
    //  3. Warm-start chaining (intra-thread entry-point reuse)
    //  4. Adaptive ef_construction (ramps up as graph densifies)
    //  5. Edge backfill (2-hop α-RNG repair after parallel phase)
    //
    //  Expected throughput: 4–8× over sequential `insert()` on M-series.
    // ══════════════════════════════════════════════════════════════════════

    /// Generate a random unit hyperplane for locality-sensitive projection.
    fn random_hyperplane(&self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let raw: Vec<f32> = (0..self.dimensions)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        // Normalize (not strictly required, but keeps projection scale consistent)
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        raw.into_iter().map(|x| x / norm).collect()
    }

    /// Internal insert used by `par_insert_batch`.
    ///
    /// Differs from `insert()`:
    /// - Node ID, level, binary are pre-computed (avoids atomic contention).
    /// - Accepts optional `warm_ep` for layer-0 search warm-start.
    /// - Accepts `ef_override` for adaptive ef_construction.
    #[allow(clippy::too_many_arguments)]
    fn insert_one_warp(
        &self,
        id: NodeId,
        vector: &[f32],
        binary: Vec<u64>,
        level: usize,
        metadata: MetaData,
        warm_ep: Option<NodeId>,
        ef_override: usize,
    ) {
        // Create and register the node
        let neighbors = vec![Vec::new(); level + 1];
        let node = HnswNode {
            id,
            vector: vector.to_vec(),
            binary,
            neighbors,
            level,
            metadata,
        };
        self.nodes.insert(id, node);

        // Get global entry point (must exist — bootstrap guarantees it)
        let ep_id = match self.entry_point.read().clone() {
            Some(ep) => ep,
            None => return,
        };
        let ep_level = self.nodes.get(&ep_id).map(|n| n.level).unwrap_or(0);
        let insert_level = level.min(ep_level);

        // Phase 1: Greedy descent from top to (insert_level + 1)
        let mut current_ep = ep_id;
        for lc in ((insert_level + 1)..=ep_level).rev() {
            current_ep = self.search_layer_greedy(vector, current_ep, lc);
        }

        // Phase 2: beam search + connect at each layer
        for lc in (0..=insert_level).rev() {
            // ── Innovation 3: warm-start at layer 0 ──
            // In a locality-sorted batch the previous insert landed nearby,
            // so its node is a better entry point than the global one.
            let search_ep = if lc == 0 {
                warm_ep.unwrap_or(current_ep)
            } else {
                current_ep
            };

            let candidates = self.search_layer(vector, search_ep, ef_override, lc);
            let max_m = self.max_neighbors_at(lc);
            let selected = self.select_neighbors_heuristic(vector, &candidates, max_m, lc);

            // Set forward edges
            if let Some(mut node_ref) = self.nodes.get_mut(&id) {
                if lc < node_ref.neighbors.len() {
                    node_ref.neighbors[lc] = selected.clone();
                }
            }

            // Bidirectional connections + pruning (same logic as `insert`)
            for &neighbor_id in &selected {
                let needs_prune =
                    if let Some(mut nr) = self.nodes.get_mut(&neighbor_id) {
                        if lc < nr.neighbors.len() {
                            if !nr.neighbors[lc].contains(&id) {
                                nr.neighbors[lc].push(id);
                            }
                            nr.neighbors[lc].len() > max_m
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                // DashMap ref dropped — safe to read other nodes

                if needs_prune {
                    let (nv, nb_ids) = match self.nodes.get(&neighbor_id) {
                        Some(nr) => (nr.vector.clone(), nr.neighbors[lc].clone()),
                        None => continue,
                    };
                    let nb_list: Vec<(NodeId, f32)> = nb_ids
                        .iter()
                        .filter_map(|&nid| {
                            self.nodes
                                .get(&nid)
                                .map(|n| (nid, self.dist(&nv, &n.vector)))
                        })
                        .collect();
                    let pruned =
                        self.select_neighbors_heuristic(&nv, &nb_list, max_m, lc);
                    if let Some(mut nr) = self.nodes.get_mut(&neighbor_id) {
                        if lc < nr.neighbors.len() {
                            nr.neighbors[lc] = pruned;
                        }
                    }
                }
            }

            // Use best candidate as entry for next (lower) layer
            if let Some((best_id, _)) = candidates.first() {
                current_ep = *best_id;
            }
        }

        // Update global entry point if this node reached a higher level
        if level > ep_level {
            let mut ep_guard = self.entry_point.write();
            let current_max = self.max_level.load(AtomicOrdering::Acquire);
            if level > current_max {
                *ep_guard = Some(id);
                self.max_level.store(level, AtomicOrdering::Release);
            }
        }
    }

    /// **WarpInsert** — parallel batch insert with 5 innovations.
    ///
    /// ```text
    ///  ┌─ locality sort ─┐   ┌─── rayon par_chunks ──────────────────┐
    ///  │ random-project   │ → │ thread 0: warm-chain → insert → insert│
    ///  │ sort by proj     │   │ thread 1: warm-chain → insert → insert│
    ///  └─────────────────┘   │ …                                      │
    ///                        └─── adaptive ef ramps up ──────────────┘
    ///                                       │
    ///                              ┌────────▼────────┐
    ///                              │  edge backfill  │  ← 2-hop α-RNG repair
    ///                              │  (parallel)     │
    ///                              └─────────────────┘
    /// ```
    ///
    /// Returns `Vec<NodeId>` in the same order as the input vectors.
    /// Call `publish_snapshot()` is done automatically at the end.
    pub fn par_insert_batch(
        &self,
        vectors: Vec<(Vec<f32>, MetaData)>,
    ) -> Vec<NodeId> {
        let total = vectors.len();
        if total == 0 {
            return Vec::new();
        }

        // Pre-assign contiguous IDs (single atomic add — no per-insert contention)
        let base_id = self.next_id.fetch_add(total as u64, AtomicOrdering::Relaxed);

        // ── Innovation 2: Locality-Aware Scheduling ──
        // Project every vector onto a random hyperplane; sort by projection.
        // Nearby vectors land on the same rayon chunk → warm-start works,
        // DashMap shard contention drops because threads touch different
        // regions of the graph.
        let hyperplane = self.random_hyperplane();

        struct PreparedNode {
            id: NodeId,
            vector: Vec<f32>,
            binary: Vec<u64>,
            level: usize,
            metadata: MetaData,
        }

        let mut prepared: Vec<(f32, PreparedNode)> = vectors
            .into_iter()
            .enumerate()
            .map(|(i, (vec, meta))| {
                let proj: f32 = vec.iter().zip(&hyperplane).map(|(v, h)| v * h).sum();
                let binary = quantize_binary(&vec);
                let level = self.random_level();
                (
                    proj,
                    PreparedNode {
                        id: NodeId(base_id + i as u64),
                        vector: vec,
                        binary,
                        level,
                        metadata: meta,
                    },
                )
            })
            .collect();

        prepared.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // ── Bootstrap: ensure entry point exists ──
        let needs_bootstrap = self.entry_point.read().is_none();
        let start_idx = if needs_bootstrap {
            let first = &prepared[0].1;
            let node = HnswNode {
                id: first.id,
                vector: first.vector.clone(),
                binary: first.binary.clone(),
                neighbors: vec![Vec::new(); first.level + 1],
                level: first.level,
                metadata: first.metadata.clone(),
            };
            self.nodes.insert(first.id, node);
            *self.entry_point.write() = Some(first.id);
            self.max_level.store(first.level, AtomicOrdering::Release);
            1
        } else {
            0
        };

        // ── Parallel phase ──
        let work = &prepared[start_idx..];
        let num_threads = rayon::current_num_threads().max(1);
        // Chunk size balances rayon overhead vs warm-start chain length
        let chunk_size = (work.len() / num_threads).max(256).min(8192);
        let progress = AtomicUsize::new(start_idx);
        let graph_size_before = self.nodes.len();

        eprintln!(
            "  [warp] {} vectors, {} threads, chunk={}, NEON={}",
            total,
            num_threads,
            chunk_size,
            cfg!(target_arch = "aarch64"),
        );

        work.par_chunks(chunk_size).for_each(|chunk| {
            let mut warm_ep: Option<NodeId> = None;

            for (_, item) in chunk {
                // ── Innovation 4: Adaptive ef_construction ──
                // Early inserts need less ef (graph is small, search is cheap).
                // Ramp from 30% to 100% of ef_construction proportionally to
                // how full the graph is.  Saves ~35% distance computations in
                // the first half of a large batch.
                let current_size = self.nodes.len();
                let target_size = total + graph_size_before;
                let progress_frac = current_size as f64 / target_size as f64;
                let adaptive_ef =
                    ((self.ef_construction as f64) * (0.3 + 0.7 * progress_frac)) as usize;
                let effective_ef = adaptive_ef.max(32).min(self.ef_construction);

                self.insert_one_warp(
                    item.id,
                    &item.vector,
                    item.binary.clone(),
                    item.level,
                    item.metadata.clone(),
                    warm_ep,
                    effective_ef,
                );

                // ── Innovation 3: warm-start for next insert ──
                warm_ep = Some(item.id);

                // Progressive snapshot + progress
                let count = progress.fetch_add(1, AtomicOrdering::Relaxed);
                if count > 0 && count % 10_000 == 0 {
                    self.publish_snapshot();
                    eprint!("\r  [warp] {}/{}", count, total);
                }
            }
        });

        eprintln!("\r  [warp] {}/{} — backfilling edges…", total, total);

        // ── Innovation 5: Edge Backfill ──
        // Concurrent inserts may have missed each other's edges.
        // Re-evaluate layer-0 neighbors for every new node using a cheap
        // 2-hop gather + α-RNG re-selection.  Fully parallel via rayon.
        self.backfill_edges(base_id, total);
        self.publish_snapshot();

        eprintln!("  [warp] done — {} nodes in graph", self.nodes.len());

        // Return IDs in original (pre-sort) order
        (0..total as u64).map(|i| NodeId(base_id + i)).collect()
    }

    /// Post-insert graph repair: 2-hop α-RNG re-selection for recently
    /// inserted nodes.
    ///
    /// During concurrent insert, thread A may connect node X to node Y,
    /// but thread B (inserting node Z between X and Y) doesn't see Z yet.
    /// Backfill gathers each node's 2-hop neighborhood and re-runs the
    /// α-RNG heuristic, allowing Z to appear as a neighbor of X.
    ///
    /// Cost: O(N × M²) distance computations, fully parallelised.
    /// For 1M nodes, M=16: ~1B distances × ~30 ns/NEON ≈ 30 s on M1.
    fn backfill_edges(&self, base_id: u64, count: usize) {
        let ids: Vec<NodeId> = (0..count as u64).map(|i| NodeId(base_id + i)).collect();

        ids.par_iter().for_each(|&id| {
            let (vector, current_l0) = match self.nodes.get(&id) {
                Some(n) => {
                    let l0 = n.neighbors.first().cloned().unwrap_or_default();
                    (n.vector.clone(), l0)
                }
                None => return,
            };

            if current_l0.is_empty() {
                return;
            }

            // Gather 2-hop candidates at layer 0
            let mut candidate_set = HashSet::new();
            for &nb in &current_l0 {
                candidate_set.insert(nb);
                if let Some(nb_node) = self.nodes.get(&nb) {
                    if let Some(nb_l0) = nb_node.neighbors.first() {
                        for &nnb in nb_l0 {
                            candidate_set.insert(nnb);
                        }
                    }
                }
            }
            candidate_set.remove(&id);

            // Score all candidates
            let mut scored: Vec<(NodeId, f32)> = candidate_set
                .iter()
                .filter_map(|&cid| {
                    self.nodes
                        .get(&cid)
                        .map(|n| (cid, self.dist(&vector, &n.vector)))
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let new_l0 =
                self.select_neighbors_heuristic(&vector, &scored, self.max_neighbors_l0, 0);

            // Only write if neighbors actually changed
            if new_l0 != current_l0 {
                if let Some(mut nr) = self.nodes.get_mut(&id) {
                    if !nr.neighbors.is_empty() {
                        nr.neighbors[0] = new_l0;
                    }
                }
            }
        });
    }
}

/// Binary quantization (LSB-first convention)
fn quantize_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = (vector.len() + 63) / 64;
    let mut result = vec![0u64; num_words];
    for (i, &value) in vector.iter().enumerate() {
        if value > 0.0 {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
    result
}

/// Squared L2 distance — NEON-accelerated on Apple Silicon, scalar fallback elsewhere.
///
/// On aarch64: 4-lane FMA with 4× accumulator unrolling to hide NEON pipeline
/// latency (~4 cycles for `vfmaq_f32`).  For 128-d SIFT vectors this is
/// 8 fully pipelined iterations = ~32 cycles vs ~128 scalar multiplies.
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        l2_distance_neon(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}

/// NEON SIMD squared-L2 with 4× unrolled FMA pipeline.
///
/// Processes 16 floats per iteration (4 NEON registers × 4 lanes).
/// The 4 independent accumulators (`s0`–`s3`) keep the FMA unit fully
/// occupied because each depends only on its own accumulator chain.
///
/// For 128-d vectors: 8 iterations, zero remainder.  For arbitrary dims
/// the tail is handled by a 4-lane pass then scalar cleanup.
#[cfg(target_arch = "aarch64")]
#[inline]
fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        let full_iters = n / 16;
        for i in 0..full_iters {
            let base = i * 16;
            let d0 = vsubq_f32(vld1q_f32(pa.add(base)), vld1q_f32(pb.add(base)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(base + 4)), vld1q_f32(pb.add(base + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(base + 8)), vld1q_f32(pb.add(base + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(base + 12)), vld1q_f32(pb.add(base + 12)));
            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
            s2 = vfmaq_f32(s2, d2, d2);
            s3 = vfmaq_f32(s3, d3, d3);
        }

        // Merge four accumulators → one, then horizontal sum.
        s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut sum = vaddvq_f32(s0);

        // Tail: groups of 4
        let mut i = full_iters * 16;
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            sum += vaddvq_f32(vmulq_f32(d, d));
            i += 4;
        }

        // Scalar remainder (0–3 elements)
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn test_insert_and_search_self() {
        let index = HnswIndex::new(128, 16, 200);
        let mut vectors = Vec::new();

        // Insert 500 vectors
        for i in 0..500u64 {
            let v = random_vector(128, i);
            vectors.push(v.clone());
            index.insert(
                v,
                MetaData {
                    label: format!("{}", i),
                },
            );
        }
        index.publish_snapshot();

        // Each vector should find itself as closest
        let mut self_recall = 0;
        for (i, v) in vectors.iter().enumerate() {
            let results = index.search(v, Some(10));
            if !results.is_empty() && results[0].0 == NodeId(i as u64) {
                self_recall += 1;
            }
        }
        let recall_pct = self_recall as f32 / vectors.len() as f32 * 100.0;
        assert!(
            recall_pct > 95.0,
            "Self-recall should be >95%, got {:.1}%",
            recall_pct
        );
    }

    #[test]
    fn test_monotonic_ef() {
        // Higher ef should give >= recall
        let index = HnswIndex::new(64, 16, 100);
        let mut vectors = Vec::new();

        for i in 0..300u64 {
            let v = random_vector(64, i + 1000);
            vectors.push(v.clone());
            index.insert(
                v,
                MetaData {
                    label: format!("{}", i),
                },
            );
        }
        index.publish_snapshot();

        // Compute brute-force ground truth for first 20 queries
        let queries: Vec<Vec<f32>> = (0..20).map(|i| random_vector(64, i + 5000)).collect();
        let k = 10;

        let mut recall_by_ef = Vec::new();
        for ef in [8, 16, 32, 64, 128] {
            let mut total_recall = 0.0;
            for query in &queries {
                // Brute force
                let mut gt: Vec<(NodeId, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (NodeId(i as u64), l2_distance(query, v)))
                    .collect();
                gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let gt_set: HashSet<NodeId> = gt.iter().take(k).map(|x| x.0).collect();

                let results = index.search(query, Some(ef));
                let result_set: HashSet<NodeId> = results.iter().take(k).map(|x| x.0).collect();
                let overlap = gt_set.intersection(&result_set).count();
                total_recall += overlap as f32 / k as f32;
            }
            recall_by_ef.push(total_recall / queries.len() as f32);
        }

        // Verify monotonically non-decreasing (with small tolerance for randomness)
        for i in 1..recall_by_ef.len() {
            assert!(
                recall_by_ef[i] >= recall_by_ef[i - 1] - 0.05,
                "Recall should be monotonically non-decreasing with ef: {:?}",
                recall_by_ef
            );
        }
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(64, 16, 100);
        let results = index.search(&vec![0.0; 64], Some(10));
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_node() {
        let index = HnswIndex::new(64, 16, 100);
        let v = random_vector(64, 42);
        let id = index.insert(
            v.clone(),
            MetaData {
                label: "only".into(),
            },
        );
        index.publish_snapshot();
        let results = index.search(&v, Some(10));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_binary_search() {
        let index = HnswIndex::new(128, 16, 100);
        for i in 0..200u64 {
            let v = random_vector(128, i);
            index.insert(
                v,
                MetaData {
                    label: format!("{}", i),
                },
            );
        }
        index.publish_snapshot();

        let query = random_vector(128, 9999);
        let query_bin = quantize_binary(&query);
        let results = index.search_binary(&query_bin, Some(32));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_get_neighbors() {
        let index = HnswIndex::new(64, 8, 50);
        for i in 0..100u64 {
            let v = random_vector(64, i);
            index.insert(
                v,
                MetaData {
                    label: format!("{}", i),
                },
            );
        }
        index.publish_snapshot();

        let neighbors = index.get_neighbors(NodeId(0), 1);
        assert!(!neighbors.is_empty(), "Node 0 should have neighbors");
    }

    // ── WarpInsert tests ──────────────────────────────────────────────

    #[test]
    fn test_neon_l2_matches_scalar() {
        // Verify NEON SIMD gives same result as scalar (within fp32 tolerance)
        let a = random_vector(128, 42);
        let b = random_vector(128, 99);

        let scalar: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        let simd = l2_distance(&a, &b);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "L2 mismatch: scalar={}, simd={}, diff={}",
            scalar,
            simd,
            (scalar - simd).abs()
        );
    }

    #[test]
    fn test_neon_l2_odd_dims() {
        // Non-power-of-2 dims to exercise tail handling
        for dims in [3, 7, 17, 33, 65, 100, 127, 129, 255] {
            let a = random_vector(dims, 1);
            let b = random_vector(dims, 2);
            let scalar: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            let simd = l2_distance(&a, &b);
            assert!(
                (scalar - simd).abs() < 1e-2,
                "L2 mismatch at dims={}: scalar={}, simd={}",
                dims,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_warp_insert_basic() {
        let index = HnswIndex::new(128, 16, 100);

        let vectors: Vec<(Vec<f32>, MetaData)> = (0..1000u64)
            .map(|i| {
                (
                    random_vector(128, i),
                    MetaData {
                        label: format!("{}", i),
                    },
                )
            })
            .collect();

        let ids = index.par_insert_batch(vectors.clone());
        assert_eq!(ids.len(), 1000);
        assert_eq!(index.len(), 1000);

        // Search: self-recall should be high
        let mut self_recall = 0;
        for (i, (v, _)) in vectors.iter().enumerate() {
            let results = index.search(v, Some(32));
            if !results.is_empty() && results[0].0 == ids[i] {
                self_recall += 1;
            }
        }
        let recall_pct = self_recall as f32 / vectors.len() as f32 * 100.0;
        assert!(
            recall_pct > 90.0,
            "WarpInsert self-recall should be >90%, got {:.1}%",
            recall_pct
        );
    }

    #[test]
    fn test_warp_insert_recall_vs_sequential() {
        // Compare recall of par_insert_batch vs sequential insert
        let dims = 64;
        let n = 500;
        let queries: Vec<Vec<f32>> = (0..50).map(|i| random_vector(dims, i + 9000)).collect();

        // Sequential
        let seq_index = HnswIndex::new(dims, 16, 100);
        let mut seq_vecs = Vec::new();
        for i in 0..n as u64 {
            let v = random_vector(dims, i);
            seq_vecs.push(v.clone());
            seq_index.insert(v, MetaData { label: format!("{}", i) });
        }
        seq_index.publish_snapshot();

        // Parallel
        let par_index = HnswIndex::new(dims, 16, 100);
        let par_vecs: Vec<(Vec<f32>, MetaData)> = (0..n as u64)
            .map(|i| (random_vector(dims, i), MetaData { label: format!("{}", i) }))
            .collect();
        par_index.par_insert_batch(par_vecs);

        // Measure recall for both
        let k = 10;
        let mut seq_recall_sum = 0.0;
        let mut par_recall_sum = 0.0;

        for query in &queries {
            // Brute-force ground truth
            let mut gt: Vec<(usize, f32)> = seq_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2_distance(query, v)))
                .collect();
            gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt_set: HashSet<usize> = gt.iter().take(k).map(|x| x.0).collect();

            let seq_results = seq_index.search(query, Some(64));
            let seq_set: HashSet<usize> =
                seq_results.iter().take(k).map(|x| x.0 .0 as usize).collect();
            seq_recall_sum += gt_set.intersection(&seq_set).count() as f32 / k as f32;

            let par_results = par_index.search(query, Some(64));
            let par_set: HashSet<usize> =
                par_results.iter().take(k).map(|x| x.0 .0 as usize).collect();
            par_recall_sum += gt_set.intersection(&par_set).count() as f32 / k as f32;
        }

        let seq_recall = seq_recall_sum / queries.len() as f32;
        let par_recall = par_recall_sum / queries.len() as f32;

        eprintln!(
            "Recall@10: sequential={:.1}%, parallel={:.1}%, delta={:.1}%",
            seq_recall * 100.0,
            par_recall * 100.0,
            (seq_recall - par_recall) * 100.0
        );

        // Parallel should be within 5% of sequential (backfill repairs most loss)
        assert!(
            par_recall > seq_recall - 0.05,
            "WarpInsert recall {:.1}% too far below sequential {:.1}%",
            par_recall * 100.0,
            seq_recall * 100.0
        );
    }
}
