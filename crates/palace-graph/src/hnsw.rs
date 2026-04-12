// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hierarchical Navigable Small World (HNSW) index — UMA-native implementation
//!
//! Key design decisions:
//! - Level assignment via `floor(-ln(uniform()) * level_mult)` (Malkov/Yashunin 2018)
//! - M_max0 = 2*M at layer 0 for denser base connectivity
//! - α-RNG neighbor selection heuristic (Vamana-style, default α=1.2)
//! - DashMap for concurrent writes, ArcSwap snapshot for wait-free reads
//! - DistanceMetric dispatch: L2 for raw features (SIFT), Cosine for embeddings

use crate::node::{cosine_distance, hamming_distance, GraphNode, MetaData};
use arc_swap::ArcSwap;
use dashmap::DashMap;
use palace_core::NodeId;
use parking_lot::RwLock;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
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
        Self::with_params(dimensions, max_neighbors, ef_construction, 1.2, HnswDistanceMetric::L2)
    }

    /// Create with cosine metric (for normalized embeddings)
    pub fn with_cosine(dimensions: usize, max_neighbors: usize, ef_construction: usize) -> Self {
        Self::with_params(dimensions, max_neighbors, ef_construction, 1.2, HnswDistanceMetric::Cosine)
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
                    if let Some(mut neighbor_ref) = self.nodes.get_mut(&neighbor_id) {
                        if lc < neighbor_ref.neighbors.len() {
                            if !neighbor_ref.neighbors[lc].contains(&id) {
                                neighbor_ref.neighbors[lc].push(id);
                            }

                            // Prune if over capacity
                            if neighbor_ref.neighbors[lc].len() > max_m {
                                let neighbor_vec = neighbor_ref.vector.clone();
                                let nb_list: Vec<(NodeId, f32)> = neighbor_ref.neighbors[lc]
                                    .iter()
                                    .filter_map(|&nid| {
                                        self.nodes.get(&nid).map(|n| (nid, self.dist(&neighbor_vec, &n.vector)))
                                    })
                                    .collect();
                                let pruned = self.select_neighbors_heuristic(&neighbor_vec, &nb_list, max_m, lc);
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
            if level > ep_level {
                *self.entry_point.write() = Some(id);
                self.max_level.store(level, AtomicOrdering::Relaxed);
            }
        } else {
            // First node — set as entry point
            *self.entry_point.write() = Some(id);
            self.max_level.store(level, AtomicOrdering::Relaxed);
        }

        // Publish snapshot for wait-free reads
        self.publish_snapshot();
        id
    }

    /// Greedy 1-NN search at a single layer (for navigational descent)
    fn search_layer_greedy(&self, query: &[f32], entry: NodeId, layer: usize) -> NodeId {
        let mut current = entry;
        let mut current_dist = self.nodes.get(&current)
            .map(|n| self.dist(query, &n.vector))
            .unwrap_or(f32::MAX);

        loop {
            let mut improved = false;
            let neighbors: Vec<NodeId> = self.nodes.get(&current)
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
        let entry_dist = self.nodes.get(&entry)
            .map(|n| self.dist(query, &n.vector))
            .unwrap_or(f32::MAX);

        let mut visited = HashSet::new();
        visited.insert(entry);

        // Min-heap: closest candidates to explore
        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate { id: entry, distance: entry_dist });

        // Max-heap: farthest in result set (for pruning)
        let mut result = BinaryHeap::new();
        result.push(FarCandidate { id: entry, distance: entry_dist });

        while let Some(Candidate { id: current, distance: current_dist }) = candidates.pop() {
            // Stop if closest unexplored is farther than farthest result
            let farthest_result_dist = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);
            if current_dist > farthest_result_dist && result.len() >= ef {
                break;
            }

            // Explore neighbors at this layer
            let neighbors: Vec<NodeId> = self.nodes.get(&current)
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
                    let d = self.dist(query, &nb.vector);

                    let farthest = result.peek().map(|r| r.distance).unwrap_or(f32::MAX);
                    if d < farthest || result.len() < ef {
                        candidates.push(Candidate { id: nb_id, distance: d });
                        result.push(FarCandidate { id: nb_id, distance: d });

                        if result.len() > ef {
                            result.pop(); // remove farthest
                        }
                    }
                }
            }
        }

        // Drain result heap into sorted vec
        let mut results: Vec<(NodeId, f32)> = result
            .into_iter()
            .map(|fc| (fc.id, fc.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// α-RNG neighbor selection heuristic
    /// Prefers close AND diverse neighbors. A candidate is accepted if:
    /// dist(candidate, query) <= alpha * dist(candidate, any_selected)
    fn select_neighbors_heuristic(
        &self,
        query: &[f32],
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
            let mut is_diverse = true;
            for sv in &selected_vecs {
                let dist_to_selected = self.dist(&cand_vec, sv);
                if *cand_dist > self.alpha * dist_to_selected {
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
    /// Returns Vec<(NodeId, f32)> sorted by distance ascending
    pub fn search(&self, query: &[f32], ef: Option<usize>) -> Vec<(NodeId, f32)> {
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
            let mut current_dist = self.nodes.get(&current)
                .map(|n| hamming_distance(&n.binary, query_binary))
                .unwrap_or(u32::MAX);
            loop {
                let mut improved = false;
                let neighbors: Vec<NodeId> = self.nodes.get(&current)
                    .map(|n| if lc < n.neighbors.len() { n.neighbors[lc].clone() } else { Vec::new() })
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
                if !improved { break; }
            }
        }

        // Beam search at layer 0 with Hamming
        let entry_dist = self.nodes.get(&current)
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

            let neighbors: Vec<NodeId> = self.nodes.get(&curr_id)
                .map(|n| if n.neighbors.is_empty() { Vec::new() } else { n.neighbors[0].clone() })
                .unwrap_or_default();

            for nb_id in neighbors {
                if visited.contains(&nb_id) { continue; }
                visited.insert(nb_id);

                if let Some(nb) = self.nodes.get(&nb_id) {
                    let d = hamming_distance(&nb.binary, query_binary);
                    let farthest = result.peek().map(|r| r.0).unwrap_or(u32::MAX);
                    if d < farthest || result.len() < ef {
                        candidates.push(std::cmp::Reverse((d, nb_id)));
                        result.push((d, nb_id));
                        if result.len() > ef { result.pop(); }
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

    /// Get vector for a node
    pub fn get_vector(&self, id: &NodeId) -> Option<Vec<f32>> {
        self.nodes.get(id).map(|n| n.vector.clone())
    }

    /// Get metadata label for a node
    pub fn get_label(&self, id: &NodeId) -> Option<String> {
        self.nodes.get(id).map(|n| n.metadata.label.clone())
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

/// Squared L2 distance
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
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
            index.insert(v, MetaData { label: format!("{}", i) });
        }

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
            index.insert(v, MetaData { label: format!("{}", i) });
        }

        // Compute brute-force ground truth for first 20 queries
        let queries: Vec<Vec<f32>> = (0..20).map(|i| random_vector(64, i + 5000)).collect();
        let k = 10;

        let mut recall_by_ef = Vec::new();
        for ef in [8, 16, 32, 64, 128] {
            let mut total_recall = 0.0;
            for query in &queries {
                // Brute force
                let mut gt: Vec<(NodeId, f32)> = vectors.iter().enumerate()
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
        let id = index.insert(v.clone(), MetaData { label: "only".into() });
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
            index.insert(v, MetaData { label: format!("{}", i) });
        }

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
            index.insert(v, MetaData { label: format!("{}", i) });
        }

        let neighbors = index.get_neighbors(NodeId(0), 1);
        assert!(!neighbors.is_empty(), "Node 0 should have neighbors");
    }
}
