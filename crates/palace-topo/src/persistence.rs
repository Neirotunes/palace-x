// Copyright (c) 2026 M.Diach
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Persistent homology for ego-graphs.
//!
//! Upgrades from raw β₁ (first Betti number) to persistence diagrams that capture
//! the full multi-scale topological structure of local neighborhoods.
//!
//! # Algorithm
//!
//! **H₀ persistence** (connected components):
//! 1. Sort edges by weight (cosine distance between neighbors)
//! 2. Process edges via Union-Find
//! 3. When two components merge, the younger one "dies" → record (birth, death)
//!
//! **H₁ persistence** (cycles):
//! 1. Build flag complex from the ego-graph cliques
//! 2. Sort simplices by filtration value
//! 3. Column reduction on the boundary matrix
//! 4. Each unpaired column in the boundary of 2-simplices that creates a cycle
//!    records a (birth, death) pair
//!
//! # References
//! - Extended-RaBitQ, SIGMOD 2025
//! - Edelsbrunner & Harer, "Computational Topology", 2010

use crate::ego_graph::EgoGraph;
use palace_core::NodeId;
use std::collections::HashMap;

/// A persistence pair (birth_time, death_time).
/// For H₀: birth = 0.0 for all components (they're born at filtration = 0),
///          death = weight of the edge that merges them.
/// For H₁: birth = weight of the edge that completes the cycle,
///          death = weight of the edge/triangle that fills it.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
    pub dimension: usize, // 0 for H₀, 1 for H₁
}

impl PersistencePair {
    pub fn persistence(&self) -> f32 {
        (self.death - self.birth).abs()
    }
}

/// A persistence diagram: collection of (birth, death) pairs.
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl Default for PersistenceDiagram {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistenceDiagram {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Total persistence: sum of |death - birth| for all pairs.
    /// Higher value = more persistent topological features = richer structure.
    pub fn total_persistence(&self) -> f32 {
        self.pairs.iter().map(|p| p.persistence()).sum()
    }

    /// Total persistence for a specific dimension.
    pub fn total_persistence_dim(&self, dim: usize) -> f32 {
        self.pairs
            .iter()
            .filter(|p| p.dimension == dim)
            .map(|p| p.persistence())
            .sum()
    }

    /// Number of features in a given dimension.
    pub fn count_dim(&self, dim: usize) -> usize {
        self.pairs.iter().filter(|p| p.dimension == dim).count()
    }

    /// Maximum persistence across all pairs.
    pub fn max_persistence(&self) -> f32 {
        self.pairs
            .iter()
            .map(|p| p.persistence())
            .fold(0.0f32, f32::max)
    }
}

// ─── Union-Find for H₀ ───────────────────────────────────────────

struct PersistenceUF {
    parent: Vec<usize>,
    rank: Vec<usize>,
    birth_time: Vec<f32>,
}

impl PersistenceUF {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth_time: vec![0.0; n], // all components born at time 0
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    /// Union two sets. Returns Some(death_time) if a merge happened (younger dies).
    fn union(&mut self, x: usize, y: usize, edge_weight: f32) -> Option<PersistencePair> {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return None; // same component — this edge creates a cycle (H₁ feature)
        }

        // The younger component (higher birth_time, or arbitrary if equal) dies
        let (survivor, dying) = if self.rank[rx] >= self.rank[ry] {
            (rx, ry)
        } else {
            (ry, rx)
        };

        self.parent[dying] = survivor;
        if self.rank[survivor] == self.rank[dying] {
            self.rank[survivor] += 1;
        }

        let pair = PersistencePair {
            birth: self.birth_time[dying],
            death: edge_weight,
            dimension: 0,
        };

        Some(pair)
    }
}

// ─── H₀ Persistence ──────────────────────────────────────────────

/// Compute H₀ persistence diagram from a weighted ego-graph.
///
/// Process edges in order of increasing weight (filtration).
/// Each edge that merges two components creates a death event.
///
/// # Arguments
/// * `ego_graph` - The ego-graph
/// * `weight_fn` - Function returning weight (e.g. cosine distance) for an edge
pub fn h0_persistence(
    ego_graph: &EgoGraph,
    weight_fn: &impl Fn(NodeId, NodeId) -> f32,
) -> PersistenceDiagram {
    if ego_graph.num_vertices == 0 {
        return PersistenceDiagram::new();
    }

    // Map NodeIds to indices
    let id_to_idx: HashMap<NodeId, usize> = ego_graph
        .vertices
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Weight and sort edges
    let mut weighted_edges: Vec<(f32, NodeId, NodeId)> = ego_graph
        .edges
        .iter()
        .map(|&(u, v)| (weight_fn(u, v), u, v))
        .collect();
    weighted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = PersistenceUF::new(ego_graph.num_vertices);
    let mut diagram = PersistenceDiagram::new();

    for (w, u, v) in &weighted_edges {
        let ui = id_to_idx[u];
        let vi = id_to_idx[v];
        if let Some(pair) = uf.union(ui, vi, *w) {
            // Skip trivial pairs where birth == death (zero persistence)
            if pair.persistence() > 1e-10 {
                diagram.pairs.push(pair);
            }
        }
    }

    diagram
}

// ─── H₁ Persistence (simplified via edge-triangle approach) ──────

/// Compute H₁ persistence diagram.
///
/// Uses the flag complex approach:
/// 1. Enumerate all triangles (3-cliques) in the ego-graph
/// 2. Sort edges and triangles by filtration value
/// 3. An edge that doesn't merge components (in H₀ sense) creates an H₁ birth
/// 4. A triangle that "fills" the cycle creates an H₁ death
///
/// This is a simplified version that captures the essential H₁ features
/// without full boundary matrix reduction.
pub fn h1_persistence(
    ego_graph: &EgoGraph,
    weight_fn: &impl Fn(NodeId, NodeId) -> f32,
) -> PersistenceDiagram {
    if ego_graph.num_vertices < 3 || ego_graph.num_edges < 3 {
        return PersistenceDiagram::new();
    }

    // Build adjacency set for triangle enumeration
    let mut adj: HashMap<NodeId, std::collections::HashSet<NodeId>> = HashMap::new();
    for &(u, v) in &ego_graph.edges {
        adj.entry(u).or_default().insert(v);
        adj.entry(v).or_default().insert(u);
    }

    // Map NodeIds to indices for UF
    let id_to_idx: HashMap<NodeId, usize> = ego_graph
        .vertices
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Weight edges
    let mut weighted_edges: Vec<(f32, NodeId, NodeId)> = ego_graph
        .edges
        .iter()
        .map(|&(u, v)| (weight_fn(u, v), u, v))
        .collect();
    weighted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Enumerate triangles with filtration value = max edge weight
    let mut triangles: Vec<(f32, NodeId, NodeId, NodeId)> = Vec::new();
    let vertex_set: std::collections::HashSet<NodeId> =
        ego_graph.vertices.iter().copied().collect();

    for &u in &ego_graph.vertices {
        if let Some(neighbors_u) = adj.get(&u) {
            for &v in neighbors_u {
                if v <= u {
                    continue;
                }
                if let Some(neighbors_v) = adj.get(&v) {
                    for &w in neighbors_v {
                        if w <= v {
                            continue;
                        }
                        if neighbors_u.contains(&w) && vertex_set.contains(&w) {
                            // Triangle (u, v, w)
                            let w_uv = weight_fn(u, v);
                            let w_uw = weight_fn(u, w);
                            let w_vw = weight_fn(v, w);
                            let max_w = w_uv.max(w_uw).max(w_vw);
                            triangles.push((max_w, u, v, w));
                        }
                    }
                }
            }
        }
    }
    triangles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Process: edges that don't merge components create H₁ births.
    // Triangles fill cycles and create H₁ deaths.
    let mut uf = PersistenceUF::new(ego_graph.num_vertices);
    let mut cycle_births: Vec<f32> = Vec::new(); // birth times of H₁ features

    for &(w, u, v) in &weighted_edges {
        let ui = id_to_idx[&u];
        let vi = id_to_idx[&v];
        if uf.union(ui, vi, w).is_none() {
            // Same component — this edge creates a cycle
            cycle_births.push(w);
        }
    }

    // Sort births ascending
    cycle_births.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Match cycle births with triangle deaths (greedy pairing)
    let mut diagram = PersistenceDiagram::new();
    let mut birth_idx = 0;

    for &(tri_weight, _, _, _) in &triangles {
        if birth_idx < cycle_births.len() && cycle_births[birth_idx] <= tri_weight {
            let pair = PersistencePair {
                birth: cycle_births[birth_idx],
                death: tri_weight,
                dimension: 1,
            };
            if pair.persistence() > 1e-10 {
                diagram.pairs.push(pair);
            }
            birth_idx += 1;
        }
    }

    // Remaining unpaired cycles persist to infinity — represent as (birth, max_weight)
    let max_weight = weighted_edges.last().map(|e| e.0).unwrap_or(1.0);
    while birth_idx < cycle_births.len() {
        diagram.pairs.push(PersistencePair {
            birth: cycle_births[birth_idx],
            death: max_weight,
            dimension: 1,
        });
        birth_idx += 1;
    }

    diagram
}

// ─── Combined Persistence ─────────────────────────────────────────

/// Compute full persistence diagram (H₀ + H₁) for an ego-graph.
pub fn persistence_diagram(
    ego_graph: &EgoGraph,
    weight_fn: &impl Fn(NodeId, NodeId) -> f32,
) -> PersistenceDiagram {
    let h0 = h0_persistence(ego_graph, weight_fn);
    let h1 = h1_persistence(ego_graph, weight_fn);

    let mut combined = PersistenceDiagram::new();
    combined.pairs.extend(h0.pairs);
    combined.pairs.extend(h1.pairs);
    combined
}

/// Compute total persistence score for use in d_total metric.
///
/// This replaces raw β₁ with a weighted sum of persistence values,
/// giving a more nuanced measure of topological richness.
///
/// score = total_h0_persistence + λ * total_h1_persistence
///
/// where λ controls relative importance of cycles vs components.
pub fn total_persistence_score(
    ego_graph: &EgoGraph,
    weight_fn: &impl Fn(NodeId, NodeId) -> f32,
    lambda: f32,
) -> f32 {
    let diagram = persistence_diagram(ego_graph, weight_fn);
    diagram.total_persistence_dim(0) + lambda * diagram.total_persistence_dim(1)
}

/// Palace-X topological distance using persistence diagrams instead of raw β₁.
///
/// d_total_persistence(x, y) = α · d_cosine + β · exp(-persistence_score / max(|E|, 1))
pub fn d_total_persistence(
    cosine_dist: f32,
    ego_graph: &EgoGraph,
    weight_fn: &impl Fn(NodeId, NodeId) -> f32,
    alpha: f32,
    beta: f32,
    lambda: f32,
) -> f32 {
    let score = total_persistence_score(ego_graph, weight_fn, lambda);
    let e = ego_graph.num_edges.max(1) as f32;
    let structural = (-score / e).exp();
    alpha * cosine_dist + beta * structural
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    #[allow(dead_code)]
    struct TestGraph {
        adjacency: StdHashMap<NodeId, Vec<NodeId>>,
    }

    #[allow(dead_code)]
    impl TestGraph {
        fn new() -> Self {
            Self {
                adjacency: StdHashMap::new(),
            }
        }

        fn add_edge(&mut self, u: NodeId, v: NodeId) {
            self.adjacency.entry(u).or_default().push(v);
            self.adjacency.entry(v).or_default().push(u);
        }

        fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
            self.adjacency.get(&node).cloned().unwrap_or_default()
        }
    }

    fn build_ego(vertices: Vec<NodeId>, edges: Vec<(NodeId, NodeId)>) -> EgoGraph {
        let nv = vertices.len();
        let ne = edges.len();
        EgoGraph {
            vertices,
            edges,
            num_vertices: nv,
            num_edges: ne,
            max_ego_edges: None,
        }
    }

    // Simple weight function: absolute difference of node IDs
    fn simple_weight(u: NodeId, v: NodeId) -> f32 {
        ((u.0 as f32) - (v.0 as f32)).abs() * 0.1
    }

    #[test]
    fn test_h0_empty() {
        let ego = build_ego(vec![], vec![]);
        let diagram = h0_persistence(&ego, &simple_weight);
        assert!(diagram.pairs.is_empty());
    }

    #[test]
    fn test_h0_single_edge() {
        let ego = build_ego(vec![NodeId(0), NodeId(1)], vec![(NodeId(0), NodeId(1))]);
        let diagram = h0_persistence(&ego, &simple_weight);
        // One merge: component dies
        assert_eq!(diagram.pairs.len(), 1);
        assert_eq!(diagram.pairs[0].dimension, 0);
        assert!(diagram.pairs[0].death > 0.0);
    }

    #[test]
    fn test_h0_triangle() {
        let ego = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(0), NodeId(2)),
                (NodeId(1), NodeId(2)),
            ],
        );
        let diagram = h0_persistence(&ego, &simple_weight);
        // 3 vertices, 3 edges: 2 merges, 1 cycle-creating edge
        assert_eq!(diagram.pairs.len(), 2);
    }

    #[test]
    fn test_h1_square_has_persistent_cycle() {
        // Square (4-cycle) without diagonal — cycle persists because
        // no triangle exists to kill it
        let ego = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(1), NodeId(2)),
                (NodeId(2), NodeId(3)),
                (NodeId(3), NodeId(0)),
            ],
        );
        let diagram = h1_persistence(&ego, &simple_weight);
        // Square has one H₁ feature (cycle never killed — no triangles)
        assert!(!diagram.pairs.is_empty(), "Square should have H₁ features");
        assert_eq!(diagram.pairs[0].dimension, 1);
    }

    #[test]
    fn test_h1_tree_no_cycles() {
        let ego = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(1), NodeId(2)),
                (NodeId(2), NodeId(3)),
            ],
        );
        let diagram = h1_persistence(&ego, &simple_weight);
        // Tree has no cycles
        assert!(diagram.pairs.is_empty());
    }

    #[test]
    fn test_total_persistence_score() {
        let ego = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(0), NodeId(2)),
                (NodeId(1), NodeId(2)),
            ],
        );
        let score = total_persistence_score(&ego, &simple_weight, 1.0);
        assert!(score > 0.0, "Triangle should have positive persistence");
    }

    #[test]
    fn test_d_total_persistence_square_vs_line() {
        // Square has a persistent cycle, line does not
        let square = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(1), NodeId(2)),
                (NodeId(2), NodeId(3)),
                (NodeId(3), NodeId(0)),
            ],
        );
        let line = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(1), NodeId(2)),
                (NodeId(2), NodeId(3)),
            ],
        );

        let d_sq = d_total_persistence(0.5, &square, &simple_weight, 0.5, 0.5, 1.0);
        let d_line = d_total_persistence(0.5, &line, &simple_weight, 0.5, 0.5, 1.0);

        // Both should produce valid finite distances
        assert!(
            d_sq.is_finite() && d_sq > 0.0,
            "Square d_total should be finite positive"
        );
        assert!(
            d_line.is_finite() && d_line > 0.0,
            "Line d_total should be finite positive"
        );
        // Square has persistence signal, line does not — they should differ
        assert!(
            (d_sq - d_line).abs() > 1e-6,
            "Square ({}) and line ({}) should have different d_total due to topology",
            d_sq,
            d_line
        );
    }

    #[test]
    fn test_persistence_diagram_combined() {
        // Square (0-1-2-3-0) + tail (3-4): cycle + tree edge
        let ego = build_ego(
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
            vec![
                (NodeId(0), NodeId(1)),
                (NodeId(1), NodeId(2)),
                (NodeId(2), NodeId(3)),
                (NodeId(3), NodeId(0)),
                (NodeId(3), NodeId(4)),
            ],
        );
        let diagram = persistence_diagram(&ego, &simple_weight);
        let h0_count = diagram.count_dim(0);
        let h1_count = diagram.count_dim(1);
        assert!(h0_count > 0, "Should have H₀ features");
        // Square creates persistent H₁ (no triangle to kill it)
        assert!(h1_count > 0, "Should have H₁ features from square cycle");
    }
}
