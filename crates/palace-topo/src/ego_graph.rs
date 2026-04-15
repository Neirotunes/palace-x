// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ego-graph construction for local neighborhoods around node pairs.
//!
//! An ego-graph is a subgraph containing a focal set of nodes plus their k-hop neighbors.
//! This module implements 2-hop ego-graphs around node pairs, which are used to analyze
//! the local topological structure for d_total distance computation.

use palace_core::NodeId;
use std::collections::{HashMap, HashSet};

/// A local subgraph around a set of nodes.
///
/// Contains the vertices and edges within a bounded neighborhood, used to compute
/// topological invariants like β₀ (connected components) and β₁ (first Betti number).
#[derive(Clone, Debug)]
pub struct EgoGraph {
    /// Node IDs in the ego-graph
    pub vertices: Vec<NodeId>,
    /// Edges in the ego-graph as (source, target) pairs
    pub edges: Vec<(NodeId, NodeId)>,
    /// Total number of vertices
    pub num_vertices: usize,
    /// Total number of edges
    pub num_edges: usize,
    /// Hard cap on ego-graph edges. When set, reduces worst-case complexity from O(K²) to O(max_ego_edges).
    /// Default: None (unlimited). Recommended: 500.
    pub max_ego_edges: Option<usize>,
}

impl EgoGraph {
    /// Build a 2-hop ego-graph for a pair of nodes (x, y).
    ///
    /// The ego-graph is constructed as follows:
    /// 1. Start with the focal set {x, y}
    /// 2. Add all 1-hop neighbors of x and y
    /// 3. Add all 1-hop neighbors of those 1-hop neighbors (2-hop from x, y)
    /// 4. Collect all edges between vertices in the subgraph
    ///
    /// # Arguments
    /// * `x` - First focal node
    /// * `y` - Second focal node
    /// * `neighbors_fn` - Function that returns immediate neighbors of a given NodeId
    ///
    /// # Returns
    /// An EgoGraph containing the 2-hop neighborhood and all edges within it.
    /// Use `.with_cap(500)` to bound edge count for complexity control.
    pub fn build_pair<F>(x: NodeId, y: NodeId, neighbors_fn: F) -> Self
    where
        F: Fn(NodeId) -> Vec<NodeId>,
    {
        let mut vertices = HashSet::new();
        let mut vertex_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Step 1: Start with focal nodes
        vertices.insert(x);
        vertices.insert(y);

        // Step 2: Get 1-hop neighbors
        let neighbors_x = neighbors_fn(x);
        let neighbors_y = neighbors_fn(y);

        for neighbor in &neighbors_x {
            vertices.insert(*neighbor);
        }
        for neighbor in &neighbors_y {
            vertices.insert(*neighbor);
        }

        vertex_neighbors.insert(x, neighbors_x.clone());
        vertex_neighbors.insert(y, neighbors_y.clone());

        // Cache neighbors for 1-hop nodes
        for &node in neighbors_x.iter().chain(neighbors_y.iter()) {
            vertex_neighbors
                .entry(node)
                .or_insert_with(|| neighbors_fn(node));
        }

        // Step 3: Add 2-hop neighbors
        let neighbors_1hop: Vec<NodeId> = vertices
            .iter()
            .copied()
            .filter(|&v| v != x && v != y)
            .collect();

        for node in neighbors_1hop {
            let two_hop = vertex_neighbors
                .entry(node)
                .or_insert_with(|| neighbors_fn(node))
                .clone();

            for neighbor in two_hop {
                vertices.insert(neighbor);
            }
        }

        // Step 4: Collect all edges between vertices in the subgraph
        let mut edges = HashSet::new();
        let vertex_set = &vertices;

        for &node in vertex_set {
            let neighbors = vertex_neighbors
                .entry(node)
                .or_insert_with(|| neighbors_fn(node))
                .clone();

            for neighbor in neighbors {
                if vertex_set.contains(&neighbor) {
                    // Normalize edge to avoid duplicates (smaller ID first)
                    let edge = if node < neighbor {
                        (node, neighbor)
                    } else {
                        (neighbor, node)
                    };
                    edges.insert(edge);
                }
            }
        }

        let num_vertices = vertices.len();
        let num_edges = edges.len();
        let vertices_vec: Vec<NodeId> = vertices.into_iter().collect();
        let edges_vec: Vec<(NodeId, NodeId)> = edges.into_iter().collect();

        EgoGraph {
            vertices: vertices_vec,
            edges: edges_vec,
            num_vertices,
            num_edges,
            max_ego_edges: None,
        }
    }

    /// Build an ego-graph around a single node with k hops.
    ///
    /// # Arguments
    /// * `node` - The focal node
    /// * `hops` - Number of hops (0 = just the node, 1 = node + neighbors, etc.)
    /// * `neighbors_fn` - Function that returns immediate neighbors of a given NodeId
    ///
    /// # Returns
    /// An EgoGraph containing the k-hop neighborhood and all edges within it.
    /// Use `.with_cap(500)` to bound edge count for complexity control.
    pub fn build_single<F>(node: NodeId, hops: usize, neighbors_fn: F) -> Self
    where
        F: Fn(NodeId) -> Vec<NodeId>,
    {
        let mut vertices = HashSet::new();
        let mut to_expand = vec![node];
        let mut expanded = HashSet::new();

        for _ in 0..=hops {
            let mut next_expand = Vec::new();

            for &v in &to_expand {
                if expanded.insert(v) {
                    vertices.insert(v);
                    let neighbors = neighbors_fn(v);
                    next_expand.extend(neighbors);
                }
            }

            to_expand = next_expand;
        }

        // Collect all edges
        let mut edges = HashSet::new();
        let mut cached_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &node_id in &vertices {
            let neighbors = cached_neighbors
                .entry(node_id)
                .or_insert_with(|| neighbors_fn(node_id))
                .clone();

            for neighbor in neighbors {
                if vertices.contains(&neighbor) {
                    let edge = if node_id < neighbor {
                        (node_id, neighbor)
                    } else {
                        (neighbor, node_id)
                    };
                    edges.insert(edge);
                }
            }
        }

        let num_vertices = vertices.len();
        let num_edges = edges.len();
        let vertices_vec: Vec<NodeId> = vertices.into_iter().collect();
        let edges_vec: Vec<(NodeId, NodeId)> = edges.into_iter().collect();

        EgoGraph {
            vertices: vertices_vec,
            edges: edges_vec,
            num_vertices,
            num_edges,
            max_ego_edges: None,
        }
    }

    /// Set a hard cap on ego-graph edges for complexity control.
    ///
    /// Truncates edges to the first `max_edges` edges and updates `num_edges`.
    /// This bounds the worst-case complexity of topological computations from O(K²)
    /// to O(max_edges), where K is the number of vertices.
    ///
    /// # Default
    /// `max_ego_edges` defaults to `None` (unlimited). Recommended: 500 edges.
    ///
    /// # Arguments
    /// * `max_edges` - Maximum number of edges to retain
    ///
    /// # Example
    /// ```ignore
    /// let ego = EgoGraph::build_pair(...)
    ///     .with_cap(500);
    /// ```
    pub fn with_cap(mut self, max_edges: usize) -> Self {
        if self.edges.len() > max_edges {
            self.edges.truncate(max_edges);
            self.num_edges = max_edges;
        }
        self.max_ego_edges = Some(max_edges);
        self
    }

    /// Set a weighted cap on ego-graph edges using a custom weight function.
    ///
    /// Computes weight for each edge, sorts by weight ascending (lower distance = higher priority),
    /// keeps only the top `max_edges`, and updates `num_edges` and `edges`.
    ///
    /// This is useful for retaining the most significant edges when edges have an associated
    /// metric (e.g., cosine distance between neighbors).
    ///
    /// # Default
    /// `max_ego_edges` defaults to `None` (unlimited). Recommended: 500 edges.
    ///
    /// # Arguments
    /// * `max_edges` - Maximum number of edges to retain
    /// * `weight_fn` - Function computing edge weight: lower values = higher priority
    ///
    /// # Example
    /// ```ignore
    /// let ego = EgoGraph::build_pair(...)
    ///     .cap_by_weight(500, |u, v| cosine_distance(u, v));
    /// ```
    pub fn cap_by_weight(
        mut self,
        max_edges: usize,
        weight_fn: impl Fn(NodeId, NodeId) -> f32,
    ) -> Self {
        if self.edges.len() > max_edges {
            // Compute weights and sort by weight ascending
            let mut weighted_edges: Vec<_> = self
                .edges
                .iter()
                .map(|&(u, v)| (weight_fn(u, v), u, v))
                .collect();
            weighted_edges
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Keep only top max_edges
            self.edges = weighted_edges
                .into_iter()
                .take(max_edges)
                .map(|(_, u, v)| (u, v))
                .collect();
            self.num_edges = self.edges.len();
        }
        self.max_ego_edges = Some(max_edges);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a test graph as an adjacency list
    struct TestGraph {
        adjacency: HashMap<NodeId, Vec<NodeId>>,
    }

    impl TestGraph {
        fn new() -> Self {
            TestGraph {
                adjacency: HashMap::new(),
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

    #[test]
    fn test_ego_graph_triangle() {
        // Build a triangle: 0 - 1 - 2 - 0
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(0));

        let ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| graph.neighbors(node));

        assert_eq!(ego.num_vertices, 3);
        assert_eq!(ego.num_edges, 3);
    }

    #[test]
    fn test_ego_graph_line() {
        // Build a line: 0 - 1 - 2 - 3
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(3));

        let ego = EgoGraph::build_pair(NodeId(1), NodeId(2), |node| graph.neighbors(node));

        // Should include: {0, 1, 2, 3}
        assert_eq!(ego.num_vertices, 4);
        // Edges: (0,1), (1,2), (2,3)
        assert_eq!(ego.num_edges, 3);
    }

    #[test]
    fn test_ego_graph_single_node() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let ego = EgoGraph::build_single(NodeId(0), 0, |node| graph.neighbors(node));

        // Just node 0 with 0 hops
        assert_eq!(ego.num_vertices, 1);
        assert_eq!(ego.num_edges, 0);
    }

    #[test]
    fn test_ego_graph_single_node_one_hop() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let ego = EgoGraph::build_single(NodeId(1), 1, |node| graph.neighbors(node));

        // Node 1 + neighbors 0, 2
        assert_eq!(ego.num_vertices, 3);
        // Edges: (0,1), (1,2)
        assert_eq!(ego.num_edges, 2);
    }
}
