// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Topological invariants: β₀ (connected components), β₁ (first Betti number), and Euler characteristic.
//!
//! The first Betti number β₁ counts the number of independent cycles in a graph.
//! It is computed as: β₁ = β₀ - χ, where β₀ is the number of connected components
//! and χ = V - E is the Euler characteristic.

use crate::ego_graph::EgoGraph;
use palace_core::NodeId;
use std::collections::HashMap;

/// Disjoint-set (Union-Find) data structure with path compression and union by rank.
///
/// Used to efficiently compute connected components in a graph.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Mapping from NodeId to index in parent/rank vectors
    node_to_index: HashMap<NodeId, usize>,
    /// Reverse mapping from index to NodeId
    index_to_node: HashMap<usize, NodeId>,
    /// Counter for assigning indices to new nodes
    next_index: usize,
}

impl UnionFind {
    /// Create a new Union-Find structure.
    fn new() -> Self {
        UnionFind {
            parent: Vec::new(),
            rank: Vec::new(),
            node_to_index: HashMap::new(),
            index_to_node: HashMap::new(),
            next_index: 0,
        }
    }

    /// Add a node to the Union-Find if it doesn't already exist.
    fn ensure_node(&mut self, node: NodeId) {
        if !self.node_to_index.contains_key(&node) {
            let index = self.next_index;
            self.node_to_index.insert(node, index);
            self.index_to_node.insert(index, node);
            self.parent.push(index);
            self.rank.push(0);
            self.next_index += 1;
        }
    }

    /// Find the root of the set containing x, with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union two sets by rank. Returns true if the sets were different (union happened).
    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
        true
    }

    /// Count the number of connected components.
    fn components(&mut self) -> usize {
        let mut roots = std::collections::HashSet::new();
        for i in 0..self.parent.len() {
            roots.insert(self.find(i));
        }
        roots.len()
    }
}

/// Compute the 0-th Betti number (β₀): the number of connected components.
///
/// β₀ is computed using a Union-Find data structure with path compression
/// and union by rank.
///
/// # Arguments
/// * `graph` - The ego-graph to analyze
///
/// # Returns
/// The number of connected components in the graph
pub fn beta_0(graph: &EgoGraph) -> usize {
    if graph.num_vertices == 0 {
        return 0;
    }

    let mut uf = UnionFind::new();

    // Ensure all vertices are in the Union-Find
    for &node in &graph.vertices {
        uf.ensure_node(node);
    }

    // Union vertices connected by edges
    for &(u, v) in &graph.edges {
        let u_idx = uf.node_to_index[&u];
        let v_idx = uf.node_to_index[&v];
        uf.union(u_idx, v_idx);
    }

    uf.components()
}

/// Compute the Euler characteristic: χ = V - E
///
/// For a graph, the Euler characteristic is the number of vertices
/// minus the number of edges.
///
/// # Arguments
/// * `graph` - The ego-graph to analyze
///
/// # Returns
/// The Euler characteristic
pub fn euler_characteristic(graph: &EgoGraph) -> i64 {
    graph.num_vertices as i64 - graph.num_edges as i64
}

/// Compute the first Betti number (β₁): the number of independent cycles.
///
/// β₁ is computed as: β₁ = β₀ - χ = β₀ - V + E
///
/// Intuitively, each edge beyond a spanning tree (which has V - β₀ edges)
/// creates one independent cycle. With β₀ connected components, a spanning
/// forest has V - β₀ edges, so excess edges = E - (V - β₀) = E - V + β₀,
/// which equals β₁.
///
/// # Arguments
/// * `graph` - The ego-graph to analyze
///
/// # Returns
/// The number of independent cycles (always non-negative)
pub fn beta_1(graph: &EgoGraph) -> usize {
    let b0 = beta_0(graph) as i64;
    let chi = euler_characteristic(graph);
    let result = b0 - chi;
    result.max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

        fn to_ego_graph(&self, vertices: Vec<NodeId>) -> EgoGraph {
            let mut edges = std::collections::HashSet::new();

            for &node in &vertices {
                for &neighbor in &self.neighbors(node) {
                    if vertices.contains(&neighbor) {
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

            EgoGraph {
                vertices: vertices.clone(),
                edges: edges.into_iter().collect(),
                num_vertices,
                num_edges,
                max_ego_edges: None,
            }
        }
    }

    #[test]
    fn test_beta_0_single_vertex() {
        let graph = TestGraph::new();
        let ego = graph.to_ego_graph(vec![NodeId(0)]);
        assert_eq!(beta_0(&ego), 1);
    }

    #[test]
    fn test_beta_0_disconnected() {
        let graph = TestGraph::new();
        // Two isolated vertices
        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1)]);
        assert_eq!(beta_0(&ego), 2);
    }

    #[test]
    fn test_beta_0_connected_line() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2)]);
        assert_eq!(beta_0(&ego), 1);
    }

    #[test]
    fn test_euler_characteristic_single_vertex() {
        let graph = TestGraph::new();
        let ego = graph.to_ego_graph(vec![NodeId(0)]);
        assert_eq!(euler_characteristic(&ego), 1);
    }

    #[test]
    fn test_euler_characteristic_line() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2)]);
        // V=3, E=2, χ = 3 - 2 = 1
        assert_eq!(euler_characteristic(&ego), 1);
    }

    #[test]
    fn test_beta_1_tree() {
        // A tree has no cycles, so β₁ = 0
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(3));
        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]);

        // V=4, E=3, β₀=1, χ=1
        // β₁ = β₀ - χ = 1 - 1 = 0
        assert_eq!(beta_1(&ego), 0);
    }

    #[test]
    fn test_beta_1_triangle() {
        // A triangle has one cycle, so β₁ = 1
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(0));
        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2)]);

        // V=3, E=3, β₀=1, χ=-0
        // β₁ = β₀ - χ = 1 - (-0) = 1 - (3 - 3) = 1 - 0 = 1
        assert_eq!(beta_1(&ego), 1);
    }

    #[test]
    fn test_beta_1_two_triangles_shared_edge() {
        // Two triangles sharing an edge: 2 cycles
        // Graph: 0-1-2-0 and 1-3-2-1
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(0));
        graph.add_edge(NodeId(1), NodeId(3));
        graph.add_edge(NodeId(3), NodeId(2));

        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]);

        // V=4, E=5, β₀=1, χ=-1
        // β₁ = β₀ - χ = 1 - (-1) = 2
        assert_eq!(beta_1(&ego), 2);
    }

    #[test]
    fn test_beta_1_two_components_with_cycle() {
        // Component 1: triangle (0-1-2-0)
        // Component 2: isolated vertex (3)
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(0));

        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]);

        // V=4, E=3, β₀=2 (two components), χ=1
        // β₁ = β₀ - χ = 2 - 1 = 1
        assert_eq!(beta_1(&ego), 1);
    }

    #[test]
    fn test_beta_1_complete_graph_k4() {
        // Complete graph on 4 vertices: K4
        // All pairs connected, E=6
        let mut graph = TestGraph::new();
        for i in 0u64..4 {
            for j in (i + 1)..4 {
                graph.add_edge(NodeId(i), NodeId(j));
            }
        }

        let ego = graph.to_ego_graph(vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)]);

        // V=4, E=6, β₀=1, χ=-2
        // β₁ = β₀ - χ = 1 - (-2) = 3
        assert_eq!(beta_1(&ego), 3);
    }

    #[test]
    fn test_beta_1_empty_graph() {
        let graph = TestGraph::new();
        let ego = graph.to_ego_graph(vec![]);
        // Empty graph: β₁ = 0
        assert_eq!(beta_1(&ego), 0);
    }
}
