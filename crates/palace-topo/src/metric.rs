//! Palace-X topological distance metric (d_total).
//!
//! Combines cosine distance with structural connectivity analysis via the first Betti number.
//! Higher cycle density in the ego-graph indicates stronger structural interconnection,
//! which reduces the overall distance.

use crate::betti::beta_1;
use crate::ego_graph::EgoGraph;

/// Compute the Palace-X topological distance metric.
///
/// d_total(x, y) = α · d_cosine(x, y) + β · exp(-β₁(C_xy) / max(|E(C_xy)|, 1))
///
/// Where:
/// - d_cosine is the cosine distance between two vectors
/// - C_xy is the ego-graph around the pair (x, y)
/// - β₁(C_xy) is the first Betti number (number of independent cycles)
/// - |E(C_xy)| is the number of edges in the ego-graph
/// - α and β are weighting parameters
///
/// Intuition:
/// - The cosine component captures semantic similarity
/// - The structural component captures topological interconnection:
///   - High β₁/|E| ratio = many cycles per edge = rich structure = lower penalty
///   - Low β₁/|E| ratio = few cycles = sparse structure = higher penalty
/// - The max(|E|, 1) guards against division by zero for empty or single-node graphs
///
/// # Arguments
/// * `cosine_dist` - Cosine distance between the two nodes (0.0 to 2.0 for typical unit vectors)
/// * `ego_graph` - The 2-hop ego-graph around the node pair
/// * `alpha` - Weight for the cosine distance component (typically 0.3-0.7)
/// * `beta` - Weight for the structural component (typically 0.3-0.7, with alpha + beta ≈ 1.0)
///
/// # Returns
/// The total distance score (lower = more similar)
///
/// # Example
/// ```ignore
/// let cosine_dist = 0.3; // Semantically close
/// let ego_graph = EgoGraph::build_pair(x, y, &neighbors_fn);
/// let d = d_total(cosine_dist, &ego_graph, 0.5, 0.5);
/// // d combines cosine similarity with structural metrics
/// ```
pub fn d_total(
    cosine_dist: f32,
    ego_graph: &EgoGraph,
    alpha: f32,
    beta: f32,
) -> f32 {
    let b1 = beta_1(ego_graph) as f32;
    let e = ego_graph.num_edges.max(1) as f32;

    // Structural component: exp(-β₁/|E|)
    // - When β₁/|E| is large: many cycles relative to edges → lower penalty (exp decays)
    // - When β₁/|E| is small: few cycles relative to edges → higher penalty (exp closer to 1)
    let structural = (-b1 / e).exp();

    alpha * cosine_dist + beta * structural
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use palace_core::NodeId;

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
            self.adjacency.entry(u).or_insert_with(Vec::new).push(v);
            self.adjacency.entry(v).or_insert_with(Vec::new).push(u);
        }

        fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
            self.adjacency
                .get(&node)
                .cloned()
                .unwrap_or_default()
        }
    }

    #[test]
    fn test_d_total_basic() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| graph.neighbors(node));

        let d = d_total(0.5, &ego, 0.5, 0.5);

        // Should be: 0.5 * 0.5 + 0.5 * exp(-β₁/|E|)
        // A line has β₁=0, so structural = exp(0) = 1.0
        // d = 0.25 + 0.5 * 1.0 = 0.75
        assert!((d - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_d_total_triangle_lower_than_line() {
        // Build a triangle (rich structure: β₁=1)
        let mut triangle_graph = TestGraph::new();
        triangle_graph.add_edge(NodeId(0), NodeId(1));
        triangle_graph.add_edge(NodeId(1), NodeId(2));
        triangle_graph.add_edge(NodeId(2), NodeId(0));

        let triangle_ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| triangle_graph.neighbors(node));

        // Build a line (sparse structure: β₁=0)
        let mut line_graph = TestGraph::new();
        line_graph.add_edge(NodeId(0), NodeId(1));
        line_graph.add_edge(NodeId(1), NodeId(2));

        let line_ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| line_graph.neighbors(node));

        let d_triangle = d_total(0.5, &triangle_ego, 0.5, 0.5);
        let d_line = d_total(0.5, &line_ego, 0.5, 0.5);

        // Triangle has β₁=1, E=3: exp(-1/3) ≈ 0.717
        // Line has β₁=0, E=2: exp(0) = 1.0
        // So triangle distance should be lower (more structure = tighter coupling)
        assert!(d_triangle < d_line, "Triangle ({}) should have lower distance than line ({})", d_triangle, d_line);
    }

    #[test]
    fn test_d_total_alpha_beta_weighting() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| graph.neighbors(node));

        // Test with alpha=1, beta=0 (cosine only)
        let d_cosine_only = d_total(0.5, &ego, 1.0, 0.0);
        assert!((d_cosine_only - 0.5).abs() < 0.001);

        // Test with alpha=0, beta=1 (structural only)
        let d_structural_only = d_total(0.5, &ego, 0.0, 1.0);
        // β₁=0 for a line, so structural = exp(0) = 1.0
        assert!((d_structural_only - 1.0).abs() < 0.001);

        // Test with alpha=0.5, beta=0.5
        let d_balanced = d_total(0.5, &ego, 0.5, 0.5);
        assert!((d_balanced - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_d_total_empty_graph() {
        let graph = TestGraph::new();
        let ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| graph.neighbors(node));

        // Empty ego-graph: β₁=0, |E|=0 → use max(|E|, 1) = 1
        let d = d_total(0.5, &ego, 0.5, 0.5);

        // Structural = exp(-0/1) = 1.0
        // d = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        assert!((d - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_d_total_increases_with_cosine_distance() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |node| graph.neighbors(node));

        let d_small = d_total(0.1, &ego, 0.5, 0.5);
        let d_large = d_total(0.9, &ego, 0.5, 0.5);

        assert!(d_small < d_large);
    }

    #[test]
    fn test_d_total_cycle_rich_vs_sparse() {
        // Complete graph K3 (triangle): β₁=1
        let mut triangle = TestGraph::new();
        triangle.add_edge(NodeId(0), NodeId(1));
        triangle.add_edge(NodeId(1), NodeId(2));
        triangle.add_edge(NodeId(2), NodeId(0));
        let tri_ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |n| triangle.neighbors(n));

        // Graph with 4 nodes, 4 edges forming a cycle: β₁=1
        let mut four_cycle = TestGraph::new();
        four_cycle.add_edge(NodeId(0), NodeId(1));
        four_cycle.add_edge(NodeId(1), NodeId(2));
        four_cycle.add_edge(NodeId(2), NodeId(3));
        four_cycle.add_edge(NodeId(3), NodeId(0));
        let cycle_ego = EgoGraph::build_pair(NodeId(0), NodeId(1), |n| four_cycle.neighbors(n));

        let d_tri = d_total(0.5, &tri_ego, 0.5, 0.5);
        let d_cycle = d_total(0.5, &cycle_ego, 0.5, 0.5);

        // Triangle: β₁=1, E=3, structural = exp(-1/3) ≈ 0.717
        // 4-cycle: β₁=1, E=4, structural = exp(-1/4) ≈ 0.778
        // So triangle should have slightly lower distance (higher cycle density)
        assert!(d_tri < d_cycle, "Triangle ({}) should have lower distance than 4-cycle ({})", d_tri, d_cycle);
    }
}
