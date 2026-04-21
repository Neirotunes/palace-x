// Copyright (c) 2026 M.Diach
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Graph node representation with binary quantization

use palace_core::NodeId;
use serde::{Deserialize, Serialize};

/// Metadata associated with a node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaData {
    pub label: String,
}

/// Represents a node in the NSW graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: NodeId,

    /// Original FP32 vector
    pub vector: Vec<f32>,

    /// Binary quantized representation
    pub binary: Vec<u64>,

    /// NSW edges (list of neighboring node IDs)
    pub neighbors: Vec<NodeId>,

    /// Metadata associated with this node
    pub metadata: MetaData,

    /// Hub score: frequency of being visited during search
    /// Higher values indicate the node is a good entry point
    pub hub_score: f32,
}

impl GraphNode {
    /// Creates a new graph node with the given parameters
    pub fn new(id: NodeId, vector: Vec<f32>, metadata: MetaData) -> Self {
        let binary = quantize_to_binary(&vector);
        GraphNode {
            id,
            vector,
            binary,
            neighbors: Vec::new(),
            metadata,
            hub_score: 0.0,
        }
    }

    /// Computes cosine distance between this node's vector and another vector
    pub fn cosine_distance(&self, other: &[f32]) -> f32 {
        cosine_distance(&self.vector, other)
    }

    /// Computes Hamming distance between this node's binary representation and another
    pub fn hamming_distance(&self, other: &[u64]) -> u32 {
        hamming_distance(&self.binary, other)
    }
}

/// Quantizes a float vector to binary (bitwise representation)
/// Simple approach: for each component, if value >= 0 set bit to 1, else 0
fn quantize_to_binary(vector: &[f32]) -> Vec<u64> {
    let mut binary = Vec::with_capacity(vector.len().div_ceil(64));

    for chunk in vector.chunks(64) {
        let mut bits: u64 = 0;
        for (i, &val) in chunk.iter().enumerate() {
            if val >= 0.0 {
                bits |= 1u64 << i;
            }
        }
        binary.push(bits);
    }

    binary
}

/// Computes cosine distance between two vectors
/// Returns a value in [0, 2], where 0 means identical and 2 means opposite
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

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
        return 2.0; // Treat zero vectors as maximally distant
    }

    let similarity = dot_product / (norm_a * norm_b);
    // Clamp to [-1, 1] to handle floating point errors
    let similarity = similarity.clamp(-1.0, 1.0);
    1.0 - similarity // Convert similarity to distance
}

/// Computes Hamming distance between two binary vectors
pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "Binary vectors must have the same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&v1, &v2);
        assert!(
            (dist - 0.0).abs() < 1e-5,
            "Identical vectors should have distance 0"
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let dist = cosine_distance(&v1, &v2);
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "Orthogonal vectors should have distance 1"
        );
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        let dist = cosine_distance(&v1, &v2);
        assert!(
            (dist - 2.0).abs() < 1e-5,
            "Opposite vectors should have distance 2"
        );
    }

    #[test]
    fn test_hamming_distance() {
        let b1 = vec![0b1010];
        let b2 = vec![0b1100];
        let dist = hamming_distance(&b1, &b2);
        assert_eq!(dist, 2, "Expected Hamming distance of 2");
    }

    #[test]
    fn test_graph_node_creation() {
        let vector = vec![1.0, 2.0, 3.0];
        let metadata = MetaData {
            label: "test".to_string(),
        };
        let node = GraphNode::new(NodeId(42), vector.clone(), metadata);
        assert_eq!(node.id, NodeId(42));
        assert_eq!(node.vector, vector);
        assert_eq!(node.neighbors.len(), 0);
        assert!(node.hub_score == 0.0);
    }
}
