// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Neighbor selection heuristic for maintaining diverse NSW edges

use crate::node::cosine_distance;
use palace_core::NodeId;

/// Selects neighbors using the HNSW-style heuristic
///
/// This heuristic maintains angular diversity while preferring closer nodes.
/// For each candidate, we check if it's closer to the base node than to any
/// already-selected neighbor. This promotes a diverse set of neighbors that
/// can serve as entry points for exploration.
///
/// Algorithm:
/// 1. Sort candidates by distance
/// 2. Iteratively select candidates that maintain diversity:
///    - Always select the closest candidate
///    - For others, only select if they are closer to base than to any selected neighbor
///
/// # Arguments
/// * `candidates` - List of (NodeId, distance) tuples
/// * `base_vector` - The vector of the node for which we're selecting neighbors
/// * `candidate_vectors` - Map of NodeId to their vectors
/// * `max` - Maximum number of neighbors to select
/// * `alpha` - Pruning parameter (default: 1.2). accept if dist(cand, base) <= alpha * dist(cand, selected)
///
/// # Returns
/// Vector of selected NodeIds
pub fn select_neighbors_heuristic(
    candidates: &[(NodeId, f32)],
    _base_vector: &[f32],
    candidate_vectors: &std::collections::HashMap<NodeId, Vec<f32>>,
    max: usize,
    alpha: f32,
) -> Vec<NodeId> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort candidates by distance
    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = Vec::new();
    let mut selected_vectors = Vec::new();

    for (candidate_id, candidate_dist) in sorted_candidates.iter() {
        if selected.len() >= max {
            break;
        }

        // Always select the first (closest) candidate
        if selected.is_empty() {
            selected.push(*candidate_id);
            if let Some(vec) = candidate_vectors.get(candidate_id) {
                selected_vectors.push(vec.clone());
            }
            continue;
        }

        // For other candidates, check if they are closer to base than to any selected neighbor
        let candidate_vec = match candidate_vectors.get(candidate_id) {
            Some(v) => v,
            None => continue,
        };

        let mut is_diverse = true;
        for selected_vec in &selected_vectors {
            let dist_to_selected = cosine_distance(candidate_vec, selected_vec);
            // Vamana alpha-pruning logic:
            // accept if dist(candidate, base) <= alpha * dist(candidate, selected)
            if *candidate_dist > alpha * dist_to_selected {
                // This candidate is closer to an already-selected neighbor (scaled by alpha) than to base
                is_diverse = false;
                break;
            }
        }

        if is_diverse {
            selected.push(*candidate_id);
            selected_vectors.push(candidate_vec.clone());
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use palace_core::NodeId;
    use std::collections::HashMap;

    #[test]
    fn test_select_neighbors_empty() {
        let base_vec = vec![1.0, 0.0];
        let candidates = vec![];
        let vectors = HashMap::new();
        let result = select_neighbors_heuristic(&candidates, &base_vec, &vectors, 5, 1.2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_neighbors_basic() {
        let base_vec = vec![1.0, 0.0];

        let mut vectors = HashMap::new();
        vectors.insert(NodeId(0), vec![1.0, 0.0]); // identical to base
        vectors.insert(NodeId(1), vec![0.9, 0.1]); // close to base
        vectors.insert(NodeId(2), vec![0.0, 1.0]); // orthogonal to base

        let candidates = vec![
            (NodeId(0), 0.0f32),
            (NodeId(1), 0.1f32),
            (NodeId(2), 1.0f32),
        ];
        let alpha = 1.0;
        let result = select_neighbors_heuristic(&candidates, &base_vec, &vectors, 2, alpha);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], NodeId(0)); // Always select closest
    }
}
