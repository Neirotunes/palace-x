// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Navigable Small World (NSW) index implementation

use crate::heuristic::select_neighbors_heuristic;
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

/// A candidate node with distance, used for min-heap operations
#[derive(Debug, Clone)]
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
        // Reverse ordering for min-heap (smallest distances first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Navigable Small World graph index
pub struct NswIndex {
    /// Sharded concurrent map for writes and direct access
    nodes: DashMap<NodeId, GraphNode>,
    /// Immutable snapshot for wait-free reads during high-concurrency reranking
    read_snapshot: ArcSwap<im::HashMap<NodeId, GraphNode>>,
    max_neighbors: usize,           // M parameter
    ef_construction: usize,         // ef during construction
    ef_search: AtomicUsize,         // ef during search
    hub_cache: RwLock<Vec<NodeId>>, // top-K hub nodes
    next_id: AtomicU64,
    dimensions: usize,
}

impl NswIndex {
    /// Creates a new NSW index
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimension
    /// * `max_neighbors` - Maximum neighbors per node (M parameter, typically 32)
    /// * `ef_construction` - Ef parameter during construction (typically 200)
    pub fn new(dimensions: usize, max_neighbors: usize, ef_construction: usize) -> Self {
        NswIndex {
            nodes: DashMap::new(),
            read_snapshot: ArcSwap::from_pointee(im::HashMap::new()),
            max_neighbors,
            ef_construction,
            ef_search: AtomicUsize::new(64), // Default ef for search
            hub_cache: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            dimensions,
        }
    }

    /// Publishes a new immutable snapshot of the graph for wait-free reads.
    /// Should be called after batch insertions to ensure search consistency.
    pub fn publish_snapshot(&self) {
        let mut new_map = im::HashMap::new();
        for entry in self.nodes.iter() {
            new_map.insert(*entry.key(), entry.value().clone());
        }
        self.read_snapshot.store(Arc::new(new_map));
    }

    /// Sets the ef parameter for search
    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, AtomicOrdering::Relaxed);
    }

    /// Inserts a new vector into the index
    pub fn insert(&self, vector: Vec<f32>, metadata: MetaData) -> NodeId {
        assert_eq!(vector.len(), self.dimensions, "Vector dimension mismatch");

        let id = self.next_id.fetch_add(1, AtomicOrdering::SeqCst);
        let node_id = NodeId(id);
        let mut node = GraphNode::new(node_id, vector, metadata);

        if self.nodes.is_empty() {
            // First node: just insert and publish snapshot
            self.nodes.insert(node_id, node);
            self.publish_snapshot();
            return node_id;
        }

        // Search context: use current snapshot for neighborhood search
        let snapshot = self.read_snapshot.load();

        // If snapshot is empty (race condition), publish and retry
        if snapshot.is_empty() {
            self.publish_snapshot();
            let snapshot = self.read_snapshot.load();
            if snapshot.is_empty() {
                // Still empty — just insert without neighbors
                self.nodes.insert(node_id, node);
                self.publish_snapshot();
                return node_id;
            }
        }

        // Find nearest neighbors for the new node
        let nearest = self.search_for_insertion_internal(&node, &snapshot);

        // Select neighbors using heuristic
        let base_vector = &node.vector;
        let mut candidate_vectors = HashMap::new();
        for (cand_id, _) in &nearest {
            if let Some(cand_node) = self.nodes.get(cand_id) {
                candidate_vectors.insert(*cand_id, cand_node.vector.clone());
            }
        }

        node.neighbors = select_neighbors_heuristic(
            &nearest,
            base_vector,
            &candidate_vectors,
            self.max_neighbors,
        );

        // Reciprocal connections: update neighbors to include this node
        for neighbor_id in &node.neighbors {
            // Take segment lock for individual neighbor
            if let Some(mut neighbor_node) = self.nodes.get_mut(neighbor_id) {
                neighbor_node.neighbors.push(node_id);

                if neighbor_node.neighbors.len() > self.max_neighbors {
                    // Pruning needed
                    let neighbor_vector = neighbor_node.vector.clone();
                    let neighbor_neighbor_ids = neighbor_node.neighbors.clone();

                    // Release the lock before doing heuristic calculation (can be slow)
                    drop(neighbor_node);

                    let mut neighbor_candidates = Vec::new();
                    let mut neighbor_candidate_vectors = HashMap::new();

                    for nid in &neighbor_neighbor_ids {
                        if let Some(n) = self.nodes.get(nid) {
                            let dist = cosine_distance(&neighbor_vector, &n.vector);
                            neighbor_candidates.push((*nid, dist));
                            neighbor_candidate_vectors.insert(*nid, n.vector.clone());
                        }
                    }

                    let pruned = select_neighbors_heuristic(
                        &neighbor_candidates,
                        &neighbor_vector,
                        &neighbor_candidate_vectors,
                        self.max_neighbors,
                    );

                    // Re-acquire lock to assign
                    if let Some(mut neighbor_node) = self.nodes.get_mut(neighbor_id) {
                        neighbor_node.neighbors = pruned;
                    }
                }
            }
        }

        self.nodes.insert(node_id, node);
        self.publish_snapshot();
        node_id
    }

    /// Searches for the nearest neighbors of a node during insertion
    fn search_for_insertion_internal(
        &self,
        query_node: &GraphNode,
        nodes: &im::HashMap<NodeId, GraphNode>,
    ) -> Vec<(NodeId, f32)> {
        let query_vec = &query_node.vector;
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Start from a random entry point
        let entry_id = {
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = nodes.keys().copied().collect();
            ids[rng.gen_range(0..ids.len())]
        };

        if let Some(entry_node) = nodes.get(&entry_id) {
            let dist = cosine_distance(query_vec, &entry_node.vector);
            candidates.push(std::cmp::Reverse(Candidate {
                id: entry_id,
                distance: dist,
            }));
            w.push(Candidate {
                id: entry_id,
                distance: dist,
            });
        }

        let mut visited = HashSet::new();
        visited.insert(entry_id);

        for _ in 0..self.ef_construction {
            // Get the nearest unvisited candidate
            let lowerbound = if let Some(candidate) = candidates.peek() {
                candidate.0.distance
            } else {
                break;
            };

            if lowerbound > w.peek().map(|c| c.distance).unwrap_or(f32::INFINITY) {
                break;
            }

            let current = candidates.pop().unwrap().0;

            if let Some(current_node) = nodes.get(&current.id) {
                for neighbor_id in &current_node.neighbors {
                    if !visited.insert(*neighbor_id) {
                        continue;
                    }

                    if let Some(neighbor_node) = nodes.get(neighbor_id) {
                        let dist = cosine_distance(query_vec, &neighbor_node.vector);

                        if dist < w.peek().map(|c| c.distance).unwrap_or(f32::INFINITY)
                            || w.len() < self.ef_construction
                        {
                            candidates.push(std::cmp::Reverse(Candidate {
                                id: *neighbor_id,
                                distance: dist,
                            }));

                            w.push(Candidate {
                                id: *neighbor_id,
                                distance: dist,
                            });

                            if w.len() > self.ef_construction {
                                // Remove the farthest
                                let mut vec: Vec<_> = w.into_iter().collect();
                                vec.sort_by(|a, b| {
                                    a.distance
                                        .partial_cmp(&b.distance)
                                        .unwrap_or(Ordering::Equal)
                                });
                                vec.pop();
                                w = vec.into_iter().collect();
                            }
                        }
                    }
                }
            }
        }

        w.into_iter().map(|c| (c.id, c.distance)).collect()
    }

    /// Searches the index for nearest neighbors of a query vector
    ///
    /// Returns a vector of (NodeId, distance) tuples sorted by distance.
    pub fn search(&self, query: &[f32], ef: Option<usize>) -> Vec<(NodeId, f32)> {
        assert_eq!(
            query.len(),
            self.dimensions,
            "Query vector dimension mismatch"
        );

        let ef = ef.unwrap_or(self.ef_search.load(AtomicOrdering::Relaxed));

        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Entry point: prefer hub if available, otherwise random
        let entry_id = {
            let hub_cache = self.hub_cache.read();
            if !hub_cache.is_empty() {
                hub_cache[0]
            } else {
                let mut rng = rand::thread_rng();
                let ids: Vec<_> = self.nodes.iter().map(|r| *r.key()).collect();
                ids[rng.gen_range(0..ids.len())]
            }
        };

        if let Some(entry_node) = self.nodes.get(&entry_id) {
            let dist = cosine_distance(query, &entry_node.vector);
            candidates.push(std::cmp::Reverse(Candidate {
                id: entry_id,
                distance: dist,
            }));
            w.push(Candidate {
                id: entry_id,
                distance: dist,
            });
        }

        let mut visited = HashSet::new();
        visited.insert(entry_id);

        for _ in 0..ef {
            let lowerbound = if let Some(candidate) = candidates.peek() {
                candidate.0.distance
            } else {
                break;
            };

            if lowerbound > w.peek().map(|c| c.distance).unwrap_or(f32::INFINITY) {
                break;
            }

            let current = candidates.pop().unwrap().0;

            let neighbor_ids: Vec<NodeId> = self
                .nodes
                .get(&current.id)
                .map(|n| n.neighbors.clone())
                .unwrap_or_default();

            for neighbor_id in neighbor_ids {
                if !visited.insert(neighbor_id) {
                    continue;
                }

                if let Some(neighbor_node) = self.nodes.get(&neighbor_id) {
                    let dist = cosine_distance(query, &neighbor_node.vector);

                    if dist < w.peek().map(|c| c.distance).unwrap_or(f32::INFINITY) || w.len() < ef
                    {
                        candidates.push(std::cmp::Reverse(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        }));

                        w.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });

                        if w.len() > ef {
                            let mut vec: Vec<_> = w.into_iter().collect();
                            vec.sort_by(|a, b| {
                                a.distance
                                    .partial_cmp(&b.distance)
                                    .unwrap_or(Ordering::Equal)
                            });
                            vec.pop();
                            w = vec.into_iter().collect();
                        }
                    }
                }
            }
        }

        let mut result: Vec<_> = w.into_iter().map(|c| (c.id, c.distance)).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }

    /// Performs binary (Hamming) search on the index
    /// Returns a vector of (NodeId, Hamming distance) tuples
    pub fn search_binary(&self, query_binary: &[u64], ef: Option<usize>) -> Vec<(NodeId, u32)> {
        let ef = ef.unwrap_or(self.ef_search.load(AtomicOrdering::Relaxed));

        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        let entry_id = {
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = self.nodes.iter().map(|r| *r.key()).collect();
            ids[rng.gen_range(0..ids.len())]
        };

        if let Some(entry_node) = self.nodes.get(&entry_id) {
            let dist = hamming_distance(query_binary, &entry_node.binary);
            candidates.push(std::cmp::Reverse((dist, entry_id)));
            w.push((dist, entry_id));
        }

        let mut visited = HashSet::new();
        visited.insert(entry_id);

        for _ in 0..ef {
            let lowerbound = candidates.peek().map(|c| c.0 .0).unwrap_or(u32::MAX);

            if lowerbound > w.peek().map(|c| c.0).unwrap_or(u32::MAX) {
                break;
            }

            let (_, current_id) = candidates.pop().unwrap().0;

            let neighbor_ids: Vec<NodeId> = self
                .nodes
                .get(&current_id)
                .map(|n| n.neighbors.clone())
                .unwrap_or_default();

            for neighbor_id in neighbor_ids {
                if !visited.insert(neighbor_id) {
                    continue;
                }

                if let Some(neighbor_node) = self.nodes.get(&neighbor_id) {
                    let dist = hamming_distance(query_binary, &neighbor_node.binary);

                    if dist < w.peek().map(|c| c.0).unwrap_or(u32::MAX) || w.len() < ef {
                        candidates.push(std::cmp::Reverse((dist, neighbor_id)));
                        w.push((dist, neighbor_id));

                        if w.len() > ef {
                            let mut vec: Vec<_> = w.into_iter().collect();
                            vec.sort_by_key(|a| a.0);
                            vec.pop();
                            w = vec.into_iter().collect();
                        }
                    }
                }
            }
        }

        let mut result: Vec<_> = w.into_iter().map(|(d, id)| (id, d)).collect();
        result.sort_by_key(|a| a.1);
        result
    }

    /// Updates hub scores based on neighbor frequency
    /// Higher-scoring nodes appear in many other nodes' neighbor lists
    pub fn update_hub_scores(&self) {
        // Count how many times each node appears in other nodes' neighbor lists
        let mut hub_counts: HashMap<NodeId, u32> = HashMap::new();

        for entry in self.nodes.iter() {
            for neighbor_id in &entry.value().neighbors {
                *hub_counts.entry(*neighbor_id).or_insert(0) += 1;
            }
        }

        // Find max count for normalization
        let max_count = *hub_counts.values().max().unwrap_or(&1) as f32;

        // Update hub scores in DashMap
        for mut entry in self.nodes.iter_mut() {
            let count = hub_counts.get(entry.key()).copied().unwrap_or(0);
            entry.value_mut().hub_score = count as f32 / max_count;
        }

        // Update hub cache with top nodes
        let mut hub_nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|n| (*n.key(), n.value().hub_score))
            .collect();
        hub_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut hub_cache = self.hub_cache.write();
        *hub_cache = hub_nodes.into_iter().take(10).map(|(id, _)| id).collect();
    }

    /// Gets all neighbors within K hops from a node (ego-graph)
    pub fn get_neighbors(&self, id: NodeId, hops: usize) -> HashSet<NodeId> {
        let nodes = self.read_snapshot.load();

        if !nodes.contains_key(&id) {
            return HashSet::new();
        }

        let mut result = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((id, 0));
        visited.insert(id);

        while let Some((current_id, current_hop)) = queue.pop_front() {
            if current_hop > 0 {
                result.insert(current_id);
            }

            if current_hop < hops {
                if let Some(node) = nodes.get(&current_id) {
                    for &neighbor_id in &node.neighbors {
                        if visited.insert(neighbor_id) {
                            queue.push_back((neighbor_id, current_hop + 1));
                        }
                    }
                }
            }
        }

        result
    }

    /// Removes a node from the index and repairs connections
    pub fn remove(&self, id: NodeId) -> bool {
        if self.nodes.remove(&id).is_none() {
            return false;
        }

        // Find all neighbors of the removed node
        let neighbors_to_repair: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|n| n.value().neighbors.contains(&id))
            .map(|n| *n.key())
            .collect();

        // Repair connections: connect neighbors to each other
        for neighbor_id in &neighbors_to_repair {
            if let Some(mut neighbor_node) = self.nodes.get_mut(neighbor_id) {
                // Remove reference to deleted node
                neighbor_node.neighbors.retain(|&nid| nid != id);

                // Try to connect to other neighbors of the deleted node
                for &other_neighbor_id in &neighbors_to_repair {
                    if other_neighbor_id != *neighbor_id
                        && !neighbor_node.neighbors.contains(&other_neighbor_id)
                        && neighbor_node.neighbors.len() < self.max_neighbors
                    {
                        neighbor_node.neighbors.push(other_neighbor_id);
                    }
                }
            }
        }

        true
    }

    /// Returns the number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Checks if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns a copy of a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<GraphNode> {
        self.nodes.get(&id).map(|entry| entry.value().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_insertion() {
        let index = NswIndex::new(128, 32, 200);
        let vector = vec![0.1; 128];
        let metadata = MetaData {
            label: "test".to_string(),
        };
        let id = index.insert(vector, metadata);
        assert_eq!(id, NodeId(0));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_multiple_insertions() {
        let index = NswIndex::new(64, 32, 200);

        for i in 0..10 {
            let mut vector = vec![0.1; 64];
            vector[0] = i as f32 * 0.1;
            let metadata = MetaData {
                label: format!("node_{}", i),
            };
            let id = index.insert(vector, metadata);
            assert_eq!(id, NodeId(i as u64));
        }

        assert_eq!(index.len(), 10);
    }

    #[test]
    fn test_search_finds_inserted_vector() {
        let index = NswIndex::new(32, 32, 200);

        // Insert a distinctive vector
        let query_vec = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let id = index.insert(
            query_vec.clone(),
            MetaData {
                label: "query".to_string(),
            },
        );

        // Insert some other vectors
        for i in 1..5 {
            let mut v = vec![0.1; 32];
            v[0] = i as f32 * 0.1;
            index.insert(
                v,
                MetaData {
                    label: format!("node_{}", i),
                },
            );
        }

        // Search for the first vector
        let results = index.search(&query_vec, Some(10));
        assert!(!results.is_empty());
        assert_eq!(results[0].0, id);
    }

    #[test]
    fn test_hub_scores_update() {
        let index = NswIndex::new(32, 16, 100);

        // Insert some vectors
        for i in 0..20 {
            let v = vec![(i as f32 * 0.1).sin(); 32];
            index.insert(
                v,
                MetaData {
                    label: format!("node_{}", i),
                },
            );
        }

        index.update_hub_scores();

        // Check that hub scores are assigned
        let node_0 = index.get_node(NodeId(0));
        assert!(node_0.is_some());
        let node = node_0.unwrap();
        assert!(node.hub_score >= 0.0 && node.hub_score <= 1.0);
    }

    #[test]
    fn test_get_neighbors_ego_graph() {
        let index = NswIndex::new(32, 16, 100);

        // Insert vectors with controlled structure
        for i in 0..10 {
            let v = vec![(i as f32 * 0.1).sin(); 32];
            index.insert(
                v,
                MetaData {
                    label: format!("node_{}", i),
                },
            );
        }

        // Get 1-hop neighbors of node 0
        let neighbors = index.get_neighbors(NodeId(0), 1);
        assert!(!neighbors.is_empty());

        // Get 2-hop neighbors (should include more nodes)
        let neighbors_2hop = index.get_neighbors(NodeId(0), 2);
        assert!(neighbors_2hop.len() >= neighbors.len());
    }

    #[test]
    fn test_search_binary() {
        let index = NswIndex::new(64, 32, 200);

        let vector = vec![1.5; 64];
        let id = index.insert(
            vector.clone(),
            MetaData {
                label: "test".to_string(),
            },
        );

        // Insert some other vectors
        for i in 1..5 {
            let v = vec![i as f32 * 0.1; 64];
            index.insert(
                v,
                MetaData {
                    label: format!("node_{}", i),
                },
            );
        }

        // Create binary representation of vector
        let binary: Vec<u64> = vector
            .chunks(64)
            .map(|chunk| {
                let mut bits = 0u64;
                for (i, &val) in chunk.iter().enumerate() {
                    if val >= 0.0 {
                        bits |= 1u64 << i;
                    }
                }
                bits
            })
            .collect();

        let results = index.search_binary(&binary, Some(10));
        assert!(!results.is_empty());
        // All vectors have positive components, so binary representations are identical.
        // Just verify the target node appears somewhere in results.
        assert!(results.iter().any(|(nid, _)| *nid == id));
    }

    #[test]
    fn test_remove_node() {
        let index = NswIndex::new(32, 16, 100);

        let _id1 = index.insert(
            vec![1.0; 32],
            MetaData {
                label: "node1".to_string(),
            },
        );
        let id2 = index.insert(
            vec![2.0; 32],
            MetaData {
                label: "node2".to_string(),
            },
        );
        let _id3 = index.insert(
            vec![3.0; 32],
            MetaData {
                label: "node3".to_string(),
            },
        );

        assert_eq!(index.len(), 3);

        let removed = index.remove(id2);
        assert!(removed);
        assert_eq!(index.len(), 2);

        let node = index.get_node(id2);
        assert!(node.is_none());
    }
}
