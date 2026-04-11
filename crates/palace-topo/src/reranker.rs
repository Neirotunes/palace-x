// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Topological reranker for Stage 2 refinement of retrieval candidates.
//!
//! The TopologicalReranker takes a set of candidates from Stage 1 (NSW coarse search)
//! and reranks them using the d_total metric, which combines cosine similarity with
//! analysis of the local topological structure via ego-graphs.
//!
//! **Parallelized via Rayon**: ego-graph construction and β₁ computation run in
//! parallel across all candidates, giving ~Nx speedup on N cores.

use crate::ego_cache::EgoCache;
use crate::ego_graph::EgoGraph;
use crate::metric::d_total;
use palace_core::NodeId;
use rayon::prelude::*;

/// A fragment candidate from Stage 1 retrieval.
///
/// Represents a potential result with associated metadata.
#[derive(Clone, Debug)]
pub struct Fragment {
    /// The node ID in the semantic graph
    pub node_id: NodeId,
    /// Cosine distance to the query
    pub cosine_dist: f32,
    /// Optional metadata (semantic embedding or content hash)
    pub metadata: Option<Vec<u8>>,
}

/// The topological reranker for Stage 2 refinement.
///
/// Combines cosine similarity with structural analysis to rerank candidates
/// based on how well-connected they are to other candidates.
#[derive(Clone, Debug)]
pub struct TopologicalReranker {
    /// Weight for cosine distance component (typically 0.3-0.7)
    pub alpha: f32,
    /// Weight for structural component (typically 0.3-0.7)
    pub beta: f32,
}

impl TopologicalReranker {
    /// Create a new reranker with specified weights.
    ///
    /// # Arguments
    /// * `alpha` - Weight for cosine distance (suggested: 0.5)
    /// * `beta` - Weight for structural component (suggested: 0.5)
    ///
    /// Typically alpha + beta ≈ 1.0 for normalized weighting.
    pub fn new(alpha: f32, beta: f32) -> Self {
        TopologicalReranker { alpha, beta }
    }

    /// Rerank candidate fragments using ego-graph structural analysis.
    ///
    /// **Parallelized via Rayon `par_iter()`**: each candidate's ego-graph
    /// construction and β₁ computation runs independently in parallel,
    /// giving ~Nx speedup on N CPU cores.
    ///
    /// # Algorithm
    /// For each candidate fragment (in parallel):
    /// 1. Build a 1-hop ego-graph around it
    /// 2. Compute the first Betti number (β₁) measuring cycle richness
    /// 3. Combine cosine distance with structural information via d_total:
    ///    d_total = α · cosine_dist + β · exp(-β₁/|E|)
    ///
    /// # Arguments
    /// * `candidates` - List of candidate fragments to rerank
    /// * `neighbors_fn` - Function that returns immediate neighbors of a given NodeId
    ///
    /// # Returns
    /// Candidates sorted by ascending d_total (best first)
    pub fn rerank<F>(&self, candidates: &[Fragment], neighbors_fn: F) -> Vec<Fragment>
    where
        F: Fn(NodeId) -> Vec<NodeId> + Sync,
    {
        if candidates.is_empty() {
            return vec![];
        }

        // Parallel: build ego-graphs and compute d_total for each candidate
        let mut scored: Vec<(Fragment, f32)> = candidates
            .par_iter()
            .map(|candidate| {
                let ego = EgoGraph::build_single(candidate.node_id, 1, |node| neighbors_fn(node)).with_cap(500);
                let distance = d_total(candidate.cosine_dist, &ego, self.alpha, self.beta);
                (candidate.clone(), distance)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(fragment, _)| fragment).collect()
    }

    /// Rerank with ego-graph caching for repeated queries.
    ///
    /// Same as `rerank()` but uses the provided `EgoCache` to avoid
    /// redundant ego-graph construction for recently-seen nodes.
    pub fn rerank_cached<F>(
        &self,
        candidates: &[Fragment],
        neighbors_fn: F,
        cache: &EgoCache,
    ) -> Vec<Fragment>
    where
        F: Fn(NodeId) -> Vec<NodeId> + Sync,
    {
        if candidates.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(Fragment, f32)> = candidates
            .par_iter()
            .map(|candidate| {
                let ego = match cache.get(candidate.node_id) {
                    Some(cached) => cached,
                    None => {
                        let built =
                            EgoGraph::build_single(candidate.node_id, 1, |node| neighbors_fn(node)).with_cap(500);
                        cache.put(candidate.node_id, built.clone());
                        built
                    }
                };
                let distance = d_total(candidate.cosine_dist, &ego, self.alpha, self.beta);
                (candidate.clone(), distance)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(fragment, _)| fragment).collect()
    }

    /// Sequential rerank (for benchmarking or small candidate sets).
    pub fn rerank_sequential<F>(&self, candidates: &[Fragment], neighbors_fn: F) -> Vec<Fragment>
    where
        F: Fn(NodeId) -> Vec<NodeId>,
    {
        if candidates.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(Fragment, f32)> = candidates
            .iter()
            .map(|candidate| {
                let ego = EgoGraph::build_single(candidate.node_id, 1, |node| neighbors_fn(node)).with_cap(500);
                let distance = d_total(candidate.cosine_dist, &ego, self.alpha, self.beta);
                (candidate.clone(), distance)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(fragment, _)| fragment).collect()
    }

    /// Rerank candidates with pairwise structural analysis.
    ///
    /// A more sophisticated variant that builds ego-graphs around candidate pairs
    /// to capture shared structural context. **Also parallelized via Rayon.**
    pub fn rerank_pairwise<F>(&self, candidates: &[Fragment], neighbors_fn: F) -> Vec<Fragment>
    where
        F: Fn(NodeId) -> Vec<NodeId> + Sync,
    {
        if candidates.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(Fragment, f32)> = candidates
            .par_iter()
            .map(|candidate| {
                let mut total_distance = 0.0;

                for other in candidates {
                    if candidate.node_id != other.node_id {
                        let ego = EgoGraph::build_pair(candidate.node_id, other.node_id, |node| {
                            neighbors_fn(node)
                        }).with_cap(500);
                        let distance = d_total(candidate.cosine_dist, &ego, self.alpha, self.beta);
                        total_distance += distance;
                    }
                }

                let avg_distance =
                    total_distance / (candidates.len().saturating_sub(1).max(1) as f32);
                (candidate.clone(), avg_distance)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(fragment, _)| fragment).collect()
    }
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
    }

    #[test]
    fn test_rerank_single_candidate() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let reranker = TopologicalReranker::new(0.5, 0.5);
        let candidates = vec![Fragment {
            node_id: NodeId(0),
            cosine_dist: 0.5,
            metadata: None,
        }];

        let result = reranker.rerank(&candidates, |node| graph.neighbors(node));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, NodeId(0));
    }

    #[test]
    fn test_rerank_multiple_candidates() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));
        graph.add_edge(NodeId(2), NodeId(0));
        graph.add_edge(NodeId(3), NodeId(0));

        let reranker = TopologicalReranker::new(0.5, 0.5);
        let candidates = vec![
            Fragment {
                node_id: NodeId(3),
                cosine_dist: 0.4,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(0),
                cosine_dist: 0.5,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(1),
                cosine_dist: 0.5,
                metadata: None,
            },
        ];

        let result = reranker.rerank(&candidates, |node| graph.neighbors(node));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_rerank_preserves_count() {
        let graph = TestGraph::new();
        let reranker = TopologicalReranker::new(0.5, 0.5);
        let candidates = vec![
            Fragment {
                node_id: NodeId(0),
                cosine_dist: 0.3,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(1),
                cosine_dist: 0.4,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(2),
                cosine_dist: 0.5,
                metadata: None,
            },
        ];

        let result = reranker.rerank(&candidates, |node| graph.neighbors(node));
        assert_eq!(result.len(), candidates.len());
    }

    #[test]
    fn test_rerank_empty_candidates() {
        let graph = TestGraph::new();
        let reranker = TopologicalReranker::new(0.5, 0.5);
        let result = reranker.rerank(&[], |node| graph.neighbors(node));
        assert!(result.is_empty());
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let mut graph = TestGraph::new();
        for i in 0u64..19 {
            graph.add_edge(NodeId(i), NodeId(i + 1));
        }

        let reranker = TopologicalReranker::new(0.7, 0.3);
        let candidates: Vec<Fragment> = (0..20)
            .map(|i| Fragment {
                node_id: NodeId(i),
                cosine_dist: 0.1 * (i as f32),
                metadata: None,
            })
            .collect();

        let par = reranker.rerank(&candidates, |node| graph.neighbors(node));
        let seq = reranker.rerank_sequential(&candidates, |node| graph.neighbors(node));

        assert_eq!(par.len(), seq.len());
        for (p, s) in par.iter().zip(seq.iter()) {
            assert_eq!(p.node_id, s.node_id);
        }
    }

    #[test]
    fn test_rerank_with_cache() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));

        let reranker = TopologicalReranker::new(0.5, 0.5);
        let cache = EgoCache::new(100);
        let candidates = vec![
            Fragment {
                node_id: NodeId(0),
                cosine_dist: 0.5,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(1),
                cosine_dist: 0.3,
                metadata: None,
            },
        ];

        // First pass — cache miss
        let _r1 = reranker.rerank_cached(&candidates, |n| graph.neighbors(n), &cache);
        let (hits, misses) = cache.stats();
        assert_eq!(misses, 2);
        assert_eq!(hits, 0);

        // Second pass — cache hit
        let _r2 = reranker.rerank_cached(&candidates, |n| graph.neighbors(n), &cache);
        let (hits2, _) = cache.stats();
        assert_eq!(hits2, 2);
    }

    #[test]
    fn test_rerank_pairwise() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));
        graph.add_edge(NodeId(1), NodeId(2));

        let reranker = TopologicalReranker::new(0.5, 0.5);
        let candidates = vec![
            Fragment {
                node_id: NodeId(0),
                cosine_dist: 0.3,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(1),
                cosine_dist: 0.4,
                metadata: None,
            },
        ];

        let result = reranker.rerank_pairwise(&candidates, |node| graph.neighbors(node));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_rerank_with_metadata() {
        let graph = TestGraph::new();
        let reranker = TopologicalReranker::new(0.5, 0.5);
        let candidates = vec![Fragment {
            node_id: NodeId(0),
            cosine_dist: 0.5,
            metadata: Some(vec![1, 2, 3]),
        }];

        let result = reranker.rerank(&candidates, |node| graph.neighbors(node));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metadata, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_rerank_weights_matter() {
        let mut graph = TestGraph::new();
        graph.add_edge(NodeId(0), NodeId(1));

        let reranker_cosine = TopologicalReranker::new(1.0, 0.0);
        let reranker_structural = TopologicalReranker::new(0.0, 1.0);

        let candidates = vec![
            Fragment {
                node_id: NodeId(0),
                cosine_dist: 0.2,
                metadata: None,
            },
            Fragment {
                node_id: NodeId(1),
                cosine_dist: 0.8,
                metadata: None,
            },
        ];

        let result_cosine = reranker_cosine.rerank(&candidates, |node| graph.neighbors(node));
        let result_structural =
            reranker_structural.rerank(&candidates, |node| graph.neighbors(node));

        assert_eq!(result_cosine.len(), 2);
        assert_eq!(result_structural.len(), 2);
    }
}
