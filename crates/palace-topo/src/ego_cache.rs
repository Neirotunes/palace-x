// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! LRU-style ego-graph cache with concurrent access.
//!
//! Pre-computed ego-graphs are cached to avoid redundant BFS traversals
//! during topological reranking. The cache is invalidated per-node when
//! the graph topology changes (insert/remove).

use crate::ego_graph::EgoGraph;
use palace_core::NodeId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe ego-graph cache with LRU eviction.
pub struct EgoCache {
    entries: RwLock<HashMap<NodeId, CacheEntry>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    clock: AtomicU64,
}

#[derive(Clone)]
struct CacheEntry {
    ego: EgoGraph,
    last_access: u64,
}

impl EgoCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            clock: AtomicU64::new(0),
        }
    }

    /// Try to get a cached ego-graph (read-only fast path).
    pub fn get(&self, node_id: NodeId) -> Option<EgoGraph> {
        let entries = self.entries.read();
        if let Some(entry) = entries.get(&node_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.ego.clone());
        }
        drop(entries);
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store an ego-graph, evicting LRU entry if at capacity.
    pub fn put(&self, node_id: NodeId, ego: EgoGraph) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.write();

        if entries.len() >= self.capacity && !entries.contains_key(&node_id) {
            if let Some((&oldest_id, _)) = entries.iter().min_by_key(|(_, e)| e.last_access) {
                entries.remove(&oldest_id);
            }
        }

        entries.insert(
            node_id,
            CacheEntry {
                ego,
                last_access: tick,
            },
        );
    }

    /// Invalidate a single node's cached ego-graph.
    pub fn invalidate(&self, node_id: NodeId) {
        self.entries.write().remove(&node_id);
    }

    /// Clear all cached entries and reset counters.
    pub fn clear(&self) {
        self.entries.write().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Return (hits, misses) counters.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            h / total * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ego(n: usize) -> EgoGraph {
        let vertices: Vec<NodeId> = (0..n as u64).map(NodeId).collect();
        EgoGraph {
            vertices: vertices.clone(),
            edges: vec![],
            num_vertices: n,
            num_edges: 0,
            max_ego_edges: None,
        }
    }

    #[test]
    fn test_cache_hit_miss() {
        let cache = EgoCache::new(10);
        assert!(cache.get(NodeId(0)).is_none());
        assert_eq!(cache.stats(), (0, 1));

        cache.put(NodeId(0), make_ego(3));
        let result = cache.get(NodeId(0));
        assert!(result.is_some());
        assert_eq!(result.unwrap().num_vertices, 3);
        assert_eq!(cache.stats(), (1, 1));
    }

    #[test]
    fn test_eviction() {
        let cache = EgoCache::new(2);
        cache.put(NodeId(0), make_ego(1));
        cache.put(NodeId(1), make_ego(2));
        assert_eq!(cache.len(), 2);

        cache.put(NodeId(2), make_ego(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(NodeId(0)).is_none());
        assert!(cache.get(NodeId(2)).is_some());
    }

    #[test]
    fn test_invalidate() {
        let cache = EgoCache::new(10);
        cache.put(NodeId(0), make_ego(3));
        assert!(cache.get(NodeId(0)).is_some());
        cache.invalidate(NodeId(0));
        // miss counter was already at 0 from initial state, now +1 from this get
        assert!(cache.get(NodeId(0)).is_none());
    }

    #[test]
    fn test_clear() {
        let cache = EgoCache::new(10);
        cache.put(NodeId(0), make_ego(1));
        cache.put(NodeId(1), make_ego(2));
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats(), (0, 0));
    }
}
