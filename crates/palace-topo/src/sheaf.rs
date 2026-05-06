// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>

//! Sheaf Cohomology: real H¹ obstruction via coboundary operator.
//!
//! Mathematics: Sheaf F over graph G = (V, E)
//!   - Stalk F(v) ∈ ℝⁿ at each vertex v
//!   - Restriction maps F_{u→e}, F_{v→e}: F(v) → F(e) for each edge e=(u,v)
//!   - Coboundary: γ(e) = F_{u→e}·x_u - F_{v→e}·x_v
//!   - H¹ obstruction: (1/|E|) · Σ_e ||γ(e)||² / (||x_u||² + ||x_v||² + ε)
//!
//! H¹ ≈ 0 → consistent knowledge (global section exists, signals agree)
//! H¹ >> 0 → topological gap (conflicting local sections → chop / no trade)
//!
//! Reference: Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves"
//! arXiv:2012.06333; Curry, "Sheaves, Cosheaves and Applications" (2014).

use palace_core::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Local section at a node for a given modality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stalk {
    pub modality_id: String,
    pub values: Vec<f32>,
}

/// Linear restriction map F_{v→e}: ℝ^{input_dim} → ℝ^{output_dim}.
///
/// Stored row-major: `matrix[i * input_dim + j] = M[i][j]`.
/// If no map is set for an edge, identity/truncation is used automatically.
#[derive(Debug, Clone)]
pub struct RestrictionMap {
    pub matrix: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl RestrictionMap {
    /// Apply: y = M · x.  len(x) must equal input_dim.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let in_d = self.input_dim.min(x.len());
        let mut out = vec![0.0f32; self.output_dim];
        for i in 0..self.output_dim {
            for j in 0..in_d {
                out[i] += self.matrix[i * self.input_dim + j] * x[j];
            }
        }
        out
    }

    /// Identity map for dimension `d`.
    pub fn identity(d: usize) -> Self {
        let mut m = vec![0.0f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        Self {
            matrix: m,
            input_dim: d,
            output_dim: d,
        }
    }

    /// Truncation projection: take the first `output_dim` components.
    pub fn truncation(input_dim: usize, output_dim: usize) -> Self {
        let d = output_dim.min(input_dim);
        let mut m = vec![0.0f32; d * input_dim];
        for i in 0..d {
            m[i * input_dim + i] = 1.0;
        }
        Self {
            matrix: m,
            input_dim,
            output_dim: d,
        }
    }
}

/// H¹ obstruction for a single edge.
#[derive(Debug, Clone)]
pub struct EdgeObstruction {
    pub u: NodeId,
    pub v: NodeId,
    /// Normalized obstruction: ||γ(e)||² / (||x_u||² + ||x_v||² + ε)
    pub obstruction: f32,
    /// Raw cocycle γ(e) = F_{u→e}·x_u - F_{v→e}·x_v
    pub cocycle: Vec<f32>,
}

/// Global H¹ result over all edges.
#[derive(Debug, Clone)]
pub struct SheafH1Result {
    /// Mean normalized H¹ ∈ [0, ∞).
    /// < 0.05 → consistent; > 0.5 → strong obstruction.
    pub h1_obstruction: f32,
    /// Per-edge breakdown for inspection.
    pub edge_obstructions: Vec<EdgeObstruction>,
    pub edge_count: usize,
}

impl SheafH1Result {
    /// `threshold` ~ 0.1 works well for cosine-normalized embeddings.
    pub fn is_consistent(&self, threshold: f32) -> bool {
        self.h1_obstruction < threshold
    }
}

pub struct SheafAnalyzer {
    /// Node stalks: node_id → list of modality stalks.
    pub stalks: HashMap<NodeId, Vec<Stalk>>,
    /// Undirected graph edges.
    edges: Vec<(NodeId, NodeId)>,
    /// Per-edge restriction maps.  Key: (min_id, max_id).
    /// Value: (map from lower-id node, map from higher-id node).
    restriction_maps: HashMap<(NodeId, NodeId), (RestrictionMap, RestrictionMap)>,
}

impl SheafAnalyzer {
    pub fn new() -> Self {
        Self {
            stalks: HashMap::new(),
            edges: Vec::new(),
            restriction_maps: HashMap::new(),
        }
    }

    pub fn add_stalk(&mut self, node: NodeId, stalk: Stalk) {
        self.stalks.entry(node).or_default().push(stalk);
    }

    /// Add an undirected edge to the sheaf graph.
    pub fn add_edge(&mut self, u: NodeId, v: NodeId) {
        self.edges.push((u, v));
    }

    /// Set explicit restriction maps for edge (u, v).
    /// `map_u` projects F(u) into edge space; `map_v` projects F(v).
    pub fn set_restriction_maps(
        &mut self,
        u: NodeId,
        v: NodeId,
        map_u: RestrictionMap,
        map_v: RestrictionMap,
    ) {
        let (key, mu, mv) = if u <= v {
            ((u, v), map_u, map_v)
        } else {
            ((v, u), map_v, map_u)
        };
        self.restriction_maps.insert(key, (mu, mv));
    }

    /// Aggregate multi-modal stalks for a node into one vector (mean over modalities).
    fn aggregate_stalk(&self, node: NodeId) -> Option<Vec<f32>> {
        let stalks = self.stalks.get(&node)?;
        if stalks.is_empty() {
            return None;
        }
        if stalks.len() == 1 {
            return Some(stalks[0].values.clone());
        }
        let min_dim = stalks.iter().map(|s| s.values.len()).min().unwrap_or(0);
        if min_dim == 0 {
            return None;
        }
        let mut agg = vec![0.0f32; min_dim];
        for s in stalks {
            for (i, &v) in s.values[..min_dim].iter().enumerate() {
                agg[i] += v;
            }
        }
        let n = stalks.len() as f32;
        for x in &mut agg {
            *x /= n;
        }
        Some(agg)
    }

    /// γ(e) = F_{u→e}·x_u − F_{v→e}·x_v
    fn compute_cocycle(&self, u: NodeId, v: NodeId, x_u: &[f32], x_v: &[f32]) -> Vec<f32> {
        let key = if u <= v { (u, v) } else { (v, u) };
        let edge_dim = x_u.len().min(x_v.len());

        let (proj_u, proj_v) = if let Some((mu, mv)) = self.restriction_maps.get(&key) {
            if u <= v {
                (mu.apply(x_u), mv.apply(x_v))
            } else {
                (mv.apply(x_u), mu.apply(x_v))
            }
        } else {
            (x_u[..edge_dim].to_vec(), x_v[..edge_dim].to_vec())
        };

        let d = proj_u.len().min(proj_v.len());
        (0..d).map(|i| proj_u[i] - proj_v[i]).collect()
    }

    /// Compute H¹ sheaf cohomology obstruction over all edges.
    ///
    /// Normalized so that:
    /// - Identical stalks everywhere → H¹ = 0.0
    /// - Perfectly anti-correlated stalks → H¹ = 2.0
    ///
    /// Stability: γ is bounded by ||x_u|| + ||x_v||, so H¹ ∈ [0, 4] in the worst case.
    /// For unit-norm embeddings H¹ ∈ [0, 2].
    pub fn compute_h1(&self) -> SheafH1Result {
        let mut edge_obstructions = Vec::with_capacity(self.edges.len());
        let mut total = 0.0f32;
        let mut count = 0usize;

        for &(u, v) in &self.edges {
            let Some(x_u) = self.aggregate_stalk(u) else {
                continue;
            };
            let Some(x_v) = self.aggregate_stalk(v) else {
                continue;
            };

            let cocycle = self.compute_cocycle(u, v, &x_u, &x_v);

            let gamma_sq: f32 = cocycle.iter().map(|x| x * x).sum();
            let norm_u_sq: f32 = x_u.iter().map(|x| x * x).sum();
            let norm_v_sq: f32 = x_v.iter().map(|x| x * x).sum();
            let obstruction = gamma_sq / (norm_u_sq + norm_v_sq + 1e-8);

            total += obstruction;
            count += 1;
            edge_obstructions.push(EdgeObstruction {
                u,
                v,
                obstruction,
                cocycle,
            });
        }

        SheafH1Result {
            h1_obstruction: if count > 0 { total / count as f32 } else { 0.0 },
            edge_obstructions,
            edge_count: count,
        }
    }

    /// Cross-modal coherence at a single node (backward-compatible, cosine-based).
    /// NOTE: This is NOT H¹ — use `compute_h1()` for real sheaf cohomology.
    pub fn compute_coherence(&self, node: NodeId) -> f32 {
        let stalks = match self.stalks.get(&node) {
            Some(s) if s.len() > 1 => s,
            _ => return 1.0,
        };
        let mut total = 0.0f32;
        let mut pairs = 0usize;
        for i in 0..stalks.len() {
            for j in (i + 1)..stalks.len() {
                total += cosine_sim(&stalks[i].values, &stalks[j].values);
                pairs += 1;
            }
        }
        if pairs == 0 {
            1.0
        } else {
            total / pairs as f32
        }
    }
}

impl Default for SheafAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0;
    }
    let dot: f32 = (0..len).map(|i| a[i] * b[i]).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        1.0
    } else {
        (dot / (na * nb)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId(id)
    }
    fn stalk(vals: Vec<f32>) -> Stalk {
        Stalk {
            modality_id: "test".into(),
            values: vals,
        }
    }

    #[test]
    fn h1_zero_for_identical_stalks() {
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(node(0), stalk(vec![1.0, 0.5, 0.3]));
        sa.add_stalk(node(1), stalk(vec![1.0, 0.5, 0.3]));
        sa.add_edge(node(0), node(1));
        let r = sa.compute_h1();
        assert!(
            r.h1_obstruction < 1e-6,
            "identical stalks → H¹ = 0, got {}",
            r.h1_obstruction
        );
    }

    #[test]
    fn h1_high_for_opposite_stalks() {
        // γ = [2,0], gamma_sq = 4, denom = 1+1+ε ≈ 2 → obstruction ≈ 2.0
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(node(0), stalk(vec![1.0, 0.0]));
        sa.add_stalk(node(1), stalk(vec![-1.0, 0.0]));
        sa.add_edge(node(0), node(1));
        let r = sa.compute_h1();
        assert!(
            r.h1_obstruction > 1.5,
            "opposite stalks → H¹ high, got {}",
            r.h1_obstruction
        );
    }

    #[test]
    fn h1_low_for_smooth_chain() {
        let mut sa = SheafAnalyzer::new();
        for i in 0u64..4 {
            sa.add_stalk(node(i), stalk(vec![i as f32 * 0.01, 0.5]));
        }
        sa.add_edge(node(0), node(1));
        sa.add_edge(node(1), node(2));
        sa.add_edge(node(2), node(3));
        let r = sa.compute_h1();
        assert!(
            r.h1_obstruction < 0.1,
            "smooth chain → low H¹, got {}",
            r.h1_obstruction
        );
    }

    #[test]
    fn h1_no_edges_returns_zero() {
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(node(0), stalk(vec![1.0, 2.0]));
        let r = sa.compute_h1();
        assert_eq!(r.h1_obstruction, 0.0);
        assert_eq!(r.edge_count, 0);
    }

    #[test]
    fn h1_missing_stalk_skips_edge() {
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(node(0), stalk(vec![1.0, 2.0]));
        // node(1) has no stalk
        sa.add_edge(node(0), node(1));
        let r = sa.compute_h1();
        assert_eq!(r.edge_count, 0);
    }

    #[test]
    fn identity_map_matches_default() {
        let vals_u = vec![1.0, 2.0, 3.0];
        let vals_v = vec![4.0, 5.0, 6.0];

        let mut sa_default = SheafAnalyzer::new();
        sa_default.add_stalk(node(0), stalk(vals_u.clone()));
        sa_default.add_stalk(node(1), stalk(vals_v.clone()));
        sa_default.add_edge(node(0), node(1));

        let mut sa_explicit = SheafAnalyzer::new();
        sa_explicit.add_stalk(node(0), stalk(vals_u.clone()));
        sa_explicit.add_stalk(node(1), stalk(vals_v.clone()));
        sa_explicit.add_edge(node(0), node(1));
        sa_explicit.set_restriction_maps(
            node(0),
            node(1),
            RestrictionMap::identity(3),
            RestrictionMap::identity(3),
        );

        let r1 = sa_default.compute_h1();
        let r2 = sa_explicit.compute_h1();
        assert!(
            (r1.h1_obstruction - r2.h1_obstruction).abs() < 1e-6,
            "identity map must match default truncation"
        );
    }

    #[test]
    fn is_consistent_near_identical() {
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(node(0), stalk(vec![1.0, 0.0]));
        sa.add_stalk(node(1), stalk(vec![1.001, 0.0]));
        sa.add_edge(node(0), node(1));
        let r = sa.compute_h1();
        assert!(
            r.is_consistent(0.1),
            "near-identical stalks should be consistent, H¹={}",
            r.h1_obstruction
        );
    }

    #[test]
    fn multi_modal_aggregation() {
        // Two modalities per node — should aggregate and compute H¹
        let mut sa = SheafAnalyzer::new();
        sa.add_stalk(
            node(0),
            Stalk {
                modality_id: "price".into(),
                values: vec![1.0, 0.0],
            },
        );
        sa.add_stalk(
            node(0),
            Stalk {
                modality_id: "volume".into(),
                values: vec![0.8, 0.2],
            },
        );
        sa.add_stalk(
            node(1),
            Stalk {
                modality_id: "price".into(),
                values: vec![1.0, 0.0],
            },
        );
        sa.add_stalk(
            node(1),
            Stalk {
                modality_id: "volume".into(),
                values: vec![0.8, 0.2],
            },
        );
        sa.add_edge(node(0), node(1));
        let r = sa.compute_h1();
        assert!(
            r.h1_obstruction < 1e-5,
            "identical multi-modal stalks → H¹≈0, got {}",
            r.h1_obstruction
        );
    }
}
