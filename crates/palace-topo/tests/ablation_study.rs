// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ablation study: α/β parameter sweep for topological reranking.
//!
//! Tests on synthetic data (random 128d vectors, n=1000) to compare:
//! 1. Pure cosine (α=1.0, β=0.0) — baseline
//! 2. Pure topological (α=0.0, β=1.0)
//! 3. Default mix (α=0.7, β=0.3)
//! 4. Grid search: α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, β = 1-α

use palace_core::NodeId;
use palace_topo::{ego_graph::EgoGraph, metric::d_total};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const N_VECTORS: usize = 1000;
const DIM: usize = 128;
const K: usize = 10;
const N_QUERIES: usize = 50;
const M: usize = 16; // max neighbors for NSW-like graph

fn generate_data(rng: &mut StdRng) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let base: Vec<Vec<f32>> = (0..N_VECTORS)
        .map(|_| (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    (base, queries)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    na = na.sqrt();
    nb = nb.sqrt();
    if na == 0.0 || nb == 0.0 {
        return 2.0;
    }
    1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
}

fn ground_truth(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, b)| (i, cosine_distance(b, query)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(i, _)| *i).collect()
}

/// Build a simple NSW-like graph: for each node, connect to M nearest neighbors
fn build_graph(base: &[Vec<f32>]) -> HashMap<NodeId, Vec<NodeId>> {
    let mut graph: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let n = base.len();

    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_distance(&base[i], &base[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<NodeId> = dists
            .iter()
            .take(M)
            .map(|(j, _)| NodeId(*j as u64))
            .collect();
        graph.insert(NodeId(i as u64), neighbors);
    }

    graph
}

fn recall_at_k(retrieved: &[usize], gt: &[usize], k: usize) -> f32 {
    let gt_set: HashSet<usize> = gt.iter().take(k).copied().collect();
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();
    hits as f32 / k.min(gt.len()) as f32
}

fn mrr_at_k(retrieved: &[usize], gt: &[usize], k: usize) -> f32 {
    let gt_set: HashSet<usize> = gt.iter().take(k).copied().collect();
    for (rank, id) in retrieved.iter().take(k).enumerate() {
        if gt_set.contains(id) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Search with topological reranking
fn search_with_rerank(
    query: &[f32],
    base: &[Vec<f32>],
    graph: &HashMap<NodeId, Vec<NodeId>>,
    alpha: f32,
    beta: f32,
    candidate_k: usize,
) -> Vec<usize> {
    // Stage 1: Get top candidates by cosine distance
    let mut dists: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, b)| (i, cosine_distance(b, query)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let candidates: Vec<(usize, f32)> = dists.into_iter().take(candidate_k).collect();

    if beta < 1e-6 {
        // Pure cosine — no reranking needed
        return candidates.iter().map(|(i, _)| *i).collect();
    }

    // Stage 2: Topological reranking
    let _query_pseudo_id = NodeId(base.len() as u64); // pseudo-ID for query
    let mut reranked: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&(idx, cos_dist)| {
            let node_id = NodeId(idx as u64);

            // Build ego-graph around candidate
            let ego = EgoGraph::build_single(node_id, 1, |id| {
                graph.get(&id).cloned().unwrap_or_default()
            })
            .with_cap(500);

            let d = d_total(cos_dist, &ego, alpha, beta);
            (idx, d)
        })
        .collect();

    reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    reranked.iter().map(|(i, _)| *i).collect()
}

#[test]
fn ablation_alpha_beta() {
    let mut rng = StdRng::seed_from_u64(42);
    let (base, queries) = generate_data(&mut rng);

    eprintln!("\n═══ ABLATION STUDY: α/β Parameter Sweep ═══");
    eprintln!(
        "Dataset: {} vectors × {}d, {} queries, k={}",
        N_VECTORS, DIM, N_QUERIES, K
    );
    eprintln!("Building kNN graph (M={})...", M);

    let graph = build_graph(&base);

    eprintln!("Graph built. Running ablation...\n");

    let configs: Vec<(&str, f32, f32)> = vec![
        ("Pure cosine (α=1.0, β=0.0)", 1.0, 0.0),
        ("Pure topological (α=0.0, β=1.0)", 0.0, 1.0),
        ("α=0.5, β=0.5", 0.5, 0.5),
        ("α=0.6, β=0.4", 0.6, 0.4),
        ("α=0.7, β=0.3 (default)", 0.7, 0.3),
        ("α=0.8, β=0.2", 0.8, 0.2),
        ("α=0.9, β=0.1", 0.9, 0.1),
    ];

    let candidate_k = 50; // rerank top-50 candidates

    eprintln!(
        "| Configuration | Recall@{} | MRR@{} | Latency/query |",
        K, K
    );
    eprintln!("|---------------|-----------|--------|---------------|");

    let mut best_recall = 0.0f32;
    let mut best_config = "";

    for (name, alpha, beta) in &configs {
        let start = Instant::now();
        let mut total_recall = 0.0f32;
        let mut total_mrr = 0.0f32;

        for query in &queries {
            let gt = ground_truth(&base, query, K);
            let retrieved = search_with_rerank(query, &base, &graph, *alpha, *beta, candidate_k);
            total_recall += recall_at_k(&retrieved, &gt, K);
            total_mrr += mrr_at_k(&retrieved, &gt, K);
        }

        let elapsed = start.elapsed();
        let avg_recall = total_recall / N_QUERIES as f32;
        let avg_mrr = total_mrr / N_QUERIES as f32;
        let latency = elapsed / N_QUERIES as u32;

        eprintln!(
            "| {:<30} | {:>7.1}% | {:>6.3} | {:>13?} |",
            name,
            avg_recall * 100.0,
            avg_mrr,
            latency,
        );

        if avg_recall > best_recall {
            best_recall = avg_recall;
            best_config = name;
        }
    }

    eprintln!(
        "\nBest configuration: {} (recall@{} = {:.1}%)",
        best_config,
        K,
        best_recall * 100.0
    );

    // The pure cosine baseline should have good recall since we're doing brute-force
    // The test passes if it runs without panic — actual results are informational
    assert!(
        best_recall > 0.0,
        "At least one config should have non-zero recall"
    );
}
