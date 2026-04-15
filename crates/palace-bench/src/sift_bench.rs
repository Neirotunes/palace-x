// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SIFT-10K benchmark suite.
//!
//! Runs recall@k and QPS benchmarks on the SIFT-10K dataset for all
//! Palace-X quantization and search configurations.

use crate::sift::{self, SiftDataset};
use palace_graph::{MetaData, NswIndex};
use palace_quant::binary::quantize_binary;
use palace_quant::hamming::hamming_distance;
use palace_quant::rabitq::{rabitq_topk, RaBitQCode, RaBitQIndex};
use std::path::Path;
use std::time::Instant;

/// Run the full SIFT-10K benchmark suite.
pub fn run_sift_benchmark(data_dir: &Path) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║       SIFT-10K Benchmark Suite                       ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let dataset = match sift::load_sift10k(data_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT-10K: {}", e);
            eprintln!("Please ensure data/siftsmall/ exists with fvecs/ivecs files.");
            return;
        }
    };

    let mut results: Vec<(String, f32, f32, f32, f64, String)> = Vec::new();

    // Compute centroid for RaBitQ
    let centroid = compute_centroid(&dataset.base);

    // ─── Brute-force cosine baseline ──────────────────────────
    {
        let (r1, r10, r100, qps) = bench_brute_force_cosine(&dataset);
        println!(
            "Brute-force cosine: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
            r1 * 100.0,
            r10 * 100.0,
            r100 * 100.0,
            qps
        );
        results.push((
            "Brute-force cosine".into(),
            r1,
            r10,
            r100,
            qps,
            "D×4 B".into(),
        ));
    }

    // ─── Naive binary (Hamming brute-force) ───────────────────
    {
        let (r1, r10, r100, qps) = bench_naive_binary(&dataset);
        println!(
            "Naive binary: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
            r1 * 100.0,
            r10 * 100.0,
            r100 * 100.0,
            qps
        );
        results.push((
            "Naive binary (Hamming)".into(),
            r1,
            r10,
            r100,
            qps,
            "D/8 B".into(),
        ));
    }

    // ─── RaBitQ 1-bit brute-force ─────────────────────────────
    {
        let (r1, r10, r100, qps) = bench_rabitq_brute(&dataset, &centroid, 1);
        println!(
            "RaBitQ 1-bit: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
            r1 * 100.0,
            r10 * 100.0,
            r100 * 100.0,
            qps
        );
        results.push((
            "RaBitQ 1-bit (brute)".into(),
            r1,
            r10,
            r100,
            qps,
            "D/8+16 B".into(),
        ));
    }

    // ─── RaBitQ 4-bit brute-force ─────────────────────────────
    {
        let (r1, r10, r100, qps) = bench_rabitq_brute(&dataset, &centroid, 4);
        println!(
            "RaBitQ 4-bit: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
            r1 * 100.0,
            r10 * 100.0,
            r100 * 100.0,
            qps
        );
        results.push((
            "RaBitQ 4-bit (brute)".into(),
            r1,
            r10,
            r100,
            qps,
            "D/2+16 B".into(),
        ));
    }

    // ─── NSW + RaBitQ reranking ───────────────────────────────
    // Build NSW index once
    let nsw = NswIndex::with_l2(dataset.dim, 32, 200);
    println!("\nBuilding NSW index ({} vectors)...", dataset.base.len());
    let t0 = Instant::now();
    for (i, vec) in dataset.base.iter().enumerate() {
        nsw.insert(
            vec.clone(),
            MetaData {
                label: format!("{}", i),
            },
        );
    }
    nsw.update_hub_scores();
    let build_time = t0.elapsed();
    println!("NSW built in {:.2}s", build_time.as_secs_f64());

    // RaBitQ index for reranking
    let rq1 = RaBitQIndex::with_centroid(dataset.dim, centroid.clone(), 42);
    let codes_1bit: Vec<RaBitQCode> = dataset.base.iter().map(|v| rq1.encode(v)).collect();

    let rq4 = RaBitQIndex::with_centroid(dataset.dim, centroid.clone(), 42);
    let codes_4bit: Vec<RaBitQCode> = dataset
        .base
        .iter()
        .map(|v| rq4.encode_multibit(v, 4))
        .collect();

    for &ef in &[32usize, 64, 128, 256] {
        // NSW search → cosine rerank
        {
            let (r1, r10, r100, qps) = bench_nsw_search(&dataset, &nsw, ef);
            let label = format!("NSW cosine (ef={})", ef);
            println!(
                "{}: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D×4+graph B".into()));
        }

        // NSW → RaBitQ 1-bit rerank
        {
            let (r1, r10, r100, qps) =
                bench_nsw_rabitq_rerank(&dataset, &nsw, &rq1, &codes_1bit, ef);
            let label = format!("NSW + RaBitQ-1bit (ef={})", ef);
            println!(
                "{}: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D/8+16+graph B".into()));
        }

        // NSW → RaBitQ 4-bit rerank
        {
            let (r1, r10, r100, qps) =
                bench_nsw_rabitq_rerank(&dataset, &nsw, &rq4, &codes_4bit, ef);
            let label = format!("NSW + RaBitQ-4bit (ef={})", ef);
            println!(
                "{}: R@1={:.1}% R@10={:.1}% R@100={:.1}% QPS={:.0}",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D/2+16+graph B".into()));
        }
    }

    // ─── Print Markdown table ─────────────────────────────────
    println!("\n{}", sift::format_results_table(&results));
}

fn compute_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n;
    }
    centroid
}

fn bench_brute_force_cosine(dataset: &SiftDataset) -> (f32, f32, f32, f64) {
    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let mut dists: Vec<(usize, f32)> = dataset
            .base
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let d: f32 = b
                    .iter()
                    .zip(query.iter())
                    .map(|(a, q)| (a - q) * (a - q))
                    .sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let retrieved: Vec<usize> = dists.iter().map(|(i, _)| *i).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}

fn bench_naive_binary(dataset: &SiftDataset) -> (f32, f32, f32, f64) {
    let binary_base: Vec<Vec<u64>> = dataset.base.iter().map(|v| quantize_binary(v)).collect();

    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let query_bin = quantize_binary(query);
        let mut dists: Vec<(usize, u32)> = binary_base
            .iter()
            .enumerate()
            .map(|(i, b)| (i, hamming_distance(&query_bin, b)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        let retrieved: Vec<usize> = dists.iter().map(|(i, _)| *i).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}

fn bench_rabitq_brute(dataset: &SiftDataset, centroid: &[f32], bits: u8) -> (f32, f32, f32, f64) {
    let rq = RaBitQIndex::with_centroid(dataset.dim, centroid.to_vec(), 42);
    let codes: Vec<RaBitQCode> = dataset
        .base
        .iter()
        .map(|v| rq.encode_multibit(v, bits))
        .collect();

    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let rq_query = rq.encode_query(query);
        let results = rabitq_topk(&rq, &rq_query, &codes, 100);
        let retrieved: Vec<usize> = results.iter().map(|(i, _)| *i).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}

fn bench_nsw_search(dataset: &SiftDataset, nsw: &NswIndex, ef: usize) -> (f32, f32, f32, f64) {
    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let results = nsw.search(query, Some(ef));
        let retrieved: Vec<usize> = results.iter().map(|(id, _)| id.0 as usize).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}

fn bench_nsw_rabitq_rerank(
    dataset: &SiftDataset,
    nsw: &NswIndex,
    rq: &RaBitQIndex,
    codes: &[RaBitQCode],
    ef: usize,
) -> (f32, f32, f32, f64) {
    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        // Stage 1: NSW graph search to get candidates
        let candidates = nsw.search(query, Some(ef));

        // Stage 2: Rerank candidates with RaBitQ asymmetric distance
        let rq_query = rq.encode_query(query);
        let mut reranked: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(id, _)| {
                let idx = id.0 as usize;
                let (est_dist, _) = rq.estimate_distance(&rq_query, &codes[idx]);
                (idx, est_dist)
            })
            .collect();
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let retrieved: Vec<usize> = reranked.iter().map(|(i, _)| *i).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}
