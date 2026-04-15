// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SIFT-1M benchmark: validates Palace-X full pipeline at scale.
//!
//! Measures HNSW, HNSW+RaBitQ (Asymmetric & RaBitQBeam), and Metal GPU
//! batch reranking against 1M 128d vectors with official ground truth.
//!
//! Usage:
//!   cargo run -p palace-bench --release -- --sift1m
//!   RUN_SIFT1M_BENCH=1 cargo run -p palace-bench --release

use crate::sift::{self, SiftDataset};
use palace_graph::{HnswDistanceMetric, HnswIndex, MetaData};
use palace_optimizer::{MetalBatchSearch, MetalDistanceMetric};
use palace_storage::{HnswRaBitQ, HnswRaBitQConfig, SearchMode};
use std::path::Path;
use std::time::Instant;

/// Run the SIFT-1M benchmark suite.
pub fn run_sift1m_benchmark(data_dir: &Path) {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║       SIFT-1M Benchmark Suite (1M × 128d)           ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let dataset = match sift::load_sift1m(data_dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT-1M: {}", e);
            return;
        }
    };

    println!(
        "Dataset: {} base, {} queries, {} ground truth, {}d\n",
        dataset.base.len(),
        dataset.queries.len(),
        dataset.ground_truth.len(),
        dataset.dim
    );

    let mut results: Vec<(String, f32, f32, f32, f64, String)> = Vec::new();

    // ─── 1. Pure HNSW (float L2) ─────────────────────────────
    {
        println!("═══ Building pure HNSW (M=16, ef_construction=200) ═══");
        let hnsw = HnswIndex::new(dataset.dim, 16, 200);

        let t0 = Instant::now();
        for (i, vec) in dataset.base.iter().enumerate() {
            hnsw.insert(
                vec.clone(),
                MetaData {
                    label: format!("{}", i),
                },
            );
            if (i + 1) % 100_000 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                println!(
                    "  Inserted {}/{}  ({:.0} vec/s)",
                    i + 1,
                    dataset.base.len(),
                    rate
                );
            }
        }
        hnsw.publish_snapshot();
        let build_time = t0.elapsed();
        println!(
            "  HNSW built in {:.1}s ({:.0} vec/s)\n",
            build_time.as_secs_f64(),
            dataset.base.len() as f64 / build_time.as_secs_f64()
        );

        for &ef in &[32usize, 64, 128, 256] {
            let (r1, r10, r100, qps) = bench_hnsw_search(&dataset, &hnsw, ef);
            // Effective ef is max(ef, 100) to match combined.search(q, 100) pool size
            let label = format!("HNSW float (ef={})", ef.max(100));
            println!(
                "  {}: R@1={:.1}% R@10={:.1}% R@100={:.1}%  {:.0} QPS",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D×4+graph".into()));
        }
    }

    // ─── 2. HNSW + RaBitQ 4-bit (Asymmetric) ─────────────────
    {
        println!("\n═══ Building HNSW+RaBitQ-4bit Asymmetric (M=16) ═══");
        let config = HnswRaBitQConfig {
            dimensions: dataset.dim,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 0,
            metric: HnswDistanceMetric::L2,
            search_mode: SearchMode::Asymmetric,
            seed: 42,
            ..Default::default()
        };
        let combined = HnswRaBitQ::new(config);

        let t0 = Instant::now();
        for (i, vec) in dataset.base.iter().enumerate() {
            combined.insert(
                vec.clone(),
                MetaData {
                    label: format!("{}", i),
                },
            );
            if (i + 1) % 100_000 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed;
                println!(
                    "  Inserted {}/{}  ({:.0} vec/s)",
                    i + 1,
                    dataset.base.len(),
                    rate
                );
            }
        }
        combined.publish_snapshot();
        combined.rebuild_with_centroid();
        let build_time = t0.elapsed();
        println!("  Built in {:.1}s", build_time.as_secs_f64());

        let (graph_b, float_b, rq_b, total_b) = combined.memory_estimate();
        println!(
            "  Memory: graph={:.1}MB float={:.1}MB rabitq={:.1}MB total={:.1}MB\n",
            graph_b as f64 / 1_048_576.0,
            float_b as f64 / 1_048_576.0,
            rq_b as f64 / 1_048_576.0,
            total_b as f64 / 1_048_576.0
        );

        for &ef in &[32usize, 64, 128, 256] {
            let (r1, r10, r100, qps) = bench_combined_search(&dataset, &combined, ef);
            let label = format!("HNSW+RaBitQ-4bit Asym (ef={})", ef);
            println!(
                "  {}: R@1={:.1}% R@10={:.1}% R@100={:.1}%  {:.0} QPS",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D/2+16+graph".into()));
        }
    }

    // ─── 3. HNSW + RaBitQ 4-bit (RaBitQBeam + float rerank top-50) ──
    {
        println!("\n═══ Building HNSW+RaBitQ-4bit Beam+Rerank (M=16) ═══");
        let config = HnswRaBitQConfig {
            dimensions: dataset.dim,
            max_neighbors: 16,
            ef_construction: 200,
            rabitq_bits: 4,
            rerank_top: 50,
            metric: HnswDistanceMetric::L2,
            search_mode: SearchMode::RaBitQBeam,
            seed: 42,
            ..Default::default()
        };
        let combined_rerank = HnswRaBitQ::new(config);

        let t0 = Instant::now();
        for (i, vec) in dataset.base.iter().enumerate() {
            combined_rerank.insert(
                vec.clone(),
                MetaData {
                    label: format!("{}", i),
                },
            );
            if (i + 1) % 100_000 == 0 {
                println!("  Inserted {}/{}", i + 1, dataset.base.len());
            }
        }
        combined_rerank.publish_snapshot();
        combined_rerank.rebuild_with_centroid();
        let build_time = t0.elapsed();
        println!("  Built in {:.1}s\n", build_time.as_secs_f64());

        for &ef in &[32usize, 64, 128, 256] {
            let (r1, r10, r100, qps) = bench_combined_search(&dataset, &combined_rerank, ef);
            let label = format!("HNSW+RaBitQ-4bit Beam+RR50 (ef={})", ef);
            println!(
                "  {}: R@1={:.1}% R@10={:.1}% R@100={:.1}%  {:.0} QPS",
                label,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );
            results.push((label, r1, r10, r100, qps, "D/2+16+graph".into()));
        }
    }

    // ─── 4. Metal GPU batch rerank ───────────────────────────
    {
        println!("\n═══ Metal GPU Batch Distance ═══");
        match MetalBatchSearch::new() {
            Some(gpu) => {
                println!("  GPU: {}", gpu.device_name());
                println!("  Calibrating GPU/CPU crossover…");
                gpu.calibrate(dataset.dim);
                println!(
                    "  Active GPU threshold: {} candidates",
                    gpu.active_threshold()
                );

                // Build flat candidate matrix for varying sizes
                for &n_cands in &[256, 1024, 4096, 16384, 65536] {
                    if n_cands > dataset.base.len() {
                        break;
                    }

                    let flat_cands: Vec<f32> = dataset.base[..n_cands]
                        .iter()
                        .flat_map(|v| v.iter().copied())
                        .collect();

                    let n_queries = dataset.queries.len().min(100); // 100 queries for GPU bench
                    let mut total_gpu_us = 0.0f64;
                    let mut total_cpu_us = 0.0f64;

                    for qi in 0..n_queries {
                        let query = &dataset.queries[qi];

                        // GPU path
                        let t0 = Instant::now();
                        let _gpu_dists = gpu.batch_distances(
                            query,
                            &flat_cands,
                            dataset.dim,
                            MetalDistanceMetric::L2,
                        );
                        total_gpu_us += t0.elapsed().as_micros() as f64;

                        // CPU path (scalar L2 for comparison)
                        let t0 = Instant::now();
                        let _cpu_dists: Vec<f32> = (0..n_cands)
                            .map(|i| {
                                let cand = &flat_cands[i * dataset.dim..(i + 1) * dataset.dim];
                                query.iter().zip(cand).map(|(a, b)| (a - b) * (a - b)).sum()
                            })
                            .collect();
                        total_cpu_us += t0.elapsed().as_micros() as f64;
                    }

                    let avg_gpu = total_gpu_us / n_queries as f64;
                    let avg_cpu = total_cpu_us / n_queries as f64;
                    let speedup = avg_cpu / avg_gpu;

                    let label = format!("Metal GPU batch ({} cands)", n_cands);
                    println!(
                        "  {}: GPU={:.0}μs  CPU={:.0}μs  speedup={:.2}x",
                        label, avg_gpu, avg_cpu, speedup
                    );
                    // Note: GPU batch is a latency benchmark, not recall — skip results table.
                }
            }
            None => {
                println!("  Metal not available, skipping GPU benchmark");
            }
        }
    }

    // ─── Results table ───────────────────────────────────────
    println!("\n{}", sift::format_results_table(&results));

    // ─── Summary ─────────────────────────────────────────────
    println!("═══ SIFT-1M Summary ═══");
    if let Some(best_hnsw) = results
        .iter()
        .filter(|r| r.0.starts_with("HNSW float"))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        println!(
            "  Best HNSW R@1:  {:.1}% ({})  @ {:.0} QPS",
            best_hnsw.1 * 100.0,
            best_hnsw.0,
            best_hnsw.4
        );
    }
    if let Some(best_hnsw) = results
        .iter()
        .filter(|r| r.0.starts_with("HNSW float"))
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    {
        println!(
            "  Best HNSW R@10: {:.1}% ({})  @ {:.0} QPS",
            best_hnsw.2 * 100.0,
            best_hnsw.0,
            best_hnsw.4
        );
    }
    if let Some(best_combined) = results
        .iter()
        .filter(|r| r.0.contains("RaBitQ"))
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    {
        println!(
            "  Best RaBitQ R@10: {:.1}% ({})  @ {:.0} QPS",
            best_combined.2 * 100.0,
            best_combined.0,
            best_combined.4
        );
    }
}

// ─── Benchmark helpers ───────────────────────────────────────

fn bench_hnsw_search(dataset: &SiftDataset, hnsw: &HnswIndex, ef: usize) -> (f32, f32, f32, f64) {
    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    // Apples-to-apples: combined.search(q, 100) forces effective ef >= 100.
    // To compare fairly, pure HNSW also needs at least 100 candidates in the pool.
    let effective_ef = ef.max(100);

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let results = hnsw.search(query, Some(effective_ef));
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

fn bench_combined_search(
    dataset: &SiftDataset,
    combined: &HnswRaBitQ,
    ef: usize,
) -> (f32, f32, f32, f64) {
    combined.set_ef_search(ef);

    let mut total_r1 = 0.0f32;
    let mut total_r10 = 0.0f32;
    let mut total_r100 = 0.0f32;

    let start = Instant::now();
    for (qi, query) in dataset.queries.iter().enumerate() {
        let results = combined.search(query, 100);
        let retrieved: Vec<usize> = results.iter().map(|r| r.node_id.0 as usize).collect();

        total_r1 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 1);
        total_r10 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 10);
        total_r100 += sift::recall_at_k(&retrieved, &dataset.ground_truth[qi], 100);
    }
    let elapsed = start.elapsed();
    let n = dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / elapsed.as_secs_f64();

    (total_r1 / n, total_r10 / n, total_r100 / n, qps)
}
