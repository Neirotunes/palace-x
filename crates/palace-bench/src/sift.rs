// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! SIFT dataset loader for fvecs/ivecs format + recall/QPS metrics.
//!
//! Parses the standard ANN benchmark format (SIFT1M, SIFT10K, GIST, etc.).
//! Downloads siftsmall (10K) from IRISA if not cached locally.

use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::process::Command;

// ─── Dataset Structure ────────────────────────────────────────────

/// A loaded SIFT dataset with base vectors, queries, and ground truth.
pub struct SiftDataset {
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    /// Ground truth: for each query, indices of true nearest neighbors
    pub ground_truth: Vec<Vec<u32>>,
    pub dim: usize,
}

// ─── File I/O ─────────────────────────────────────────────────────

/// Loads vectors from a .fvecs file.
pub fn load_fvecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut results = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut buf = vec![0u8; dim * 4];
        reader.read_exact(&mut buf)?;

        let vec: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        results.push(vec);
    }

    Ok(results)
}

/// Loads indices from a .ivecs file.
pub fn load_ivecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut results = Vec::new();

    loop {
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut buf = vec![0u8; dim * 4];
        reader.read_exact(&mut buf)?;

        let vec: Vec<u32> = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        results.push(vec);
    }

    Ok(results)
}

// ─── Dataset Loading ──────────────────────────────────────────────

/// Load SIFT-10K (siftsmall) dataset. Downloads if not cached.
pub fn load_sift10k(data_dir: &Path) -> io::Result<SiftDataset> {
    let sift_dir = data_dir.join("siftsmall");
    let base_path = sift_dir.join("siftsmall_base.fvecs");

    if !base_path.exists() {
        download_siftsmall(data_dir)?;
    }

    let base = load_fvecs(sift_dir.join("siftsmall_base.fvecs"))?;
    let queries = load_fvecs(sift_dir.join("siftsmall_query.fvecs"))?;
    let ground_truth = load_ivecs(sift_dir.join("siftsmall_groundtruth.ivecs"))?;

    let dim = if base.is_empty() { 128 } else { base[0].len() };

    eprintln!(
        "SIFT-10K loaded: {} base ({}d), {} queries, {} ground truth",
        base.len(),
        dim,
        queries.len(),
        ground_truth.len()
    );

    Ok(SiftDataset {
        base,
        queries,
        ground_truth,
        dim,
    })
}

/// Load SIFT-1M dataset. Downloads if not cached (~170 MB compressed).
pub fn load_sift1m(data_dir: &Path) -> io::Result<SiftDataset> {
    let sift_dir = data_dir.join("sift");
    let base_path = sift_dir.join("sift_base.fvecs");

    if !base_path.exists() {
        download_sift1m(data_dir)?;
    }

    let base = load_fvecs(sift_dir.join("sift_base.fvecs"))?;
    let queries = load_fvecs(sift_dir.join("sift_query.fvecs"))?;
    let ground_truth = load_ivecs(sift_dir.join("sift_groundtruth.ivecs"))?;

    let dim = if base.is_empty() { 128 } else { base[0].len() };

    eprintln!(
        "SIFT-1M loaded: {} base ({}d), {} queries, {} ground truth",
        base.len(),
        dim,
        queries.len(),
        ground_truth.len()
    );

    Ok(SiftDataset {
        base,
        queries,
        ground_truth,
        dim,
    })
}

fn download_sift1m(data_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(data_dir)?;
    let tar_path = data_dir.join("sift.tar.gz");

    if !tar_path.exists() {
        eprintln!("Downloading SIFT-1M dataset (~170 MB)...");
        let urls = [
            "http://corpus-texmex.irisa.fr/sift.tar.gz",
            "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        ];

        let mut ok = false;
        for url in &urls {
            eprintln!("  Trying: {}", url);
            let status = Command::new("curl")
                .args(["-fSL", "--connect-timeout", "60", "-o"])
                .arg(&tar_path)
                .arg(url)
                .status();
            if matches!(status, Ok(s) if s.success()) {
                ok = true;
                break;
            }
            let _ = fs::remove_file(&tar_path);
        }
        if !ok {
            return Err(io::Error::other(
                "Failed to download SIFT-1M. Download manually:\n\
                 curl -O http://corpus-texmex.irisa.fr/sift.tar.gz\n\
                 tar xzf sift.tar.gz -C data/",
            ));
        }
    }

    eprintln!("Extracting SIFT-1M...");
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(&tar_path)
        .arg("-C")
        .arg(data_dir)
        .status()?;
    if !status.success() {
        return Err(io::Error::other("tar extract failed"));
    }
    Ok(())
}

fn download_siftsmall(data_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(data_dir)?;
    let tar_path = data_dir.join("siftsmall.tar.gz");

    if !tar_path.exists() {
        eprintln!("Downloading siftsmall dataset...");
        let urls = [
            "http://corpus-texmex.irisa.fr/siftsmall.tar.gz",
            "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        ];

        let mut ok = false;
        for url in &urls {
            eprintln!("  Trying: {}", url);
            let status = Command::new("curl")
                .args(["-fSL", "--connect-timeout", "30", "-o"])
                .arg(&tar_path)
                .arg(url)
                .status();
            if matches!(status, Ok(s) if s.success()) {
                ok = true;
                break;
            }
            let _ = fs::remove_file(&tar_path);
        }
        if !ok {
            return Err(io::Error::other(
                "Failed to download siftsmall. Download manually to data/siftsmall/",
            ));
        }
    }

    eprintln!("Extracting siftsmall...");
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(&tar_path)
        .arg("-C")
        .arg(data_dir)
        .status()?;
    if !status.success() {
        return Err(io::Error::other("tar extract failed"));
    }
    Ok(())
}

// ─── Metrics ──────────────────────────────────────────────────────

/// Recall@k: fraction of true top-k found in retrieved set.
pub fn recall_at_k(retrieved: &[usize], ground_truth: &[u32], k: usize) -> f32 {
    let k_actual = k.min(ground_truth.len()).min(retrieved.len());
    if k_actual == 0 {
        return 0.0;
    }
    let gt_set: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|&x| x as usize).collect();
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();
    hits as f32 / k_actual as f32
}

/// Mean Reciprocal Rank @k.
#[allow(dead_code)]
pub fn mrr_at_k(retrieved: &[usize], ground_truth: &[u32], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|&x| x as usize).collect();
    for (rank, id) in retrieved.iter().take(k).enumerate() {
        if gt_set.contains(id) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Brute-force ground truth using L2 distance.
#[allow(dead_code)]
pub fn compute_ground_truth_l2(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let d: f32 = b.iter().zip(q.iter()).map(|(a, q)| (a - q) * (a - q)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i as u32).collect()
        })
        .collect()
}

/// Format results as a Markdown table.
pub fn format_results_table(results: &[(String, f32, f32, f32, f64, String)]) -> String {
    let mut t = String::new();
    t.push_str("| Method | Recall@1 | Recall@10 | Recall@100 | QPS | Memory/vec |\n");
    t.push_str("|--------|----------|-----------|------------|-----|------------|\n");
    for (method, r1, r10, r100, qps, mem) in results {
        t.push_str(&format!(
            "| {} | {:.1}% | {:.1}% | {:.1}% | {:.0} | {} |\n",
            method,
            r1 * 100.0,
            r10 * 100.0,
            r100 * 100.0,
            qps,
            mem
        ));
    }
    t
}
