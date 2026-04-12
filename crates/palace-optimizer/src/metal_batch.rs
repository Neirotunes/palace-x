// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Metal GPU batch distance computation for HNSW layer-0 reranking.
//!
//! On Apple Silicon, GPU and CPU share the same physical memory (UMA), so
//! there is zero copy overhead when the GPU reads vector data that the CPU
//! already allocated. This module exploits that by dispatching thousands of
//! L2-distance computations to Metal compute shaders in a single GPU pass.
//!
//! ## Architecture
//!
//! ```text
//!   ┌─────────────┐      shared UMA buffer      ┌─────────────┐
//!   │ CPU (Rust)   │  ──── query + candidates ──▶ │ GPU (Metal)  │
//!   │              │  ◀─── distances[] ────────── │ L2 kernel    │
//!   └─────────────┘      zero-copy MTLBuffer     └─────────────┘
//! ```
//!
//! ## Kernel Design
//!
//! - Each threadgroup processes one candidate vector against the query.
//! - Threads within a threadgroup cooperatively reduce the L2 sum using
//!   SIMD shuffle + threadgroup shared memory for the final reduction.
//! - Threadgroup size = min(dims, max_threads), with a 256-thread cap
//!   to stay within Apple GPU limits.
//!
//! ## When to use
//!
//! GPU batch pays off when candidate count ≥ ~256 (amortizes dispatch
//! overhead). For smaller candidate sets, CPU NEON path is faster.
//! The `MetalBatchSearch` struct exposes a `should_use_gpu()` heuristic.

use metal::*;
use objc::rc::autoreleasepool;
use std::sync::atomic::{AtomicU64, Ordering};

/// Minimum candidate count to justify GPU dispatch overhead.
/// Below this, CPU NEON is faster due to dispatch + sync latency.
const GPU_THRESHOLD: usize = 256;

/// Maximum dimensions supported by the shader (compile-time threadgroup limit).
/// For dims > 1024, the shader falls back to a loop within each thread.
const MAX_THREADGROUP_DIM: u64 = 256;

// ──────────────────────────────────────────────────────────────────────
// Metal Shader Source
// ──────────────────────────────────────────────────────────────────────

const BATCH_L2_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

/// Batch squared-L2 distance: one threadgroup per candidate vector.
///
/// Layout:
///   buffer(0) = query        [dims floats]
///   buffer(1) = candidates   [num_candidates * dims floats, row-major]
///   buffer(2) = distances    [num_candidates floats, output]
///   buffer(3) = params       {dims: uint, num_candidates: uint}
///
/// Each threadgroup computes dist(query, candidates[group_id]).
/// Threads within the group each handle a slice of dimensions,
/// then reduce via threadgroup shared memory.
kernel void batch_l2_distance(
    device const float*  query        [[buffer(0)]],
    device const float*  candidates   [[buffer(1)]],
    device       float*  distances    [[buffer(2)]],
    constant     uint2&  params       [[buffer(3)]],   // (dims, num_candidates)
    uint                 cand_idx     [[threadgroup_position_in_grid]],
    uint                 tid          [[thread_index_in_threadgroup]],
    uint                 tg_size      [[threads_per_threadgroup]])
{
    uint dims           = params.x;
    uint num_candidates = params.y;

    if (cand_idx >= num_candidates) return;

    // Each thread accumulates partial sum over its slice of dimensions
    device const float* cand = candidates + cand_idx * dims;
    float partial_sum = 0.0f;
    for (uint d = tid; d < dims; d += tg_size) {
        float diff = query[d] - cand[d];
        partial_sum += diff * diff;
    }

    // Threadgroup reduction via SIMD shuffle (warp-level) + shared memory
    // Apple GPUs have SIMD width 32
    threadgroup float shared_sums[256];
    shared_sums[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        distances[cand_idx] = shared_sums[0];
    }
}

/// Batch cosine distance: 1 - (dot(a,b) / (|a|*|b|))
/// Same layout as batch_l2_distance.
kernel void batch_cosine_distance(
    device const float*  query        [[buffer(0)]],
    device const float*  candidates   [[buffer(1)]],
    device       float*  distances    [[buffer(2)]],
    constant     uint2&  params       [[buffer(3)]],
    uint                 cand_idx     [[threadgroup_position_in_grid]],
    uint                 tid          [[thread_index_in_threadgroup]],
    uint                 tg_size      [[threads_per_threadgroup]])
{
    uint dims           = params.x;
    uint num_candidates = params.y;

    if (cand_idx >= num_candidates) return;

    device const float* cand = candidates + cand_idx * dims;

    float dot_sum  = 0.0f;
    float norm_q   = 0.0f;
    float norm_c   = 0.0f;

    for (uint d = tid; d < dims; d += tg_size) {
        float q = query[d];
        float c = cand[d];
        dot_sum += q * c;
        norm_q  += q * q;
        norm_c  += c * c;
    }

    // Pack three partial sums into shared memory for reduction
    threadgroup float shared_dot[256];
    threadgroup float shared_nq[256];
    threadgroup float shared_nc[256];

    shared_dot[tid] = dot_sum;
    shared_nq[tid]  = norm_q;
    shared_nc[tid]  = norm_c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_nq[tid]  += shared_nq[tid + stride];
            shared_nc[tid]  += shared_nc[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float denom = sqrt(shared_nq[0] * shared_nc[0]);
        if (denom < 1e-10f) {
            distances[cand_idx] = 2.0f;  // max cosine distance for zero vectors
        } else {
            distances[cand_idx] = 1.0f - shared_dot[0] / denom;
        }
    }
}
";

// ──────────────────────────────────────────────────────────────────────
// Rust Wrapper
// ──────────────────────────────────────────────────────────────────────

/// Distance metric selector for Metal kernels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetalDistanceMetric {
    L2,
    Cosine,
}

/// GPU batch distance computation pipeline.
///
/// Holds the compiled Metal compute pipelines and a command queue.
/// Thread-safe: the Metal command queue serializes GPU submissions.
pub struct MetalBatchSearch {
    device: Device,
    command_queue: CommandQueue,
    l2_pipeline: ComputePipelineState,
    cosine_pipeline: ComputePipelineState,
    /// Total GPU dispatches (for stats)
    dispatch_count: AtomicU64,
    /// Total candidate vectors processed on GPU
    vectors_processed: AtomicU64,
}

impl MetalBatchSearch {
    /// Initialize Metal pipelines. Returns `None` if no Metal GPU is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(BATCH_L2_SHADER, &opts)
            .map_err(|e| eprintln!("Metal shader compilation failed: {}", e))
            .ok()?;

        let l2_fn = library.get_function("batch_l2_distance", None).ok()?;
        let cos_fn = library.get_function("batch_cosine_distance", None).ok()?;

        let l2_pipeline = device
            .new_compute_pipeline_state_with_function(&l2_fn)
            .ok()?;
        let cosine_pipeline = device
            .new_compute_pipeline_state_with_function(&cos_fn)
            .ok()?;

        Some(Self {
            device,
            command_queue,
            l2_pipeline,
            cosine_pipeline,
            dispatch_count: AtomicU64::new(0),
            vectors_processed: AtomicU64::new(0),
        })
    }

    /// Should we dispatch to GPU? True when candidate count ≥ threshold.
    #[inline]
    pub fn should_use_gpu(num_candidates: usize) -> bool {
        num_candidates >= GPU_THRESHOLD
    }

    /// GPU device name (e.g. "Apple M3 Max")
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Compute squared-L2 distances from `query` to each row in `candidates`.
    ///
    /// # Arguments
    /// - `query`: `[dims]` floats
    /// - `candidates`: `[num_candidates][dims]` row-major floats
    /// - `dims`: vector dimensionality
    /// - `metric`: L2 or Cosine
    ///
    /// # Returns
    /// `Vec<f32>` of length `num_candidates` — distance from query to each candidate.
    ///
    /// # Panics
    /// - If `query.len() != dims`
    /// - If `candidates.len() != num_candidates * dims`
    pub fn batch_distances(
        &self,
        query: &[f32],
        candidates: &[f32],
        dims: usize,
        metric: MetalDistanceMetric,
    ) -> Vec<f32> {
        assert_eq!(query.len(), dims, "query length must equal dims");
        assert_eq!(
            candidates.len() % dims,
            0,
            "candidates length must be multiple of dims"
        );
        let num_candidates = candidates.len() / dims;
        if num_candidates == 0 {
            return Vec::new();
        }

        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        self.vectors_processed
            .fetch_add(num_candidates as u64, Ordering::Relaxed);

        autoreleasepool(|| {
            // Create UMA shared buffers (zero-copy on Apple Silicon)
            let query_buf = self.device.new_buffer_with_data(
                query.as_ptr() as *const _,
                (dims * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cand_buf = self.device.new_buffer_with_data(
                candidates.as_ptr() as *const _,
                (candidates.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let dist_buf = self.device.new_buffer(
                (num_candidates * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Params: (dims, num_candidates) as uint2
            let params: [u32; 2] = [dims as u32, num_candidates as u32];
            let params_buf = self.device.new_buffer_with_data(
                params.as_ptr() as *const _,
                (2 * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let pipeline = match metric {
                MetalDistanceMetric::L2 => &self.l2_pipeline,
                MetalDistanceMetric::Cosine => &self.cosine_pipeline,
            };
            encoder.set_compute_pipeline_state(pipeline);

            encoder.set_buffer(0, Some(&query_buf), 0);
            encoder.set_buffer(1, Some(&cand_buf), 0);
            encoder.set_buffer(2, Some(&dist_buf), 0);
            encoder.set_buffer(3, Some(&params_buf), 0);

            // 1D dispatch: one threadgroup per candidate, threads cooperate on dims
            let tg_width = MAX_THREADGROUP_DIM.min(dims as u64);
            let threads_per_group = MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            };
            let num_threadgroups = MTLSize {
                width: num_candidates as u64,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read results back (zero-copy — just read the shared buffer)
            let result_ptr = dist_buf.contents() as *const f32;
            let mut distances = vec![0.0f32; num_candidates];
            unsafe {
                std::ptr::copy_nonoverlapping(result_ptr, distances.as_mut_ptr(), num_candidates);
            }
            distances
        })
    }

    /// Batch distances returning (index, distance) pairs sorted by distance ascending.
    ///
    /// Convenience method that pairs each distance with its candidate index
    /// and returns the top-k closest.
    pub fn batch_distances_topk(
        &self,
        query: &[f32],
        candidates: &[f32],
        dims: usize,
        metric: MetalDistanceMetric,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let distances = self.batch_distances(query, candidates, dims, metric);
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        // Partial sort for top-k (O(n) average via select_nth_unstable)
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);
        }
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
    }

    /// Statistics: (dispatch_count, vectors_processed)
    pub fn stats(&self) -> (u64, u64) {
        (
            self.dispatch_count.load(Ordering::Relaxed),
            self.vectors_processed.load(Ordering::Relaxed),
        )
    }

    /// Reset statistics counters
    pub fn reset_stats(&self) {
        self.dispatch_count.store(0, Ordering::Relaxed);
        self.vectors_processed.store(0, Ordering::Relaxed);
    }
}

// ──────────────────────────────────────────────────────────────────────
// GPU-accelerated HNSW layer-0 reranking
// ──────────────────────────────────────────────────────────────────────

/// Results from GPU-accelerated reranking
#[derive(Debug)]
pub struct GpuRerankResult {
    /// (original_candidate_index, node_id_placeholder, distance)
    pub ranked: Vec<(usize, f32)>,
    /// Whether GPU or CPU path was used
    pub used_gpu: bool,
}

/// Rerank a set of candidate vectors against a query using GPU batch compute.
///
/// Falls back to CPU scalar path if:
/// - Metal is not available (non-Apple hardware)
/// - Candidate count < GPU_THRESHOLD
///
/// # Arguments
/// - `gpu`: Optional Metal pipeline (pass `None` to force CPU)
/// - `query`: query vector
/// - `candidate_vecs`: flat row-major candidate vectors
/// - `dims`: vector dimensionality
/// - `metric`: L2 or Cosine
/// - `k`: number of results to return
pub fn gpu_rerank(
    gpu: Option<&MetalBatchSearch>,
    query: &[f32],
    candidate_vecs: &[f32],
    dims: usize,
    metric: MetalDistanceMetric,
    k: usize,
) -> GpuRerankResult {
    let num_candidates = candidate_vecs.len() / dims;

    // GPU path
    if let Some(gpu) = gpu {
        if MetalBatchSearch::should_use_gpu(num_candidates) {
            let ranked = gpu.batch_distances_topk(query, candidate_vecs, dims, metric, k);
            return GpuRerankResult {
                ranked,
                used_gpu: true,
            };
        }
    }

    // CPU fallback — scalar L2/cosine
    let mut indexed: Vec<(usize, f32)> = (0..num_candidates)
        .map(|i| {
            let cand = &candidate_vecs[i * dims..(i + 1) * dims];
            let dist = match metric {
                MetalDistanceMetric::L2 => {
                    query
                        .iter()
                        .zip(cand.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum()
                }
                MetalDistanceMetric::Cosine => {
                    let dot: f32 = query.iter().zip(cand.iter()).map(|(a, b)| a * b).sum();
                    let nq: f32 = query.iter().map(|a| a * a).sum();
                    let nc: f32 = cand.iter().map(|a| a * a).sum();
                    let denom = (nq * nc).sqrt();
                    if denom < 1e-10 {
                        2.0
                    } else {
                        1.0 - dot / denom
                    }
                }
            };
            (i, dist)
        })
        .collect();

    if k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
    }
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    GpuRerankResult {
        ranked: indexed,
        used_gpu: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dims: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut vecs = Vec::with_capacity(n * dims);
        for i in 0..n {
            for d in 0..dims {
                let mut hasher = DefaultHasher::new();
                (seed, i, d).hash(&mut hasher);
                let h = hasher.finish();
                // Pseudo-random float in [-1, 1]
                vecs.push((h as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0);
            }
        }
        vecs
    }

    #[test]
    fn test_cpu_fallback_l2() {
        let dims = 128;
        let n = 100;
        let query: Vec<f32> = vec![0.5; dims];
        let candidates = make_vectors(n, dims, 42);

        let result = gpu_rerank(None, &query, &candidates, dims, MetalDistanceMetric::L2, 10);
        assert!(!result.used_gpu);
        assert_eq!(result.ranked.len(), 10);

        // Verify distances are sorted ascending
        for w in result.ranked.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6, "Results not sorted: {} > {}", w[0].1, w[1].1);
        }
    }

    #[test]
    fn test_cpu_fallback_cosine() {
        let dims = 64;
        let n = 50;
        let query: Vec<f32> = (0..dims).map(|i| (i as f32).sin()).collect();
        let candidates = make_vectors(n, dims, 99);

        let result = gpu_rerank(None, &query, &candidates, dims, MetalDistanceMetric::Cosine, 5);
        assert!(!result.used_gpu);
        assert_eq!(result.ranked.len(), 5);

        // Cosine distances should be in [0, 2]
        for (_, d) in &result.ranked {
            assert!(*d >= -0.01 && *d <= 2.01, "Cosine distance out of range: {}", d);
        }
    }

    #[test]
    fn test_gpu_threshold() {
        assert!(!MetalBatchSearch::should_use_gpu(100));
        assert!(!MetalBatchSearch::should_use_gpu(255));
        assert!(MetalBatchSearch::should_use_gpu(256));
        assert!(MetalBatchSearch::should_use_gpu(10_000));
    }

    #[test]
    fn test_cpu_fallback_identical_query() {
        // Distance to self should be ~0
        let dims = 64;
        let query: Vec<f32> = vec![1.0; dims];
        // Put the query as the first candidate
        let mut candidates = query.clone();
        // Add some other vectors
        candidates.extend(make_vectors(10, dims, 7));

        let result = gpu_rerank(None, &query, &candidates, dims, MetalDistanceMetric::L2, 5);
        // First result should be index 0 with distance ~0
        assert_eq!(result.ranked[0].0, 0);
        assert!(result.ranked[0].1 < 1e-6, "Self-distance should be ~0, got {}", result.ranked[0].1);
    }

    // GPU tests — only run on Apple Silicon with Metal support.
    // `MetalBatchSearch::new()` returns None on other platforms.

    #[test]
    fn test_metal_init() {
        // This test just verifies the constructor doesn't panic.
        // On non-Metal platforms, it returns None gracefully.
        let _gpu = MetalBatchSearch::new();
    }

    #[test]
    fn test_gpu_l2_if_available() {
        let gpu = match MetalBatchSearch::new() {
            Some(g) => g,
            None => {
                eprintln!("Metal not available, skipping GPU test");
                return;
            }
        };

        let dims = 128;
        let n = 512; // Above GPU_THRESHOLD
        let query: Vec<f32> = vec![0.5; dims];
        let candidates = make_vectors(n, dims, 42);

        let gpu_dists = gpu.batch_distances(&query, &candidates, dims, MetalDistanceMetric::L2);
        assert_eq!(gpu_dists.len(), n);

        // Verify against CPU
        for i in 0..n {
            let cand = &candidates[i * dims..(i + 1) * dims];
            let cpu_dist: f32 = query.iter().zip(cand).map(|(a, b)| (a - b) * (a - b)).sum();
            let diff = (gpu_dists[i] - cpu_dist).abs();
            assert!(
                diff < 0.1, // Metal fp32 rounding can differ slightly
                "GPU/CPU mismatch at {}: gpu={}, cpu={}, diff={}",
                i, gpu_dists[i], cpu_dist, diff
            );
        }

        let (dispatches, processed) = gpu.stats();
        assert_eq!(dispatches, 1);
        assert_eq!(processed, n as u64);
    }

    #[test]
    fn test_gpu_topk_if_available() {
        let gpu = match MetalBatchSearch::new() {
            Some(g) => g,
            None => return,
        };

        let dims = 128;
        let n = 1000;
        let query: Vec<f32> = vec![0.0; dims];
        let candidates = make_vectors(n, dims, 123);

        let topk = gpu.batch_distances_topk(&query, &candidates, dims, MetalDistanceMetric::L2, 10);
        assert_eq!(topk.len(), 10);

        // Verify sorted
        for w in topk.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
    }
}
