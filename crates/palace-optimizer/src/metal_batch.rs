// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Conservative default — used only if calibration hasn't run or is disabled.
/// Empirically the real crossover on M-series is far higher (8K–64K range),
/// depending on chip generation and candidate dimensionality.
const GPU_THRESHOLD: usize = 256;

/// Upper bound for the auto-calibrated threshold — we never disable GPU
/// entirely even if no crossover is found in the probed range.
const CALIBRATION_MAX_THRESHOLD: usize = 131_072;

/// Maximum dimensions supported by the shader (compile-time threadgroup limit).
/// For dims > 1024, the shader falls back to a loop within each thread.
const MAX_THREADGROUP_DIM: u64 = 256;

/// Initial pre-allocation: 4096 candidates × 384 dims × 4 bytes ≈ 6 MB.
/// Grows on demand if exceeded.
const PREALLOC_CANDIDATES: usize = 4096;
const PREALLOC_DIMS: usize = 384;

// ──────────────────────────────────────────────────────────────────────
// Metal Shader Source
// ──────────────────────────────────────────────────────────────────────

/// Threadgroup memory budget (Apple Silicon: 32 KB per threadgroup)
///
/// ## L2 kernel
///   - query cache:  MAX_THREADGROUP_DIM × 4 = 1024 B  (query loaded once)
///   - SIMD partial: 8 × 4                   =   32 B  (max 8 SIMD groups)
///   - Total:                                   1056 B  (3.2% of 32 KB)
///
/// ## Cosine kernel
///   - query cache:  MAX_THREADGROUP_DIM × 4 = 1024 B
///   - SIMD partial: 8 × 4 × 3              =   96 B  (dot, nq, nc)
///   - Total:                                   1120 B  (3.4% of 32 KB)
///
/// Headroom: ~30 KB free for future multi-candidate tiling.
const BATCH_L2_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

// ── helpers ──────────────────────────────────────────────────────────

/// Two-stage reduction: simd_sum (no barrier) → cross-SIMD via shared mem
/// (one barrier).  Replaces the old log2(N)-barrier tree.
///
/// Apple GPU SIMD width = 32.  For tg_size=256 that is 8 SIMD groups,
/// so shared[] only needs 8 entries (32 B).
inline float fast_reduce(
    float                val,
    threadgroup float*   shared,    // [8] – one slot per SIMD group
    uint                 tid,
    uint                 tg_size)
{
    // Stage 1: intra-SIMD (hardware shuffle, zero barriers)
    float warp_sum = simd_sum(val);

    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    uint num_groups = (tg_size + 31) / 32;

    // Stage 2: one representative per SIMD group writes to shared
    if (simd_lane == 0) {
        shared[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // single barrier

    // Stage 3: first SIMD group reduces across groups
    float total = 0.0f;
    if (tid < num_groups) {
        total = shared[tid];
    }
    total = simd_sum(total);   // num_groups ≤ 8, fits in one SIMD
    return total;
}

// ── L2 kernel ────────────────────────────────────────────────────────

/// Batch squared-L2 distance — one threadgroup per candidate.
///
/// Optimisations vs v0.2:
///  1. Query cached in threadgroup memory (read once from device, not N times)
///  2. simd_sum + 1 barrier replaces 8-barrier tree reduction
///  3. shared[] is 8 floats (32 B) instead of 256 (1 KB)
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

    // ── cache query in threadgroup memory ──
    threadgroup float tg_query[256];   // 1 KB – covers dims ≤ 256
    for (uint d = tid; d < dims; d += tg_size) {
        tg_query[d] = query[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── per-thread partial L2 ──
    device const float* cand = candidates + cand_idx * dims;
    float partial_sum = 0.0f;
    for (uint d = tid; d < dims; d += tg_size) {
        float diff = tg_query[d] - cand[d];
        partial_sum += diff * diff;
    }

    // ── fast two-stage reduction ──
    threadgroup float simd_scratch[8];
    float total = fast_reduce(partial_sum, simd_scratch, tid, tg_size);

    if (tid == 0) {
        distances[cand_idx] = total;
    }
}

/// Batch cosine distance: 1 - (dot(a,b) / (|a|·|b|))
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

    // ── cache query ──
    threadgroup float tg_query[256];
    for (uint d = tid; d < dims; d += tg_size) {
        tg_query[d] = query[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── per-thread partial sums ──
    device const float* cand = candidates + cand_idx * dims;
    float dot_sum = 0.0f;
    float norm_q  = 0.0f;
    float norm_c  = 0.0f;

    for (uint d = tid; d < dims; d += tg_size) {
        float q = tg_query[d];
        float c = cand[d];
        dot_sum += q * c;
        norm_q  += q * q;
        norm_c  += c * c;
    }

    // ── three parallel fast reductions ──
    threadgroup float scratch_dot[8];
    threadgroup float scratch_nq[8];
    threadgroup float scratch_nc[8];

    float total_dot = fast_reduce(dot_sum, scratch_dot, tid, tg_size);
    float total_nq  = fast_reduce(norm_q,  scratch_nq,  tid, tg_size);
    float total_nc  = fast_reduce(norm_c,  scratch_nc,  tid, tg_size);

    if (tid == 0) {
        float denom = sqrt(total_nq * total_nc);
        if (denom < 1e-10f) {
            distances[cand_idx] = 2.0f;
        } else {
            distances[cand_idx] = 1.0f - total_dot / denom;
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

/// Pre-allocated Metal buffers, reused across dispatch calls.
/// Avoids ObjC runtime overhead of per-call buffer creation.
struct ReusableBuffers {
    query_buf: Buffer,
    cand_buf: Buffer,
    dist_buf: Buffer,
    params_buf: Buffer,
    /// Current capacity in bytes for candidates buffer
    cand_capacity: usize,
    /// Current capacity in floats for distance output
    dist_capacity: usize,
    /// Current capacity in bytes for query buffer
    query_capacity: usize,
}

/// GPU batch distance computation pipeline.
///
/// Holds the compiled Metal compute pipelines, a command queue,
/// and pre-allocated reusable buffers.  Thread-safe: the Mutex
/// serializes buffer access (Metal command queue already serializes
/// GPU work, so contention is minimal).
pub struct MetalBatchSearch {
    device: Device,
    command_queue: CommandQueue,
    l2_pipeline: ComputePipelineState,
    cosine_pipeline: ComputePipelineState,
    /// Pre-allocated shared-mode buffers (protected by Mutex)
    buffers: Mutex<ReusableBuffers>,
    /// Total GPU dispatches (for stats)
    dispatch_count: AtomicU64,
    /// Total candidate vectors processed on GPU
    vectors_processed: AtomicU64,
    /// Empirically measured crossover point (set by `calibrate()`).
    /// 0 = not calibrated, fall back to static `GPU_THRESHOLD`.
    calibrated_threshold: AtomicUsize,
}

impl MetalBatchSearch {
    /// Initialize Metal pipelines and pre-allocate buffers.
    /// Returns `None` if no Metal GPU is available.
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

        let query_bytes = PREALLOC_DIMS * std::mem::size_of::<f32>();
        let cand_bytes = PREALLOC_CANDIDATES * PREALLOC_DIMS * std::mem::size_of::<f32>();
        let dist_bytes = PREALLOC_CANDIDATES * std::mem::size_of::<f32>();
        let params_bytes = 2 * std::mem::size_of::<u32>();

        let buffers = ReusableBuffers {
            query_buf: device.new_buffer(query_bytes as u64, MTLResourceOptions::StorageModeShared),
            cand_buf: device.new_buffer(cand_bytes as u64, MTLResourceOptions::StorageModeShared),
            dist_buf: device.new_buffer(dist_bytes as u64, MTLResourceOptions::StorageModeShared),
            params_buf: device
                .new_buffer(params_bytes as u64, MTLResourceOptions::StorageModeShared),
            cand_capacity: cand_bytes,
            dist_capacity: PREALLOC_CANDIDATES,
            query_capacity: query_bytes,
        };

        Some(Self {
            device,
            command_queue,
            l2_pipeline,
            cosine_pipeline,
            buffers: Mutex::new(buffers),
            dispatch_count: AtomicU64::new(0),
            vectors_processed: AtomicU64::new(0),
            calibrated_threshold: AtomicUsize::new(0),
        })
    }

    /// Static default — kept for backward compatibility and for call sites
    /// that don't hold a `MetalBatchSearch` handle.  Prefer the instance
    /// method `should_use_gpu_for()` which consults the calibrated threshold.
    #[inline]
    pub fn should_use_gpu(num_candidates: usize) -> bool {
        num_candidates >= GPU_THRESHOLD
    }

    /// Instance variant that consults the calibrated threshold when available.
    #[inline]
    pub fn should_use_gpu_for(&self, num_candidates: usize) -> bool {
        let t = self.calibrated_threshold.load(Ordering::Relaxed);
        let threshold = if t == 0 { GPU_THRESHOLD } else { t };
        num_candidates >= threshold
    }

    /// Currently active threshold (calibrated or default).
    pub fn active_threshold(&self) -> usize {
        let t = self.calibrated_threshold.load(Ordering::Relaxed);
        if t == 0 {
            GPU_THRESHOLD
        } else {
            t
        }
    }

    /// Empirically find the GPU vs CPU crossover for this device + `dims`.
    ///
    /// Runs a short warm-up then times both paths across a log-spaced set of
    /// candidate counts, picks the smallest `n` where GPU beats CPU by at
    /// least 10 %, and stores it as the active threshold.
    ///
    /// If no crossover is found up to `CALIBRATION_MAX_THRESHOLD`, sets the
    /// threshold to that maximum — the GPU path will effectively be disabled
    /// for realistic rerank sizes, which is the correct conservative choice.
    ///
    /// Cost: ~50 ms on M-series.  Call once at startup (or first use).
    pub fn calibrate(&self, dims: usize) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Deterministic pseudo-random data — no external deps.
        let make_vec = |seed: u64, n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let mut h = DefaultHasher::new();
                (seed, i).hash(&mut h);
                v.push((h.finish() as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0);
            }
            v
        };

        let query = make_vec(0xC0FFEE, dims);
        let max_cands = 65_536usize;
        let cands = make_vec(0xDEADBEEF, max_cands * dims);

        // Warm-up: first dispatch pays JIT / driver-cache cost.
        let _ = self.batch_distances(&query, &cands[..1024 * dims], dims, MetalDistanceMetric::L2);

        let sizes = [256usize, 1024, 4096, 16_384, 65_536];
        let reps = 5;
        let mut crossover: Option<usize> = None;

        for &n in &sizes {
            let cand_slice = &cands[..n * dims];

            // GPU timing
            let t0 = Instant::now();
            for _ in 0..reps {
                let _ = self.batch_distances(&query, cand_slice, dims, MetalDistanceMetric::L2);
            }
            let gpu_us = t0.elapsed().as_micros() as f64 / reps as f64;

            // CPU scalar timing (same work the fallback path does)
            let t0 = Instant::now();
            for _ in 0..reps {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let c = &cand_slice[i * dims..(i + 1) * dims];
                    out[i] = query.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                }
                std::hint::black_box(out);
            }
            let cpu_us = t0.elapsed().as_micros() as f64 / reps as f64;

            if gpu_us * 1.10 < cpu_us && crossover.is_none() {
                crossover = Some(n);
            }
        }

        let chosen = crossover.unwrap_or(CALIBRATION_MAX_THRESHOLD);
        self.calibrated_threshold.store(chosen, Ordering::Relaxed);
        eprintln!(
            "[metal_batch] calibrated GPU threshold: {} candidates (dims={})",
            chosen, dims
        );
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

        let query_bytes = dims * std::mem::size_of::<f32>();
        let cand_bytes = candidates.len() * std::mem::size_of::<f32>();
        let dist_bytes = num_candidates * std::mem::size_of::<f32>();

        let mut bufs = self.buffers.lock().unwrap();

        // Grow pre-allocated buffers if this call exceeds capacity.
        // Growth is rare after warm-up — typically 0-1 reallocs per session.
        if query_bytes > bufs.query_capacity {
            bufs.query_buf = self
                .device
                .new_buffer(query_bytes as u64, MTLResourceOptions::StorageModeShared);
            bufs.query_capacity = query_bytes;
        }
        if cand_bytes > bufs.cand_capacity {
            bufs.cand_buf = self
                .device
                .new_buffer(cand_bytes as u64, MTLResourceOptions::StorageModeShared);
            bufs.cand_capacity = cand_bytes;
        }
        if num_candidates > bufs.dist_capacity {
            bufs.dist_buf = self
                .device
                .new_buffer(dist_bytes as u64, MTLResourceOptions::StorageModeShared);
            bufs.dist_capacity = num_candidates;
        }

        // Write data directly into pre-allocated shared buffers.
        // On UMA this is a simple memcpy within the same physical RAM —
        // no bus transfer, no ObjC buffer object allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                bufs.query_buf.contents() as *mut f32,
                dims,
            );
            std::ptr::copy_nonoverlapping(
                candidates.as_ptr(),
                bufs.cand_buf.contents() as *mut f32,
                candidates.len(),
            );
            let params: [u32; 2] = [dims as u32, num_candidates as u32];
            std::ptr::copy_nonoverlapping(
                params.as_ptr(),
                bufs.params_buf.contents() as *mut u32,
                2,
            );
        }

        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let pipeline = match metric {
                MetalDistanceMetric::L2 => &self.l2_pipeline,
                MetalDistanceMetric::Cosine => &self.cosine_pipeline,
            };
            encoder.set_compute_pipeline_state(pipeline);

            encoder.set_buffer(0, Some(&bufs.query_buf), 0);
            encoder.set_buffer(1, Some(&bufs.cand_buf), 0);
            encoder.set_buffer(2, Some(&bufs.dist_buf), 0);
            encoder.set_buffer(3, Some(&bufs.params_buf), 0);

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

            // Read results from the shared buffer (same physical RAM, no copy overhead)
            let result_ptr = bufs.dist_buf.contents() as *const f32;
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

    // GPU path — prefer calibrated threshold if available
    if let Some(gpu) = gpu {
        if gpu.should_use_gpu_for(num_candidates) {
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
                MetalDistanceMetric::L2 => query
                    .iter()
                    .zip(cand.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum(),
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
            assert!(
                w[0].1 <= w[1].1 + 1e-6,
                "Results not sorted: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_cpu_fallback_cosine() {
        let dims = 64;
        let n = 50;
        let query: Vec<f32> = (0..dims).map(|i| (i as f32).sin()).collect();
        let candidates = make_vectors(n, dims, 99);

        let result = gpu_rerank(
            None,
            &query,
            &candidates,
            dims,
            MetalDistanceMetric::Cosine,
            5,
        );
        assert!(!result.used_gpu);
        assert_eq!(result.ranked.len(), 5);

        // Cosine distances should be in [0, 2]
        for (_, d) in &result.ranked {
            assert!(
                *d >= -0.01 && *d <= 2.01,
                "Cosine distance out of range: {}",
                d
            );
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
        assert!(
            result.ranked[0].1 < 1e-6,
            "Self-distance should be ~0, got {}",
            result.ranked[0].1
        );
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
                i,
                gpu_dists[i],
                cpu_dist,
                diff
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
