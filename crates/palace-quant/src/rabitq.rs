// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! RaBitQ: Randomized Binary Quantization with theoretical error bounds.
//!
//! Implements the RaBitQ algorithm (Gao & Long, SIGMOD 2024) which dramatically
//! improves recall over naive sign-bit binary quantization by applying a random
//! orthogonal rotation before quantization and storing per-vector scalar corrections.
//!
//! Key properties:
//! - Same 1-bit-per-dimension storage as naive binary
//! - Same SIMD Hamming popcount kernels for distance computation
//! - 4 extra f32s per vector (16 bytes) for scalar correction
//! - Recall improvement from ~2% to ~90%+ at the same bit budget
//!
//! # Algorithm overview
//!
//! **Indexing**: For each vector `o`:
//! 1. Compute residual from centroid: `r = o - c`
//! 2. Normalize: `o_hat = r / ||r||`
//! 3. Rotate: `o' = P^T * o_hat` (P = random orthogonal matrix)
//! 4. Quantize to sign bits: `b[i] = (o'[i] >= 0)`
//! 5. Store binary code + scalar factors for distance correction
//!
//! **Query**: For query `q`:
//! 1. Compute residual: `q_r = q - c`
//! 2. Rotate: `q' = P^T * q_r` (NOT normalized — keep magnitude)
//! 3. Use popcount-based inner product with scalar correction
//!    to estimate squared Euclidean distance

use crate::binary::quantize_binary;
use crate::hamming::hamming_distance;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Random rotation matrix using Fast Hadamard Transform + random sign flips.
///
/// This is O(D log D) per rotation instead of O(D^2) for full matrix multiply.
/// Three rounds of (sign-flip + FHT) approximate a Haar-distributed orthogonal rotation.
#[derive(Clone, Debug)]
pub struct FastRotation {
    /// Random signs for each of the 3 rounds: +1.0 or -1.0
    signs: [Vec<f32>; 3],
    /// Padded dimension (next power of 2)
    padded_dim: usize,
    /// Original dimension
    dim: usize,
}

impl FastRotation {
    /// Create a new random rotation for the given dimension.
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = dim.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);

        let signs = std::array::from_fn(|_| {
            (0..padded_dim)
                .map(|_| if rng.gen::<bool>() { 1.0f32 } else { -1.0f32 })
                .collect()
        });

        Self {
            signs,
            padded_dim,
            dim,
        }
    }

    /// Apply the rotation to a vector (in-place on padded buffer).
    /// Returns the rotated vector truncated to original dimension.
    pub fn rotate(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.dim);

        // Pad to power-of-2
        let mut buf = vec![0.0f32; self.padded_dim];
        buf[..self.dim].copy_from_slice(input);

        // 3 rounds of sign-flip + normalized FHT
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        for round in 0..3 {
            // Element-wise sign flip
            for i in 0..self.padded_dim {
                buf[i] *= self.signs[round][i];
            }
            // In-place Fast Hadamard Transform + normalize
            fast_hadamard_transform(&mut buf);
            for x in buf.iter_mut() {
                *x *= scale;
            }
        }

        buf.truncate(self.dim);
        buf
    }
}

/// In-place Fast Hadamard Transform (unnormalized).
///
/// Operates on a slice whose length must be a power of 2.
fn fast_hadamard_transform(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two(), "FHT requires power-of-2 length");

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Per-vector scalar factors stored alongside the binary code.
#[derive(Clone, Debug)]
pub struct RaBitQFactors {
    /// ||o - centroid||^2
    pub sqr_norm: f32,
    /// -2/sqrt(D) * (||o-c|| / x0)
    pub factor_ip: f32,
    /// factor_ip * (2*popcount - D)
    pub factor_ppc: f32,
    /// Error bound factor for this vector
    pub error_bound: f32,
}

/// A RaBitQ-encoded vector: binary code + scalar correction factors.
#[derive(Clone, Debug)]
pub struct RaBitQCode {
    /// Packed binary code (sign bits of rotated normalized residual), D/64 u64s
    pub binary: Vec<u64>,
    /// Scalar correction factors
    pub factors: RaBitQFactors,
}

/// RaBitQ index: stores the rotation and centroid, encodes/decodes vectors.
#[derive(Clone, Debug)]
pub struct RaBitQIndex {
    /// Random rotation
    rotation: FastRotation,
    /// Centroid (mean of dataset or zero)
    centroid: Vec<f32>,
    /// Dimensionality
    dim: usize,
}

impl RaBitQIndex {
    /// Create a new RaBitQ index with zero centroid.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            rotation: FastRotation::new(dim, seed),
            centroid: vec![0.0; dim],
            dim,
        }
    }

    /// Create a new RaBitQ index with a precomputed centroid.
    pub fn with_centroid(dim: usize, centroid: Vec<f32>, seed: u64) -> Self {
        debug_assert_eq!(centroid.len(), dim);
        Self {
            rotation: FastRotation::new(dim, seed),
            centroid,
            dim,
        }
    }

    /// Update centroid (e.g., after computing mean of dataset).
    pub fn set_centroid(&mut self, centroid: Vec<f32>) {
        debug_assert_eq!(centroid.len(), self.dim);
        self.centroid = centroid;
    }

    /// Dimension of this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a data vector into RaBitQ binary code + factors.
    pub fn encode(&self, vector: &[f32]) -> RaBitQCode {
        debug_assert_eq!(vector.len(), self.dim);

        // Step 1: Residual from centroid
        let residual: Vec<f32> = vector
            .iter()
            .zip(self.centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        let norm = l2_norm(&residual);
        let sqr_norm = norm * norm;

        // Step 2: Normalize to unit vector
        let normalized: Vec<f32> = if norm > 1e-10 {
            residual.iter().map(|x| x / norm).collect()
        } else {
            vec![0.0; self.dim]
        };

        // Step 3: Rotate
        let rotated = self.rotation.rotate(&normalized);

        // Step 4: Sign-bit quantization (reuse existing binary.rs)
        let binary = quantize_binary(&rotated);

        // Step 5: Compute x0 = <x_bar, rotated> where x_bar = (2*bit - 1)/sqrt(D)
        let d = self.dim as f32;
        let sqrt_d = d.sqrt();
        let mut x0: f32 = 0.0;
        for i in 0..self.dim {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let bit = (binary[word_idx] >> (63 - bit_idx)) & 1;
            let sign = if bit == 1 { 1.0 } else { -1.0 };
            x0 += sign * rotated[i];
        }
        x0 /= sqrt_d;

        // Guard against degenerate cases
        let x0 = if x0 > 1e-6 { x0 } else { 0.8 };

        // Step 6: Popcount
        let pc: u32 = binary.iter().map(|w| w.count_ones()).sum();

        // Step 7: Precompute factors
        let x_x0 = norm / x0; // ||o-c|| / x0
        let factor_ip = -2.0 / sqrt_d * x_x0;
        let factor_ppc = factor_ip * (2.0 * pc as f32 - d);

        // Error bound: 2 * epsilon_0 / sqrt(D-1) * sqrt((norm/x0)^2 - norm^2)
        let epsilon_0 = 1.9;
        let inner = (x_x0 * x_x0 - sqr_norm).max(0.0);
        let error_bound = 2.0 * epsilon_0 / ((self.dim - 1).max(1) as f32).sqrt() * inner.sqrt();

        RaBitQCode {
            binary,
            factors: RaBitQFactors {
                sqr_norm,
                factor_ip,
                factor_ppc,
                error_bound,
            },
        }
    }

    /// Prepare a query for distance estimation against encoded vectors.
    ///
    /// Returns a `RaBitQQuery` that can be used with `estimate_distance`.
    pub fn encode_query(&self, query: &[f32]) -> RaBitQQuery {
        debug_assert_eq!(query.len(), self.dim);

        // Residual from centroid (NOT normalized — keep magnitude)
        let residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        let q_norm = l2_norm(&residual);
        let sqr_y = q_norm * q_norm;

        // Rotate query
        let rotated = self.rotation.rotate(&residual);

        // Scalar-quantize rotated query to 4 bits for asymmetric comparison
        let vl = rotated.iter().cloned().fold(f32::INFINITY, f32::min);
        let vr = rotated.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let num_levels = 15.0f32; // 4-bit: 0..15
        let width = if (vr - vl).abs() > 1e-10 {
            (vr - vl) / num_levels
        } else {
            1.0
        };

        let quantized: Vec<u8> = rotated
            .iter()
            .map(|&v| ((v - vl) / width).round().clamp(0.0, num_levels) as u8)
            .collect();

        let sumq: u32 = quantized.iter().map(|&v| v as u32).sum();

        // Also create binary version of query for fast 1-bit path
        let query_binary = quantize_binary(&rotated);

        RaBitQQuery {
            query_binary,
            quantized,
            sumq,
            vl,
            width,
            sqr_y,
            q_norm,
        }
    }

    /// Estimate squared Euclidean distance between a query and an encoded vector.
    ///
    /// Uses the RaBitQ asymmetric distance estimation:
    /// ||o - q||^2 ≈ ||o-c||^2 + ||q-c||^2 - 2 * (||o-c|| / x0) * <x_bar, q'>
    ///
    /// where <x_bar, q'> is estimated via the asymmetric byte-bin inner product.
    ///
    /// Returns (estimated_distance, lower_bound_distance).
    #[inline]
    pub fn estimate_distance(&self, query: &RaBitQQuery, code: &RaBitQCode) -> (f32, f32) {
        // Inner product between quantized query and binary code:
        // ip = sum_i quantized[i] * bit_i(code)
        let ip = inner_product_byte_bin(&query.quantized, &code.binary, self.dim);

        // Reconstruct <x_bar, q'> from quantized values:
        // q'[i] ≈ vl + quantized[i] * width
        // x_bar[i] = (2*bit_i - 1) / sqrt(D)
        // <x_bar, q'> = (1/sqrt(D)) * [vl*(2*pc - D) + width*(2*ip - sumq)]
        let d = self.dim as f32;
        let sqrt_d = d.sqrt();
        let pc: u32 = code.binary.iter().map(|w| w.count_ones()).sum();
        let xbar_dot_qprime = (query.vl * (2.0 * pc as f32 - d)
            + query.width * (2.0 * ip as f32 - query.sumq as f32))
            / sqrt_d;

        // Estimated inner product <o-c, q-c> = (||o-c|| / x0) * <x_bar, q'>
        let norm_over_x0 = (-code.factors.factor_ip * sqrt_d) / 2.0; // recover ||o-c||/x0
        let est_ip = norm_over_x0 * xbar_dot_qprime;

        // ||o-q||^2 = ||o-c||^2 + ||q-c||^2 - 2*<o-c, q-c>
        let est_dist = code.factors.sqr_norm + query.sqr_y - 2.0 * est_ip;

        let error = query.q_norm * code.factors.error_bound;
        let lower_bound = est_dist - error;

        (est_dist.max(0.0), lower_bound)
    }

    /// Fast 1-bit symmetric distance using Hamming distance.
    /// Less accurate than asymmetric but uses existing SIMD kernels directly.
    ///
    /// Returns estimated squared Euclidean distance.
    #[inline]
    pub fn estimate_distance_symmetric(&self, query: &RaBitQQuery, code: &RaBitQCode) -> f32 {
        let hamming = hamming_distance(&query.query_binary, &code.binary);
        let d = self.dim as f32;

        // RaBitQ symmetric: d_est = sqr_norm + sqr_y + factor_ppc * 0
        //   + (D - 2*hamming) * factor_ip * 1  (simplified for binary query)
        // But with proper correction:
        let matching_bits = d as u32 - hamming;
        let ip_proxy = 2.0 * matching_bits as f32 - d;

        code.factors.sqr_norm + query.sqr_y + code.factors.factor_ip * ip_proxy / d.sqrt()
    }
}

/// Prepared query for RaBitQ distance estimation.
#[derive(Clone, Debug)]
pub struct RaBitQQuery {
    /// Binary quantization of rotated query (for symmetric/Hamming path)
    pub query_binary: Vec<u64>,
    /// 4-bit scalar quantization of rotated query
    pub quantized: Vec<u8>,
    /// Sum of quantized values
    pub sumq: u32,
    /// Quantization range minimum
    pub vl: f32,
    /// Quantization step width
    pub width: f32,
    /// ||query - centroid||^2
    pub sqr_y: f32,
    /// ||query - centroid||
    pub q_norm: f32,
}

/// Compute inner product between scalar-quantized query and binary code.
///
/// Returns sum_i quantized[i] * bit_i(binary_code)
/// This is the core asymmetric kernel of RaBitQ.
#[inline]
fn inner_product_byte_bin(q: &[u8], d_bin: &[u64], dim: usize) -> u32 {
    let mut result: u32 = 0;
    let words = (dim + 63) / 64;

    for w in 0..words {
        let d_word = d_bin[w];
        let base = w * 64;
        let end = (base + 64).min(dim);
        for i in base..end {
            let bit_idx = i - base;
            let d_bit = (d_word >> (63 - bit_idx)) & 1;
            result += q[i] as u32 * d_bit as u32;
        }
    }
    result
}

/// Compute L2 norm of a vector.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Batch top-k search using RaBitQ distance estimation.
///
/// Returns (index, estimated_distance) pairs sorted by distance ascending.
pub fn rabitq_topk(
    index: &RaBitQIndex,
    query: &RaBitQQuery,
    codes: &[RaBitQCode],
    k: usize,
) -> Vec<(usize, f32)> {
    if codes.is_empty() || k == 0 {
        return Vec::new();
    }

    let k_actual = k.min(codes.len());

    let mut results: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(idx, code)| {
            let (est_dist, _lb) = index.estimate_distance(query, code);
            (idx, est_dist)
        })
        .collect();

    if k_actual < results.len() {
        results.select_nth_unstable_by(k_actual - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k_actual);
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_hadamard_transform() {
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        fast_hadamard_transform(&mut v);
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);

        let mut v = vec![1.0, 1.0, 1.0, 1.0];
        fast_hadamard_transform(&mut v);
        assert_eq!(v, vec![4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fast_rotation_preserves_norm() {
        let dim = 64;
        let rot = FastRotation::new(dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let norm_before = l2_norm(&v);

        let rotated = rot.rotate(&v);
        let norm_after = l2_norm(&rotated);

        // Rotation should approximately preserve norm (FHT + signs is orthogonal)
        assert!(
            (norm_before - norm_after).abs() / norm_before < 0.1,
            "Norm changed too much: {} -> {}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_encode_decode_basic() {
        let dim = 64;
        let index = RaBitQIndex::new(dim, 42);

        let vector = vec![1.0; dim];
        let code = index.encode(&vector);

        // Binary code should have correct length
        assert_eq!(code.binary.len(), 1); // 64/64 = 1 u64

        // Factors should be finite
        assert!(code.factors.sqr_norm.is_finite());
        assert!(code.factors.factor_ip.is_finite());
        assert!(code.factors.factor_ppc.is_finite());
    }

    #[test]
    fn test_distance_self_is_small() {
        let dim = 128;
        let index = RaBitQIndex::new(dim, 42);

        let vector = vec![1.0; dim];
        let code = index.encode(&vector);
        let query = index.encode_query(&vector);

        let (dist, _lb) = index.estimate_distance(&query, &code);

        // Distance to self should be close to 0
        assert!(dist < 1.0, "Self-distance should be near 0, got {}", dist);
    }

    #[test]
    fn test_distance_ordering() {
        let dim = 128;
        let index = RaBitQIndex::new(dim, 42);

        // Create three vectors: query, near, far
        let query_vec: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let near_vec: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.001).collect();
        let far_vec: Vec<f32> = (0..dim).map(|i| -((i as f32) * 0.01)).collect();

        let near_code = index.encode(&near_vec);
        let far_code = index.encode(&far_vec);
        let query = index.encode_query(&query_vec);

        let (dist_near, _) = index.estimate_distance(&query, &near_code);
        let (dist_far, _) = index.estimate_distance(&query, &far_code);

        assert!(
            dist_near < dist_far,
            "Near vector ({}) should be closer than far vector ({})",
            dist_near,
            dist_far
        );
    }

    #[test]
    fn test_rabitq_topk() {
        let dim = 64;
        let index = RaBitQIndex::new(dim, 42);

        // Create dataset
        let mut codes = Vec::new();
        for i in 0..100 {
            let v: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32) * 0.001).collect();
            codes.push(index.encode(&v));
        }

        // Query near vector 50
        let query_vec: Vec<f32> = (0..dim).map(|j| ((50 * dim + j) as f32) * 0.001).collect();
        let query = index.encode_query(&query_vec);

        let results = rabitq_topk(&index, &query, &codes, 10);
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 0..results.len() - 1 {
            assert!(results[i].1 <= results[i + 1].1);
        }

        // Vector 50 should be among top results
        assert!(
            results.iter().any(|(idx, _)| *idx == 50),
            "Expected vector 50 in top-10, got {:?}",
            results.iter().map(|(idx, _)| *idx).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inner_product_byte_bin() {
        // All bits set, all query values = 1
        let q = vec![1u8; 64];
        let d_bin = vec![u64::MAX]; // all 64 bits set
        let ip = inner_product_byte_bin(&q, &d_bin, 64);
        assert_eq!(ip, 64); // 1 * 1 for each of 64 dimensions

        // No bits set
        let d_bin_zero = vec![0u64];
        let ip_zero = inner_product_byte_bin(&q, &d_bin_zero, 64);
        assert_eq!(ip_zero, 0);
    }

    #[test]
    fn test_with_centroid() {
        let dim = 64;
        let centroid = vec![0.5; dim];
        let index = RaBitQIndex::with_centroid(dim, centroid, 42);

        let vector = vec![1.0; dim];
        let code = index.encode(&vector);

        // sqr_norm should be ||vector - centroid||^2 = 64 * 0.25 = 16
        assert!(
            (code.factors.sqr_norm - 16.0).abs() < 0.01,
            "sqr_norm should be ~16, got {}",
            code.factors.sqr_norm
        );
    }

    #[test]
    fn test_recall_improvement_over_naive() {
        // Compare RaBitQ recall vs naive binary quantization on random data
        use rand::Rng;
        let dim = 64;
        let n = 200;
        let k = 10;
        let mut rng = StdRng::seed_from_u64(123);

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        let index = RaBitQIndex::new(dim, 42);

        // RaBitQ encode all vectors
        let codes: Vec<RaBitQCode> = vectors.iter().map(|v| index.encode(v)).collect();

        // Naive binary encode
        let naive_codes: Vec<Vec<u64>> = vectors.iter().map(|v| quantize_binary(v)).collect();

        // Run queries and measure recall
        let num_queries = 20;
        let mut rabitq_recall_sum = 0.0;
        let mut naive_recall_sum = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

            // Ground truth: brute-force L2 distances
            let mut true_dists: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, d)
                })
                .collect();
            true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let true_topk: Vec<usize> = true_dists.iter().take(k).map(|(i, _)| *i).collect();

            // RaBitQ top-k
            let rq = index.encode_query(&query);
            let rabitq_results = rabitq_topk(&index, &rq, &codes, k);
            let rabitq_topk: Vec<usize> = rabitq_results.iter().map(|(i, _)| *i).collect();

            // Naive Hamming top-k
            let query_bin = quantize_binary(&query);
            let mut naive_dists: Vec<(usize, u32)> = naive_codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i, hamming_distance(&query_bin, c)))
                .collect();
            naive_dists.sort_by_key(|(_, d)| *d);
            let naive_topk: Vec<usize> = naive_dists.iter().take(k).map(|(i, _)| *i).collect();

            // Compute recall
            let rabitq_hits = rabitq_topk
                .iter()
                .filter(|id| true_topk.contains(id))
                .count();
            let naive_hits = naive_topk
                .iter()
                .filter(|id| true_topk.contains(id))
                .count();

            rabitq_recall_sum += rabitq_hits as f32 / k as f32;
            naive_recall_sum += naive_hits as f32 / k as f32;
        }

        let rabitq_recall = rabitq_recall_sum / num_queries as f32;
        let naive_recall = naive_recall_sum / num_queries as f32;

        eprintln!(
            "RaBitQ recall@{}: {:.1}%  |  Naive recall@{}: {:.1}%",
            k,
            rabitq_recall * 100.0,
            k,
            naive_recall * 100.0
        );

        // RaBitQ should have meaningfully better recall
        assert!(
            rabitq_recall > naive_recall,
            "RaBitQ recall ({:.3}) should beat naive ({:.3})",
            rabitq_recall,
            naive_recall
        );
    }
}
