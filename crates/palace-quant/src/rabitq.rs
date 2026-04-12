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
        assert_eq!(input.len(), self.dim, "RaBitQ: input dimension mismatch");

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
    assert!(n.is_power_of_two(), "FHT requires power-of-2 length");

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
    /// ||o - centroid||²
    pub sqr_norm: f32,
    /// ||o - centroid||
    pub norm: f32,
    /// Quantization quality: ⟨x', x_bar⟩ / √D  where x' is the rotated
    /// normalized residual and x_bar = sign(x') ∈ {-1,+1}^D.
    pub x0: f32,
    /// Error bound factor for this vector
    pub error_bound: f32,
}

/// A RaBitQ-encoded vector: binary code + scalar correction factors.
#[derive(Clone, Debug)]
pub struct RaBitQCode {
    /// Packed binary codes. For multibit, stores bit-planes sequentially.
    /// 1-bit: D/64 u64s
    /// 4-bit: 4 * (D/64) u64s
    pub binary: Vec<u64>,
    /// Number of bits per dimension used for encoding
    pub bits: u8,
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

    /// Update the centroid of the index using a representative sample of vectors.
    pub fn update_centroid(&mut self, samples: &[Vec<f32>]) {
        if samples.is_empty() { return; }
        let mut new_centroid = vec![0.0; self.dim];
        for s in samples {
            for (i, v) in s.iter().enumerate() {
                new_centroid[i] += v;
            }
        }
        for i in 0..self.dim {
            new_centroid[i] /= samples.len() as f32;
        }
        self.centroid = new_centroid;
    }

    /// Create a new RaBitQ index with a precomputed centroid.
    pub fn with_centroid(dim: usize, centroid: Vec<f32>, seed: u64) -> Self {
        assert_eq!(centroid.len(), dim, "RaBitQ: centroid dimension mismatch");
        Self {
            rotation: FastRotation::new(dim, seed),
            centroid,
            dim,
        }
    }

    /// Update centroid (e.g., after computing mean of dataset).
    pub fn set_centroid(&mut self, centroid: Vec<f32>) {
        assert_eq!(centroid.len(), self.dim, "RaBitQ: centroid dimension mismatch");
        self.centroid = centroid;
    }

    /// Dimension of this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a data vector into RaBitQ binary code + factors (1-bit default).
    pub fn encode(&self, vector: &[f32]) -> RaBitQCode {
        self.encode_multibit(vector, 1)
    }

    /// Encode a data vector into RaBitQ multi-bit code + factors.
    ///
    /// # Arguments
    /// * `vector` - Data vector to encode
    /// * `bits` - Number of bits per dimension (1 or 4 supported)
    pub fn encode_multibit(&self, vector: &[f32], bits: u8) -> RaBitQCode {
        assert_eq!(vector.len(), self.dim, "RaBitQ: vector dimension mismatch");
        assert!(bits == 1 || bits == 4 || bits == 7, "RaBitQ: unsupported bit-depth {}, expected 1, 4, or 7", bits);

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

        // Step 4: Multi-bit quantization
        let mut binary = Vec::new();
        let words_per_plane = (self.dim + 63) / 64;

        if bits == 1 {
            binary = quantize_binary(&rotated);
        } else {
            // Encode into bit-planes
            // For 4-bit, we quantize each dimension to 0..15
            // To maintain RaBitQ properties, we should ideally use a randomized
            // or well-distributed quantization. For now, we'll use linear quant.
            let mut quantized = vec![0u8; self.dim];
            
            for i in 0..self.dim {
                // Centered quantization for RaBitQ
                // We map [-range, range] to [0, 15], centered at 0.0 mapping to 7.5
                let val = (rotated[i] * 10.0).clamp(-7.5, 7.5);
                quantized[i] = (val + 7.5).round() as u8;
            }

            // Extract bit-planes
            for b in 0..bits {
                let mut plane = vec![0u64; words_per_plane];
                for i in 0..self.dim {
                    let bit = (quantized[i] >> b) & 1;
                    if bit == 1 {
                        plane[i / 64] |= 1u64 << (i % 64);
                    }
                }
                binary.extend(plane);
            }
        }

        // Step 5: Compute x0 for distance correction
        // For multi-bit, we use the 1st bit-plane (sign bit) for the x0 correction
        // as per standard RaBitQ if we want to keep the same factor structure.
        let sign_bits = if bits == 1 {
            &binary[..words_per_plane]
        } else {
            // For multi-bit, the highest bit-plane is the "sign" if we use offset binary,
            // but here we just used linear quant. Let's use the most significant plane.
            let msb_plane_idx = (bits - 1) as usize * words_per_plane;
            &binary[msb_plane_idx..msb_plane_idx + words_per_plane]
        };

        // Compute quantization quality x0 = ⟨x', x_bar⟩ / √D
        let d = self.dim as f32;
        let mut dot_x_xbar: f32 = 0.0;
        for i in 0..self.dim {
            let bit = (sign_bits[i / 64] >> (i % 64)) & 1;
            let sign = if bit == 1 { 1.0f32 } else { -1.0f32 };
            dot_x_xbar += sign * rotated[i];
        }
        let x0 = dot_x_xbar / d.sqrt();
        // Clamp x0 away from zero (degenerate vectors)
        let x0 = if x0 > 1e-6 { x0 } else { 0.8 };

        // Error bound: ε ≈ 2·ε₀·||o-c||·√(1/x0² - 1) / √(D-1)
        let epsilon_0 = 1.9;
        let inv_x0_sq_minus_1 = (1.0 / (x0 * x0) - 1.0).max(0.0);
        let error_bound = 2.0 * epsilon_0 * norm
            * inv_x0_sq_minus_1.sqrt()
            / ((self.dim - 1).max(1) as f32).sqrt();

        RaBitQCode {
            binary,
            bits,
            factors: RaBitQFactors {
                sqr_norm,
                norm,
                x0,
                error_bound,
            },
        }
    }

    /// Prepare a query for distance estimation against encoded vectors.
    ///
    /// The query residual is rotated WITHOUT normalization — the magnitude
    /// must be preserved for correct asymmetric distance estimation.
    pub fn encode_query(&self, query: &[f32]) -> RaBitQQuery {
        assert_eq!(query.len(), self.dim, "RaBitQ: query dimension mismatch");

        let residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        let sqr_y: f32 = residual.iter().map(|x| x * x).sum();
        let q_norm = sqr_y.sqrt();

        // Rotate the raw residual — do NOT normalize
        let rotated_vector = self.rotation.rotate(&residual);

        RaBitQQuery {
            rotated_vector,
            sqr_y,
            q_norm,
        }
    }

    /// Estimate squared Euclidean distance between a query and an encoded vector.
    ///
    /// Uses the RaBitQ asymmetric distance estimation:
    ///
    /// ```text
    /// ⟨o-c, q-c⟩ ≈ ||o-c|| · (x0/√D) · ⟨x_bar, q'⟩
    /// ||o - q||²  ≈ ||o-c||² + ||q-c||² - 2·||o-c||·(x0/√D)·⟨x_bar, q'⟩
    /// ```
    ///
    /// where q' = P^T·(q-c) is the rotated RAW query residual (not normalized),
    /// and x_bar ∈ {-1,+1}^D is the sign-bit quantization of the rotated
    /// normalized database vector.
    ///
    /// Returns (estimated_distance, lower_bound_distance).
    #[inline]
    pub fn estimate_distance(&self, query: &RaBitQQuery, code: &RaBitQCode) -> (f32, f32) {
        let d = self.dim as f32;
        let words_per_plane = (self.dim + 63) / 64;

        // Select sign bits (MSB plane for multi-bit encoding)
        let msb_plane_idx = if code.bits == 1 {
            0
        } else {
            (code.bits - 1) as usize * words_per_plane
        };
        let sign_bits = &code.binary[msb_plane_idx..msb_plane_idx + words_per_plane];

        // Compute ⟨x_bar, q'⟩ — inner product of sign-quantized database
        // vector with the rotated raw query residual
        let mut x_dot_q: f32 = 0.0;
        for i in 0..self.dim {
            let bit = (sign_bits[i / 64] >> (i % 64)) & 1;
            let val = query.rotated_vector[i];
            if bit == 1 {
                x_dot_q += val;
            } else {
                x_dot_q -= val;
            }
        }

        // Estimate the real inner product ⟨o-c, q-c⟩:
        //   ⟨o-c, q-c⟩ = ||o-c|| · ⟨x_hat, q-c⟩
        //               = ||o-c|| · ⟨x', q'⟩              (rotation preserves IP)
        //
        // RaBitQ reconstruction (Gao & Long, 2024):
        //   ⟨x', q'⟩ ≈ (1 / (x0·√D)) · ⟨x_bar, q'⟩
        //
        // This gives an unbiased estimate for self-distance (exactly 0) and
        // preserves ranking fidelity for cross-vector distances.
        let est_inner_product =
            code.factors.norm / (code.factors.x0 * d.sqrt()) * x_dot_q;

        let est_dist = code.factors.sqr_norm + query.sqr_y - 2.0 * est_inner_product;
        let lower_bound = est_dist - code.factors.error_bound * query.q_norm;

        (est_dist.max(0.0), lower_bound)
    }
}


/// Prepared query for RaBitQ distance estimation.
#[derive(Clone, Debug)]
pub struct RaBitQQuery {
    /// Full rotated query vector (for high-precision asymmetric path)
    pub rotated_vector: Vec<f32>,
    /// ||query - centroid||^2
    pub sqr_y: f32,
    /// ||query - centroid||
    pub q_norm: f32,
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
    use crate::hamming::hamming_distance;

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

        // Factors should be finite and valid
        assert!(code.factors.sqr_norm.is_finite());
        assert!(code.factors.norm.is_finite());
        assert!(code.factors.x0.is_finite());
        assert!(code.factors.x0 > 0.0, "x0 should be positive");
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
