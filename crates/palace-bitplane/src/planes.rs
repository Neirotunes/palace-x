// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! Bit-plane decomposition for f32 vectors.
//!
//! IEEE 754 f32 layout: [sign (1 bit) | exponent (8 bits) | mantissa (23 bits)]
//!
//! This module decomposes vectors into independent bit planes for tiered storage:
//! - Sign plane: coarse-grained magnitude sign
//! - Exponent planes: magnitude/scale information (8 planes)
//! - Mantissa planes: fine-grained precision (23 planes, optional loading)

use std::fmt;

/// Represents a vector decomposed into bit planes.
///
/// Sign + Exponent planes = coarse representation (stored in RAM)
/// Mantissa planes = fine precision (loaded on demand)
#[derive(Clone)]
pub struct BitPlaneVector {
    pub dimensions: usize,
    pub sign_plane: Vec<u8>,           // Packed: 1 bit per dimension
    pub exponent_planes: Vec<Vec<u8>>, // 8 planes, each ceil(dimensions/8) bytes
    pub mantissa_planes: Vec<Vec<u8>>, // 23 planes, each ceil(dimensions/8) bytes
}

impl fmt::Debug for BitPlaneVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitPlaneVector")
            .field("dimensions", &self.dimensions)
            .field("sign_plane_bytes", &self.sign_plane.len())
            .field("exponent_planes", &self.exponent_planes.len())
            .field("mantissa_planes", &self.mantissa_planes.len())
            .finish()
    }
}

impl BitPlaneVector {
    const NUM_EXPONENT_BITS: usize = 8;
    const NUM_MANTISSA_BITS: usize = 23;
    const TOTAL_PLANES: usize = 1 + 8 + 23; // sign + exponent + mantissa

    /// Decompose an f32 vector into bit planes.
    pub fn from_f32(vector: &[f32]) -> Self {
        let dimensions = vector.len();
        let bytes_per_plane = dimensions.div_ceil(8);

        let mut sign_plane = vec![0u8; bytes_per_plane];
        let mut exponent_planes = vec![vec![0u8; bytes_per_plane]; Self::NUM_EXPONENT_BITS];
        let mut mantissa_planes = vec![vec![0u8; bytes_per_plane]; Self::NUM_MANTISSA_BITS];

        for (dim, &value) in vector.iter().enumerate() {
            let bits = value.to_bits();

            // Extract and store sign bit (bit 31)
            let sign_bit = (bits >> 31) & 1;
            let byte_idx = dim / 8;
            let bit_idx = dim % 8;
            if sign_bit != 0 {
                sign_plane[byte_idx] |= 1 << bit_idx;
            }

            // Extract and store exponent bits (bits 30-23)
            #[allow(clippy::needless_range_loop)]
            for exp_plane in 0..Self::NUM_EXPONENT_BITS {
                let exp_bit = (bits >> (23 + exp_plane)) & 1;
                if exp_bit != 0 {
                    exponent_planes[exp_plane][byte_idx] |= 1 << bit_idx;
                }
            }

            // Extract and store mantissa bits (bits 22-0)
            #[allow(clippy::needless_range_loop)]
            for mant_plane in 0..Self::NUM_MANTISSA_BITS {
                let mant_bit = (bits >> mant_plane) & 1;
                if mant_bit != 0 {
                    mantissa_planes[mant_plane][byte_idx] |= 1 << bit_idx;
                }
            }
        }

        BitPlaneVector {
            dimensions,
            sign_plane,
            exponent_planes,
            mantissa_planes,
        }
    }

    /// Reconstruct full f32 vector from all planes.
    pub fn reconstruct_full(&self) -> Vec<f32> {
        self.reconstruct_partial(Self::NUM_MANTISSA_BITS as u8)
    }

    /// Reconstruct coarse vector (sign + exponent only).
    ///
    /// This gives the magnitude and sign but no mantissa precision.
    /// Useful for approximate search and initial filtering.
    pub fn reconstruct_coarse(&self) -> Vec<f32> {
        self.reconstruct_partial(0)
    }

    /// Reconstruct with N mantissa bits (0 = coarse, 23 = full precision).
    ///
    /// # Arguments
    /// * `mantissa_bits` - Number of mantissa bits to include (0-23)
    ///
    /// # Panics
    /// Panics if `mantissa_bits > 23`.
    pub fn reconstruct_partial(&self, mantissa_bits: u8) -> Vec<f32> {
        assert!(
            mantissa_bits <= Self::NUM_MANTISSA_BITS as u8,
            "mantissa_bits must be <= 23"
        );

        let mut result = Vec::with_capacity(self.dimensions);

        for dim in 0..self.dimensions {
            let byte_idx = dim / 8;
            let bit_idx = dim % 8;

            // Reconstruct sign bit
            let sign_bit = (self.sign_plane[byte_idx] >> bit_idx) & 1;

            // Reconstruct exponent bits
            let mut exponent_bits = 0u8;
            for exp_plane in 0..Self::NUM_EXPONENT_BITS {
                let bit = (self.exponent_planes[exp_plane][byte_idx] >> bit_idx) & 1;
                exponent_bits |= bit << exp_plane;
            }

            // Reconstruct mantissa bits
            let mut mantissa_bits_val = 0u32;
            for mant_plane in 0..mantissa_bits as usize {
                let bit = (self.mantissa_planes[mant_plane][byte_idx] >> bit_idx) & 1;
                mantissa_bits_val |= (bit as u32) << mant_plane;
            }

            // Reconstruct the full 32-bit representation
            let reconstructed_bits =
                ((sign_bit as u32) << 31) | ((exponent_bits as u32) << 23) | mantissa_bits_val;

            result.push(f32::from_bits(reconstructed_bits));
        }

        result
    }

    /// Calculate storage size in bytes for coarse planes only (sign + exponent).
    pub fn coarse_size_bytes(&self) -> usize {
        let bytes_per_plane = self.dimensions.div_ceil(8);
        // 1 sign plane + 8 exponent planes
        (1 + Self::NUM_EXPONENT_BITS) * bytes_per_plane
    }

    /// Calculate storage size in bytes for all planes.
    pub fn total_size_bytes(&self) -> usize {
        let bytes_per_plane = self.dimensions.div_ceil(8);
        Self::TOTAL_PLANES * bytes_per_plane
    }

    /// Compression ratio compared to original f32 vector.
    ///
    /// # Arguments
    /// * `mantissa_bits` - Number of mantissa bits included (0-23)
    ///
    /// # Returns
    /// Ratio of plane storage to original f32 storage.
    /// Values < 1.0 indicate compression.
    pub fn compression_ratio(&self, mantissa_bits: u8) -> f32 {
        let original_bytes = (self.dimensions * 4) as f32;
        let bytes_per_plane = self.dimensions.div_ceil(8) as f32;
        let planes_used = 1 + Self::NUM_EXPONENT_BITS + (mantissa_bits as usize);
        let used_bytes = (planes_used as f32) * bytes_per_plane;
        used_bytes / original_bytes
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_reconstruct_full() {
        let original = vec![1.0, 2.5, -3.14, 0.0, 100.0];
        let bpv = BitPlaneVector::from_f32(&original);
        let reconstructed = bpv.reconstruct_full();

        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-6,
                "Expected {}, got {}",
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_coarse_reconstruction() {
        let original = vec![1.0, 2.5, -3.14, 0.0, 100.0];
        let bpv = BitPlaneVector::from_f32(&original);
        let coarse = bpv.reconstruct_coarse();

        // Coarse should preserve sign and approximate magnitude
        assert_eq!(coarse.len(), original.len());
        for (orig, coarse_val) in original.iter().zip(coarse.iter()) {
            // Sign should match
            assert_eq!(orig.is_sign_positive(), coarse_val.is_sign_positive());
            // Coarse = sign + exponent only (power-of-2), so up to 50% error is expected
            if orig.abs() > 0.1 {
                let error_ratio = (orig.abs() - coarse_val.abs()).abs() / orig.abs();
                assert!(
                    error_ratio < 0.50,
                    "Coarse error too large for {}: {:.4}",
                    orig,
                    error_ratio
                );
            }
        }
    }

    #[test]
    fn test_partial_reconstruction_monotonic() {
        let original = vec![1.5, -2.75, 3.14159, -0.5];
        let bpv = BitPlaneVector::from_f32(&original);

        let mut previous_error = f32::INFINITY;
        for bits in 0..=23 {
            let reconstructed = bpv.reconstruct_partial(bits);
            let error: f32 = original
                .iter()
                .zip(reconstructed.iter())
                .map(|(o, r)| (o - r).abs())
                .sum();

            // Error should decrease (monotonically) as we add more bits
            assert!(
                error <= previous_error,
                "Error increased at {} bits: {} > {}",
                bits,
                error,
                previous_error
            );
            previous_error = error;
        }
    }

    #[test]
    fn test_compression_ratio() {
        let vector = vec![1.0; 1024];
        let bpv = BitPlaneVector::from_f32(&vector);

        let coarse_ratio = bpv.compression_ratio(0);
        let full_ratio = bpv.compression_ratio(23);

        // For a vector of 1024 f32s:
        // Original: 4096 bytes
        // Coarse (1 + 8 planes): 9 * 128 = 1152 bytes -> ~0.28
        // Full (1 + 8 + 23 planes): 32 * 128 = 4096 bytes -> 1.0
        assert!(
            coarse_ratio < 0.3,
            "Coarse ratio too high: {}",
            coarse_ratio
        );
        assert!((full_ratio - 1.0).abs() < 0.01, "Full ratio should be ~1.0");
        assert!(
            coarse_ratio < full_ratio,
            "Coarse should be smaller than full"
        );
    }

    #[test]
    fn test_edge_cases() {
        let cases = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::EPSILON,
        ];

        let bpv = BitPlaneVector::from_f32(&cases);
        let reconstructed = bpv.reconstruct_full();

        for (orig, recon) in cases.iter().zip(reconstructed.iter()) {
            // Use bit comparison for zero/sign-zero
            if orig.is_nan() && recon.is_nan() {
                continue;
            }
            assert!(
                (orig - recon).abs() < 1e-6 || (orig.is_infinite() && recon.is_infinite()),
                "Mismatch: {} vs {}",
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_dimensions_various_sizes() {
        for dim in [1, 7, 8, 9, 15, 16, 17, 256, 1000] {
            let vector: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
            let bpv = BitPlaneVector::from_f32(&vector);
            let reconstructed = bpv.reconstruct_full();

            assert_eq!(
                reconstructed.len(),
                dim,
                "Dimension mismatch for size {}",
                dim
            );
            for (orig, recon) in vector.iter().zip(reconstructed.iter()) {
                assert!(
                    (orig - recon).abs() < 1e-5,
                    "Dimension {}: {} != {}",
                    dim,
                    orig,
                    recon
                );
            }
        }
    }

    #[test]
    fn test_storage_sizes() {
        let vector = vec![1.0; 256];
        let bpv = BitPlaneVector::from_f32(&vector);

        let coarse = bpv.coarse_size_bytes();
        let total = bpv.total_size_bytes();

        // For 256 dimensions: 32 bytes per plane
        assert_eq!(coarse, 9 * 32, "Coarse size incorrect");
        assert_eq!(total, 32 * 32, "Total size incorrect");
    }
}
