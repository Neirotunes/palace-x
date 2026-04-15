// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binary quantization: converts f32 vectors to packed u64 bit vectors.
//!
//! Each dimension is quantized to a single bit based on whether it exceeds a threshold (0.0).
//! 64 dimensions are packed into each u64, LSB-first (dimension i → bit position i%64).

/// Quantizes a single f32 vector to packed u64 representation.
///
/// Each u64 contains 64 binary digits packed LSB-first.
/// Dimension `d` maps to bit `d % 64` in word `d / 64`.
///
/// # Arguments
/// * `vector` - Input f32 slice to quantize
///
/// # Returns
/// Vec<u64> where each u64 packs 64 binary bits (threshold at 0.0)
pub fn quantize_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = vector.len().div_ceil(64);
    let mut result = vec![0u64; num_words];

    for (i, &value) in vector.iter().enumerate() {
        if value > 0.0 {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }

    result
}

/// Quantizes multiple vectors in a slice.
///
/// # Arguments
/// * `vectors` - Slice of f32 slices to quantize
///
/// # Returns
/// Vec<Vec<u64>> where each inner Vec contains packed bits for one vector
pub fn quantize_binary_slice(vectors: &[&[f32]]) -> Vec<Vec<u64>> {
    vectors.iter().map(|v| quantize_binary(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_single_word() {
        // 64 positive values -> all 1s
        let v: Vec<f32> = vec![1.0; 64];
        let result = quantize_binary(&v);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], u64::MAX);
    }

    #[test]
    fn test_quantize_all_zeros() {
        // All zero or negative -> all 0s
        let v: Vec<f32> = vec![0.0; 64];
        let result = quantize_binary(&v);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_quantize_alternating() {
        // Alternating positive/negative
        let v: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let result = quantize_binary(&v);
        assert_eq!(result.len(), 1);
        // Even indices (0,2,4,...) are set LSB-first: 0x5555...
        assert_eq!(result[0], 0x5555555555555555);
    }

    #[test]
    fn test_quantize_multiple_words() {
        let v: Vec<f32> = vec![1.0; 200];
        let result = quantize_binary(&v);
        assert_eq!(result.len(), 4); // ceil(200/64) = 4
        assert_eq!(result[0], u64::MAX);
        assert_eq!(result[1], u64::MAX);
        assert_eq!(result[2], u64::MAX);
        // Only 200 - 3*64 = 8 bits set in last word (LSB-first, bits 0..7)
        assert_eq!(result[3], 0x00000000000000FF);
    }

    #[test]
    fn test_quantize_partial_word() {
        let v: Vec<f32> = vec![1.0; 96];
        let result = quantize_binary(&v);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], u64::MAX);
        // Only 96 - 64 = 32 bits set in last word (LSB-first, bits 0..31)
        assert_eq!(result[1], 0x00000000FFFFFFFF);
    }

    #[test]
    fn test_quantize_slice() {
        let vectors = [vec![1.0; 64], vec![0.0; 64]];
        let vecs_ref: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let result = quantize_binary_slice(&vecs_ref);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0][0], u64::MAX);
        assert_eq!(result[1][0], 0);
    }

    #[test]
    fn test_quantize_first_dim() {
        // First dimension is 1, rest are 0
        let mut v = vec![0.0; 64];
        v[0] = 1.0;
        let result = quantize_binary(&v);
        // Dimension 0 → bit 0 (LSB)
        assert_eq!(result[0], 1u64);
    }

    #[test]
    fn test_quantize_last_dim() {
        // Last dimension is 1, rest are 0
        let mut v = vec![0.0; 64];
        v[63] = 1.0;
        let result = quantize_binary(&v);
        // Dimension 63 → bit 63
        assert_eq!(result[0], 1u64 << 63);
    }
}
