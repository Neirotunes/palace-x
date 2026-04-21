// Copyright (c) 2026 M.Diach
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fast cosine distance computation for f32 vectors.
//!
//! Computes cosine distance as 1.0 - cosine_similarity, where:
//! cosine_similarity = dot(a, b) / (norm(a) * norm(b))

/// Compute dot product of two f32 vectors
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input vectors must have equal length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 norm (Euclidean length) of a vector
#[inline]
fn norm_l2(v: &[f32]) -> f32 {
    let sum_sq: f32 = v.iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}

/// Compute cosine distance: 1.0 - (dot(a,b) / (norm(a) * norm(b)))
///
/// Returns a value in [0, 2], where 0 means identical direction and 2 means opposite.
/// Returns 1.0 for zero-norm vectors (undefined similarity).
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
/// Cosine distance in range [0, 2]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input vectors must have equal length");

    let norm_a = norm_l2(a);
    let norm_b = norm_l2(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Undefined similarity -> neutral distance
    }

    let dot = dot_product(a, b);
    let similarity = dot / (norm_a * norm_b);

    // Clamp to [-1, 1] for numerical stability
    let clamped = similarity.clamp(-1.0, 1.0);
    1.0 - clamped
}

/// Compute cosine similarity: dot(a,b) / (norm(a) * norm(b))
///
/// Returns a value in [-1, 1], where 1 means identical direction.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
/// Cosine similarity in range [-1, 1]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input vectors must have equal length");

    let norm_a = norm_l2(a);
    let norm_b = norm_l2(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let dot = dot_product(a, b);
    let similarity = dot / (norm_a * norm_b);
    similarity.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_perpendicular() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_scaled_vectors() {
        // Cosine distance should be invariant to scaling
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // 2x scaled
        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&a, &b);
        assert_eq!(dist, 1.0); // Undefined -> neutral
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_from_similarity() {
        // Verify distance = 1 - similarity
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let dist = cosine_distance(&a, &b);
        let sim = cosine_similarity(&a, &b);
        assert!((dist - (1.0 - sim)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_large_vectors() {
        let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i as f32) * 2.0).collect();
        let dist = cosine_distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_random_vectors() {
        let a = vec![0.5, -0.3, 0.8, -0.2, 0.1];
        let b = vec![0.3, 0.4, -0.5, 0.9, -0.2];
        let dist = cosine_distance(&a, &b);
        // Distance should be in [0, 2]
        assert!((0.0..=2.0).contains(&dist));
    }
}
