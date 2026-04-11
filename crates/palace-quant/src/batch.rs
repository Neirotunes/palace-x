//! Batch operations for Hamming distance queries.
//!
//! Efficiently compute top-k nearest neighbors using Hamming distance.

use crate::hamming::hamming_distance;

/// Result of a batch Hamming distance query: (candidate_index, distance)
pub type HammingResult = (usize, u32);

/// Find top-k nearest candidates by Hamming distance
///
/// Computes Hamming distance from query to all candidates and returns the k
/// nearest results sorted by distance (ascending).
///
/// # Arguments
/// * `query` - Query vector (u64 slice)
/// * `candidates` - Slice of candidate vectors
/// * `k` - Number of results to return
///
/// # Returns
/// Vec of (candidate_index, distance) tuples, sorted by distance ascending.
/// Returns fewer than k results if fewer candidates exist.
pub fn batch_hamming_topk(query: &[u64], candidates: &[&[u64]], k: usize) -> Vec<HammingResult> {
    // Quick bounds checking
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let k_actual = k.min(candidates.len());

    // Compute distances for all candidates
    let mut results: Vec<HammingResult> = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| (idx, hamming_distance(query, candidate)))
        .collect();

    // Partial sort to find top-k
    // For small k relative to n, this is faster than full sort
    if k_actual < results.len() {
        // Use nth_element-like behavior: partition by k-th smallest
        results.select_nth_unstable_by_key(k_actual - 1, |(_idx, dist)| *dist);
        results.truncate(k_actual);
    }

    // Sort results by distance
    results.sort_by_key(|(_idx, dist)| *dist);

    results
}

/// Find the single nearest candidate by Hamming distance
///
/// # Arguments
/// * `query` - Query vector (u64 slice)
/// * `candidates` - Slice of candidate vectors
///
/// # Returns
/// Option with (candidate_index, distance) or None if no candidates
pub fn batch_hamming_nearest(query: &[u64], candidates: &[&[u64]]) -> Option<HammingResult> {
    if candidates.is_empty() {
        return None;
    }

    let (best_idx, best_dist) = candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| (idx, hamming_distance(query, candidate)))
        .min_by_key(|(_idx, dist)| *dist)?;

    Some((best_idx, best_dist))
}

/// Compute Hamming distances from query to all candidates
///
/// # Arguments
/// * `query` - Query vector (u64 slice)
/// * `candidates` - Slice of candidate vectors
///
/// # Returns
/// Vec of (candidate_index, distance) tuples, unsorted
pub fn batch_hamming_all(query: &[u64], candidates: &[&[u64]]) -> Vec<HammingResult> {
    candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| (idx, hamming_distance(query, candidate)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::quantize_binary;

    #[test]
    fn test_batch_topk_basic() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let cand1 = vec![0xFFFFFFFFFFFFFFFF]; // distance 0
        let cand2 = vec![0x0000000000000000]; // distance 64
        let cand3 = vec![0xAAAAAAAAAAAAAAAA]; // distance 32
        let candidates = vec![&cand1[..], &cand2[..], &cand3[..]];

        let results = batch_hamming_topk(&query, &candidates, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // cand1, distance 0
        assert_eq!(results[0].1, 0);
        assert_eq!(results[1].0, 2); // cand3, distance 32
        assert_eq!(results[1].1, 32);
    }

    #[test]
    fn test_batch_topk_k_larger_than_candidates() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let cand1 = vec![0xFFFFFFFFFFFFFFFF];
        let cand2 = vec![0x0000000000000000];
        let candidates = vec![&cand1[..], &cand2[..]];

        let results = batch_hamming_topk(&query, &candidates, 10);
        assert_eq!(results.len(), 2); // Only 2 candidates exist
    }

    #[test]
    fn test_batch_topk_empty() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let candidates: Vec<&[u64]> = vec![];

        let results = batch_hamming_topk(&query, &candidates, 5);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_topk_k_zero() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let cand1 = vec![0xFFFFFFFFFFFFFFFF];
        let candidates = vec![&cand1[..]];

        let results = batch_hamming_topk(&query, &candidates, 0);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_nearest() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let cand1 = vec![0x0000000000000000];
        let cand2 = vec![0xFFFFFFFFFFFFFFFF];
        let cand3 = vec![0xAAAAAAAAAAAAAAAA];
        let candidates = vec![&cand1[..], &cand2[..], &cand3[..]];

        let result = batch_hamming_nearest(&query, &candidates);
        assert!(result.is_some());
        let (idx, dist) = result.unwrap();
        assert_eq!(idx, 1); // cand2 is identical
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_batch_nearest_empty() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let candidates: Vec<&[u64]> = vec![];

        let result = batch_hamming_nearest(&query, &candidates);
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_all() {
        let query = vec![0xFFFFFFFFFFFFFFFF];
        let cand1 = vec![0xFFFFFFFFFFFFFFFF];
        let cand2 = vec![0x0000000000000000];
        let candidates = vec![&cand1[..], &cand2[..]];

        let results = batch_hamming_all(&query, &candidates);
        assert_eq!(results.len(), 2);
        // Results are unsorted
        assert!(results.iter().any(|(idx, dist)| *idx == 0 && *dist == 0));
        assert!(results.iter().any(|(idx, dist)| *idx == 1 && *dist == 64));
    }

    #[test]
    fn test_batch_topk_sorted() {
        let query = vec![0x0000000000000000];
        let candidates: Vec<Vec<u64>> = vec![
            vec![0x0000000000000001], // distance 1
            vec![0x000000000000000F], // distance 4
            vec![0x00000000000000FF], // distance 8
            vec![0x0000000000000000], // distance 0
        ];
        let candidates_ref: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();

        let results = batch_hamming_topk(&query, &candidates_ref, 4);
        assert_eq!(results.len(), 4);

        // Verify sorted by distance
        assert_eq!(results[0].1, 0);
        assert_eq!(results[1].1, 1);
        assert_eq!(results[2].1, 4);
        assert_eq!(results[3].1, 8);
    }

    #[test]
    fn test_batch_hamming_with_real_quantization() {
        // Quantize some random-ish vectors
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let v3 = vec![-1.0, -2.0, -3.0, -4.0, -5.0];

        let q1 = quantize_binary(&v1);
        let q2 = quantize_binary(&v2);
        let q3 = quantize_binary(&v3);

        let candidates = vec![&q2[..], &q3[..]];
        let results = batch_hamming_topk(&q1, &candidates, 2);

        assert_eq!(results.len(), 2);
        // q2 should be closer to q1 than q3 (similar signs)
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn test_batch_topk_large() {
        let query = vec![0xAAAAAAAAAAAAAAAA; 16];
        let mut candidates = Vec::new();

        for i in 0..1000 {
            let pattern = if i % 2 == 0 { 0xAAAAAAAAAAAAAAAA } else { 0x5555555555555555 };
            candidates.push(vec![pattern; 16]);
        }

        let candidates_ref: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let results = batch_hamming_topk(&query, &candidates_ref, 10);

        assert_eq!(results.len(), 10);
        // All results with pattern 0xAAAA should have distance 0
        assert_eq!(results[0].1, 0);
        // Results should be sorted
        for i in 0..results.len() - 1 {
            assert!(results[i].1 <= results[i + 1].1);
        }
    }
}
