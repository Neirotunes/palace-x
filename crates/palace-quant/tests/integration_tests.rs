// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests verifying correctness and correlation between metrics
#![allow(clippy::useless_vec)]

use palace_quant::{
    batch::batch_hamming_topk, binary::quantize_binary, cosine::cosine_distance,
    hamming::hamming_distance,
};

#[test]
fn test_quantization_and_hamming_roundtrip() {
    // Verify that quantization and Hamming distance work together
    let v1 = vec![0.5, -0.3, 0.8, -0.2, 0.1, 1.5, -0.9, 0.0];
    let v2 = vec![0.4, -0.4, 0.7, -0.1, 0.2, 1.4, -0.8, 0.1];

    let q1 = quantize_binary(&v1);
    let q2 = quantize_binary(&v2);

    let dist = hamming_distance(&q1, &q2);
    assert!(dist <= 64); // Maximum possible with 1 word
    assert!(dist > 0); // Should have some differences

    // Different vector should have different distance
    let v3 = vec![-0.5, 0.3, -0.8, 0.2, -0.1, -1.5, 0.9, 0.0];
    let q3 = quantize_binary(&v3);
    let dist2 = hamming_distance(&q1, &q3);

    assert_ne!(dist, dist2);
}

#[test]
fn test_hamming_cosine_correlation() {
    // Verify that Hamming distance on quantized vectors correlates with
    // cosine distance on original vectors.
    //
    // The intuition: vectors with similar direction should have similar signs,
    // and thus similar binary quantizations.

    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1], // Very similar
        vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], // Similar magnitudes, opposite order
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], // Opposite signs
    ];

    let query = &vectors[0];
    let quantized: Vec<Vec<u64>> = vectors.iter().map(|v| quantize_binary(v)).collect();

    // Compute pairwise correlations
    for i in 1..vectors.len() {
        let hamming_dist = hamming_distance(&quantized[0], &quantized[i]);
        let cosine_dist = cosine_distance(query, &vectors[i]);

        // Higher cosine distance (less similar) should correlate with higher Hamming distance
        println!(
            "vs[{}]: hamming={}, cosine={:.4}",
            i, hamming_dist, cosine_dist
        );
    }

    // vectors[0] vs vectors[1] should be very close in both metrics
    let h01 = hamming_distance(&quantized[0], &quantized[1]);
    let c01 = cosine_distance(query, &vectors[1]);

    // vectors[0] vs vectors[3] should be far in both metrics
    let h03 = hamming_distance(&quantized[0], &quantized[3]);
    let c03 = cosine_distance(query, &vectors[3]);

    // Verify correlation: closer in cosine -> closer in Hamming
    assert!(
        h01 < h03,
        "Similar vectors should have smaller Hamming distance"
    );
    assert!(
        c01 < c03,
        "Similar vectors should have smaller cosine distance"
    );
}

#[test]
fn test_batch_topk_returns_sorted_results() {
    let query_vec = vec![1.0, 0.5, -0.5, 0.2];
    let query = quantize_binary(&query_vec);

    let candidates = vec![
        vec![1.0, 0.5, -0.5, 0.2],   // identical
        vec![1.0, 0.5, -0.5, -0.2],  // one bit different
        vec![1.0, 0.5, 0.5, 0.2],    // one bit different
        vec![-1.0, -0.5, 0.5, -0.2], // all bits different
        vec![0.9, 0.4, -0.6, 0.1],   // similar but slightly different
    ];

    let quantized: Vec<Vec<u64>> = candidates.iter().map(|v| quantize_binary(v)).collect();
    let candidates_ref: Vec<&[u64]> = quantized.iter().map(|c| c.as_slice()).collect();

    let results = batch_hamming_topk(&query, &candidates_ref, 5);

    assert_eq!(results.len(), 5);

    // Verify results are sorted by distance (ascending)
    for i in 0..results.len() - 1 {
        assert!(
            results[i].1 <= results[i + 1].1,
            "Results must be sorted by distance: {} > {}",
            results[i].1,
            results[i + 1].1
        );
    }

    // First result should be identical or very close
    assert!(
        results[0].1 < 5,
        "Top result should have very small distance"
    );
}

#[test]
fn test_batch_topk_with_larger_vectors() {
    // Test with 256-dimensional vectors
    let query_vec: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
    let query = quantize_binary(&query_vec);

    let mut candidates = Vec::new();
    for j in 0..100 {
        let cand: Vec<f32> = (0..256)
            .map(|i| ((i as f32) + j as f32 * 0.1).sin())
            .collect();
        candidates.push(quantize_binary(&cand));
    }

    let candidates_ref: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
    let results = batch_hamming_topk(&query, &candidates_ref, 10);

    assert_eq!(results.len(), 10);

    // Results should be sorted
    for i in 0..results.len() - 1 {
        assert!(results[i].1 <= results[i + 1].1);
    }

    // All indices should be in valid range
    for (idx, _dist) in &results {
        assert!(*idx < 100);
    }
}

#[test]
fn test_all_backends_agree_on_random_data() {
    // Generate some random-ish data and verify all backends produce same results
    use palace_quant::hamming::hamming_distance_scalar;

    let a = vec![
        0xDEADBEEFCAFEBABE,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
        0x1111111111111111,
        0x2222222222222222,
    ];

    let b = vec![
        0xCAFEBABEDEADBEEF,
        0x9876543210FEDCBA,
        0x0123456789ABCDEF,
        0xFFFFFFFFFFFFFFFF,
        0x0000000000000000,
    ];

    let scalar = hamming_distance_scalar(&a, &b);
    let auto = hamming_distance(&a, &b);

    assert_eq!(scalar, auto, "Auto-dispatch and scalar must agree");

    // Also verify with avx512/neon if available
    #[cfg(target_arch = "x86_64")]
    {
        use palace_quant::hamming::hamming_distance_avx512;
        let avx512 = hamming_distance_avx512(&a, &b);
        assert_eq!(scalar, avx512, "AVX-512 must match scalar");
    }

    #[cfg(target_arch = "aarch64")]
    {
        use palace_quant::hamming::hamming_distance_neon;
        let neon = hamming_distance_neon(&a, &b);
        assert_eq!(scalar, neon, "NEON must match scalar");
    }
}

#[test]
fn test_quantization_preserves_direction() {
    // Verify that quantization reasonably preserves vector direction
    // by checking that similar vectors stay similar after quantization

    let v_base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v_similar = vec![1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05];
    let v_opposite = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];

    let q_base = quantize_binary(&v_base);
    let q_similar = quantize_binary(&v_similar);
    let q_opposite = quantize_binary(&v_opposite);

    let dist_similar = hamming_distance(&q_base, &q_similar);
    let dist_opposite = hamming_distance(&q_base, &q_opposite);

    // Similar vectors should have smaller Hamming distance
    assert!(
        dist_similar < dist_opposite,
        "Similar vectors should be closer: {} vs {}",
        dist_similar,
        dist_opposite
    );

    // Similar vectors should be very close (ideally 0)
    assert!(
        dist_similar <= 2,
        "Very similar vectors should have tiny Hamming distance, got {}",
        dist_similar
    );
}

#[test]
fn test_batch_empty_and_edge_cases() {
    let query = quantize_binary(&[1.0, 2.0, 3.0]);

    // Empty candidates
    let empty: Vec<&[u64]> = vec![];
    let results = batch_hamming_topk(&query, &empty, 10);
    assert_eq!(results.len(), 0);

    // k=0
    let cand = quantize_binary(&[1.0, 2.0, 3.0]);
    let cand_ref = vec![&cand[..]];
    let results = batch_hamming_topk(&query, &cand_ref, 0);
    assert_eq!(results.len(), 0);

    // k > candidates
    let results = batch_hamming_topk(&query, &cand_ref, 100);
    assert_eq!(results.len(), 1);
}
