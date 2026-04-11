//! Hamming distance computation with multiple backends and runtime feature detection.
//!
//! Provides scalar, AVX-512, NEON, and auto-detecting backends for maximum performance.

use std::sync::atomic::{AtomicU8, Ordering};

/// Backend selection indicator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HammingBackend {
    Scalar,
    Avx512,
    Neon,
}

/// Cached backend detection state
static BACKEND_CACHE: AtomicU8 = AtomicU8::new(0); // 0 = undetected, 1 = scalar, 2 = avx512, 3 = neon

/// Scalar Hamming distance using u64::count_ones()
#[inline]
fn hamming_scalar(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "Input slices must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// AVX-512 Hamming distance using _mm512_popcnt_epi64
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512vpopcntdq")]
unsafe fn hamming_avx512(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len(), "Input slices must have equal length");

    let mut sum: u64 = 0;
    let mut i = 0;

    // Process 8 u64s at a time (512 bits)
    while i + 8 <= a.len() {
        let a_vec = _mm512_loadu_epi64(a.as_ptr().add(i) as *const i64);
        let b_vec = _mm512_loadu_epi64(b.as_ptr().add(i) as *const i64);
        let xor_vec = _mm512_xor_epi64(a_vec, b_vec);
        let popcount_vec = _mm512_popcnt_epi64(xor_vec);
        let reduced = _mm512_reduce_add_epi64(popcount_vec);
        sum += reduced as u64;
        i += 8;
    }

    // Handle remainder with scalar
    while i < a.len() {
        sum += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }

    sum as u32
}

/// NEON Hamming distance using vcntq_u8 with 512-bit unrolling (Silicon-Native)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len(), "Input slices must have equal length");

    let mut i = 0usize;
    let len = a.len();
    let mut total_sum: u32 = 0;

    // Process in 512-bit blocks (8 u64s)
    while i + 8 <= len {
        let mut acc0 = vdupq_n_u8(0);
        let mut acc1 = vdupq_n_u8(0);
        let mut acc2 = vdupq_n_u8(0);
        let mut acc3 = vdupq_n_u8(0);

        // Sub-loop to avoid u8 overflow for extremely large vectors
        // Max iterations per sub-loop: 31 (31 * 8 = 248 bits in a byte lane)
        let end = (i + 8 * 31).min(len - (len % 8));
        while i + 8 <= end {
            // Load 512 bits from A and B directly from memory
            let a0 = vld1q_u64(a.as_ptr().add(i));
            let a1 = vld1q_u64(a.as_ptr().add(i + 2));
            let a2 = vld1q_u64(a.as_ptr().add(i + 4));
            let a3 = vld1q_u64(a.as_ptr().add(i + 6));

            let b0 = vld1q_u64(b.as_ptr().add(i));
            let b1 = vld1q_u64(b.as_ptr().add(i + 2));
            let b2 = vld1q_u64(b.as_ptr().add(i + 4));
            let b3 = vld1q_u64(b.as_ptr().add(i + 6));

            // XOR and popcount
            acc0 = vaddq_u8(acc0, vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a0, b0))));
            acc1 = vaddq_u8(acc1, vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a1, b1))));
            acc2 = vaddq_u8(acc2, vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a2, b2))));
            acc3 = vaddq_u8(acc3, vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a3, b3))));

            i += 8;
        }

        // Reduce each accumulator separately to avoid u8 overflow when combining
        for acc in [acc0, acc1, acc2, acc3] {
            let sum16 = vpaddlq_u8(acc);
            let sum32 = vpaddlq_u16(sum16);
            let sum64 = vpaddlq_u32(sum32);
            total_sum += (vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1)) as u32;
        }
    }

    // Handle remaining blocks (128-bit)
    while i + 2 <= len {
        let a_vec = vld1q_u64(a.as_ptr().add(i));
        let b_vec = vld1q_u64(b.as_ptr().add(i));
        let xor = veorq_u64(a_vec, b_vec);
        let counts = vcntq_u8(vreinterpretq_u8_u64(xor));
        total_sum += vaddvq_u8(counts) as u32;
        i += 2;
    }

    // Handle scalar remainder
    while i < len {
        total_sum += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    total_sum
}

/// Detect and cache the best available backend at runtime
fn detect_backend() -> HammingBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return HammingBackend::Avx512;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64
        return HammingBackend::Neon;
    }

    HammingBackend::Scalar
}

/// Get the cached backend, detecting on first call
fn get_cached_backend() -> HammingBackend {
    let cached = BACKEND_CACHE.load(Ordering::Relaxed);
    if cached != 0 {
        return match cached {
            2 => HammingBackend::Avx512,
            3 => HammingBackend::Neon,
            _ => HammingBackend::Scalar,
        };
    }

    // First call: detect
    let backend = detect_backend();
    let cache_val = match backend {
        HammingBackend::Scalar => 1,
        HammingBackend::Avx512 => 2,
        HammingBackend::Neon => 3,
    };
    BACKEND_CACHE.store(cache_val, Ordering::Relaxed);
    backend
}

/// Compute Hamming distance with automatic backend selection
///
/// On first call, detects available SIMD features and caches the result.
/// Subsequent calls use the cached backend.
///
/// # Arguments
/// * `a` - First u64 slice
/// * `b` - Second u64 slice (must have same length as `a`)
///
/// # Returns
/// Hamming distance (number of differing bits)
#[inline]
pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    match get_cached_backend() {
        HammingBackend::Scalar => hamming_scalar(a, b),
        #[cfg(target_arch = "x86_64")]
        HammingBackend::Avx512 => unsafe { hamming_avx512(a, b) },
        #[cfg(not(target_arch = "x86_64"))]
        HammingBackend::Avx512 => hamming_scalar(a, b),
        #[cfg(target_arch = "aarch64")]
        HammingBackend::Neon => unsafe { hamming_neon(a, b) },
        #[cfg(not(target_arch = "aarch64"))]
        HammingBackend::Neon => hamming_scalar(a, b),
    }
}

/// Compute Hamming distance using the scalar backend explicitly
#[inline]
pub fn hamming_distance_scalar(a: &[u64], b: &[u64]) -> u32 {
    hamming_scalar(a, b)
}

/// Compute Hamming distance using AVX-512 if available on this platform
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn hamming_distance_avx512(a: &[u64], b: &[u64]) -> u32 {
    if is_x86_feature_detected!("avx512vpopcntdq") {
        unsafe { hamming_avx512(a, b) }
    } else {
        hamming_scalar(a, b)
    }
}

/// Compute Hamming distance using NEON if available on this platform
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn hamming_distance_neon(a: &[u64], b: &[u64]) -> u32 {
    unsafe { hamming_neon(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let a = vec![0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA];
        let b = vec![0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA];
        assert_eq!(hamming_distance_scalar(&a, &b), 0);
    }

    #[test]
    fn test_hamming_completely_different() {
        let a = vec![u64::MAX];
        let b = vec![0u64];
        assert_eq!(hamming_distance_scalar(&a, &b), 64);
    }

    #[test]
    fn test_hamming_single_bit_diff() {
        let a = vec![0u64];
        let b = vec![1u64];
        assert_eq!(hamming_distance_scalar(&a, &b), 1);
    }

    #[test]
    fn test_hamming_multiple_words() {
        let a = vec![u64::MAX, 0, u64::MAX];
        let b = vec![0, u64::MAX, 0];
        // First word: 64 differences
        // Second word: 64 differences
        // Third word: 64 differences
        assert_eq!(hamming_distance_scalar(&a, &b), 192);
    }

    #[test]
    fn test_hamming_auto_dispatch() {
        let a = vec![0xDEADBEEFDEADBEEF, 0xCAFEBABECAFEBABE];
        let b = vec![0xBEEFDEADBEEFDEAD, 0xBABECAFEBABECAFE];

        let scalar_result = hamming_distance_scalar(&a, &b);
        let auto_result = hamming_distance(&a, &b);

        // All backends should agree
        assert_eq!(scalar_result, auto_result);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hamming_avx512_matches_scalar() {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            let a = vec![0x123456789ABCDEF0, 0xFEDCBA9876543210, 0x0F0F0F0F0F0F0F0F];
            let b = vec![0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0x123456789ABCDEF0];

            let scalar = hamming_distance_scalar(&a, &b);
            let avx512 = hamming_distance_avx512(&a, &b);
            assert_eq!(scalar, avx512, "AVX-512 and scalar results must match");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_hamming_neon_matches_scalar() {
        let a = vec![0x123456789ABCDEF0, 0xFEDCBA9876543210, 0x0F0F0F0F0F0F0F0F];
        let b = vec![0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0x123456789ABCDEF0];

        let scalar = hamming_distance_scalar(&a, &b);
        let neon = hamming_distance_neon(&a, &b);
        assert_eq!(scalar, neon, "NEON and scalar results must match");
    }

    #[test]
    fn test_hamming_popcount_correctness() {
        // Verify against manual popcount
        let a = vec![0xAAAAAAAAAAAAAAAA]; // 32 ones
        let b = vec![0x5555555555555555]; // complement, 32 ones in different positions
                                          // XOR gives all ones = 64 differences
        assert_eq!(hamming_distance_scalar(&a, &b), 64);
    }

    #[test]
    fn test_hamming_large_vector() {
        // Test with a larger vector to trigger multiple iterations
        let a: Vec<u64> = (0..256).map(|i| i as u64).collect();
        let b: Vec<u64> = (0..256).map(|i| (i as u64).wrapping_add(1)).collect();

        let result = hamming_distance(&a, &b);
        let expected = hamming_distance_scalar(&a, &b);
        assert_eq!(result, expected);
    }
}
