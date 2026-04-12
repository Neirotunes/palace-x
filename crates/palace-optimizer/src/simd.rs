// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SoA Particle representation for SIMD
pub struct ParticleSoA {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
}

/// Updates positions of particles using NEON intrinsics.
/// Processes 4 f32 values (2 particles x 2D or 4 particles x 1D) per iteration.
///
/// # Safety
/// All slices must have the same length. Caller must ensure this.
///
/// # Panics
/// Panics if slice lengths are mismatched.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn update_particles_neon(
    pos_x: &mut [f32],
    pos_y: &mut [f32],
    vel_x: &[f32],
    vel_y: &[f32],
    dt: f32,
) {
    let n = pos_x.len();
    assert_eq!(pos_y.len(), n, "pos_y length mismatch");
    assert_eq!(vel_x.len(), n, "vel_x length mismatch");
    assert_eq!(vel_y.len(), n, "vel_y length mismatch");

    let delta = vdupq_n_f32(dt);
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;

        // Load chunks of 4
        let cx = vld1q_f32(pos_x.as_ptr().add(idx));
        let cy = vld1q_f32(pos_y.as_ptr().add(idx));
        let cvx = vld1q_f32(vel_x.as_ptr().add(idx));
        let cvy = vld1q_f32(vel_y.as_ptr().add(idx));

        // Fused Multiply-Add: pos += vel * dt
        let next_x = vfmaq_f32(cx, cvx, delta);
        let next_y = vfmaq_f32(cy, cvy, delta);

        // Store back
        vst1q_f32(pos_x.as_mut_ptr().add(idx), next_x);
        vst1q_f32(pos_y.as_mut_ptr().add(idx), next_y);
    }

    // Scalar fallback for remainder
    for i in (chunks * 4)..pos_x.len() {
        pos_x[i] += vel_x[i] * dt;
        pos_y[i] += vel_y[i] * dt;
    }
}

/// Frustum culling for 4 spheres at a time using NEON.
/// Returns a bitmask where each bit represents visibility.
///
/// # Safety
/// All slices must have the same length.
///
/// # Panics
/// Panics if slice lengths are mismatched or length exceeds 32 (bitmask capacity).
#[cfg(target_arch = "aarch64")]
pub unsafe fn sphere_frustum_cull_neon(
    cx: &[f32],
    cy: &[f32],
    cz: &[f32],
    radii: &[f32],
    plane: [f32; 4], // nx, ny, nz, d
) -> u32 {
    let n = cx.len();
    assert_eq!(cy.len(), n, "cy length mismatch");
    assert_eq!(cz.len(), n, "cz length mismatch");
    assert_eq!(radii.len(), n, "radii length mismatch");
    assert!(n <= 32, "sphere_frustum_cull_neon: max 32 spheres (bitmask is u32)");
    let nx = vdupq_n_f32(plane[0]);
    let ny = vdupq_n_f32(plane[1]);
    let nz = vdupq_n_f32(plane[2]);
    let nd = vdupq_n_f32(plane[3]);
    let zero = vdupq_n_f32(0.0);

    let mut visible_mask = 0u32;
    let chunks = cx.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let v_cx = vld1q_f32(cx.as_ptr().add(idx));
        let v_cy = vld1q_f32(cy.as_ptr().add(idx));
        let v_cz = vld1q_f32(cz.as_ptr().add(idx));
        let v_r = vld1q_f32(radii.as_ptr().add(idx));

        // dist = dot(normal, center) + d
        let dot = vfmaq_f32(nd, nx, v_cx);
        let dot = vfmaq_f32(dot, ny, v_cy);
        let dot = vfmaq_f32(dot, nz, v_cz);

        // visible = dist + radius > 0
        let signed_dist = vaddq_f32(dot, v_r);
        let res_u32 = vcgtq_f32(signed_dist, zero);

        if vgetq_lane_u32::<0>(res_u32) != 0 {
            visible_mask |= 1 << (idx + 0);
        }
        if vgetq_lane_u32::<1>(res_u32) != 0 {
            visible_mask |= 1 << (idx + 1);
        }
        if vgetq_lane_u32::<2>(res_u32) != 0 {
            visible_mask |= 1 << (idx + 2);
        }
        if vgetq_lane_u32::<3>(res_u32) != 0 {
            visible_mask |= 1 << (idx + 3);
        }
    }

    visible_mask
}

/// Computes the dot product of two vectors using NEON.
///
/// # Safety
/// Both slices must have the same length.
///
/// # Panics
/// Panics if slice lengths differ.
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product_neon: length mismatch ({} vs {})", a.len(), b.len());
    let mut sumv = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        sumv = vfmaq_f32(sumv, va, vb);
    }

    // Horizontal add of the 128-bit vector
    let mut sum = vaddvq_f32(sumv);

    // Remaining elements
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

/// Computes the cosine distance using NEON SIMD.
#[cfg(target_arch = "aarch64")]
pub unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_neon(a, b);
    let norm_a = dot_product_neon(a, a).sqrt();
    let norm_b = dot_product_neon(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let similarity = dot / (norm_a * norm_b);
    ((1.0 - similarity) / 2.0).max(0.0).min(1.0)
}
