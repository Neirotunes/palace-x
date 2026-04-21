// Copyright (c) 2026 M.Diach
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # palace-quant
//! Binary quantization and SIMD-accelerated Hamming distance module for Palace-X.
//!
//! Provides ultra-fast binary quantization of f32 vectors and multiple backends for Hamming
//! distance computation with runtime feature detection and fallbacks.
//!
//! ## RaBitQ (SIGMOD 2024)
//! The `rabitq` module implements Randomized Binary Quantization with theoretical error bounds,
//! achieving ~90%+ recall at 1-bit-per-dimension using random orthogonal rotations and
//! per-vector scalar corrections. Same SIMD Hamming kernels, dramatically better accuracy.

pub mod batch;
pub mod binary;
pub mod cosine;
pub mod hamming;
pub mod rabitq;

pub use batch::batch_hamming_topk;
pub use binary::{quantize_binary, quantize_binary_slice};
pub use cosine::cosine_distance;
pub use hamming::{hamming_distance, HammingBackend};
pub use rabitq::{RaBitQCode, RaBitQIndex, RaBitQQuery};
