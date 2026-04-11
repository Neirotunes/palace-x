//! # palace-quant
//! Binary quantization and SIMD-accelerated Hamming distance module for Palace-X.
//!
//! Provides ultra-fast binary quantization of f32 vectors and multiple backends for Hamming
//! distance computation with runtime feature detection and fallbacks.

pub mod binary;
pub mod hamming;
pub mod cosine;
pub mod batch;

pub use binary::{quantize_binary, quantize_binary_slice};
pub use hamming::{hamming_distance, HammingBackend};
pub use cosine::cosine_distance;
pub use batch::batch_hamming_topk;
