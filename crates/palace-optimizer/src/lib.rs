// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! # Silicon-Native Game Optimizer
//!
//! DARPA-grade architectural optimizations for Apple Silicon (M1/M2/M3).
//! Provides UMA-aware memory allocation, P-core thread pinning, NEON SIMD physics,
//! thermal management, and speculative prefetching.

#[cfg(target_arch = "aarch64")]
pub mod arena;
#[cfg(target_arch = "aarch64")]
pub mod graphics;
#[cfg(target_arch = "aarch64")]
pub mod prefetch;
#[cfg(target_arch = "aarch64")]
pub mod simd;
#[cfg(target_arch = "aarch64")]
pub mod thermal;
#[cfg(target_arch = "aarch64")]
pub mod threads;

#[cfg(not(target_arch = "aarch64"))]
compile_error!("palace-optimizer requires aarch64 architecture (Apple Silicon)");

// Re-export common types
#[cfg(target_arch = "aarch64")]
pub use arena::{FrameAllocator, UmaArena};
#[cfg(target_arch = "aarch64")]
pub use prefetch::SpeculativePrefetcher;
#[cfg(target_arch = "aarch64")]
pub use thermal::ThermalGuard;
#[cfg(target_arch = "aarch64")]
pub use threads::{
    create_game_thread_pool, pin_current_thread, set_thread_qos, CoreType, QosClass,
};
