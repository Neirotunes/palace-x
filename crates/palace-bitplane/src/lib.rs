// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! # palace-bitplane
//!
//! Bit-plane disaggregation module for the Palace-X project.
//! Implements precision-proportional fetch for vectors by decomposing them into
//! independent bit planes.
//!
//! ## Architecture
//!
//! Vectors are decomposed into bit planes:
//! - **Sign plane**: 1 bit per dimension (packed)
//! - **Exponent planes**: 8 planes for IEEE 754 exponent
//! - **Mantissa planes**: 23 planes for IEEE 754 mantissa (loaded on demand)
//!
//! The coarse representation (sign + exponent) is kept in RAM,
//! while mantissa planes are tiered to disk in production.

pub mod degradation;
pub mod planes;
pub mod store;

pub use degradation::{DegradedResult, PrecisionLevel};
pub use planes::BitPlaneVector;
pub use store::{BitPlaneStore, CoarsePlanes, FinePlanes};
