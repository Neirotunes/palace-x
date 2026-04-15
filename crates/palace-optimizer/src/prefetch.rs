// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

pub type AssetId = u64;

/// SpeculativePrefetcher learns memory access patterns using a Markov Table
/// and hints the M1 L2 prefetcher using ARM64 assembly hints.
pub struct SpeculativePrefetcher {
    last_asset: AtomicU64,
    /// Markov table: (PrevAsset, CurrentAsset) -> Probability/Count
    markov_table: RwLock<HashMap<(AssetId, AssetId), f32>>,
}

impl Default for SpeculativePrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeculativePrefetcher {
    pub fn new() -> Self {
        Self {
            last_asset: AtomicU64::new(0),
            markov_table: RwLock::new(HashMap::new()),
        }
    }

    pub fn record_access(&self, asset_id: AssetId) {
        let prev = self.last_asset.swap(asset_id, Ordering::SeqCst);
        if prev != 0 && prev != asset_id {
            if let Ok(mut table) = self.markov_table.write() {
                let prob = table.entry((prev, asset_id)).or_insert(0.0);
                *prob = (*prob * 0.9 + 0.1).min(1.0);
            }
        }
    }

    /// Predict the most likely next asset based on the current one.
    pub fn predict_next(&self, current: AssetId) -> Option<AssetId> {
        let table = self.markov_table.read().ok()?;
        table
            .iter()
            .filter(|((prev, _), _)| *prev == current)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|((_, next), _)| *next)
    }

    /// Issues an ARM64 prefetch hint for a predicted asset.
    pub fn prefetch_predicted(&self, _asset_id: AssetId, raw_ptr: *const u8) {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // PRFM PLDL1KEEP, [ptr]
                // Stable inline assembly for ARM64 prefetch hint
                core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) raw_ptr);
            }
        }
    }
}
