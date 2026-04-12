// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

use std::sync::atomic::{AtomicBool, Ordering};

/// ThermalGuard monitors the system's thermal state and preemptively throttles
/// work to avoid hardware-triggered downclocking.
pub struct ThermalGuard {
    threshold_celsius: f32,
    hysteresis: f32,
    throttle_active: AtomicBool,
}

impl ThermalGuard {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold_celsius: threshold,
            hysteresis: 5.0,
            throttle_active: AtomicBool::new(false),
        }
    }

    /// Checks temperature and returns whether we should throttle.
    ///
    /// Uses Acquire/Release memory ordering to ensure consistent visibility
    /// of throttle state across cores (critical on ARM where Relaxed loads
    /// can observe stale values from other cores' store buffers).
    pub fn should_throttle(&self) -> bool {
        let temp = self.read_temperature();
        let is_throttled = self.throttle_active.load(Ordering::Acquire);

        if temp > self.threshold_celsius && !is_throttled {
            self.throttle_active.store(true, Ordering::Release);
            true
        } else if temp < (self.threshold_celsius - self.hysteresis) && is_throttled {
            self.throttle_active.store(false, Ordering::Release);
            false
        } else {
            is_throttled
        }
    }

    fn read_temperature(&self) -> f32 {
        // Placeholder: M1 high-perf temp monitoring
        45.0
    }
}

impl Default for ThermalGuard {
    fn default() -> Self {
        Self::new(85.0) // Throttle at 85°C
    }
}
