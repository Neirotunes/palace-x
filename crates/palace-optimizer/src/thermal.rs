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
    pub fn should_throttle(&self) -> bool {
        let temp = self.read_temperature();
        let is_throttled = self.throttle_active.load(Ordering::Relaxed);

        if temp > self.threshold_celsius && !is_throttled {
            self.throttle_active.store(true, Ordering::SeqCst);
            true
        } else if temp < (self.threshold_celsius - self.hysteresis) && is_throttled {
            self.throttle_active.store(false, Ordering::SeqCst);
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
