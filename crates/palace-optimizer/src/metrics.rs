// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

#[cfg(target_arch = "aarch64")]
pub mod timing {
    pub struct MachTimer {
        timebase_info: mach2::mach_time::mach_timebase_info_data_t,
        start: u64,
    }

    impl MachTimer {
        pub fn new() -> Self {
            let mut info = mach2::mach_time::mach_timebase_info_data_t { numer: 0, denom: 0 };
            unsafe {
                mach2::mach_time::mach_timebase_info(&mut info);
            }
            Self {
                timebase_info: info,
                start: unsafe { mach2::mach_time::mach_absolute_time() },
            }
        }

        pub fn elapsed_nanos(&self) -> u64 {
            let end = unsafe { mach2::mach_time::mach_absolute_time() };
            let elapsed = end - self.start;
            (elapsed * self.timebase_info.numer as u64) / self.timebase_info.denom as u64
        }
    }
}
