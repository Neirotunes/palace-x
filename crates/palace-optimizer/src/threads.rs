use rayon::ThreadPool;
use std::mem;

// Constants from <mach/thread_policy.h>
const THREAD_AFFINITY_POLICY: i32 = 4;
const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;

#[repr(C)]
struct ThreadAffinityPolicy {
    affinity_tag: i32,
}

/// P-cores on M1/M2/M3 generally have affinity tag 1, E-cores tag 2.
#[derive(Debug, Clone, Copy)]
pub enum CoreType {
    Performance = 1, // Firestorm/Avalanche
    Efficiency = 2,  // Icestorm/Blizzard
}

/// Pins the current thread to a specific core type using Mach affinity policies.
pub fn pin_current_thread(core_type: CoreType) -> Result<(), i32> {
    let policy = ThreadAffinityPolicy {
        affinity_tag: core_type as i32,
    };

    let thread = unsafe { libc::pthread_self() };

    let result = unsafe {
        thread_policy_set(
            pthread_mach_thread_np(thread),
            THREAD_AFFINITY_POLICY,
            &policy as *const _ as *const i32,
            THREAD_AFFINITY_POLICY_COUNT,
        )
    };

    if result == 0 {
        Ok(())
    } else {
        Err(result)
    }
}

// Bindings to Mach kernel functions
extern "C" {
    fn thread_policy_set(thread: u32, flavor: i32, policy_info: *const i32, count: u32) -> i32;

    fn pthread_mach_thread_np(thread: libc::pthread_t) -> u32;
}

/// macOS QoS classes - impact the scheduler's behavior and energy efficiency.
#[repr(u32)]
pub enum QosClass {
    UserInteractive = 0x21, // QOS_CLASS_USER_INTERACTIVE
    UserInitiated = 0x19,   // QOS_CLASS_USER_INITIATED
    Utility = 0x11,         // QOS_CLASS_UTILITY
    Background = 0x09,      // QOS_CLASS_BACKGROUND
}

/// Sets the QoS class and scheduling priority for the current thread.
pub fn set_thread_qos(class: QosClass) {
    unsafe {
        // High-priority real-time scheduling
        let mut param: libc::sched_param = mem::zeroed();
        param.sched_priority = 47;
        libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param);
    }
}

/// Creates a Rayon thread pool with workers pinned to P-cores.
pub fn create_game_thread_pool(num_p_cores: usize) -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_p_cores)
        .start_handler(|_idx| {
            // Pin worker to P-cores
            let _ = pin_current_thread(CoreType::Performance);
            // Set highest QoS
            set_thread_qos(QosClass::UserInteractive);
        })
        .build()
        .expect("Failed to create game thread pool")
}
