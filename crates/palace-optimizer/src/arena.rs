// Copyright (c) 2026 M.Diach
// Proprietary — All Rights Reserved

use bumpalo::Bump;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Arena allocator, aligned to M1 L2 cache lines (128 bytes).
/// Mapped via mmap with MAP_SHARED for UMA zero-copy between CPU and GPU.
pub struct UmaArena {
    base: *mut u8,
    capacity: usize,
    offset: AtomicUsize,
}

impl UmaArena {
    /// Allocates memory directly via mmap, bypassing the standard glibc/malloc heap.
    pub fn new(capacity: usize) -> Self {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                // MAP_SHARED is critical: GPU sees these pages without copying
                libc::MAP_ANON | libc::MAP_SHARED,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed for UmaArena");

        // Hint to the kernel that these pages will be needed immediately
        unsafe { libc::madvise(ptr, capacity, libc::MADV_WILLNEED) };

        Self {
            base: ptr as *mut u8,
            capacity,
            offset: AtomicUsize::new(0),
        }
    }

    /// Raw allocation of n bytes with specified alignment.
    ///
    /// Returns `None` if the arena is full or alignment arithmetic overflows.
    /// Uses a bounded CAS retry loop (max 256 attempts) to avoid stack overflow
    /// under high contention.
    pub fn alloc_raw(&self, size: usize, align: usize) -> Option<*mut u8> {
        // Bounded retry loop instead of recursive CAS (prevents stack overflow)
        for _ in 0..256 {
            let current_offset = self.offset.load(Ordering::Acquire);

            // Checked alignment arithmetic to prevent overflow wrapping
            let aligned_offset = current_offset
                .checked_add(align - 1)
                .map(|v| v & !(align - 1))?;

            let new_offset = aligned_offset.checked_add(size)?;

            if new_offset > self.capacity {
                return None;
            }

            match self.offset.compare_exchange_weak(
                current_offset,
                new_offset,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(unsafe { self.base.add(aligned_offset) }),
                Err(_) => continue, // CAS failed, retry
            }
        }
        None // Exceeded retry limit under extreme contention
    }

    /// Lock-free bump allocation - O(1), no mutexes.
    #[inline(always)]
    pub fn alloc_aligned<T>(&self) -> Option<*mut T> {
        const CACHE_LINE: usize = 128;
        let size = std::mem::size_of::<T>();
        let align = CACHE_LINE.max(std::mem::align_of::<T>());

        self.alloc_raw(size, align).map(|p| p.cast::<T>())
    }

    /// Reset arena - total frame cleanup in O(1).
    #[inline(always)]
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
}

unsafe impl Send for UmaArena {}
unsafe impl Sync for UmaArena {}

impl Drop for UmaArena {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.capacity) };
    }
}

/// FrameAllocator for complex frame-bound structures.
pub struct FrameAllocator {
    pub bump: Bump,
    /// Pre-allocated pool for frequent allocations (e.g., particles, fragments)
    pub pool: UmaArena,
}

impl FrameAllocator {
    pub fn new(pool_capacity: usize) -> Self {
        Self {
            bump: Bump::new(),
            pool: UmaArena::new(pool_capacity),
        }
    }

    pub fn begin_frame(&mut self) {
        // Reset the entire frame in O(1)
        self.bump.reset();
        self.pool.reset();
    }
}
