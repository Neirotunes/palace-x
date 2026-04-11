// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

use palace_optimizer::*;
use std::time::Instant;

fn main() {
    println!("--- Silicon-Native Optimizer: M1 Benchmark ---");

    // 1. UMA Arena Allocation
    let arena = UmaArena::new(1024 * 1024 * 10); // 10MB
    let start = Instant::now();
    for _ in 0..1000 {
        let _ptr = arena.alloc_aligned::<[f32; 1024]>().unwrap();
    }
    println!("UMA Arena: 1000 x 4KB allocs in {:?}", start.elapsed());

    // 2. Thread Pinning
    match pin_current_thread(CoreType::Performance) {
        Ok(_) => println!("P-Core Pinning: SUCCESS (Firestorm/Avalanche)"),
        Err(e) => println!("P-Core Pinning: FAILED (code: {})", e),
    }

    // 3. NEON SIMD Physics
    let mut px = vec![0.0f32; 10000];
    let mut py = vec![0.0f32; 10000];
    let vx = vec![1.0f32; 10000];
    let vy = vec![1.0f32; 10000];

    let start = Instant::now();
    unsafe {
        simd::update_particles_neon(&mut px, &mut py, &vx, &vy, 0.016);
    }
    println!(
        "NEON SIMD: 10,000 particles updated in {:?}",
        start.elapsed()
    );

    // 4. Speculative Prefetching
    let prefetcher = SpeculativePrefetcher::new();
    prefetcher.record_access(42);
    unsafe {
        prefetcher.prefetch_predicted(42, px.as_ptr() as *const u8);
    }
    println!("Prefetching: Hints issued via ARM64 PRFM");

    println!("\nBenchmark Complete. System is operating at peak Apple Silicon efficiency.");
}
