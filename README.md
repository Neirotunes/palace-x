<div align="center">

# P A L A C E - X
### Topological Vector Search Engine for Apple Silicon

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon-lightgrey.svg)](https://developer.apple.com/apple-silicon/)
[![SIMD: NEON / AVX-512](https://img.shields.io/badge/SIMD-NEON%20%2F%20AVX--512-green.svg)](#neon-simd-kernels)

---

**HNSW graph traversal + RaBitQ compressed distances + NEON SIMD + Metal GPU.**
**99.9% recall @ 5,700 QPS.** Purpose-built for M1/M2/M3 unified memory.

</div>

<br/>

## Benchmark Results (SIFT-128, 10K vectors, Apple M1)

All numbers measured on Apple M1 8GB, `--release` with LTO. Reproducible via `cargo run -p palace-bench --release`.

<!-- BENCHMARK_TABLE_START -->

#### HNSW Graph Search

| Method | R@1 | R@10 | QPS | Memory/vec |
|--------|-----|------|-----|------------|
| **HNSW L2 (ef=32)** | **100.0%** | **99.8%** | **5,696** | D×4+graph |
| HNSW L2 (ef=64) | 100.0% | 99.9% | 3,657 | D×4+graph |
| HNSW L2 (ef=128) | 100.0% | 100.0% | 2,324 | D×4+graph |
| HNSW L2 (ef=256) | 100.0% | 100.0% | 1,286 | D×4+graph |

#### RaBitQ Quantization (brute-force)

| Method | R@1 | R@10 | QPS | Memory/vec |
|--------|-----|------|-----|------------|
| Brute-force L2 | 100.0% | 100.0% | 735 | D×4 B |
| RaBitQ 1-bit | 51.0% | 54.0% | 2,589 | D/8+16 B |
| **RaBitQ 4-bit** | **70.0%** | **75.6%** | **852** | **D/2+16 B** |

#### NEON SIMD Kernels (dims=384)

| Kernel | Throughput | Latency |
|--------|-----------|---------|
| RaBitQ 4-bit estimate | **2.79M ops/s** | 358 ns |
| RaBitQ 1-bit estimate | **7.60M ops/s** | 131 ns |
| Hamming NEON vcntq_u8 | 195M ops/s | 5 ns |

#### Metal GPU Batch Distance (dims=384)

| Candidates | GPU (μs) | CPU (μs) | Speedup |
|------------|----------|----------|---------|
| 256 | 596 | 106 | 0.2× |
| 4,000 | 2,267 | 1,685 | 0.7× |
| 16,000 | 6,699 | 6,877 | **1.0×** |
| 64,000 | 19,476 | 26,839 | **1.4×** |

#### UMA Cache-Aware HNSW (dims=384, 5K vectors)

| Method | QPS | Latency |
|--------|-----|---------|
| Standard HNSW | 788 | 1.27ms |
| **UMA Prefetch HNSW** | **1,008** | **992μs** |

> **Speedup: 1.28×** with ARM64 `prfm pldl1keep` speculative prefetch. 100% recall parity.

<!-- BENCHMARK_TABLE_END -->

<br/>

## Architecture

```
Query Vector (f32)
       │
       ▼
┌──────────────────────┐
│  Phase 1: HNSW       │  Greedy descent through upper layers
│  Greedy Descent      │  Float L2, O(log N) nodes
└────────┬─────────────┘
         │
┌────────▼─────────────┐
│  Phase 2: RaBitQ     │  Layer-0 beam search with compressed distances
│  Beam Search         │  4-bit, NEON vtstq_u32 branchless masking
└────────┬─────────────┘
         │
┌────────▼─────────────┐
│  Phase 3: Topo       │  β₁ Betti number on 2-hop ego-graphs
│  Reranking (opt.)    │  Penalizes hubs, rewards local clusters
└────────┬─────────────┘
         │
         ▼
   Final Top-K Results
   99.9% R@10, sub-ms
```

### HNSW+RaBitQ Combined Pipeline (v0.3)

The `HnswRaBitQ` index combines graph traversal quality with memory efficiency:

- **Phase 1** — HNSW greedy descent through upper layers (float L2, O(log N) nodes)
- **Phase 2** — Layer-0 beam search using RaBitQ 4-bit estimated distances (8× cheaper per eval)
- **Phase 3** — Optional float L2 rerank of top-k for maximum precision

```rust
use palace_storage::{HnswRaBitQ, HnswRaBitQConfig};

let config = HnswRaBitQConfig {
    dimensions: 128,
    max_neighbors: 16,
    ef_construction: 200,
    rabitq_bits: 4,
    rerank_top: 50, // float rerank top 50
    ..Default::default()
};

let index = HnswRaBitQ::new(config);
for vec in vectors {
    index.insert(vec, metadata);
}
index.publish_snapshot();

let results = index.search(&query, 10);
```

### Crates

| Crate | Description |
|-------|-------------|
| `palace-core` | Foundation types: `NodeId`, `Fragment`, `MetaData`, `SearchConfig` |
| `palace-quant` | RaBitQ (1-bit & 4-bit), NEON vtstq_u32 kernels, SIMD Hamming |
| `palace-graph` | HNSW with α-RNG pruning (α=1.2), UMA hot/cold tier, ArcSwap |
| `palace-topo` | Ego-graph β₁, persistent homology (H₀/H₁), d_total metric |
| `palace-bitplane` | IEEE 754 bit-plane disaggregation for progressive retrieval |
| `palace-storage` | MemoryPalace + **HnswRaBitQ combined pipeline** |
| `palace-engine` | Async actor pipeline, batch ingestion, graceful shutdown |
| `palace-bench` | Full benchmark suite with SIFT-128 recall/QPS tables |
| `palace-optimizer` | Metal GPU batch, UMA arena, thermal guard, NEON SIMD |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | HNSW edges per node (M_max0 = 2M at layer 0) |
| `ef_construction` | 200 | Beam width during index build |
| `ef_search` | 64 | Beam width during search (tunable) |
| `alpha` | 1.2 | Vamana α-RNG pruning relaxation factor |
| `rabitq_bits` | 4 | RaBitQ bit depth (1, 2, 4, or 7) |
| `rerank_top` | 0 | Float rerank candidates (0 = disabled) |

<br/>

## Quick Start

```bash
# Clone and build
git clone https://github.com/Neirotunes/palace-x.git
cd palace-x
cargo build --release

# Run full benchmark suite
cargo run -p palace-bench --release

# Run tests
cargo test --workspace

# SIFT-only benchmarks
cargo run -p palace-bench --release -- --sift
```

### Programmatic Usage

```rust
use palace_graph::{HnswIndex, MetaData};

// Build HNSW index (M=16, ef_c=200, L2 metric)
let index = HnswIndex::new(128, 16, 200);

// Insert vectors
for (i, vec) in vectors.iter().enumerate() {
    index.insert(vec.clone(), MetaData { label: format!("{}", i) });
}
index.publish_snapshot(); // Required after batch insertion

// Search — 99.8% R@10 at ef=32
let results = index.search(&query, Some(32));
```

<br/>

## RaBitQ Multi-Bit Quantization

RaBitQ (SIGMOD 2024) applies a random orthogonal rotation before quantization, preserving inner products while enabling SIMD-friendly binary operations.

**v0.2**: 4-bit weighted bit-plane inner product with NEON acceleration:

```
⟨x_recon, q'⟩ = (1/half_max) · [Σ_k 2^k · plane_ip_k - half_max · Σ q'_i]
```

**v0.3**: NEON `vtstq_u32` branchless masked sums — 7.5× speedup over scalar. Processes 8 f32 values per cycle with dual accumulator pipeline saturation.

| Method | Storage/vec | R@10 | Description |
|--------|------------|------|-------------|
| Naive binary | D/8 B | 15.5% | Sign-bit + Hamming |
| RaBitQ 1-bit | D/8+16 B | 54.0% | Random rotation + scalar correction |
| **RaBitQ 4-bit** | **D/2+16 B** | **75.6%** | 4 bit-planes + NEON weighted popcount |
| Full FP32 | D×4 B | 100.0% | Brute-force baseline |

<br/>

## Apple Silicon Optimizations

**UMA Cache-Aware HNSW** — Upper layers packed into contiguous `Vec<f32>`, fitting in L2/SLC cache. Layer 0 stays in DashMap for concurrent writes. ARM64 `prfm pldl1keep` issued 1 hop ahead.

**Metal GPU Batch Distance** — Compute shaders for L2/cosine with UMA zero-copy (`StorageModeShared`). Auto-dispatches for ≥256 candidates.

**NEON SIMD** — `vtstq_u32` for branchless 4-bit-to-mask expansion in RaBitQ. `vcntq_u8` for Hamming popcount. Dual accumulator for M1 pipeline saturation.

<br/>

## Commercial Edition

Palace-X is dual-licensed. The community edition (this repo) is **AGPL-3.0**. A commercial license is available for proprietary deployments.

| | Community | Commercial |
|--|-----------|-----------|
| HNSW (99.9% R@10) | ✓ | ✓ |
| RaBitQ 1-bit & 4-bit | ✓ | ✓ |
| NEON SIMD kernels | ✓ | ✓ |
| Topological reranking | ✓ | ✓ |
| Metal GPU batch | — | ✓ |
| UMA cache-aware HNSW | — | ✓ |
| HNSW+RaBitQ pipeline | — | ✓ |
| Thermal scheduling | — | ✓ |
| Priority support | — | ✓ |

**Contact:** [max@neirosynth.com](mailto:max@neirosynth.com?subject=Palace-X%20Commercial%20License)

<br/>

<div align="center">

Copyright (c) 2026 Maksym Dyachenko — Licensed under [AGPL-3.0-or-later](LICENSE)

*Optimized for Apple Silicon (M1-M4) and AVX-512.*

</div>
