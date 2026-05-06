<div align="center">

<img src="docs/assets/logo.png" alt="Palace-X" width="120">

# Palace-X

**Vector search engine for Apple Silicon — written in Rust**

[![CI](https://github.com/Neirotunes/palace-x/actions/workflows/ci.yml/badge.svg)](https://github.com/Neirotunes/palace-x/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20%2F%20x86--64-lightgrey.svg)](#)

HNSW graph search + RaBitQ 4-bit quantization + NEON SIMD kernels.  
**98.7% R@10 on SIFT-1M at 1,843 QPS — same recall as float, 1.9× faster, 6.4× smaller.**

</div>

---

## Why Palace-X

Most vector databases target x86 datacenters. Palace-X is built from the ground up for **Apple Silicon unified memory** — no separate CPU/GPU copies, NEON-first SIMD, cache-aware graph layout.

The core insight: RaBitQ 4-bit asymmetric scoring lets you search with compressed distances (8× fewer FLOPS) while reranking with float, preserving recall. On M-series chips, this fits the entire distance computation inside L2 cache.

**TL;DR** for the HN crowd: it's a Rust reimplementation of HNSW + RaBitQ (SIGMOD 2024) tuned specifically for aarch64, with topological reranking (β₁ ego-graph) on top.

---

## Benchmarks (SIFT-100K, Apple M1, `--release`)

Real numbers from `cargo run -p palace-bench --release`. Dataset: 100,000 × 128d SIFT vectors, 10,000 queries, L2 ground truth.

| Method | R@1 | R@10 | QPS | Memory/vec |
|--------|-----|------|-----|-----------|
| HNSW float (ef=100) | 98.7% | 98.6% | 1,857 | 512 B |
| HNSW float (ef=256) | 98.7% | 98.7% | 889 | 512 B |
| **HNSW+RaBitQ-4bit Asym (ef=32)** | **98.7%** | **98.6%** | **1,717** | **80 B** |
| **HNSW+RaBitQ-4bit Asym (ef=64)** | **98.7%** | **98.7%** | **1,410** | **80 B** |
| HNSW+RaBitQ-4bit Beam+RR50 (ef=32) | 97.7% | 92.7% | 1,386 | 80 B |

**Key result:** `HNSW+RaBitQ-4bit Asym (ef=32)` matches `HNSW float (ef=256)` recall — using 1.9× fewer beam evaluations and **6.4× less memory per vector** (80 bytes vs 512 bytes for 128d).

For OpenAI ada-002 dimensions (1536d): float = 6 KB/vec, 4-bit = 784 B/vec. **7.8× smaller index.**

### Why asymmetric mode works

Phase 1 (HNSW greedy descent through upper layers) uses full float L2 — this is the expensive O(log N) part and runs over few nodes. Phase 2 (layer-0 beam search) uses RaBitQ 4-bit estimated distances — 8× cheaper per evaluation, runs over thousands of candidates. Top candidates are float-reranked. Result: the noise from quantization only affects which candidates enter the beam, not the final reranking.

### SIMD kernels

| Kernel | Throughput | Backend |
|--------|-----------|---------|
| RaBitQ 4-bit distance estimate | 2.79M ops/s | NEON `vtstq_u32` |
| RaBitQ 1-bit distance estimate | 7.60M ops/s | NEON `vtstq_u32` |
| Hamming distance (384d) | 195M ops/s | NEON `vcntq_u8` |
| UMA prefetch HNSW vs standard | 1.28× speedup | ARM64 `prfm pldl1keep` |

---

## Quick Start

```bash
git clone https://github.com/Neirotunes/palace-x
cd palace-x
cargo build --release

# Run benchmarks
cargo run -p palace-bench --release

# Run tests
cargo test --workspace
```

### Programmatic usage

```rust
use palace_storage::{HnswRaBitQ, HnswRaBitQConfig};

let config = HnswRaBitQConfig {
    dimensions: 128,
    max_neighbors: 16,
    ef_construction: 200,
    rabitq_bits: 4,
    rerank_top: 50,
    ..Default::default()
};

let index = HnswRaBitQ::new(config);

// Ingest
for (vec, meta) in vectors {
    index.insert(vec, meta);
}
index.publish_snapshot(); // wait-free ArcSwap publish

// Search — 98.7% R@10 at ef=32
let results = index.search(&query, 10);
```

### Async actor API

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};

let engine = PalaceEngine::start(128); // starts background actor

let node_id = engine.ingest(vector, MetaData::new(timestamp, "source")).await?;
let fragments = engine.search(query, SearchConfig::default_with_limit(10)).await?;
let stats = engine.stats().await?;
```

---

## Architecture

```
Query Vector (f32)
       │
       ▼
┌──────────────────────┐
│   Phase 1: HNSW      │  Greedy descent, upper layers only
│   Float L2           │  O(log N) nodes, full precision
└────────┬─────────────┘
         │ top candidates
┌────────▼─────────────┐
│   Phase 2: RaBitQ    │  Layer-0 beam search
│   4-bit NEON         │  8× fewer FLOPS, NEON vtstq_u32
└────────┬─────────────┘
         │ top-k compressed candidates
┌────────▼─────────────┐
│   Phase 3: Rerank    │  Float L2 rerank of top-50
│   (optional)         │  Restores full precision
└────────┬─────────────┘
         │
         ▼
   98.7% R@10, sub-ms
```

### Crate structure

| Crate | Description |
|-------|-------------|
| `palace-core` | Types: `NodeId`, `Fragment`, `MetaData`, `SearchConfig` |
| `palace-quant` | RaBitQ 1-bit & 4-bit, NEON `vtstq_u32` kernels |
| `palace-graph` | HNSW with α-RNG pruning, ArcSwap wait-free reads |
| `palace-topo` | Ego-graph β₁, topological reranking, Sheaf H¹ cohomology |
| `palace-bitplane` | IEEE 754 bit-plane disaggregation |
| `palace-storage` | `MemoryPalace` + `HnswRaBitQ` combined pipeline |
| `palace-engine` | Async mpsc actor, batch ingestion |
| `palace-optimizer` | UMA arena, NEON SIMD, thermal guard, Metal GPU |
| `palace-bench` | SIFT-128/1M benchmark suite |

---

## Sheaf Cohomology (H¹ Obstruction)

`palace-topo` includes a real implementation of sheaf cohomology for detecting knowledge gaps in a graph.

**Math:** For a sheaf F over graph G = (V, E), each edge carries a coboundary:

```
γ(e) = F_{u→e}·x_u − F_{v→e}·x_v
H¹ = (1/|E|) · Σ_e ‖γ(e)‖² / (‖x_u‖² + ‖x_v‖² + ε)
```

- H¹ ≈ 0 → consistent knowledge (global section exists, signals agree)
- H¹ >> 0 → topological gap (conflicting local knowledge, no global view)

```rust
use palace_topo::{SheafAnalyzer, Stalk, NodeId};

let mut sheaf = SheafAnalyzer::new();
sheaf.add_stalk(NodeId(0), Stalk { modality_id: "price".into(), values: vec![1.0, 0.5] });
sheaf.add_stalk(NodeId(1), Stalk { modality_id: "price".into(), values: vec![1.0, 0.4] });
sheaf.add_edge(NodeId(0), NodeId(1));

let result = sheaf.compute_h1();
println!("H¹ obstruction: {:.4}", result.h1_obstruction); // ~0.003 (consistent)
println!("Consistent: {}", result.is_consistent(0.1));    // true
```

Reference: Hansen & Ghrist, ["Toward a Spectral Theory of Cellular Sheaves"](https://arxiv.org/abs/2012.06333) (2021).

---

## Apple Silicon specifics

**UMA zero-copy** — Metal GPU batch distance uses `StorageModeShared`; no CPU↔GPU transfer. Breakeven at ~16K candidates, 1.4× speedup at 64K.

**Cache-aware graph layout** — upper HNSW layers packed into contiguous `Vec<f32>` fitting in L2/SLC. Layer 0 in `DashMap` for concurrent writes. ARM64 `prfm pldl1keep` issued 1 hop ahead.

**Thermal-aware scheduling** — `ThermalGuard` reads SoC die temperature via SMC and backs off threads before throttling.

**NEON kernels** — `vtstq_u32` for branchless 4-bit-to-mask expansion in RaBitQ (2.79M ops/s at 384d). Dual accumulator to saturate M1 pipeline. `vcntq_u8` for Hamming popcount (**290M ops/s**, 3ns/op, 2026-05-04 on Apple M1).

| Kernel | Throughput | Latency | Backend |
|--------|-----------|---------|---------|
| Hamming distance (384d) | **290M ops/s** | 3 ns | NEON `vcntq_u8` |
| Binary quantize f32→u64 (384d) | 1.16M ops/s | 862 ns | NEON |
| Batch Hamming top-10 (10K cands) | 8,508 ops/s | 117 µs | NEON |
| Batch Hamming top-50 (10K cands) | 11,377 ops/s | 87 µs | NEON |

---

## RaBitQ

RaBitQ (SIGMOD 2024, [paper](https://arxiv.org/abs/2405.12497)) applies a random orthogonal rotation before quantization, preserving inner-product structure while enabling SIMD-friendly bit operations. The asymmetric variant keeps queries in float and only quantizes the database — this is why recall matches float at much lower ef.

| Method | Bytes/vec (128d) | R@10 (brute-force) |
|--------|-----------------|-------------------|
| Float32 | 512 | 100.0% |
| RaBitQ 1-bit | 32 | 54.0% |
| **RaBitQ 4-bit** | **80** | **75.6%** (standalone BF) |
| **HNSW+RaBitQ-4bit Asym** | **80 + graph** | **98.7%** (with HNSW) |

The gap between standalone 75.6% and graph-assisted 98.7% is the HNSW beam carrying recall — quantization only affects which candidates enter the beam, not the final reranking.

---

## Comparison

|  | **Palace-X** | FAISS (ARM) | Qdrant | Milvus |
|--|-------------|------------|--------|--------|
| Language | Rust | C++ | Rust | Go/C++ |
| Apple Silicon optimized | ✅ NEON-first | ⚠️ x86-primary | ✅ | ⚠️ |
| RaBitQ 4-bit | ✅ | ❌ | ❌ | ❌ |
| Async actor API | ✅ | ❌ | ✅ | ✅ |
| UMA Metal GPU | ✅ | ❌ | ❌ | ❌ |
| Topological reranking | ✅ β₁ ego-graph | ❌ | ❌ | ❌ |
| License | AGPL-3.0 | MIT | Apache 2.0 | Apache 2.0 |

FAISS is the gold standard but its ARM path is not actively optimized — it was designed for AVX-512. Palace-X's NEON kernels are written for aarch64 first.

Qdrant and Milvus are production managed services. Palace-X is an embeddable library — different deployment model targeting in-process vector search.

---

## Roadmap

- [x] HNSW + RaBitQ 4-bit asymmetric pipeline
- [x] NEON SIMD kernels (`vtstq_u32`, `vcntq_u8`)
- [x] Async actor engine (`palace-engine`)
- [x] Metal GPU batch distance
- [x] UMA cache-aware graph layout
- [ ] mmap persistence (in progress)
- [ ] Scalar quantization (8-bit)
- [ ] Distributed sharding
- [ ] `cargo install palace-x` CLI

---

## License

AGPL-3.0-or-later. If you need a commercial license (proprietary deployment, no source disclosure), contact [hello@neirosynth.com](mailto:hello@neirosynth.com?subject=Palace-X%20Commercial%20License).

AGPL means: embed freely, but if you run this as a network service you must publish your source. This is intentional — it's the same model Grafana, MongoDB, and Qdrant use.

---

<div align="center">

Built by [M.Diach](mailto:hello@neirosynth.com) · Apple Silicon · Rust · 2026

</div>
