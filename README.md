<div align="center">

# P A L A C E - X
### Topological Memory Engine for Autonomous AI Agents

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon-lightgrey.svg)](https://developer.apple.com/apple-silicon/)
[![SIMD: NEON / AVX-512](https://img.shields.io/badge/SIMD-NEON%20%2F%20AVX--512-green.svg)](#simd-backends)

---

**Palace-X** is a hardware-accelerated vector search engine that leverages **Algebraic Topology** to redefine semantic search. By replacing standard LLM-in-the-loop classification with topological distillation, it identifies structural connectivity that pure cosine similarity misses.

</div>

<br/>

## Benchmark Results (SIFT-128, 10K vectors)

All numbers measured on SIFT-10K dataset (128 dimensions, 10,000 base vectors, 100 queries).
Reproducible via `cargo run -p palace-bench --release`.

<!-- BENCHMARK_TABLE_START -->

| Method | R@1 | R@10 | R@100 | QPS | Memory/vec |
|--------|-----|------|-------|-----|------------|
| Brute-force L2 (baseline) | **100.0%** | **100.0%** | **100.0%** | 389 | D×4 B |
| RaBitQ 1-bit (brute) | 52.0% | 54.0% | 65.0% | 346 | D/8+16 B |
| RaBitQ 4-bit (brute) | 52.0% | 54.0% | 65.0% | 342 | D/2+16 B |
| Naive binary / Hamming (brute) | 26.0% | 15.5% | 15.2% | 3,667 | D/8 B |
| NSW L2 (ef=32) | 1.0% | 1.0% | 3.0% | 6,494 | D×4+graph |
| NSW L2 (ef=256) | 1.0% | 1.9% | 9.2% | 1,247 | D×4+graph |
| NSW + RaBitQ-1bit (ef=256) | 0.0% | 1.5% | 8.2% | 1,683 | D/8+16+graph |

> **Note:** NSW graph recall is currently limited by flat-graph construction quality
> (early nodes get poor neighborhoods). HNSW-style hierarchical build is planned for v0.2.
> The RaBitQ brute-force results demonstrate the quantization quality independent of graph issues.

```bash
# Reproduce these numbers
cargo run -p palace-bench --release -- --sift
```

<!-- BENCHMARK_TABLE_END -->

### Ablation: α/β Parameter Sweep

The topological reranking metric $d_{total}$ combines cosine distance with structural connectivity:

$$d_{total}(x, y) = \alpha \cdot d_{cosine}(x, y) + \beta \cdot \exp(-\beta_1(C_{xy}) / |E(C_{xy})|)$$

Run the ablation study to measure the effect of β₁ reranking vs pure cosine:

```bash
cargo test -p palace-topo --test ablation_study -- --nocapture
```

### Vamana α-Pruning Comparison

The neighbor selection heuristic uses α-RNG pruning (Vamana-style). Comparison of α values:

```bash
cargo test -p palace-graph --test recall_test test_alpha_pruning -- --nocapture
```

<br/>

## Why Topological Reranking?

Pure vector similarity often fails to capture the "contextual bridge" between concepts. Palace-X computes the **First Betti Number (β₁)** on 2-hop ego-graphs to measure structural connectivity. High cycle density indicates robust semantic relationships.

The `palace-topo` module now supports two topological metrics:

1. **Raw β₁** — number of independent cycles in the ego-graph (fast, simple)
2. **Persistent homology** — full H₀/H₁ persistence diagrams that capture multi-scale topological features. More nuanced than raw β₁ but computationally heavier.

<br/>

## Architecture

```
Query Vector (f32)
       │
       ▼
┌──────────────────┐
│  Stage 1: Coarse │  NSW graph search with Hamming distance
│  (SIMD-fast)     │  Entry via hub-highway cache
│  ef candidates   │  α-RNG pruned neighbor selection
└────────┬─────────┘
         │
┌────────▼──────────────┐
│  Stage 2: Topological  │  Ego-graph analysis (capped at 500 edges)
│  Reranking (optional)  │  β₁ or persistence diagram scoring
│  d_total metric        │  Parallel via Rayon + LRU cache
└────────┬──────────────┘
         │
         ▼
   Final Top-K Results
```

### Crates

| Crate | Description |
|-------|-------------|
| `palace-core` | Foundation types: `NodeId`, `Fragment`, `MetaData`, `SearchConfig` |
| `palace-quant` | RaBitQ (1-bit & 4-bit) with FHT rotation + SIMD Hamming |
| `palace-graph` | Hub-Highway NSW with Vamana α-pruning (`α=1.2` default) |
| `palace-topo` | Ego-graph analysis, β₁, persistent homology (H₀/H₁), d_total |
| `palace-bitplane` | IEEE 754 bit-plane disaggregation for progressive retrieval |
| `palace-storage` | Async `MemoryPalace` storage provider |
| `palace-engine` | High-level orchestration and async/await interface |
| `palace-bench` | SIFT-128 benchmarks with recall/QPS measurement |
| `palace-optimizer` | Silicon-native SIMD + threading (M1-M4 / AVX-512) |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` (max_neighbors) | 32 | NSW edges per node |
| `ef_construction` | 200 | Beam width during index build |
| `ef_search` | 64 | Beam width during search |
| `alpha` (pruning) | 1.2 | Vamana α-RNG relaxation factor |
| `α` (d_total) | 0.7 | Cosine distance weight |
| `β` (d_total) | 0.3 | Topological penalty weight |
| `max_ego_edges` | 500 | Hard cap on ego-graph edges (bounds O(K²)) |

<br/>

## Quick Start

```bash
# Clone and build
git clone https://github.com/Neirotunes/palace-x.git
cd palace-x
cargo build --release

# Run tests
cargo test --workspace

# Run benchmarks (downloads SIFT-10K automatically)
mkdir -p data
cargo run -p palace-bench --release

# Run ablation study
cargo test -p palace-topo --test ablation_study -- --nocapture

# Run α-pruning comparison
cargo test -p palace-graph --test recall_test test_alpha_pruning -- --nocapture
```

### Programmatic Usage

```rust
use palace_graph::{NswIndex, MetaData};
use palace_quant::rabitq::RaBitQIndex;

// Build index with α=1.2 pruning
let index = NswIndex::with_alpha(128, 32, 200, 1.2);

// Insert vectors
for (i, vec) in vectors.iter().enumerate() {
    index.insert(vec.clone(), MetaData { label: format!("{}", i) });
}
index.update_hub_scores();

// Search
let results = index.search(&query, Some(64)); // ef=64

// Optional: RaBitQ reranking for better recall
let rq = RaBitQIndex::with_centroid(128, centroid, 42);
let codes: Vec<_> = vectors.iter().map(|v| rq.encode_multibit(v, 4)).collect();
let rq_query = rq.encode_query(&query);
// Rerank candidates with 4-bit asymmetric distance...
```

<br/>

## Quantization Methods

| Method | Storage/vec | R@10 (SIFT-10K) | Description |
|--------|------------|-----------------|-------------|
| Naive binary | D/8 B | 15.5% | Sign-bit quantization + Hamming |
| RaBitQ 1-bit | D/8 + 16 B | 54.0% | Random rotation + scalar correction |
| RaBitQ 4-bit | D/2 + 16 B | 54.0%* | 4 bit-planes + asymmetric distance |
| Full FP32 | D×4 B | 100.0% | Brute-force baseline |

\* RaBitQ 4-bit currently uses only the MSB sign plane for distance estimation, so recall matches 1-bit. Multi-bit distance computation is planned for v0.2.

RaBitQ uses a Fast Hadamard Transform (FHT) for O(D log D) random rotation instead of O(D²) matrix multiplication. The asymmetric distance formula `est_ip = norm/(x0·√D) · ⟨x_bar, q'⟩` keeps the query unquantized for higher accuracy.

<br/>

## Persistent Homology

The `palace-topo::persistence` module upgrades the topological metric from raw β₁ to full persistence diagrams:

- **H₀ persistence**: Tracks connected component merges via weight-sorted Union-Find
- **H₁ persistence**: Detects cycle birth/death via flag complex triangle enumeration
- **Total persistence score**: Weighted sum of feature lifetimes, replaces β₁ in d_total

```rust
use palace_topo::persistence::{persistence_diagram, d_total_persistence};

let diagram = persistence_diagram(&ego_graph, &weight_fn);
println!("H₀ features: {}", diagram.count_dim(0));
println!("H₁ features: {}", diagram.count_dim(1));
println!("Total persistence: {}", diagram.total_persistence());

// Use in distance metric
let d = d_total_persistence(cosine_dist, &ego_graph, &weight_fn, 0.7, 0.3, 1.0);
```

<br/>

<div align="center">

### Built for the Future of Autonomous Systems
Copyright (c) 2026 M. Diach — Licensed under [AGPL-3.0-or-later](LICENSE)

*Optimized for Apple Silicon (M1-M4) and AVX-512 based hardware.*

</div>
