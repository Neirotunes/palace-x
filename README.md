# Palace-X

> Copyright (c) 2026 M.Diach — Licensed under [AGPL-3.0-or-later](LICENSE)

**High-performance hierarchical memory system for autonomous AI agents.**

Palace-X transforms the "memory palace" metaphor into a deterministic engineering construct.
It replaces LLM-in-the-loop classification with hardware-accelerated topological distillation and binary quantization, achieving 2-4x lower latency than Python-based RAG systems while maintaining structural awareness through Betti number reranking.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    palace-engine*                    │
│              (Async Actor Pipeline)                  │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │              palace-storage*                   │  │
│  │           (MemoryPalace impl)                 │  │
│  │                                               │  │
│  │  Stage 1: NSW Coarse Search ─────────────┐   │  │
│  │    palace-graph (Hub-Highway NSW)         │   │  │
│  │    palace-quant (Binary Hamming SIMD)     │   │  │
│  │                                           │   │  │
│  │  Stage 2: Topological Rerank ◄────────────┘   │  │
│  │    palace-topo (β₁ ego-graphs)                │  │
│  │    palace-bitplane* (Precision Fetch)          │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  palace-core (Types, Traits, Errors)                │
└─────────────────────────────────────────────────────┘

* Available in Palace-X Pro
```

## Crates

| Crate | Description |
|-------|-------------|
| `palace-core` | Foundation types: `NodeId`, `Fragment`, `MetaData`, `MemoryProvider` trait, error types |
| `palace-quant` | Binary quantization + SIMD Hamming distance (AVX-512 VPOPCNTDQ / NEON / scalar fallback) |
| `palace-graph` | Flat NSW index with Hub-Highway optimization (arXiv:2412.01940) |
| `palace-topo` | Topological reranking: ego-graph construction, β₁ via Union-Find, normalized d_total metric |

Additional modules (palace-storage, palace-engine, palace-bitplane) are available in [Palace-X Pro](https://github.com/dyachmx/palace-x-pro).

## The Math

Palace-X uses a two-stage retrieval pipeline:

**Stage 1 — Coarse Search** (5-15 ms on 10k docs):
Binary-quantized NSW search using Hamming distance via SIMD POPCNT.

**Stage 2 — Topological Rerank** (top-K candidates only):
For each candidate, build a 2-hop ego-graph and compute the normalized first Betti number:

```
d_total(x, y) = α · d_cosine(x, y) + β · exp(−β₁(C_xy) / max(|E(C_xy)|, 1))
```

Where:
- `β₁ = β₀ − χ = β₀ − V + E` (number of independent cycles, via Union-Find in O(V+E))
- `|E|` = edge count in the ego-graph (normalization for continuous spectrum)
- Higher cycle density → stronger structural connection → lower distance

**Graceful Degradation:**
- If mantissa planes unavailable (NVMe failure) → fall back to coarse search
- If ego-graph too large or corrupted → fall back to pure cosine ranking

## Building

```bash
# Debug build
cargo build

# Release build (LTO + native CPU)
cargo build --release

# Run all tests
cargo test --workspace
```

## SIMD Support

Palace-X auto-detects CPU features at runtime:

| Feature | Instruction | Platform | Throughput |
|---------|-------------|----------|------------|
| AVX-512 VPOPCNTDQ | `_mm512_popcnt_epi64` | Intel Ice Lake+ / Sapphire Rapids | ~200 GB/s |
| NEON | `vcntq_u8` | Apple M1-M4 / ARM servers | ~40 GB/s |
| Scalar fallback | `u64::count_ones()` | Any platform | ~1.5 GB/s |

Detection is cached in a `static AtomicU8` — zero overhead after first call.

## Performance Targets

| Operation | MemPalace Raw (Python) | Palace-X (Rust) |
|-----------|----------------------|-----------------|
| Search Latency (10k docs) | 20-50 ms | 5-15 ms |
| Hamming Throughput | ~1.5 GB/s (NumPy) | 40-200 GB/s (SIMD) |
| Memory per 1M nodes (384d) | ~6 GB (FP32) | ~1.2 GB (BitPlane) |

## References

- [Hub-Highway Hypothesis](https://arxiv.org/abs/2412.01940) — "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'"
- [TRACE: Bit-Plane Disaggregation](https://arxiv.org/abs/2509.03377) — Precision-proportional memory access
- [LongMemEval (ICLR 2025)](https://arxiv.org/abs/2410.10813) — Long-term memory benchmark
- [MemPalace](https://github.com/milla-jovovich/mempalace) — Original Python implementation

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) for details.
