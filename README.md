<p align="center">
  <strong>P A L A C E - X</strong><br>
  <em>Topological Memory Engine for Autonomous AI Agents</em>
</p>

<p align="center">
  <a href="https://github.com/Neirotunes/palace-x/actions"><img src="https://github.com/Neirotunes/palace-x/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/SIMD-NEON%20%7C%20AVX--512-green.svg" alt="SIMD">
  <img src="https://img.shields.io/badge/tests-106%20passing-brightgreen.svg" alt="Tests">
</p>

> Copyright (c) 2026 M.Diach — Licensed under [AGPL-3.0-or-later](LICENSE)

---

**Palace-X** is the first vector search engine that uses **algebraic topology** for result reranking. It replaces LLM-in-the-loop classification with hardware-accelerated topological distillation, computing Betti numbers on ego-graphs to measure structural connectivity between query and candidate nodes.

The engine combines three SOTA techniques in a single Rust pipeline:
- **RaBitQ** (SIGMOD 2024) — randomized binary quantization with error bounds
- **Hub-Highway NSW** (arXiv:2412.01940) — flat graph search via hub node acceleration
- **Topological reranking** — first Betti number β₁ as a continuous structural distance metric

## Key Results

| Metric | Value | Method |
|--------|-------|--------|
| Hamming throughput | **448M ops/s** | NEON `vcntq_u8` with 4-way unrolling |
| Cosine distance | **NEON FMA** | Fused single-pass dot/norm via `vfmaq_f32` |
| RaBitQ recall@10 | **45%** → **90%+** target | vs 2% naive sign-bit (same bit budget) |
| Storage | **D/8 + 16 bytes/vec** | 27x compression over FP32 |
| Quantization | **O(D log D)** | Fast Hadamard Transform rotation |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Query Pipeline                        │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Stage 1: Coarse Search                              │ │
│  │                                                     │ │
│  │   RaBitQ encode ──→ SIMD Hamming ──→ NSW traverse   │ │
│  │   (FHT rotation)    (NEON/AVX-512)   (Hub-Highway)  │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │ top-K candidates           │
│  ┌──────────────────────────▼──────────────────────────┐ │
│  │ Stage 2: Topological Rerank                         │ │
│  │                                                     │ │
│  │   NEON cosine ──→ ego-graph β₁ ──→ d_total score    │ │
│  │   (fused FMA)     (Union-Find)     (α·cos + β·topo) │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  palace-core ─── palace-quant ─── palace-graph ─── palace-topo  │
└──────────────────────────────────────────────────────────┘
```

## Crates

| Crate | What it does | Key innovation |
|-------|-------------|----------------|
| [`palace-core`](crates/palace-core) | Types, traits, error handling | `MemoryProvider` async trait |
| [`palace-quant`](crates/palace-quant) | Binary quantization + SIMD distance | **RaBitQ** with FHT rotation + NEON auto-dispatch |
| [`palace-graph`](crates/palace-graph) | NSW index + neighbor heuristic | Hub-Highway entry points + Vamana-style pruning |
| [`palace-topo`](crates/palace-topo) | Topological reranking | β₁ ego-graphs + Rayon parallel + LRU ego-cache |

> Storage, engine, bit-plane, and Silicon optimizer modules available in [Palace-X Pro](https://github.com/Neirotunes/palace-x-pro).

## The Math

**Reranking metric** — combines angular distance with topological structure:

```
d_total(x, y) = α · d_cosine(x, y) + β · exp(−β₁(C_xy) / |E(C_xy)|)
```

Where β₁ is the first Betti number (independent cycles) of the 2-hop ego-graph C_xy, computed in O(V+E) via Union-Find. High cycle density indicates robust structural connectivity — a signal invisible to pure cosine similarity.

**RaBitQ encoding** — provably accurate binary quantization:

```
encode(o):  r = o − c  →  ô = r/‖r‖  →  o' = Pᵀô  →  b = sign(o')
            ‖ store: b (D/8 bytes) + 4 scalar factors (16 bytes) ‖

query(q):   q' = Pᵀ(q − c)  →  4-bit scalar quantize  →  asymmetric IP
            ‖ distance ≈ ‖o−c‖² + ‖q−c‖² − 2·(‖o−c‖/x₀)·⟨x̄, q'⟩ ‖
```

P is a random orthogonal matrix via 3× Fast Hadamard Transform (O(D log D)).

## SIMD Backends

| Feature | Instruction | Platform | Throughput |
|---------|-------------|----------|------------|
| AVX-512 VPOPCNTDQ | `_mm512_popcnt_epi64` | Intel Ice Lake+ | ~200 GB/s |
| NEON | `vcntq_u8` + `vfmaq_f32` | Apple M1–M4 | ~40 GB/s |
| Scalar | `u64::count_ones()` | Any platform | ~1.5 GB/s |

Runtime detection cached in `static AtomicU8` — zero overhead after first call.

## Quick Start

```bash
cargo test --workspace          # 106 tests
cargo bench                     # micro-benchmarks

# Use in your project:
# Cargo.toml:
# [dependencies]
# palace-core  = { git = "https://github.com/Neirotunes/palace-x" }
# palace-quant = { git = "https://github.com/Neirotunes/palace-x" }
# palace-graph = { git = "https://github.com/Neirotunes/palace-x" }
# palace-topo  = { git = "https://github.com/Neirotunes/palace-x" }
```

## References

- **RaBitQ**: Gao & Long, *"RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound"*, SIGMOD 2024 — [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
- **Hub-Highway**: *"Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'"* — [arXiv:2412.01940](https://arxiv.org/abs/2412.01940)
- **TRACE**: *"Bit-Plane Disaggregation for Precision-Proportional Memory Access"* — [arXiv:2509.03377](https://arxiv.org/abs/2509.03377)
- **LongMemEval**: ICLR 2025 — [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) for full text.

Built by [M.Diach](https://github.com/Neirotunes) | [NOTICE](NOTICE)
