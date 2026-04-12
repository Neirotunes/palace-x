# Palace-X v0.2 — Engineering Blueprint

**Codename: HIERARCHY**
**Target: SOTA 2027 / DARPA TRL-5 / Edge AI Vector Search**
**Author: M. Diach / Neirotunes**
**Date: April 2026**

---

## Mission Statement

Transform Palace-X from a working proof-of-concept (TRL 3-4) into a competitive vector search engine (TRL 5-6) that demonstrates:

1. **90%+ Recall@10** on SIFT-1M with sub-millisecond latency
2. **UMA-native memory hierarchy** that outperforms generic HNSW on Apple Silicon
3. **Topological reranking that provably improves** recall on approximate graphs
4. **Metal GPU batch search** demonstrating CPU-GPU zero-copy advantage

---

## Current State (v0.1 — measured)

| Component | Status | Key Metric |
|-----------|--------|------------|
| RaBitQ 1-bit | Working | 54% R@10 SIFT-10K (brute) |
| RaBitQ 4-bit | Partial | Same as 1-bit (MSB only) |
| NSW graph | Broken | ~1% R@10 (flat construction) |
| Topological reranking | Works but untestable | Degrades on perfect kNN graph |
| SiliconOptimizer | 4/6 modules complete | UMA, P-core, NEON, Prefetch done |
| CI/CD | Complete | Recall gate, coverage, release |

**Critical blocker:** NSW graph quality prevents testing the full pipeline.

---

## v0.2 Deliverables (ordered by dependency)

### D1: UMA-Native HNSW (Priority: CRITICAL)

**Goal:** 90%+ Recall@10 on SIFT-1M

**Why UMA-native, not generic HNSW:**
Standard HNSW treats all layers equally. On Apple Silicon with Unified Memory Architecture, we can exploit the cache hierarchy:

- **Layer 0 (dense, all nodes):** Main memory (DRAM), accessed via prefetcher
- **Layers 1+ (sparse, navigational):** Hot in L2/SLC cache via UmaArena priority allocation
- **Entry point:** Pinned in L1 via `prfm pldl1keep`

This means the Speculative Prefetcher can work **2 levels ahead** — while CPU traverses layer L, prefetcher warms layer L-1 into cache.

**Implementation plan:**

```
crates/palace-graph/src/hnsw.rs (NEW)
```

**Struct:**
```rust
pub struct HnswIndex {
    layers: Vec<HnswLayer>,          // layer[0] = dense, layer[max] = sparse
    entry_point: AtomicU64,           // top-layer node
    max_level: usize,                 // ln(N) / ln(M)
    level_mult: f64,                  // 1/ln(M), typically ~0.36
    
    // UMA-aware allocation
    arena_hot: UmaArena,              // Small arena for layers 1+ (fits in SLC)
    arena_cold: UmaArena,             // Large arena for layer 0
    
    // Existing
    nodes: DashMap<NodeId, HnswNode>,
    metric: DistanceMetric,
    ef_construction: usize,
    max_neighbors_per_layer: Vec<usize>,  // M for layer 0, M/2 for layers 1+
}

pub struct HnswNode {
    vector: Vec<f32>,
    level: usize,                     // max layer this node appears in
    neighbors: Vec<Vec<NodeId>>,      // neighbors[layer] = neighbor list
    binary: Vec<u64>,                 // RaBitQ code for coarse filter
    metadata: MetaData,
}

pub struct HnswLayer {
    nodes: Vec<NodeId>,               // sorted for cache locality
    arena_ptr: *const u8,             // UMA arena region for this layer
}
```

**Algorithm:**
```
INSERT(v, metadata):
  1. Assign level = floor(-ln(uniform()) × level_mult)
  2. Start from entry_point at top layer
  3. For each layer from top to (level+1):
     - Greedy search for 1 nearest neighbor (navigational)
     - Prefetch next layer: prfm pldl1keep(arena_hot[layer-1])
  4. For each layer from min(level, max_level) to 0:
     - Beam search with ef_construction candidates
     - Select neighbors via α-RNG heuristic (α=1.2)
     - Bidirectional links
  5. If level > max_level: update entry_point
  6. Publish snapshot for wait-free reads

SEARCH(query, ef):
  1. Start from entry_point at top layer
  2. Greedy descent through sparse layers (1 neighbor per step)
     - Prefetch layer below at each step
  3. At layer 0: beam search with ef candidates
  4. Return top-K sorted by distance
```

**Key decisions:**
- Level distribution: `P(level=l) = (1/M)^l` — standard Malkov/Yashunin
- M_max0 = 2×M (layer 0 gets double connections)
- Use DistanceMetric::L2 for SIFT, Cosine for embeddings
- Preserve `search_binary()` path for Hamming pre-filter

**Validation targets:**
| Dataset | R@1 | R@10 | R@100 | QPS |
|---------|-----|------|-------|-----|
| SIFT-10K | >95% | >98% | 100% | >5,000 |
| SIFT-1M | >85% | >92% | >98% | >1,000 |

---

### D2: 4-Bit Multi-Plane Distance (Priority: HIGH)

**Goal:** 75-85% R@10 brute-force (up from 54%)

**Current bug:** `estimate_distance()` extracts only the MSB sign plane from 4-bit codes. The 3 lower bit-planes are stored but ignored.

**Fix in `rabitq.rs`:**

```rust
// CURRENT (broken): uses only MSB plane
let msb_plane_start = (code.bits as usize - 1) * words_per_plane;
let sign_words = &code.binary[msb_plane_start..msb_plane_start + words_per_plane];

// FIXED: use all 4 planes for weighted inner product
// Each plane represents a binary digit: val = Σ bit_k × 2^k
// Reconstruct approximate value per dimension, compute proper inner product
```

**Two approaches (evaluate both):**

**A. Weighted bit-plane summation (simple):**
```
For each dimension i:
  approx[i] = Σ_{k=0}^{bits-1} bit_k(i) × 2^k  →  maps to [0, 15]
  rescale to [-1, 1]: x_approx[i] = (approx[i] / 7.5) - 1.0
  
est_ip = norm / (x0_4bit × √D) × ⟨x_approx, q'⟩
```

**B. Per-plane asymmetric (faster, SIMD-friendly):**
```
For each plane k (0..3):
  ip_k = popcount-based ⟨sign_plane_k, q'⟩
  
est_ip = norm / (x0 × √D) × Σ_k (2^k × ip_k)
```

Approach B is better — stays in integer domain, no per-dimension loop, pure popcount.

**Implementation:**
```rust
pub fn estimate_distance_multibit(&self, query: &RaBitQQuery, code: &RaBitQCode) -> (f32, f32) {
    let d = self.dim as f32;
    let words_per_plane = (self.dim + 63) / 64;
    
    let mut weighted_ip: f32 = 0.0;
    for k in 0..code.bits as usize {
        let plane_start = k * words_per_plane;
        let plane = &code.binary[plane_start..plane_start + words_per_plane];
        
        // Compute ⟨sign_plane_k, q'⟩ via popcount
        let ip_k = asymmetric_inner_product(plane, &query.rotated_vector);
        weighted_ip += (1 << k) as f32 * ip_k;
    }
    
    // Scale factor accounts for [0, 2^bits-1] quantization range
    let scale = 1.0 / ((1 << code.bits) as f32 - 1.0);
    let est_inner_product = code.factors.norm / (code.factors.x0 * d.sqrt()) 
        * weighted_ip * scale;
    
    let est_dist = code.factors.sqr_norm + query.sqr_y - 2.0 * est_inner_product;
    let lower_bound = est_dist - code.factors.error_bound * query.q_norm;
    (est_dist, lower_bound)
}
```

**Also fix x0 computation for 4-bit:** Currently x0 is computed from sign bits only. For 4-bit, x0 should measure quantization quality against the full 4-bit reconstruction.

**Validation:** R@10 should jump from 54% to 75-85% on SIFT-10K.

---

### D3: SIFT-1M Benchmark (Priority: HIGH)

**Goal:** Prove scaling advantage — memory bandwidth savings at scale

**Why:** At 10K vectors, CPU spends more time on dispatch than computation. At 1M, data exceeds L3/SLC (32MB on M1 Pro), and memory bandwidth becomes the bottleneck. This is where 32× compression (float32 → 1-bit) gives explosive QPS gains.

**Implementation:**
```
crates/palace-bench/src/sift_bench.rs — extend to handle 1M
```

**Dataset:** fvecs format from http://corpus-texmex.irisa.fr/
- `sift_base.fvecs` — 1,000,000 × 128d base vectors (~488 MB)
- `sift_query.fvecs` — 10,000 queries
- `sift_groundtruth.ivecs` — ground truth for all queries

**Benchmark matrix:**
| Method | Expected R@10 | Expected QPS | Memory |
|--------|---------------|-------------|--------|
| HNSW float32 | 95%+ | 2,000-5,000 | ~580 MB |
| HNSW + RaBitQ 1-bit rerank | 85%+ | 5,000-15,000 | ~20 MB + graph |
| HNSW + RaBitQ 4-bit rerank | 90%+ | 3,000-10,000 | ~80 MB + graph |
| Brute-force float32 | 100% | 50-100 | 488 MB |

**Key metric for investors:** QPS/$ on M1 vs. GPU-based solutions (Milvus on A100).

---

### D4: Metal GPU Batch Search (Priority: MEDIUM)

**Goal:** Demonstrate UMA advantage — CPU builds graph, GPU sieves candidates

**Architecture:**
```
CPU (P-cores):
  HNSW graph traversal → produces candidate set (100-1000 IDs)

GPU (Metal Compute):
  Load candidate vectors from SAME PHYSICAL MEMORY (UMA, no copy!)
  Parallel RaBitQ distance computation
  Return top-K with distances

CPU:
  Optional topological reranking on top-K
```

**Why this matters:** On discrete GPU systems (NVIDIA), candidate vectors must be copied CPU→GPU. On Apple Silicon, the UmaArena mmap(MAP_SHARED) memory is **already accessible to both CPU and GPU**. Zero-copy = zero latency overhead for the handoff.

**Implementation:**
```
crates/palace-optimizer/src/metal_search.rs (NEW)

pub struct MetalBatchSearcher {
    device: metal::Device,
    queue: metal::CommandQueue,
    rabitq_pipeline: metal::ComputePipelineState,
    // Shared buffer pointing into UmaArena
    vector_buffer: metal::Buffer,  // MTLBufferFromExistingPointer
}

impl MetalBatchSearcher {
    pub fn batch_rabitq_distance(
        &self, 
        query: &RaBitQQuery,
        candidate_codes: &[RaBitQCode],
    ) -> Vec<(usize, f32)>;
}
```

**Metal shader (MSL):**
```metal
kernel void rabitq_distance(
    device const uint64_t* codes [[buffer(0)]],
    device const float* query_rotated [[buffer(1)]],
    device float* distances [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread computes distance for one candidate
    // popcount via popcount() intrinsic
    // Asymmetric inner product accumulation
}
```

**Validation:** Batch of 1000 candidates should be faster on GPU than sequential CPU for D≥128.

---

### D5: Topological Reranking Validation (Priority: MEDIUM)

**Goal:** Prove β₁ reranking improves recall on approximate graphs

**Current problem:** On perfect kNN graph, topo reranking hurts (100% → 70%). This is expected — it can only help when the coarse search makes mistakes.

**Test protocol (requires D1 first):**
```
1. Build HNSW index on SIFT-1M
2. Search with ef=32 (intentionally low → ~80% R@10)
3. Rerank top-100 candidates with d_total (α=0.8, β=0.2)
4. Measure R@10 after reranking

Hypothesis: R@10 should improve by 3-8% because β₁ detects
structurally-connected nodes that cosine alone missed.
```

**If hypothesis fails:** Topological reranking may only help on domain-specific graphs (knowledge graphs, citation networks) where structural connectivity carries semantic meaning. Document honestly.

**Parameter sweep:**
- α ∈ {0.7, 0.8, 0.9, 0.95}
- β ∈ {0.05, 0.1, 0.2, 0.3}
- ef_search ∈ {16, 32, 64, 128} (controls how "approximate" the coarse search is)

---

## Integration Architecture (v0.2 target state)

```
Query Vector (f32)
       │
       ▼
┌──────────────────────────────┐
│  HNSW: Hierarchical Descent  │  Layers 2+: L2/SLC cache (UmaArena hot)
│  (UMA cache-aware)           │  Layer 0: Main memory (UmaArena cold)
│  Prefetcher: 2 levels ahead  │  SpeculativePrefetcher → prfm pldl1keep
│  ef candidates at layer 0    │
└────────────┬─────────────────┘
             │ candidate IDs (100-1000)
             ▼
┌──────────────────────────────┐
│  RaBitQ Reranking            │  CPU path: sequential asymmetric distance
│  (4-bit multi-plane)         │  GPU path: Metal batch (UMA zero-copy)
│  est_ip = Σ 2^k · ip_k      │  Choose based on batch size threshold
└────────────┬─────────────────┘
             │ top-K (10-100)
             ▼
┌──────────────────────────────┐
│  Topological Reranking       │  Ego-graph extraction (1-hop, capped 500)
│  (optional, β > 0)          │  β₁ via Union-Find + Euler
│  d_total = α·d + β·f(β₁)   │  Rayon parallel + EgoCache
└────────────┬─────────────────┘
             │
             ▼
       Final Top-K Results
```

---

## Testing Strategy

### Regression Gates (CI)
- All 185+ existing tests pass
- SIFT-10K R@10 ≥ 45% (current gate)
- **NEW:** SIFT-10K HNSW R@10 ≥ 90%
- **NEW:** RaBitQ 4-bit R@10 > RaBitQ 1-bit R@10

### Benchmark Suite
```bash
# Quick (SIFT-10K, ~30s)
cargo run -p palace-bench --release -- --sift

# Full (SIFT-1M, ~5min)
cargo run -p palace-bench --release -- --sift1m

# Ablation (topo reranking sweep)
cargo test -p palace-topo --test ablation_study -- --nocapture

# α-pruning
cargo test -p palace-graph --test recall_test test_alpha_pruning -- --nocapture
```

### Property-Based Tests
- HNSW: insert N random vectors → search each → must find self in top-1
- HNSW: recall monotonically increases with ef_search
- RaBitQ 4-bit: recall ≥ RaBitQ 1-bit for same vectors
- Topological reranking: never worse than random reordering

---

## Sprint Plan

**Sprint 1 (1-2 weeks): HNSW Core**
- [ ] `hnsw.rs` — struct, insert, search, level assignment
- [ ] Unit tests: self-recall, monotonic ef
- [ ] SIFT-10K validation: R@10 ≥ 90%
- [ ] Integrate into MemoryPalace as replacement for NSW

**Sprint 2 (1 week): 4-Bit + SIFT-1M**
- [ ] Fix `estimate_distance` for multi-plane
- [ ] Validate: 4-bit R@10 > 1-bit R@10
- [ ] SIFT-1M loader + benchmark
- [ ] QPS scaling analysis (10K → 100K → 1M)

**Sprint 3 (1 week): UMA Integration**
- [ ] Cache-aware layer allocation in UmaArena
- [ ] Prefetcher integration with HNSW traversal
- [ ] Thermal guard integration (real IOKit temperature)
- [ ] Benchmark: UMA-native vs generic HNSW latency comparison

**Sprint 4 (1-2 weeks): Metal + Validation**
- [ ] Metal compute shader for batch RaBitQ distance
- [ ] UMA zero-copy buffer from arena pointer
- [ ] Topological reranking validation on HNSW
- [ ] Final benchmark table for README / pitch deck

---

## Success Criteria

| Metric | v0.1 (actual) | v0.2 (target) | SOTA reference |
|--------|---------------|---------------|----------------|
| R@10 SIFT-10K | 54% (brute) / 1% (graph) | 98% (HNSW) | 99%+ (hnswlib) |
| R@10 SIFT-1M | not tested | 92%+ | 95%+ (Milvus) |
| QPS SIFT-1M | not tested | 5,000+ | 10,000+ (GPU) |
| Memory/vec 128d | 512 B (float) | 16 B (1-bit) + graph | 16 B (ScaNN) |
| Build time 1M | not tested | <60s | ~30s (hnswlib) |
| TRL | 3-4 | 5-6 | — |

---

## Investor Narrative

> "Palace-X is the first vector search engine built for Apple Silicon from the ground up. While competitors bolt GPU acceleration onto x86 architectures, we exploit Unified Memory Architecture to achieve zero-copy CPU-GPU data sharing, cache-hierarchy-aware graph traversal, and kernel-level thread scheduling. Our RaBitQ implementation delivers competitive recall at 32× compression, and our topological reranking module — based on persistent homology — can detect structural relationships that pure cosine similarity misses. We're targeting the Edge AI market where GPU servers aren't available but M-series chips are everywhere."

---

*This document is the source of truth for v0.2 development. Update it as decisions change.*
