# Palace-X v0.2 Meta-Prompt

**Використовуй цей промт на початку кожної сесії розробки Palace-X v0.2.**

---

## Контекст

Ти працюєш над Palace-X — Rust vector search engine з topological reranking, оптимізований для Apple Silicon (M1-M4). Репозиторій: `Neirotunes/palace-x-pro` (приватний).

### Поточний стан (v0.1, квітень 2026)

- **11K LOC Rust**, 9 крейтів, 185+ тестів, CI з recall regression gate
- **RaBitQ 1-bit**: працює, 54% R@10 на SIFT-10K (3 критичні баги виправлені)
- **RaBitQ 4-bit**: зберігає 4 бітові площини, але `estimate_distance()` використовує тільки MSB → recall = 1-bit
- **NSW граф**: ~1% R@10, flat construction проблема (ранні ноди бідне сусідство)
- **Topological reranking**: працює, але на ідеальному kNN графі **знижує** recall (100% → 70%)
- **SiliconOptimizer**: UMA arena, P-core pinning, NEON SIMD, Speculative Prefetcher — все працює. Thermal/Metal — стаби.

### Архітектура

```
crates/
├── palace-core       # NodeId, Fragment, MetaData, SearchConfig, traits
├── palace-quant      # RaBitQ (1/4/7-bit), FHT rotation, binary quantization, Hamming SIMD
├── palace-graph      # NSW index (DashMap + ArcSwap snapshot), DistanceMetric, α-RNG pruning
├── palace-topo       # EgoGraph, β₁ Betti number, persistent homology H₀/H₁, d_total, reranker
├── palace-bitplane   # IEEE 754 bit-plane disaggregation
├── palace-storage    # MemoryPalace (2-stage retrieval: Hamming coarse → topo rerank)
├── palace-engine     # Async actor (tokio mpsc), P-core pinning, prefetcher integration
├── palace-optimizer  # UmaArena (mmap), threads (Mach affinity), SIMD (NEON), prefetch (Markov), thermal, Metal
├── palace-bench      # SIFT-10K benchmark, --sift flag
```

### Ключові формули

```
RaBitQ asymmetric distance:
  est_ip = norm / (x0 · √D) · ⟨x_bar, q'⟩
  est_dist = ||o-c||² + ||q-c||² - 2·est_ip
  
  де: x_bar = sign(rotated normalized data), q' = rotated RAW query residual
  x0 = ⟨x', x_bar⟩/√D (quantization quality, clamped ≥ 0.8)

Topological distance:
  d_total = α·d_cosine + β·exp(-β₁/|E|)
  
  де: β₁ = E - V + β₀ (first Betti number, independent cycles)

HNSW level assignment:
  level = floor(-ln(uniform()) × 1/ln(M))
```

### Bit-packing convention: LSB-first
```
dimension i → bit position (i % 64) in word (i / 64)
result[i / 64] |= 1u64 << (i % 64)
```

---

## Цілі v0.2 (BLUEPRINT_V02.md — source of truth)

1. **UMA-Native HNSW** — cache-aware layering: upper layers hot в L2/SLC (UmaArena), lower в RAM. Prefetcher працює на 2 рівні наперед.
2. **4-bit multi-plane distance** — використати всі 4 площини: `est_ip = Σ_k 2^k · popcount_ip(plane_k, q')`. Очікуваний R@10: 75-85%.
3. **SIFT-1M benchmark** — довести scaling (memory bandwidth savings at scale).
4. **Metal GPU batch search** — UMA zero-copy: candidate codes вже в shared memory.
5. **Topological reranking validation** — перевірити що β₁ допомагає на approximate HNSW (де ef низький).

---

## Правила роботи

### Код

- **Rust 1.75+**, workspace з shared dependencies (Cargo.toml)
- **DashMap** для concurrent writes, **ArcSwap** для wait-free reads
- **Rayon** для data parallelism (reranking)
- **tokio** для async engine (actor pattern)
- SIMD: `#[cfg(target_arch = "aarch64")]` з scalar fallback
- Без `unsafe` без коментаря чому
- LSB-first bit packing ЗАВЖДИ

### Тестування

```bash
# Повний тест (має проходити ЗАВЖДИ)
cargo test --workspace

# Бенчмарк SIFT-10K
cargo run -p palace-bench --release -- --sift

# Ablation
cargo test -p palace-topo --test ablation_study -- --nocapture
```

- Recall regression gate: R@10 ≥ 45% (CI автоматично)
- Кожен баг-фікс = regression test
- Property-based: insert → search self = top-1

### Git

- Branch: `feat/<name>`, `fix/<name>`
- PR template з recall impact секцією
- Squash merge в main
- Не пушити без зеленого CI

---

## Спринти

### Sprint 1: HNSW Core
**Файл: `crates/palace-graph/src/hnsw.rs` (NEW)**

Створити `HnswIndex` з:
- Level assignment: `floor(-ln(rand) / ln(M))`
- Insert: greedy descent через sparse layers → beam search на layer 0 → α-RNG pruning
- Search: greedy descent → beam expansion на layer 0
- UMA-aware: `arena_hot` для layers 1+ (fits SLC), `arena_cold` для layer 0
- Prefetcher: `prfm pldl1keep` на наступний рівень під час traversal
- Зберегти `search_binary()` path для Hamming pre-filter
- M_max0 = 2×M (layer 0 подвійні зв'язки)

**Валідація:** SIFT-10K R@10 ≥ 90%, self-recall = 100%

### Sprint 2: 4-Bit + SIFT-1M
**Файл: `crates/palace-quant/src/rabitq.rs` (MODIFY)**

Додати `estimate_distance_multibit()`:
```rust
for k in 0..bits {
    ip_k = asymmetric_ip(plane[k], query.rotated_vector);
    weighted_ip += (1 << k) as f32 * ip_k;
}
est_ip = norm / (x0 * √D) * weighted_ip * scale;
```

Перерахувати x0 для 4-bit reconstruction quality.

**Файл: `crates/palace-bench/src/sift_bench.rs` (MODIFY)**
- Loader для SIFT-1M (fvecs, 488 MB)
- QPS scaling: 10K → 100K → 1M

### Sprint 3: UMA Integration
- Cache-aware layer allocation
- Prefetcher → HNSW traversal integration
- Thermal: real IOKit temperature замість hardcoded 45°C
- A/B: UMA-native vs generic HNSW latency

### Sprint 4: Metal + Topo Validation
- Metal compute shader для batch RaBitQ
- `MTLBuffer(bytesNoCopy:)` від UmaArena pointer
- Topo reranking на HNSW з низьким ef (intentionally approximate)
- Final benchmark table

---

## Targets (SOTA 2027 context)

| Metric | v0.1 | v0.2 target | SOTA ref |
|--------|------|-------------|----------|
| R@10 SIFT-10K | 54% brute / 1% graph | 98% HNSW | 99%+ hnswlib |
| R@10 SIFT-1M | — | 92%+ | 95%+ Milvus |
| QPS SIFT-1M | — | 5,000+ | 10,000+ GPU |
| Memory/vec | 512 B float | 16 B (1-bit+graph) | 16 B ScaNN |
| TRL | 3-4 | 5-6 | — |

---

## Чого НЕ робити

- Не оптимізувати NSW — він deprecated, замінюється HNSW
- Не писати generic HNSW — робити UMA-native з самого початку
- Не тестувати тільки на 10K — scaling story вимагає 1M
- Не ігнорувати negative results — якщо topo reranking не допомагає на HNSW, документувати чесно
- Не забувати LSB-first — це вже коштувало 4 години debug

---

## Pitch (одним реченням)

"Найшвидший vector search для Edge AI на Apple Silicon — 32× стиснення через RaBitQ, zero-copy CPU-GPU через UMA, topological reranking через persistent homology."
