# Contributing to Palace-X

## Development Setup

```bash
# Rust 1.75+ required
rustup update stable

# Build
cargo build --workspace

# Full test suite (185+ tests)
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```

## Branch Strategy

- `main` — stable, CI must pass
- Feature branches — `feat/<name>`, `fix/<name>`, `bench/<name>`
- All changes go through PR with at least one review

## PR Process

1. Create a feature branch from `main`
2. Make changes, ensure all tests pass locally
3. Open PR using the template — fill in the recall impact section if touching search code
4. CI must be green before merge
5. Squash-merge to `main`

## Code Guidelines

### Recall-Critical Code

Any change to `palace-quant`, `palace-graph`, or `palace-topo` can affect search recall. Before submitting:

```bash
# Run SIFT-10K benchmark
cargo run -p palace-bench --release -- --sift
```

Include R@10 numbers in your PR description (before/after). CI runs this automatically, but catching regressions locally saves time.

### Current recall baselines (SIFT-10K, 128d, 10K vectors):

| Method | R@10 |
|--------|------|
| RaBitQ 1-bit (brute) | 54.0% |
| Naive binary Hamming | 15.5% |
| Float L2 brute-force | 100.0% |

### SIMD

- All SIMD paths must have a scalar fallback
- Use `#[cfg(target_arch)]` gates, not runtime detection (for now)
- NEON (AArch64) and AVX-512 (x86_64) are the primary targets
- Test on both `ubuntu-latest` and `macos-latest` — CI does this

### Safety

- No `unsafe` without a comment explaining why it's necessary
- Prefer safe abstractions from `std::simd` where possible
- Bit manipulation code must document the packing convention (LSB-first)

### Tests

- Unit tests live in the same file (`#[cfg(test)]` module)
- Integration tests go in `tests/`
- Benchmarks go in `palace-bench`
- If you fix a bug, add a regression test

## Architecture Decision Records

Major design decisions should be documented as comments in the relevant module. Key decisions so far:

- **LSB-first bit packing** (`binary.rs`): dimension `i` → bit `i % 64` in word `i / 64`
- **FHT over random matrix** (`rabitq.rs`): O(D log D) vs O(D²), deterministic from seed
- **Asymmetric distance** (`rabitq.rs`): query stays unquantized, only database vectors compressed
- **α-RNG pruning** (`nsw.rs`): Vamana-style with configurable α (default 1.2)

## Running Specific Tests

```bash
# Ablation study (topological reranking effect)
cargo test -p palace-topo --test ablation_study -- --nocapture

# Alpha-pruning comparison
cargo test -p palace-graph --test recall_test test_alpha_pruning -- --nocapture

# RaBitQ unit tests only
cargo test -p palace-quant rabitq -- --nocapture
```

## Release Process

Releases are tagged with semver: `v0.1.0`, `v0.2.0-alpha`, etc.

```bash
git tag v0.2.0
git push origin v0.2.0
# GitHub Actions builds artifacts and creates release automatically
```
