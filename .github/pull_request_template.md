## What

<!-- One sentence: what does this PR do? -->

## Why

<!-- Motivation: what problem does it solve, what issue does it close? -->

## Changes

<!-- Key changes, max 5 bullet points. Link to relevant files. -->

-

## Recall Impact

<!-- Did you run `cargo run -p palace-bench --release -- --sift`? -->
<!-- If this touches palace-quant, palace-graph, or palace-topo — paste the R@10 numbers. -->

- [ ] N/A — no search/quantization changes
- [ ] Benchmark run attached (paste below or link artifact)

```
R@10 before:
R@10 after:
```

## Review Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace --all-targets` clean
- [ ] `cargo fmt --all -- --check` clean
- [ ] No new `unsafe` without justification
- [ ] SIMD paths have scalar fallback
- [ ] New public API has doc comments
- [ ] README updated (if user-facing change)
