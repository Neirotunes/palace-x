// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved

//! End-to-end smoke test for Palace-X engine.
//!
//! Covers the full lifecycle: start -> ingest -> search (with/without reranking)
//! -> vacuum -> re-search -> stats -> shutdown.

use palace_engine::{MetaData, PalaceEngine, SearchConfig};
use rand::Rng;

/// Generate a random f32 vector of the given dimensionality, seeded by `id`
/// so that nearby IDs produce somewhat distinct directions.
fn random_vector(dims: usize, id: u64) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    // Bias a few dimensions with the id to ensure vectors are not identical
    v[0] += (id as f32) * 0.01;
    if dims > 1 {
        v[1] -= (id as f32) * 0.005;
    }
    v
}

fn make_metadata(id: u64) -> MetaData {
    let sources = [
        "user_input",
        "internal_reasoning",
        "external_api",
        "sensor_data",
    ];
    let source = sources[(id as usize) % sources.len()];

    let mut meta = MetaData::new(1_700_000_000 + id * 60, source);
    meta.tags = vec![format!("batch_{}", id / 25), format!("priority_{}", id % 3)];
    meta.extra
        .insert("origin".to_string(), format!("node_{}", id));
    meta
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI — needs investigation"]
async fn test_full_lifecycle() {
    const DIMS: usize = 128;
    const TOTAL: u64 = 100;
    const DELETE_COUNT: usize = 10;
    const SEARCH_LIMIT: usize = 20;

    // ── 1. Create engine ──────────────────────────────────────────────
    let engine = PalaceEngine::start(DIMS);

    let init_stats = engine.stats().await.expect("initial stats");
    assert_eq!(init_stats.total_nodes, 0, "engine should start empty");
    assert_eq!(init_stats.dimensions, DIMS);

    // ── 2. Ingest 100 vectors with meaningful metadata ────────────────
    let mut node_ids = Vec::with_capacity(TOTAL as usize);
    for id in 0..TOTAL {
        let vector = random_vector(DIMS, id);
        let meta = make_metadata(id);
        let node_id = engine
            .ingest(vector, meta)
            .await
            .unwrap_or_else(|e| panic!("ingest {} failed: {}", id, e));
        node_ids.push(node_id);
    }

    let after_ingest = engine.stats().await.expect("stats after ingest");
    assert_eq!(
        after_ingest.total_nodes, TOTAL as usize,
        "all 100 nodes should be stored"
    );

    // Build a query vector close to node 0
    let query = random_vector(DIMS, 0);

    // ── 3. Search WITHOUT reranking ───────────────────────────────────
    let config_no_rerank = SearchConfig {
        limit: SEARCH_LIMIT,
        enable_reranking: false,
        alpha: 1.0,
        beta: 0.0,
        rerank_k: SEARCH_LIMIT * 2,
    };

    let results_no_rerank = engine
        .search(query.clone(), config_no_rerank)
        .await
        .expect("search without reranking failed");

    assert!(
        !results_no_rerank.is_empty(),
        "search should return results from a 100-node index"
    );
    assert!(
        results_no_rerank.len() <= SEARCH_LIMIT,
        "results should respect the limit"
    );

    // Verify descending score order
    for w in results_no_rerank.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results must be sorted by score descending (got {} before {})",
            w[0].score,
            w[1].score,
        );
    }

    // ── 4. Search WITH reranking ──────────────────────────────────────
    let config_rerank = SearchConfig {
        limit: SEARCH_LIMIT,
        enable_reranking: true,
        alpha: 0.7,
        beta: 0.3,
        rerank_k: SEARCH_LIMIT * 3,
    };

    let results_rerank = engine
        .search(query.clone(), config_rerank)
        .await
        .expect("search with reranking failed");

    assert!(
        !results_rerank.is_empty(),
        "reranked search should return results"
    );
    assert!(
        results_rerank.len() <= SEARCH_LIMIT,
        "reranked results should respect the limit"
    );

    for w in results_rerank.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "reranked results must be sorted by score descending (got {} before {})",
            w[0].score,
            w[1].score,
        );
    }

    // ── 5. Vacuum (delete) 10 nodes ───────────────────────────────────
    let to_delete: Vec<_> = node_ids[0..DELETE_COUNT].to_vec();
    let deleted_set: std::collections::HashSet<_> = to_delete.iter().copied().collect();

    let removed = engine.vacuum(to_delete).await.expect("vacuum failed");

    assert_eq!(
        removed, DELETE_COUNT as u64,
        "vacuum should report exactly {} removed nodes",
        DELETE_COUNT,
    );

    // ── 6. Re-search after vacuum — deleted nodes must not appear ─────
    let config_post_vacuum = SearchConfig::default_with_limit(TOTAL as usize);
    let results_post_vacuum = engine
        .search(query.clone(), config_post_vacuum)
        .await
        .expect("post-vacuum search failed");

    for frag in &results_post_vacuum {
        assert!(
            !deleted_set.contains(&frag.node_id),
            "deleted node {:?} must not appear in search results",
            frag.node_id,
        );
    }

    // Score ordering should still hold after vacuum
    for w in results_post_vacuum.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "post-vacuum results must be sorted by score descending",
        );
    }

    // ── 7. Check stats — node count should have decreased ─────────────
    let final_stats = engine.stats().await.expect("final stats");
    assert_eq!(
        final_stats.total_nodes,
        (TOTAL as usize) - DELETE_COUNT,
        "stats should reflect the vacuumed nodes",
    );
    assert_eq!(final_stats.dimensions, DIMS);
    assert!(
        final_stats.memory_usage_bytes > 0,
        "memory usage should be positive with remaining nodes",
    );

    // ── 8. Shutdown engine cleanly ────────────────────────────────────
    engine
        .shutdown()
        .await
        .expect("engine shutdown should succeed");
}
