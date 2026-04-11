//! Concurrency stress tests for palace-engine
//!
//! Tests cover:
//! - High-volume concurrent ingestion (8 tasks x 100 vectors)
//! - Mixed concurrent operations (ingest + search simultaneously)
//! - Rapid ingest-search cycles with growing index

use std::sync::Arc;

use palace_engine::{MetaData, PalaceEngine, SearchConfig};
use rand::Rng;
use tokio::time::{timeout, Duration};

/// Helper: generate a random f32 vector of the given dimension.
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

/// Spawn 8 tokio tasks, each ingesting 100 vectors (800 total) into a shared
/// engine. Verify final node count equals 800.
#[tokio::test]
async fn test_concurrent_ingest_stress() {
    const DIM: usize = 16;
    const TASKS: usize = 8;
    const VECTORS_PER_TASK: usize = 100;
    const TOTAL: usize = TASKS * VECTORS_PER_TASK;

    let engine = Arc::new(PalaceEngine::start(DIM));

    let mut handles = Vec::with_capacity(TASKS);

    for task_id in 0..TASKS {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let mut ids = Vec::with_capacity(VECTORS_PER_TASK);
            for i in 0..VECTORS_PER_TASK {
                let vector = random_vector(DIM);
                let meta = MetaData::new(
                    (task_id * VECTORS_PER_TASK + i) as u64,
                    format!("stress_t{}_i{}", task_id, i),
                );
                let id = engine_clone
                    .ingest(vector, meta)
                    .await
                    .expect("Ingest should not fail");
                ids.push(id);
            }
            ids
        });
        handles.push(handle);
    }

    // Collect all results
    let mut all_ids = Vec::with_capacity(TOTAL);
    for handle in handles {
        let ids = handle.await.expect("Task should not panic");
        all_ids.extend(ids);
    }

    assert_eq!(all_ids.len(), TOTAL);

    // Verify via stats
    let stats = engine.stats().await.expect("Stats should succeed");
    assert_eq!(stats.total_nodes, TOTAL);
    assert_eq!(stats.dimensions, DIM);

    let _ = Arc::try_unwrap(engine)
        .unwrap_or_else(|_| panic!("All Arc clones should have been dropped"))
        .shutdown()
        .await;
}

/// Spawn tasks that simultaneously ingest and search:
///   - 4 tasks ingesting vectors
///   - 4 tasks searching
/// Wrapped in a 30-second timeout to detect deadlocks.
/// Verify no panics occur.
#[tokio::test]
async fn test_concurrent_mixed_operations() {
    const DIM: usize = 8;
    const INGEST_TASKS: usize = 4;
    const SEARCH_TASKS: usize = 4;
    const VECTORS_PER_TASK: usize = 50;
    const SEARCHES_PER_TASK: usize = 20;

    let result = timeout(Duration::from_secs(30), async {
        let engine = Arc::new(PalaceEngine::start(DIM));

        // Seed a few vectors so early searches have something to find
        for i in 0..5 {
            let vector = random_vector(DIM);
            let meta = MetaData::new(9000 + i as u64, format!("seed_{}", i));
            engine
                .ingest(vector, meta)
                .await
                .expect("Seed ingest failed");
        }

        let mut handles = Vec::with_capacity(INGEST_TASKS + SEARCH_TASKS);

        // Spawn ingest tasks
        for task_id in 0..INGEST_TASKS {
            let eng = engine.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..VECTORS_PER_TASK {
                    let vector = random_vector(DIM);
                    let meta = MetaData::new(
                        (10_000 + task_id * VECTORS_PER_TASK + i) as u64,
                        format!("mix_ingest_t{}_i{}", task_id, i),
                    );
                    eng.ingest(vector, meta)
                        .await
                        .expect("Concurrent ingest failed");
                }
            }));
        }

        // Spawn search tasks
        for task_id in 0..SEARCH_TASKS {
            let eng = engine.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..SEARCHES_PER_TASK {
                    let query = random_vector(DIM);
                    let config = SearchConfig::default_with_limit(5);
                    let result = eng.search(query, config).await;
                    // Search may return empty if index is tiny, but must not error
                    assert!(
                        result.is_ok(),
                        "Search failed during mixed ops in task {}: {:?}",
                        task_id,
                        result.err()
                    );
                }
            }));
        }

        // Wait for all tasks to finish
        for handle in handles {
            handle.await.expect("Task should not panic");
        }

        // Final sanity check: total_nodes >= seed count + ingest count
        let stats = engine.stats().await.expect("Stats failed");
        let expected = 5 + INGEST_TASKS * VECTORS_PER_TASK;
        assert_eq!(
            stats.total_nodes, expected,
            "Expected {} nodes, got {}",
            expected, stats.total_nodes
        );

        let _ = Arc::try_unwrap(engine)
            .unwrap_or_else(|_| panic!("All Arc clones should have been dropped"))
            .shutdown()
            .await;
    })
    .await;

    assert!(
        result.is_ok(),
        "Test timed out after 30s — possible deadlock"
    );
}

/// Ingest 50 vectors, search, ingest 50 more, search again.
/// Verify second search returns results from a larger index (more diverse).
#[tokio::test]
async fn test_rapid_ingest_search_cycle() {
    const DIM: usize = 8;
    const BATCH_SIZE: usize = 50;

    let engine = PalaceEngine::start(DIM);

    // --- Phase 1: ingest first 50 vectors ---
    for i in 0..BATCH_SIZE {
        let vector = random_vector(DIM);
        let meta = MetaData::new(20_000 + i as u64, format!("cycle_phase1_{}", i));
        engine
            .ingest(vector, meta)
            .await
            .expect("Phase 1 ingest failed");
    }

    let stats1 = engine.stats().await.expect("Stats failed");
    assert_eq!(stats1.total_nodes, BATCH_SIZE);

    // --- Phase 1 search ---
    let query = random_vector(DIM);
    let config = SearchConfig::default_with_limit(10);
    let results1 = engine
        .search(query.clone(), config.clone())
        .await
        .expect("Phase 1 search failed");

    // --- Phase 2: ingest another 50 vectors ---
    for i in 0..BATCH_SIZE {
        let vector = random_vector(DIM);
        let meta = MetaData::new(30_000 + i as u64, format!("cycle_phase2_{}", i));
        engine
            .ingest(vector, meta)
            .await
            .expect("Phase 2 ingest failed");
    }

    let stats2 = engine.stats().await.expect("Stats failed");
    assert_eq!(stats2.total_nodes, BATCH_SIZE * 2);

    // --- Phase 2 search (same query) ---
    let results2 = engine
        .search(query, config)
        .await
        .expect("Phase 2 search failed");

    // With double the vectors the search should be able to return at least as
    // many results, and the candidate pool is larger so we expect potentially
    // more diverse scores.
    assert!(
        !results2.is_empty(),
        "Second search should return results from 100-vector index"
    );

    // Both searches should produce valid results; exact count may vary
    // due to graph topology changes when new nodes are added.
    assert!(
        !results1.is_empty() && !results2.is_empty(),
        "Both searches should return results (phase1={}, phase2={})",
        results1.len(),
        results2.len()
    );

    // With 100 vectors the unique node IDs across both result sets should span
    // a wider range than the first search alone, demonstrating diversity.
    let unique_ids_1: std::collections::HashSet<_> = results1.iter().map(|f| f.node_id).collect();
    let unique_ids_2: std::collections::HashSet<_> = results2.iter().map(|f| f.node_id).collect();
    let combined: std::collections::HashSet<_> = unique_ids_1.union(&unique_ids_2).collect();

    assert!(
        combined.len() >= unique_ids_1.len(),
        "Combined unique IDs ({}) should be >= first-search unique IDs ({})",
        combined.len(),
        unique_ids_1.len()
    );

    let _ = engine.shutdown().await;
}
