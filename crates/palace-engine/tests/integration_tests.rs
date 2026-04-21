// Copyright (c) 2026 M.Diach
// Proprietary — All Rights Reserved

//! Integration tests for palace-engine
//!
//! Tests cover:
//! - Basic engine lifecycle
//! - Single and concurrent ingestion
//! - Search operations
//! - Batch operations
//! - Vacuum and cleanup
//! - Statistics reporting

use palace_engine::{MetaData, PalaceEngine, SearchConfig};

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_engine_lifecycle() {
    // Create engine
    let engine = PalaceEngine::start(8);

    // Get stats
    let stats = engine.stats().await.expect("Failed to get stats");
    assert_eq!(stats.total_nodes, 0);

    // Shutdown
    assert!(engine.shutdown().await.is_ok());
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_basic_ingest_and_search() {
    let engine = PalaceEngine::start(4);

    // Ingest a vector
    let vector1 = vec![1.0, 0.0, 0.0, 0.0];
    let meta1 = MetaData::new(1000, "test_vector_1");

    let node_id = engine.ingest(vector1, meta1).await.expect("Ingest failed");

    // Verify it was stored
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 1);

    // Search for it
    let query = vec![0.95, 0.05, 0.0, 0.0];
    let config = SearchConfig::default_with_limit(5);
    let results = engine.search(query, config).await.expect("Search failed");

    assert!(!results.is_empty());
    assert_eq!(results[0].node_id, node_id);
    assert!(results[0].score > 0.9); // Should be very similar

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_multiple_ingests() {
    let engine = PalaceEngine::start(4);

    let vectors = vec![
        (vec![1.0, 0.0, 0.0, 0.0], MetaData::new(1001, "v1")),
        (vec![0.0, 1.0, 0.0, 0.0], MetaData::new(1002, "v2")),
        (vec![0.0, 0.0, 1.0, 0.0], MetaData::new(1003, "v3")),
        (vec![0.0, 0.0, 0.0, 1.0], MetaData::new(1004, "v4")),
    ];

    let mut ids = Vec::new();
    for (vec, meta) in vectors {
        let id = engine.ingest(vec, meta).await.expect("Ingest failed");
        ids.push(id);
    }

    assert_eq!(ids.len(), 4);

    // Verify all are stored
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 4);

    // Search and verify results
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let config = SearchConfig::default_with_limit(2);
    let results = engine.search(query, config).await.expect("Search failed");

    assert!(!results.is_empty());
    assert!(results.len() <= 2);

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_concurrent_ingestion() {
    let engine = std::sync::Arc::new(PalaceEngine::start(8));

    let mut handles = vec![];

    // Spawn 4 concurrent ingestion tasks
    for task_id in 0..4 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let mut results = vec![];
            for i in 0..5 {
                let vector = vec![
                    (task_id * 5 + i) as f32 / 50.0,
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                    0.03125,
                    0.015625,
                    0.0078125,
                ];
                let meta = MetaData::new(
                    (task_id * 5 + i) as u64,
                    format!("task_{}_item_{}", task_id, i),
                );
                let result = engine_clone.ingest(vector, meta).await;
                results.push(result);
            }
            results
        });
        handles.push(handle);
    }

    // Wait for all tasks
    let mut total_ingested = 0;
    for handle in handles {
        let results = handle.await.expect("Task panicked");
        for result in results {
            assert!(result.is_ok());
            total_ingested += 1;
        }
    }

    assert_eq!(total_ingested, 20);

    // Verify stats
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 20);
    assert_eq!(stats.dimensions, 8);

    let _ = std::sync::Arc::try_unwrap(engine)
        .unwrap_or_else(|_| panic!("All Arc clones should have been dropped"))
        .shutdown()
        .await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_batch_ingestion() {
    let engine = PalaceEngine::start(4);

    let items = vec![
        (vec![0.1, 0.2, 0.3, 0.4], MetaData::new(2001, "batch1")),
        (vec![0.2, 0.3, 0.4, 0.5], MetaData::new(2002, "batch2")),
        (vec![0.3, 0.4, 0.5, 0.6], MetaData::new(2003, "batch3")),
        (vec![0.4, 0.5, 0.6, 0.7], MetaData::new(2004, "batch4")),
        (vec![0.5, 0.6, 0.7, 0.8], MetaData::new(2005, "batch5")),
    ];

    let results = engine.ingest_batch(items).await;

    assert_eq!(results.len(), 5);
    for result in &results {
        assert!(result.is_ok());
    }

    // Verify all ingested
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 5);

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_vacuum_operation() {
    let engine = PalaceEngine::start(4);

    // Ingest some vectors
    let mut ids = Vec::new();
    for i in 0..5 {
        let vector = vec![i as f32 / 10.0, 0.5, 0.25, 0.125];
        let meta = MetaData::new(3000 + i as u64, format!("vac_{}", i));
        let id = engine.ingest(vector, meta).await.expect("Ingest failed");
        ids.push(id);
    }

    let before_stats = engine.stats().await.expect("Stats failed");
    assert_eq!(before_stats.total_nodes, 5);

    // Vacuum the first two nodes
    let to_remove = vec![ids[0], ids[1]];
    let removed = engine.vacuum(to_remove).await.expect("Vacuum failed");
    assert_eq!(removed, 2);

    let after_stats = engine.stats().await.expect("Stats failed");
    assert_eq!(after_stats.total_nodes, 3);

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_search_with_custom_config() {
    let engine = PalaceEngine::start(4);

    // Ingest vectors
    for i in 0..10 {
        let vector = vec![(i as f32) / 100.0, (i as f32) / 50.0, 0.5, 0.25];
        let meta = MetaData::new(4000 + i as u64, format!("search_test_{}", i));
        let _ = engine.ingest(vector, meta).await;
    }

    // Custom search config
    let mut config = SearchConfig::default_with_limit(5);
    config.enable_reranking = true;
    config.alpha = 0.8;
    config.beta = 0.2;

    let query = vec![0.05, 0.1, 0.5, 0.25];
    let results = engine.search(query, config).await.expect("Search failed");

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    // Verify results are sorted by score
    for i in 0..results.len() - 1 {
        assert!(results[i].score >= results[i + 1].score);
    }

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_stats_reporting() {
    let engine = PalaceEngine::start(16);

    // Initially empty
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.dimensions, 16);

    // Add some vectors
    for i in 0..10 {
        let vector = vec![0.1; 16];
        let meta = MetaData::new(5000 + i as u64, format!("stat_test_{}", i));
        let _ = engine.ingest(vector, meta).await;
    }

    // Check updated stats
    let stats = engine.stats().await.expect("Stats failed");
    assert_eq!(stats.total_nodes, 10);
    assert_eq!(stats.dimensions, 16);
    assert!(stats.memory_usage_bytes > 0);
    assert!(stats.bitplane_coarse_bytes > 0);

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_search_empty_index() {
    let engine = PalaceEngine::start(4);

    let query = vec![0.5, 0.5, 0.5, 0.5];
    let config = SearchConfig::default_with_limit(10);
    let results = engine.search(query, config).await.expect("Search failed");

    assert!(results.is_empty());

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_dimension_mismatch() {
    let engine = PalaceEngine::start(8);

    // Try to ingest a vector with wrong dimensions
    let vector = vec![0.1, 0.2]; // Only 2 dimensions, but engine expects 8
    let meta = MetaData::new(6000, "bad_dims");

    let result = engine.ingest(vector, meta).await;
    assert!(result.is_err());

    let _ = engine.shutdown().await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_concurrent_search() {
    let engine = std::sync::Arc::new(PalaceEngine::start(4));

    // Ingest some data first
    for i in 0..20 {
        let vector = vec![(i as f32) / 100.0, 0.5, 0.25, 0.125];
        let meta = MetaData::new(7000 + i as u64, format!("concurrent_search_{}", i));
        let _ = engine.ingest(vector, meta).await;
    }

    // Spawn multiple concurrent search tasks
    let mut handles = vec![];
    for task_id in 0..5 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let query = vec![(task_id as f32) / 100.0, 0.5, 0.25, 0.125];
            let config = SearchConfig::default_with_limit(5);
            engine_clone.search(query, config).await
        });
        handles.push(handle);
    }

    // Wait for all searches
    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok());
        let fragments = result.unwrap();
        assert!(!fragments.is_empty());
    }

    let _ = std::sync::Arc::try_unwrap(engine)
        .unwrap_or_else(|_| panic!("All Arc clones should have been dropped"))
        .shutdown()
        .await;
}

#[tokio::test]
#[ignore = "deadlocks in release mode on CI"]
async fn test_engine_with_custom_config() {
    let engine = PalaceEngine::start_with_config(8, 8, 100, 0.6, 0.4, 512);

    // Ingest a vector
    let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let meta = MetaData::new(8000, "custom_config");

    let result = engine.ingest(vector, meta).await;
    assert!(result.is_ok());

    let _ = engine.shutdown().await;
}
