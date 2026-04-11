//! Edge case and fuzz tests for MemoryPalace.
//!
//! Validates correct behavior with extreme, degenerate, and boundary-condition inputs.
//!
//! NOTE: This test requires `palace-storage` and `tokio` as dev-dependencies
//! in palace-core/Cargo.toml:
//!   [dev-dependencies]
//!   palace-storage = { workspace = true }
//!   tokio = { version = "1", features = ["full", "test-util"] }

use palace_core::{MemoryProvider, MetaData, SearchConfig};
use palace_storage::MemoryPalace;

#[tokio::test]
async fn test_zero_vector() {
    let palace = MemoryPalace::new(128);
    let vector = vec![0.0_f32; 128];
    let meta = MetaData::new(1000, "zero_vector");

    // Should not panic even though cosine distance is undefined for zero vectors
    let result = palace.ingest(vector, meta).await;
    assert!(
        result.is_ok(),
        "Ingesting a zero vector should not panic or error"
    );
}

#[tokio::test]
async fn test_very_large_values() {
    let palace = MemoryPalace::new(128);
    let vector = vec![f32::MAX; 128];
    let meta = MetaData::new(1001, "large_values");

    let result = palace.ingest(vector, meta).await;
    assert!(
        result.is_ok(),
        "Ingesting a vector with f32::MAX values should not panic: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_very_small_values() {
    let palace = MemoryPalace::new(128);
    let vector = vec![f32::MIN_POSITIVE; 128];
    let meta = MetaData::new(1002, "small_values");

    let result = palace.ingest(vector, meta).await;
    assert!(
        result.is_ok(),
        "Ingesting a vector with f32::MIN_POSITIVE values should not panic: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_negative_values() {
    let palace = MemoryPalace::new(128);
    let vector = vec![-0.5_f32; 128];
    let meta = MetaData::new(1003, "negative_values");

    let result = palace.ingest(vector, meta).await;
    assert!(
        result.is_ok(),
        "Ingesting an all-negative vector should not panic: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn test_single_node_search() {
    let palace = MemoryPalace::new(128);
    let vector = vec![0.42_f32; 128];
    let meta = MetaData::new(1004, "single_node");

    let id = palace
        .ingest(vector.clone(), meta)
        .await
        .expect("ingest failed");

    // Search for the same vector; should find it
    let mut config = SearchConfig::default_with_limit(1);
    config.enable_reranking = false; // disable reranking for simplicity

    let results = palace
        .retrieve(&vector, &config)
        .await
        .expect("retrieve failed");
    assert!(
        !results.is_empty(),
        "Searching a single-node index should return a result"
    );
    assert_eq!(
        results[0].node_id, id,
        "The only result should be the single ingested node"
    );
}

#[tokio::test]
async fn test_duplicate_vectors() {
    let palace = MemoryPalace::new(128);
    let vector = vec![0.7_f32; 128];

    let meta1 = MetaData::new(1005, "dup_1");
    let meta2 = MetaData::new(1006, "dup_2");

    let id1 = palace
        .ingest(vector.clone(), meta1)
        .await
        .expect("first ingest failed");
    let id2 = palace
        .ingest(vector.clone(), meta2)
        .await
        .expect("second ingest failed");

    // Both should be stored with different IDs
    assert_ne!(id1, id2, "Duplicate vectors must receive different NodeIds");
    assert_eq!(
        palace.len().await,
        2,
        "Both duplicate vectors should be stored"
    );
}

#[tokio::test]
async fn test_dimension_mismatch_rejected() {
    let palace = MemoryPalace::new(128);

    // Try to ingest a vector with wrong dimensions
    let wrong_dim_vector = vec![0.1_f32; 64];
    let meta = MetaData::new(1007, "wrong_dim");

    let result = palace.ingest(wrong_dim_vector, meta).await;
    assert!(
        result.is_err(),
        "Ingesting a vector with mismatched dimensions should return an error"
    );

    // Also try too-large dimensions
    let too_large_vector = vec![0.1_f32; 256];
    let meta2 = MetaData::new(1008, "too_large_dim");

    let result2 = palace.ingest(too_large_vector, meta2).await;
    assert!(
        result2.is_err(),
        "Ingesting a vector with too many dimensions should return an error"
    );

    // Index should remain empty since all ingests failed
    assert_eq!(palace.len().await, 0, "No vectors should have been stored");
}
