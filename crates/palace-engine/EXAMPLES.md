# Palace-Engine API Examples

This document provides practical examples of using the palace-engine crate.

## Basic Usage

### Create and Use Engine

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine for 768-dimensional vectors
    let engine = PalaceEngine::start(768);

    // Ingest a vector
    let embedding = vec![0.1; 768];
    let metadata = MetaData::new(1000, "my_source");
    let node_id = engine.ingest(embedding, metadata).await?;
    println!("Stored vector with ID: {:?}", node_id);

    // Search for similar vectors
    let query = vec![0.1; 768];
    let config = SearchConfig::default_with_limit(10);
    let results = engine.search(query, config).await?;
    println!("Found {} similar vectors", results.len());

    // Shutdown gracefully
    engine.shutdown().await?;
    Ok(())
}
```

## Batch Operations

### Efficient Batch Ingestion

```rust
use palace_engine::PalaceEngine;
use palace_core::MetaData;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = PalaceEngine::start(768);

    let vectors = vec![
        (vec![0.1; 768], MetaData::new(1000, "doc_1")),
        (vec![0.2; 768], MetaData::new(1001, "doc_2")),
        (vec![0.3; 768], MetaData::new(1002, "doc_3")),
    ];

    // Batch ingest multiple vectors concurrently
    let results = engine.ingest_batch(vectors).await;

    // Check for errors
    if PalaceEngine::batch_has_errors(&results) {
        println!("Some ingestions failed");
        let error_count = PalaceEngine::batch_error_count(&results);
        println!("Errors: {}", error_count);
    }

    // Extract successful IDs
    let ids = PalaceEngine::batch_success_ids(&results);
    println!("Successfully stored {} vectors", ids.len());

    engine.shutdown().await?;
    Ok(())
}
```

## Concurrent Access from Multiple Tasks

### Multi-Task Ingestion

```rust
use palace_engine::PalaceEngine;
use palace_core::MetaData;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Arc::new(PalaceEngine::start(128));

    // Spawn 10 concurrent ingestion tasks
    let mut handles = vec![];
    for task_id in 0..10 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let vector = vec![(task_id as f32) / 100.0; 128];
            let metadata = MetaData::new(task_id as u64, &format!("task_{}", task_id));
            engine_clone.ingest(vector, metadata).await
        });
        handles.push(handle);
    }

    // Wait for all tasks
    let mut success = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => success += 1,
            _ => println!("Task failed"),
        }
    }

    println!("Successfully ingested {} vectors", success);
    
    // Engine automatically shuts down when Arc is dropped
    drop(engine);
    Ok(())
}
```

### Multi-Task Search

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Arc::new(PalaceEngine::start(128));

    // First, ingest some data
    for i in 0..50 {
        let vector = vec![(i as f32) / 100.0; 128];
        let metadata = MetaData::new(i as u64, "data");
        let _ = engine.ingest(vector, metadata).await;
    }

    // Now spawn concurrent search tasks
    let mut handles = vec![];
    for task_id in 0..5 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let query = vec![(task_id as f32) / 100.0; 128];
            let config = SearchConfig::default_with_limit(5);
            engine_clone.search(query, config).await
        });
        handles.push(handle);
    }

    // Process results
    for (idx, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(fragments)) => {
                println!("Search {} found {} fragments", idx, fragments.len());
            }
            Ok(Err(e)) => println!("Search {} error: {}", idx, e),
            Err(e) => println!("Task {} panicked: {}", idx, e),
        }
    }

    Ok(())
}
```

## Custom Configuration

### Create Engine with Custom Parameters

```rust
use palace_engine::PalaceEngine;
use palace_core::MetaData;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create with custom configuration
    let engine = PalaceEngine::start_with_config(
        768,        // dimensions
        32,         // max_neighbors (NSW M parameter)
        400,        // ef_construction (NSW ef parameter)
        0.8,        // alpha (higher weight for cosine similarity)
        0.2,        // beta (lower weight for topological distance)
        2048,       // channel_size (handle more concurrent requests)
    );

    // Use engine as normal
    let vector = vec![0.1; 768];
    let metadata = MetaData::new(1000, "custom");
    engine.ingest(vector, metadata).await?;

    engine.shutdown().await?;
    Ok(())
}
```

## Search Configurations

### Different Search Strategies

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = PalaceEngine::start(128);

    // Ingest some vectors
    for i in 0..100 {
        let vector = vec![(i as f32) / 100.0; 128];
        let _ = engine.ingest(vector, MetaData::new(i as u64, "data")).await;
    }

    // Strategy 1: Fast coarse search only
    let mut config = SearchConfig::default_with_limit(5);
    config.enable_reranking = false;
    let fast_results = engine.search(vec![0.5; 128], config).await?;
    println!("Fast search: {} results", fast_results.len());

    // Strategy 2: High-quality with reranking
    let mut config = SearchConfig::default_with_limit(5);
    config.enable_reranking = true;
    config.alpha = 0.7;     // Weight cosine similarity more
    config.beta = 0.3;      // Weight topological distance less
    config.rerank_k = 20;   // Consider top 20 for reranking
    let quality_results = engine.search(vec![0.5; 128], config).await?;
    println!("Quality search: {} results", quality_results.len());

    // Strategy 3: Topologically-focused
    let mut config = SearchConfig::default_with_limit(5);
    config.enable_reranking = true;
    config.alpha = 0.3;     // Weight cosine less
    config.beta = 0.7;      // Weight topological more
    let topo_results = engine.search(vec![0.5; 128], config).await?;
    println!("Topo search: {} results", topo_results.len());

    engine.shutdown().await?;
    Ok(())
}
```

## Monitoring and Diagnostics

### Statistics and Monitoring

```rust
use palace_engine::PalaceEngine;
use palace_core::MetaData;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = PalaceEngine::start(256);

    let start = Instant::now();

    // Ingest batch of vectors
    for i in 0..1000 {
        let vector = vec![(i as f32) / 10000.0; 256];
        let _ = engine.ingest(vector, MetaData::new(i as u64, "benchmark")).await;
    }

    let ingest_time = start.elapsed();
    println!("Ingested 1000 vectors in {:?}", ingest_time);

    // Get statistics
    let stats = engine.stats().await?;
    println!("Total nodes: {}", stats.total_nodes);
    println!("Dimensions: {}", stats.dimensions);
    println!("Memory usage: {} bytes", stats.memory_usage_bytes);
    println!("Bitplane storage: {} bytes", stats.bitplane_coarse_bytes);
    println!("Hub score (avg): {:.3}", stats.avg_hub_score);
    println!("Hub score (max): {:.3}", stats.max_hub_score);
    println!("Hub nodes: {}", stats.hub_count);

    engine.shutdown().await?;
    Ok(())
}
```

## Maintenance Operations

### Vacuuming (Removing Nodes)

```rust
use palace_engine::PalaceEngine;
use palace_core::MetaData;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = PalaceEngine::start(128);

    // Ingest vectors
    let mut node_ids = vec![];
    for i in 0..10 {
        let vector = vec![0.1; 128];
        let metadata = MetaData::new(i as u64, "data");
        let id = engine.ingest(vector, metadata).await?;
        node_ids.push(id);
    }

    println!("Before vacuum:");
    let stats = engine.stats().await?;
    println!("  Nodes: {}", stats.total_nodes);

    // Remove some nodes
    let to_remove = vec![node_ids[0], node_ids[1], node_ids[2]];
    let removed = engine.vacuum(to_remove).await?;
    println!("Removed {} nodes", removed);

    println!("After vacuum:");
    let stats = engine.stats().await?;
    println!("  Nodes: {}", stats.total_nodes);

    engine.shutdown().await?;
    Ok(())
}
```

## Error Handling

### Graceful Error Handling

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, MemoryError};

#[tokio::main]
async fn main() {
    let engine = PalaceEngine::start(128);

    // Try to ingest with wrong dimensions
    let bad_vector = vec![0.1, 0.2]; // Only 2 dimensions
    let metadata = MetaData::new(1000, "bad");

    match engine.ingest(bad_vector, metadata).await {
        Ok(id) => println!("Stored: {:?}", id),
        Err(MemoryError::DimensionMismatch { expected, got }) => {
            println!("Dimension mismatch: expected {}, got {}", expected, got);
        }
        Err(e) => println!("Other error: {}", e),
    }

    // Try search on empty engine
    match engine.search(vec![0.1; 128], Default::default()).await {
        Ok(results) => {
            if results.is_empty() {
                println!("No results in empty index");
            }
        }
        Err(e) => println!("Search error: {}", e),
    }

    let _ = engine.shutdown().await;
}
```

## Production Pattern: Worker Pool

### Multiple Workers with Shared Engine

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Arc::new(PalaceEngine::start(768));
    let semaphore = Arc::new(Semaphore::new(10)); // Limit concurrent ops

    let mut tasks = vec![];

    // Spawn 100 operations with concurrency limit
    for i in 0..100 {
        let engine = engine.clone();
        let sem = semaphore.clone();
        
        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            
            let vector = vec![(i as f32) / 100.0; 768];
            let metadata = MetaData::new(i as u64, "worker");
            
            engine.ingest(vector, metadata).await
        });
        
        tasks.push(task);
    }

    // Wait for all
    let mut success = 0;
    for task in tasks {
        if task.await.is_ok_and(|r| r.is_ok()) {
            success += 1;
        }
    }

    println!("Successfully processed {} operations", success);
    engine.shutdown().await?;
    Ok(())
}
```

## Testing Pattern

### Unit Test with Engine

```rust
#[cfg(test)]
mod tests {
    use palace_engine::PalaceEngine;
    use palace_core::{MetaData, SearchConfig};

    #[tokio::test]
    async fn test_search_quality() {
        let engine = PalaceEngine::start(128);

        // Ingest reference vector
        let ref_vec = vec![1.0; 128];
        let ref_id = engine
            .ingest(ref_vec.clone(), MetaData::new(1, "reference"))
            .await
            .unwrap();

        // Search for it
        let config = SearchConfig::default_with_limit(1);
        let results = engine.search(ref_vec, config).await.unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].node_id, ref_id);

        let _ = engine.shutdown().await;
    }
}
```
