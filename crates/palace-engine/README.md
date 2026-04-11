# palace-engine

Async actor pipeline orchestrator for the Palace-X project.

Wraps `MemoryPalace` in an async actor system for safe concurrent ingestion, search, and maintenance operations.

## Architecture

The engine implements an actor model pattern:

- **Single Async Task**: Commands are serialized through a single `tokio::spawn`'d task
- **mpsc Channel**: Senders communicate with the actor via bounded MPSC channel (default 1024)
- **Zero-Lock**: No locks on the hot path; MemoryPalace handles internal synchronization
- **Graceful Shutdown**: Coordinated shutdown with `shutdown()` method

### Command Types

- **Ingest**: Add a vector with metadata
- **Search**: Query for similar vectors with configuration
- **Vacuum**: Remove nodes from the index
- **Stats**: Get engine statistics
- **Shutdown**: Gracefully shut down the actor

## Usage

### Basic Example

```rust
use palace_engine::PalaceEngine;
use palace_core::{MetaData, SearchConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine with 768-dimensional vectors
    let engine = PalaceEngine::start(768);

    // Ingest a vector
    let vector = vec![0.1; 768];
    let metadata = MetaData::new(1234567890, "user_input");
    let node_id = engine.ingest(vector, metadata).await?;

    // Search
    let query = vec![0.1; 768];
    let config = SearchConfig::default_with_limit(10);
    let results = engine.search(query, config).await?;
    println!("Found {} similar fragments", results.len());

    // Get statistics
    let stats = engine.stats().await?;
    println!("Stored {} vectors", stats.total_nodes);

    // Graceful shutdown
    engine.shutdown().await?;
    Ok(())
}
```

### Batch Ingestion

```rust
let items = vec![
    (vec![0.1; 768], MetaData::new(1000, "source1")),
    (vec![0.2; 768], MetaData::new(1001, "source2")),
];

let results = engine.ingest_batch(items).await;
let success_ids = PalaceEngine::batch_success_ids(&results);
```

### Concurrent Operations

The engine is `Clone + Send + Sync` via `Arc`, allowing safe sharing across tasks:

```rust
let engine = Arc::new(PalaceEngine::start(768));

// Spawn multiple workers
let mut handles = vec![];
for i in 0..10 {
    let engine = engine.clone();
    let handle = tokio::spawn(async move {
        let vector = vec![0.1; 768];
        let meta = MetaData::new(i as u64, "worker");
        engine.ingest(vector, meta).await
    });
    handles.push(handle);
}

// Wait for completion
for handle in handles {
    let _ = handle.await;
}
```

## Configuration

Create an engine with custom parameters:

```rust
let engine = PalaceEngine::start_with_config(
    768,          // dimensions
    16,           // max_neighbors in NSW
    200,          // ef_construction for NSW
    0.7,          // alpha (cosine weight)
    0.3,          // beta (topological weight)
    1024,         // channel_size
);
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Ingest | O(log N) | NSW insertion |
| Search | O(ef log N) + O(k log k) | Two-stage retrieval |
| Vacuum | O(m * degree) | m = nodes to remove |
| Stats | O(1) | Cached estimates |

## Testing

The crate includes comprehensive tests:

```bash
cargo test -p palace-engine
```

### Test Coverage

- Basic lifecycle (create, shutdown)
- Single and concurrent ingestion
- Search with various configurations
- Batch operations
- Vacuum operations
- Statistics reporting
- Dimension validation
- Empty index handling

## Error Handling

All async operations return `Result<T, MemoryError>` with variants:

- `IndexFull`: Index capacity exceeded
- `NodeNotFound`: Node not in index
- `DimensionMismatch`: Vector dimension mismatch
- `StorageError`: General storage errors
- `CompressionError`: Compression operation failed
- `GraphCorrupted`: Internal graph inconsistency

## Implementation Notes

### Actor Loop

The actor loop processes commands sequentially:

```rust
while let Some(cmd) = rx.recv().await {
    match cmd {
        Command::Ingest { vector, metadata, reply } => {
            let result = palace.ingest(vector, metadata).await;
            let _ = reply.send(result);
        }
        // ... other commands
    }
}
```

### Send-Reply Pattern

Each command carries a `oneshot::Sender` for returning results:

```rust
pub async fn ingest(&self, vector: Vec<f32>, metadata: MetaData) 
    -> Result<NodeId, MemoryError> 
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    self.sender.send(Command::Ingest { vector, metadata, reply: tx }).await?;
    rx.await?
}
```

### Batch Pipelining

The `ingest_batch` helper creates multiple ingestion futures:

```rust
let futures: Vec<_> = items
    .into_iter()
    .map(|(vector, metadata)| self.ingest(vector, metadata))
    .collect();
futures::future::join_all(futures).await
```

While the engine processes commands sequentially, pipelining reduces latency for many small requests.

## Integration with Palace-X

The engine integrates three Palace-X subsystems:

1. **palace-storage** (`MemoryPalace`): In-memory hierarchical index
2. **palace-core**: Type definitions and error types
3. **palace-graph**: NSW (Navigable Small World) graph
4. **palace-topo**: Topological reranking
5. **palace-bitplane**: Precision-proportional bit-plane storage

## License

MIT OR Apache-2.0
