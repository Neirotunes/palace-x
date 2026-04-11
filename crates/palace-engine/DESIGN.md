# Palace-Engine Design Document

## Overview

Palace-Engine is an async actor-based pipeline orchestrator that wraps `MemoryPalace` (the hierarchical memory implementation) in a safe, concurrent interface for the Palace-X project.

The engine enables multiple concurrent clients to interact with a single in-memory hierarchical index without locks on the hot path.

## Architecture

### Core Components

1. **PalaceEngine** (`src/engine.rs`)
   - Main public API
   - Manages actor lifecycle and command channel
   - Provides methods for ingestion, search, vacuuming, and stats

2. **Command Enum** (`src/actor.rs`)
   - Message protocol between clients and actor
   - Carries request data and oneshot reply channels
   - Types: Ingest, Search, Vacuum, Stats, Shutdown

3. **Actor Task** (`src/engine.rs::run_loop`)
   - Single async task spawned with `tokio::spawn`
   - Processes commands sequentially from MPSC channel
   - Owns the MemoryPalace instance
   - Sends results back via oneshot channels

4. **Batch Operations** (`src/batch.rs`)
   - Convenience layer for pipelined concurrent ingestion
   - Uses `futures::future::join_all` for throughput

### Actor Model Benefits

- **Thread-Safe**: No explicit locks on hot path; sequential processing eliminates races
- **Scalable**: Single actor can serve many concurrent clients
- **Graceful Shutdown**: Coordinated shutdown via `Shutdown` command
- **Predictable**: Sequential processing makes behavior easy to reason about
- **Composable**: Commands can be combined into higher-level workflows

## Design Patterns

### 1. Send-Reply Pattern

Each command carries a `oneshot::Sender` for returning results:

```rust
pub enum Command {
    Ingest {
        vector: Vec<f32>,
        metadata: MetaData,
        reply: oneshot::Sender<Result<NodeId, MemoryError>>,
    },
    // ...
}
```

The client:
1. Creates a `oneshot::channel()`
2. Sends a command with the sender
3. Awaits the receiver
4. Gets the result

### 2. Bounded Channel

The MPSC channel has a bounded capacity (default 1024):

```rust
let (tx, rx) = mpsc::channel(1024);
```

Benefits:
- Prevents unbounded memory growth
- Applies backpressure when channel is full
- Sender returns `SendError` if channel is closed

### 3. Zero-Copy Vectors

Vectors are moved directly into commands:

```rust
pub async fn ingest(&self, vector: Vec<f32>, metadata: MetaData) 
    -> Result<NodeId, MemoryError>
{
    let (tx, rx) = tokio::sync::oneshot::channel();
    self.sender.send(Command::Ingest { vector, metadata, reply: tx }).await?;
    rx.await?
}
```

No cloning or serialization overhead.

### 4. Arc-Based Sharing

For concurrent access from multiple tasks:

```rust
let engine = Arc::new(PalaceEngine::start(768));

for _ in 0..10 {
    let engine = engine.clone();
    tokio::spawn(async move {
        engine.ingest(vec, meta).await
    });
}
```

The `Arc` is dropped when all tasks complete; the last drop triggers engine shutdown.

## Control Flow

### Ingestion

```
Client Task                    Actor Task              MemoryPalace
    |                              |                       |
    +---- Ingest Command --------->|                       |
    |        (with reply sender)    |                       |
    |                              +-- ingest() ----------->|
    |                              |                       |
    |                              |<-- Result -----------+
    |                              |                       |
    |<----- oneshot::send(Ok(id)) --                       |
    |                              |                       |
```

### Concurrent Ingestion

```
Task 1 ---\
Task 2 ----+--- Ingest Command 1 ---\
Task 3 ----+--- Ingest Command 2 ----+---> Actor Task
Task 4 ----+--- Ingest Command 3 ----+     (Sequential
Task 5 ----/--- Ingest Command 4 ---/       Processing)
```

Each client sends independently; the actor processes sequentially.

### Search

```
Client                         Actor               MemoryPalace
  |                             |                      |
  +---- Search Command -------->|                      |
  |     (query, config)         |                      |
  |                             +-- retrieve() ------->|
  |                             |   (two-stage)       |
  |                             |<- Fragments -------+
  |                             |                      |
  |<----- oneshot::send() -------                      |
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Ingest | O(log N) | NSW insertion with binary quantization |
| Search (Stage 1) | O(ef log N) | ef ≈ rerank_k; binary Hamming search |
| Search (Stage 2) | O(k log k) | Topological reranking; k = rerank_k |
| Vacuum | O(m * degree) | m = nodes to delete; degree ≈ 32 |
| Stats | O(1) | Cached estimates |

### Space Complexity

- Per-node NSW: ~(dimensions * 4) + (max_neighbors * 8) bytes
- Per-node BitPlane: ~9 * dimensions / 8 bytes
- Total: O(N * dimensions) where N = node count

### Throughput

Batch ingestion via pipelined sends:

```
Without Batch:  1 -> wait -> 2 -> wait -> 3 -> wait
With Batch:     1, 2, 3 pipelined; wait once
```

## Concurrency Model

### Safe Concurrent Access

The engine is `Send + Sync` through `Arc`:

```rust
// Engine type: PalaceEngine
// Wrapped type: Arc<PalaceEngine> is Send + Sync

impl<T: Send + Sync> Send for Arc<T> {}
impl<T: Send + Sync> Sync for Arc<T> {}
```

### Data Race Prevention

MemoryPalace uses internal synchronization:
- `parking_lot::RwLock` for vector/metadata stores
- Atomic operations for node ID generation

The actor serializes commands, preventing concurrent access to MemoryPalace methods.

### Shutdown Safety

Graceful shutdown ensures:
1. No new commands are accepted after `shutdown()` is called
2. In-flight commands complete
3. All handles are properly dropped
4. MemoryPalace is properly cleaned up

```rust
pub async fn shutdown(self) -> Result<(), MemoryError> {
    let _ = self.sender.send(Command::Shutdown).await;
    self.handle.await?;
    Ok(())
}
```

## Error Handling

### Channel Errors

If the actor task crashes or the engine is shut down:
```rust
self.sender.send(cmd).await
    .map_err(|_| MemoryError::StorageError("Engine shutdown".into()))
```

### Oneshot Errors

If the reply channel is dropped:
```rust
rx.await.map_err(|_| MemoryError::StorageError("Engine dropped".into()))
```

### MemoryPalace Errors

Errors from MemoryPalace are propagated directly:
- `DimensionMismatch`
- `IndexFull`
- `StorageError`
- `CompressionError`
- `GraphCorrupted`

## Testing Strategy

### Unit Tests (in-module)

- `test_engine_creation`: Lifecycle
- `test_basic_ingest`: Single ingestion
- `test_stats`: Statistics retrieval

### Integration Tests (`tests/integration_tests.rs`)

- **Lifecycle**: Create, use, shutdown
- **Ingestion**: Single, multiple, concurrent
- **Search**: Basic, with config, concurrent, empty index
- **Batch**: Simple batch, with search, concurrent batches
- **Vacuum**: Removal and recount
- **Statistics**: Empty, populated, scaling
- **Error Handling**: Dimension mismatch, shutdown
- **Configuration**: Custom parameters

### Test Coverage

Total test count: 16+ tests covering:
- Happy paths
- Error conditions
- Concurrency scenarios
- Edge cases (empty index, dimension mismatch)

## Future Enhancements

### Possible Extensions

1. **Metrics Integration**
   ```rust
   pub struct EngineMetrics {
       ingestions: Counter,
       searches: Counter,
       latency_histogram: Histogram,
   }
   ```

2. **Persistence**
   - Snapshot/restore of index state
   - WAL (write-ahead log) for durability

3. **Replication**
   - Multi-engine coordination
   - Leader-follower replication

4. **Adaptive Batching**
   - Auto-batch based on arrival rate
   - Tune channel size dynamically

5. **Monitoring**
   - Actor task health checks
   - Channel congestion alerts
   - Memory usage tracking

## Implementation Details

### Command Processing Loop

```rust
async fn run_loop(mut palace: MemoryPalace, mut rx: mpsc::Receiver<Command>) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Command::Ingest { vector, metadata, reply } => {
                let result = palace.ingest(vector, metadata).await;
                let _ = reply.send(result);
            }
            // ... other commands
            Command::Shutdown => break,
        }
    }
}
```

Key points:
- `rx.recv().await` blocks until a command arrives
- Pattern match routes to appropriate handler
- Each handler calls MemoryPalace method
- Result is sent back via oneshot
- Loop exits on `Shutdown` command

### Channel Semantics

MPSC (Multiple Sender, Single Receiver):
- Many clients can send commands
- One actor receives and processes
- Bounded capacity prevents DOS
- Dropped sender means new senders can still be created (until channel is closed)

## References

- **Actor Model**: https://en.wikipedia.org/wiki/Actor_model
- **Tokio MPSC**: https://docs.rs/tokio/latest/tokio/sync/mpsc/
- **Oneshot Channels**: https://docs.rs/tokio/latest/tokio/sync/oneshot/
- **Rust Async**: https://rust-lang.github.io/async-book/

## Testing & Validation

All code paths are tested:
- Happy path (all operations succeed)
- Error paths (dimension mismatch, shutdown)
- Concurrency (multiple tasks, Arc sharing)
- Lifecycle (create, use, shutdown)

See `tests/integration_tests.rs` for full test suite.
