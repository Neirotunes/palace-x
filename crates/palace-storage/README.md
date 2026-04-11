# palace-storage

Concrete in-memory `MemoryProvider` implementation for the Palace-X project. This crate combines NSW (Navigable Small World) indexing, topological reranking, and bit-plane storage into a complete two-stage retrieval pipeline for hierarchical memory in autonomous agents.

## Architecture

### Core Components

1. **NSW Index** (`palace-graph`)
   - Flat Navigable Small World graph with Hub-Highway optimization
   - O(log N) amortized insertion and search
   - Configurable neighborhood size (M) and construction parameters (ef)

2. **Topological Reranking** (`palace-topo`)
   - Structure-aware candidate refinement using ego-graphs
   - Combines cosine similarity with topological density (β₁ metric)
   - Stage 2 refinement for improved relevance ranking

3. **Bit-Plane Storage** (`palace-bitplane`)
   - Precision-proportional vector retrieval
   - Coarse planes (sign + exponent) in RAM
   - Mantissa planes tiered to disk (in production)

### Two-Stage Retrieval Pipeline

**Stage 1: Coarse Search**
- Binary/Hamming distance search via NSW
- Fast candidate generation
- Uses `nsw.search()` with configurable ef parameter

**Stage 2: Topological Reranking** (Optional)
- For top-K candidates from Stage 1
- Builds 2-hop ego-graphs around each candidate
- Computes d_total metric combining:
  - Cosine similarity (α weight)
  - Topological density (β weight)
- Returns reranked results sorted by combined score

**Graceful Degradation**
- If reranking fails: fall back to cosine-only ranking
- Logged as degraded mode event via `tracing::warn!`

## Usage

### Basic Creation

```rust
use palace_storage::MemoryPalace;
use palace_core::{MetaData, SearchConfig};

// Create with default parameters
let palace = MemoryPalace::new(128); // 128-dimensional vectors

// Or with custom configuration
let palace = MemoryPalace::with_config(
    128,           // dimensions
    32,            // max_neighbors (M parameter)
    200,           // ef_construction
    0.7,           // alpha (cosine weight)
    0.3,           // beta (topological weight)
);
```

### Ingestion

```rust
let vector = vec![0.1; 128];
let meta = MetaData::new(
    1234567890,    // timestamp
    "user_input",  // source
);

let node_id = palace.ingest(vector, meta).await?;
```

### Retrieval

```rust
let query = vec![0.2; 128];
let config = SearchConfig {
    limit: 10,
    enable_reranking: true,
    alpha: 0.7,
    beta: 0.3,
    rerank_k: 20,  // Stage 1 candidate count
};

let results = palace.retrieve(&query, &config).await?;
```

### Deletion and Management

```rust
// Vacuum (delete) nodes
let deleted = palace.vacuum(&[node_id1, node_id2]).await?;

// Get current size
let count = palace.len().await;

// Inspect statistics
let stats = palace.stats();
println!("Nodes: {}", stats.total_nodes);
println!("Memory: {} bytes", stats.memory_usage_bytes);
println!("Hub score: {}", stats.avg_hub_score);
```

## Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimensions` | User-specified | Vector embedding dimensionality |
| `max_neighbors` | 32 | NSW M parameter (max neighbors per node) |
| `ef_construction` | 200 | NSW construction-phase ef parameter |
| `alpha` | 0.7 | Cosine similarity weight in reranking |
| `beta` | 0.3 | Topological distance weight in reranking |

### SearchConfig Defaults

```rust
let config = SearchConfig::default_with_limit(10);
// Results in:
// - limit: 10
// - enable_reranking: true
// - alpha: 0.7
// - beta: 0.3
// - rerank_k: 20 (limit * 2)
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Ingest | O(log N) | NSW insertion + bit-plane storage |
| Retrieve (Stage 1) | O(ef * log N) | NSW search with ef ≈ rerank_k |
| Retrieve (Stage 2) | O(rerank_k * graph_size) | Ego-graph construction + scoring |
| Vacuum | O(k * degree) | k = nodes to delete, degree = avg neighbors |
| len() | O(1) | Direct NSW counter |

### Space Complexity

Per node:
- **NSW Graph**: ~(dimensions * 4) + (max_neighbors * 8) + overhead
- **Bit-Plane Coarse**: ~(dimensions / 8) + 8 * (dimensions / 8) bytes
- **Vector Store**: dimensions * 4 bytes (f32)
- **Metadata**: variable (source string, tags, etc.)

## Error Handling

### MemoryError Types

```rust
pub enum MemoryError {
    DimensionMismatch { expected, got },
    IndexFull { capacity },
    StorageError(String),
    GraphCorrupted(String),
    DegradedMode { reason, fallback_used },
    NodeNotFound { id },
}
```

All async operations return `Result<T, MemoryError>`:
- Dimension validation on ingest and query
- Graceful degradation when reranking unavailable
- Storage layer error propagation

## Testing

Run the test suite:

```bash
cargo test -p palace-storage
```

Key test coverage:
- ✓ Ingest dimension validation
- ✓ Round-trip ingest/retrieve
- ✓ Reranking enabled vs disabled
- ✓ Vacuum removes nodes correctly
- ✓ Statistics reporting
- ✓ Cosine distance computation
- ✓ Unique ID generation
- ✓ Empty index handling

## Statistics and Monitoring

The `stats()` method provides introspection:

```rust
pub struct PalaceStats {
    pub total_nodes: usize,
    pub dimensions: usize,
    pub memory_usage_bytes: usize,
    pub bitplane_coarse_bytes: usize,
    pub avg_hub_score: f32,
    pub max_hub_score: f32,
    pub hub_count: usize,
}
```

Memory estimation formulas:
- NSW: (dims * 4) + (max_neighbors * 8) + 64 bytes per node
- BitPlane coarse: (dims / 8) * 9 bytes per node
- Total includes hash map overhead and locks

## Implementation Notes

### Design Decisions

1. **HashMap-based Storage**: Used for simplicity and flexibility
   - In production, would use custom memory-mapped structures
   - Current impl suitable for agents with moderate memory constraints

2. **Two-Stage Pipeline**:
   - Stage 1 uses NSW's native cosine distance (fast)
   - Stage 2 builds ego-graphs for structural analysis
   - Config allows disabling reranking for speed vs quality tradeoff

3. **Graceful Degradation**:
   - If ego-graph construction fails, falls back to cosine ranking
   - Logged via tracing module for monitoring

4. **Node ID Mapping**:
   - palace-core uses `NodeId(u64)` newtype
   - palace-graph uses `NodeId = u64` alias
   - Conversion via `.0` accessor throughout

### Future Optimizations

- [ ] Memory-mapped bit-plane storage for mantissa planes
- [ ] Incremental hub score updates (currently static)
- [ ] Lazy ego-graph construction with caching
- [ ] Concurrent reranking for multiple queries
- [ ] Compression of metadata strings
- [ ] Tombstone-based deletion instead of immediate removal

## Dependencies

- `palace-core`: Core traits and types
- `palace-graph`: NSW index implementation
- `palace-topo`: Topological reranking
- `palace-bitplane`: Bit-plane storage
- `palace-quant`: Quantization utilities (future)
- `tokio`: Async runtime
- `parking_lot`: Efficient locks
- `tracing`: Structured logging
- `async-trait`: Async trait implementations

## License

MIT OR Apache-2.0 (same as Palace-X)
