// Copyright (c) 2026 Maksym Dyachenko <max@neirosynth.com>
// Proprietary — All Rights Reserved
//
// palace-mcp — MCP (Model Context Protocol) server for Palace-X.
//
// Exposes three tools over JSON-RPC 2.0 / stdio:
//   • palace_ingest  — add a vector + metadata to the index
//   • palace_search  — ANN search, returns top-k fragments
//   • palace_stats   — index diagnostics (node count, memory, hub scores)
//
// Usage (Claude Code ~/.mcp.json):
//   {
//     "mcpServers": {
//       "palace": {
//         "command": "cargo",
//         "args": ["run", "--bin", "palace-mcp", "--manifest-path", "<path>/Cargo.toml"],
//         "env": { "PALACE_DIMS": "128" }
//       }
//     }
//   }

use palace_core::{MetaData, SearchConfig};
use palace_engine::PalaceEngine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// ─── Text store ──────────────────────────────────────────────────────────────

/// Stores original text alongside ingested vectors for retrieval.
type TextStore = Arc<RwLock<HashMap<u64, String>>>;

// ─── JSON-RPC 2.0 wire types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Request {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

impl Response {
    fn ok(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0", id, result: Some(result), error: None }
    }
    fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError { code, message: message.into() }),
        }
    }
}

// ─── Tool schemas ─────────────────────────────────────────────────────────────

fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "palace_ingest",
                "description": "Ingest a float32 vector with metadata into the Palace-X index. Returns the assigned NodeId.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vector": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Float32 embedding vector. Length must match PALACE_DIMS (default 128)."
                        },
                        "text": {
                            "type": "string",
                            "description": "Original text content associated with this vector. Stored alongside the vector and returned in search results."
                        },
                        "source": {
                            "type": "string",
                            "description": "Origin label for the fragment (e.g. 'user_input', 'document_chunk').",
                            "default": "mcp"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional searchable tags.",
                            "default": []
                        }
                    },
                    "required": ["vector"]
                }
            },
            {
                "name": "palace_search",
                "description": "Approximate nearest-neighbor search in the Palace-X index. Returns up to `limit` scored fragments.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vector": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Query float32 vector. Length must match PALACE_DIMS."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["vector"]
                }
            },
            {
                "name": "palace_stats",
                "description": "Return diagnostic statistics for the Palace-X index: node count, dimensions, memory usage, hub distribution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

// ─── Dimension validation ────────────────────────────────────────────────────

fn validate_vector(raw: &[Value], expected_dims: usize, tool: &str) -> Result<Vec<f32>, String> {
    if raw.len() != expected_dims {
        return Err(format!(
            "{}: vector length {} does not match expected dimensions {}",
            tool,
            raw.len(),
            expected_dims
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| format!("{}: vector[{}] is not a number", tool, i))
        })
        .collect()
}

// ─── Tool dispatch ────────────────────────────────────────────────────────────

async fn handle_ingest(
    engine: &PalaceEngine,
    texts: &TextStore,
    dims: usize,
    params: Option<Value>,
) -> Result<Value, String> {
    let params = params.ok_or("palace_ingest requires parameters")?;

    let raw_vec = params
        .get("vector")
        .and_then(|v| v.as_array())
        .ok_or("palace_ingest: 'vector' must be an array of numbers")?;
    let vector = validate_vector(raw_vec, dims, "palace_ingest")?;

    // text content (optional)
    let text = params
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // metadata
    let source = params
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("mcp")
        .to_string();
    let tags: Vec<String> = params
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut metadata = MetaData::new(ts, source);
    metadata.tags = tags;
    if let Some(ref t) = text {
        metadata.extra.insert("text".to_string(), t.clone());
    }

    let node_id = engine.ingest(vector, metadata).await.map_err(|e| e.to_string())?;

    // Store text for retrieval
    if let Some(t) = text {
        texts.write().await.insert(node_id.0, t);
    }

    Ok(json!({ "node_id": node_id.0 }))
}

async fn handle_search(
    engine: &PalaceEngine,
    texts: &TextStore,
    dims: usize,
    params: Option<Value>,
) -> Result<Value, String> {
    let params = params.ok_or("palace_search requires parameters")?;

    let raw_vec = params
        .get("vector")
        .and_then(|v| v.as_array())
        .ok_or("palace_search: 'vector' must be an array of numbers")?;
    let vector = validate_vector(raw_vec, dims, "palace_search")?;

    let limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .max(1)
        .min(100) as usize;

    let config = SearchConfig::default_with_limit(limit);
    let fragments = engine.search(vector, config).await.map_err(|e| e.to_string())?;

    let store = texts.read().await;
    let results: Vec<Value> = fragments
        .iter()
        .map(|f| {
            let mut entry = json!({
                "node_id": f.node_id.0,
                "score":   f.score,
                "source":  f.metadata.source,
                "tags":    f.metadata.tags,
                "timestamp": f.metadata.timestamp,
            });
            if let Some(t) = store.get(&f.node_id.0) {
                entry["text"] = json!(t);
            }
            entry
        })
        .collect();

    let count = results.len();
    Ok(json!({ "results": results, "count": count }))
}

async fn handle_stats(engine: &PalaceEngine, texts: &TextStore) -> Result<Value, String> {
    let stats = engine.stats().await.map_err(|e| e.to_string())?;
    let text_count = texts.read().await.len();
    Ok(json!({
        "total_nodes":           stats.total_nodes,
        "dimensions":            stats.dimensions,
        "memory_usage_bytes":    stats.memory_usage_bytes,
        "bitplane_coarse_bytes": stats.bitplane_coarse_bytes,
        "avg_hub_score":         stats.avg_hub_score,
        "max_hub_score":         stats.max_hub_score,
        "hub_count":             stats.hub_count,
        "text_fragments":        text_count,
    }))
}

// ─── MCP method dispatch ──────────────────────────────────────────────────────

async fn dispatch(
    engine: Arc<PalaceEngine>,
    texts: TextStore,
    dims: usize,
    req: Request,
) -> Response {
    let id = req.id.unwrap_or(Value::Null);

    match req.method.as_str() {
        // ── MCP lifecycle ─────────────────────────────────────────────────────
        "initialize" => {
            info!("MCP initialize");
            Response::ok(
                id,
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": { "subscribe": false, "listChanged": false }
                    },
                    "serverInfo": {
                        "name":    "palace-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
        }
        "notifications/initialized" => {
            debug!("Client initialized");
            Response::ok(id, Value::Null)
        }

        // ── Tool listing ──────────────────────────────────────────────────────
        "tools/list" => {
            debug!("tools/list");
            Response::ok(id, tools_list())
        }

        // ── Tool calls ────────────────────────────────────────────────────────
        "tools/call" => {
            let params = req.params.unwrap_or(Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tool_args = params.get("arguments").cloned();

            debug!("tools/call name={}", tool_name);

            let result = match tool_name {
                "palace_ingest" => handle_ingest(&engine, &texts, dims, tool_args).await,
                "palace_search" => handle_search(&engine, &texts, dims, tool_args).await,
                "palace_stats"  => handle_stats(&engine, &texts).await,
                other => Err(format!("Unknown tool: {}", other)),
            };

            match result {
                Ok(data) => Response::ok(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&data)
                                .unwrap_or_else(|_| data.to_string())
                        }]
                    }),
                ),
                Err(msg) => {
                    warn!("Tool error: {}", msg);
                    Response::ok(
                        id,
                        json!({
                            "content": [{ "type": "text", "text": msg }],
                            "isError": true
                        }),
                    )
                }
            }
        }

        // ── Resources ─────────────────────────────────────────────────────────
        "resources/list" => {
            debug!("resources/list");
            let stats = engine.stats().await;
            let total = stats.map(|s| s.total_nodes).unwrap_or(0);
            Response::ok(
                id,
                json!({
                    "resources": [{
                        "uri": "palace://index/stats",
                        "name": "Palace-X Index Stats",
                        "description": format!(
                            "Index with {} vectors, {}d",
                            total, dims
                        ),
                        "mimeType": "application/json"
                    }]
                }),
            )
        }
        "resources/read" => {
            let uri = req
                .params
                .as_ref()
                .and_then(|p| p.get("uri"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            debug!("resources/read uri={}", uri);
            match uri {
                "palace://index/stats" => {
                    match handle_stats(&engine, &texts).await {
                        Ok(data) => Response::ok(
                            id,
                            json!({
                                "contents": [{
                                    "uri": uri,
                                    "mimeType": "application/json",
                                    "text": serde_json::to_string_pretty(&data)
                                        .unwrap_or_else(|_| data.to_string())
                                }]
                            }),
                        ),
                        Err(msg) => Response::err(id, -32000, msg),
                    }
                }
                other => Response::err(id, -32002, format!("Unknown resource: {}", other)),
            }
        }

        // ── Ping ──────────────────────────────────────────────────────────────
        "ping" => Response::ok(id, json!({})),

        // ── Unknown ───────────────────────────────────────────────────────────
        other => {
            warn!("Unknown method: {}", other);
            Response::err(id, -32601, format!("Method not found: {}", other))
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Logging goes to stderr so stdout stays clean for JSON-RPC.
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "palace_mcp=info".to_string()),
        )
        .init();

    // Dimensionality from env (default 128)
    let dims: usize = std::env::var("PALACE_DIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    info!("Starting palace-mcp, dims={}", dims);
    let engine = Arc::new(PalaceEngine::start(dims));
    let texts: TextStore = Arc::new(RwLock::new(HashMap::new()));

    // Async stdio — does not block the tokio runtime
    let stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut lines = stdin.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                error!("JSON parse error: {}", e);
                let resp = Response::err(Value::Null, -32700, format!("Parse error: {}", e));
                let out = format!("{}\n", serde_json::to_string(&resp).unwrap());
                let _ = stdout.write_all(out.as_bytes()).await;
                let _ = stdout.flush().await;
                continue;
            }
        };

        // MCP notifications must not receive a response
        let is_notification = req.id.is_none()
            || req.method.starts_with("notifications/");
        let resp = dispatch(Arc::clone(&engine), Arc::clone(&texts), dims, req).await;

        if !is_notification {
            match serde_json::to_string(&resp) {
                Ok(json) => {
                    let out = format!("{}\n", json);
                    if let Err(e) = stdout.write_all(out.as_bytes()).await {
                        error!("write error: {}", e);
                        break;
                    }
                    if let Err(e) = stdout.flush().await {
                        error!("flush error: {}", e);
                        break;
                    }
                }
                Err(e) => error!("serialize error: {}", e),
            }
        }
    }

    info!("palace-mcp shutting down");
}
