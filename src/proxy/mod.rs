pub mod sse;

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::cache::{CacheEngine, CacheHit};
use crate::canonical::canonicalize_request;
use crate::config::EdgeConfig;
use crate::error::Result;
use crate::metrics::EdgeMetrics;

use self::sse::{
    extract_completion_from_json, extract_sse_deltas, synthesize_json_response,
    synthesize_sse_stream,
};

#[derive(Clone)]
pub struct AppState {
    pub cache: Arc<CacheEngine>,
    pub metrics: Arc<EdgeMetrics>,
    pub upstream: String,
    pub client: reqwest::Client,
    pub stream_delay_ms: u64,
    pub max_body_bytes: usize,
}

/// Build the Axum router (exposed for integration tests).
pub fn build_router(config: &EdgeConfig) -> Result<Router> {
    let upstream = config.validated_upstream()?;
    let metrics = Arc::new(EdgeMetrics::default());
    let cache = Arc::new(CacheEngine::new(
        config.max_memory_mb,
        config.ttl_secs,
        config.similarity_threshold,
        Arc::clone(&metrics),
    ));
    let state = AppState {
        cache,
        metrics,
        upstream,
        client: reqwest::Client::new(),
        stream_delay_ms: config.stream_delay_ms,
        max_body_bytes: config.max_body_mb.max(1) * 1024 * 1024,
    };

    Ok(Router::new()
        .route("/healthz", get(handle_healthz))
        .route("/metrics", get(handle_metrics))
        .route("/v1/chat/completions", post(handle_llm_post))
        .route("/v1/completions", post(handle_llm_post))
        .route("/v1/messages", post(handle_llm_post))
        .route("/api/chat", post(handle_llm_post))
        .route("/api/generate", post(handle_llm_post))
        .route("/{*path}", any(handle_passthrough))
        .with_state(state))
}

/// Run the reverse proxy server until SIGINT/SIGTERM.
pub async fn run_server(config: EdgeConfig) -> Result<()> {
    let router = build_router(&config)?;
    let listener = TcpListener::bind(config.bind).await?;
    tracing::info!(
        "llm-cache-edge listening on http://{} -> upstream {}",
        config.bind,
        config.upstream
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn handle_healthz(State(state): State<AppState>) -> Response {
    let body = serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": state.metrics.uptime_secs(),
        "hit_rate": state.metrics.hit_rate(),
        "entries": state.metrics.entries_count.load(Ordering::Relaxed),
        "bytes_used": state.metrics.bytes_used.load(Ordering::Relaxed),
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

async fn handle_metrics(State(state): State<AppState>) -> Response {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.metrics.render_prometheus(),
    )
        .into_response()
}

async fn handle_llm_post(State(state): State<AppState>, req: Request) -> Response {
    let started = Instant::now();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/v1/chat/completions".to_string());
    let req_headers = req.headers().clone();

    let body_bytes = match axum::body::to_bytes(req.into_body(), state.max_body_bytes).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
            )
                .into_response();
        }
    };

    let Some(canon) = canonicalize_request(&body_bytes) else {
        return forward_raw(
            &state,
            Method::POST,
            &path_and_query,
            &req_headers,
            body_bytes.to_vec(),
        )
        .await;
    };

    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    if let Some(hit) = state.cache.lookup(&canon) {
        let elapsed_us = started.elapsed().as_micros();
        let (status_label, sim_str, entry) = match hit {
            CacheHit::L1Exact(entry) => {
                state.metrics.l1_hits_total.fetch_add(1, Ordering::Relaxed);
                ("HIT-L1", "1.0000".to_string(), entry)
            }
            CacheHit::L2Semantic { entry, similarity } => {
                state.metrics.l2_hits_total.fetch_add(1, Ordering::Relaxed);
                ("HIT-L2", format!("{similarity:.4}"), entry)
            }
        };
        state
            .metrics
            .tokens_saved_total
            .fetch_add(entry.tokens_estimate, Ordering::Relaxed);

        let mut resp = if canon.stream {
            let stream = synthesize_sse_stream(
                canon.model,
                entry.completion_text.clone(),
                state.stream_delay_ms,
            );
            let mut r = Response::new(Body::from_stream(stream));
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            r.headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        } else {
            let json_bytes = entry.raw_json_body.clone().unwrap_or_else(|| {
                synthesize_json_response(
                    &canon.model,
                    &entry.completion_text,
                    entry.tokens_estimate,
                )
            });
            let mut r = Response::new(Body::from(json_bytes));
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            r
        };

        attach_cache_headers(resp.headers_mut(), status_label, &sim_str, elapsed_us);
        return resp;
    }

    // Cache MISS: forward upstream
    state.metrics.misses_total.fetch_add(1, Ordering::Relaxed);
    let target_url = format!("{}{}", state.upstream, path_and_query);
    let mut builder = state.client.post(&target_url);
    for (name, value) in &req_headers {
        if is_hop_by_hop(name) || name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(name.as_str(), value.as_bytes());
    }
    builder = builder.body(body_bytes.to_vec());

    let upstream_resp = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            state
                .metrics
                .upstream_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            )
                .into_response();
        }
    };

    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut resp_headers = copy_upstream_headers(upstream_resp.headers());
    let is_sse = upstream_resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));

    if is_sse && status.is_success() {
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        let cache = Arc::clone(&state.cache);
        let mut upstream_stream = upstream_resp.bytes_stream();

        tokio::spawn(async move {
            let mut accumulated = String::new();
            let mut saw_done = false;
            while let Some(chunk_res) = upstream_stream.next().await {
                let Ok(chunk) = chunk_res else {
                    break;
                };
                if let Ok(text) = std::str::from_utf8(&chunk) {
                    if extract_sse_deltas(text, &mut accumulated) {
                        saw_done = true;
                    }
                }
                if tx.send(chunk).await.is_err() {
                    break;
                }
            }
            if (saw_done || !accumulated.is_empty()) && !accumulated.is_empty() {
                let tokens = (accumulated.len() as u64 / 4).max(1);
                cache.insert(&canon, accumulated, None, tokens);
            }
        });

        let out_stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|b| (Ok::<Bytes, Infallible>(b), rx))
        });
        let mut resp = Response::new(Body::from_stream(out_stream));
        *resp.status_mut() = status;
        *resp.headers_mut() = resp_headers;
        attach_cache_headers(
            resp.headers_mut(),
            "MISS",
            "0.0000",
            started.elapsed().as_micros(),
        );
        return resp;
    }

    let resp_bytes = match upstream_resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            state
                .metrics
                .upstream_errors_total
                .fetch_add(1, Ordering::Relaxed);
            return (
                StatusCode::BAD_GATEWAY,
                format!("failed reading upstream response: {e}"),
            )
                .into_response();
        }
    };

    if status.is_success() {
        if let Some((text, tokens)) = extract_completion_from_json(&resp_bytes) {
            state
                .cache
                .insert(&canon, text, Some(resp_bytes.to_vec()), tokens);
        }
    }

    attach_cache_headers(
        &mut resp_headers,
        "MISS",
        "0.0000",
        started.elapsed().as_micros(),
    );
    let mut resp = Response::new(Body::from(resp_bytes));
    *resp.status_mut() = status;
    *resp.headers_mut() = resp_headers;
    resp
}

async fn handle_passthrough(State(state): State<AppState>, req: Request) -> Response {
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let req_headers = req.headers().clone();
    let body_bytes = match axum::body::to_bytes(req.into_body(), state.max_body_bytes).await {
        Ok(b) => b.to_vec(),
        Err(e) => return (StatusCode::BAD_REQUEST, format!("invalid body: {e}")).into_response(),
    };
    forward_raw(&state, method, &path_and_query, &req_headers, body_bytes).await
}

async fn forward_raw(
    state: &AppState,
    method: Method,
    path_and_query: &str,
    req_headers: &HeaderMap,
    body_bytes: Vec<u8>,
) -> Response {
    let target_url = format!("{}{}", state.upstream, path_and_query);
    let Ok(req_method) = reqwest::Method::from_bytes(method.as_str().as_bytes()) else {
        return (StatusCode::METHOD_NOT_ALLOWED, "unsupported method").into_response();
    };
    let mut builder = state.client.request(req_method, &target_url);
    for (name, value) in req_headers {
        if is_hop_by_hop(name) || name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(name.as_str(), value.as_bytes());
    }
    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }
    match builder.send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let headers = copy_upstream_headers(r.headers());
            let bytes = r.bytes().await.unwrap_or_default();
            let mut resp = Response::new(Body::from(bytes));
            *resp.status_mut() = status;
            *resp.headers_mut() = headers;
            resp
        }
        Err(e) => {
            state
                .metrics
                .upstream_errors_total
                .fetch_add(1, Ordering::Relaxed);
            (StatusCode::BAD_GATEWAY, format!("upstream failed: {e}")).into_response()
        }
    }
}

fn attach_cache_headers(headers: &mut HeaderMap, status: &str, similarity: &str, latency_us: u128) {
    if let Ok(v) = HeaderValue::from_str(status) {
        headers.insert(HeaderName::from_static("x-cache-status"), v);
    }
    if let Ok(v) = HeaderValue::from_str(similarity) {
        headers.insert(HeaderName::from_static("x-cache-similarity"), v);
    }
    if let Ok(v) = HeaderValue::from_str(&latency_us.to_string()) {
        headers.insert(HeaderName::from_static("x-cache-latency-us"), v);
    }
}

fn copy_upstream_headers(src: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in src {
        if is_hop_by_hop(name) || name == header::CONTENT_LENGTH {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            out.insert(n, v);
        }
    }
    out
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}
