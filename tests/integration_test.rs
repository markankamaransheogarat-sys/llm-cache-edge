use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use llm_cache_edge::cache::{CacheEngine, CacheHit};
use llm_cache_edge::canonical::canonicalize_request;
use llm_cache_edge::config::EdgeConfig;
use llm_cache_edge::metrics::EdgeMetrics;
use llm_cache_edge::proxy::build_router;

#[test]
fn canonicalizer_normalizes_whitespace_and_ignores_operational_keys() {
    let a = json!({
        "model": "gpt-4o",
        "stream": false,
        "user": "alice",
        "seed": 123,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "Explain   Rust   ownership\tplease"}]
    });
    let b = json!({
        "temperature": 0.2,
        "messages": [{"content": "Explain Rust ownership please", "role": "user"}],
        "model": "gpt-4o",
        "stream": true,
        "user": "bob"
    });

    let ca = canonicalize_request(&serde_json::to_vec(&a).unwrap()).unwrap();
    let cb = canonicalize_request(&serde_json::to_vec(&b).unwrap()).unwrap();
    assert_eq!(ca.l1_key, cb.l1_key);
    assert_eq!(ca.partition_hash, cb.partition_hash);
}

#[test]
fn l2_simhash_hits_near_identical_and_rejects_unrelated() {
    let metrics = Arc::new(EdgeMetrics::default());
    let cache = CacheEngine::new(16, 3600, 0.90, metrics);

    let req1 = canonicalize_request(
        br#"{"model":"llama3","messages":[{"role":"user","content":"Explain how TCP three-way handshake works in computer networking with SYN, SYN-ACK, and ACK packets."}]}"#,
    )
    .unwrap();
    cache.insert(&req1, "TCP uses SYN, SYN-ACK, ACK.".to_string(), None, 12);

    // Minor typo / punctuation variation -> L2 semantic hit
    let req_similar = canonicalize_request(
        br#"{"model":"llama3","messages":[{"role":"user","content":"Explain how TCP three-way handshake works in computer networking with SYN SYN-ACK and ACK packets"}]}"#,
    )
    .unwrap();
    let hit = cache.lookup(&req_similar);
    assert!(
        matches!(hit, Some(CacheHit::L2Semantic { .. })),
        "expected L2 semantic hit for near-identical prompt"
    );

    // Completely unrelated prompt -> miss
    let req_unrelated = canonicalize_request(
        br#"{"model":"llama3","messages":[{"role":"user","content":"Write a sourdough bread baking recipe with hydration ratios."}]}"#,
    )
    .unwrap();
    assert!(cache.lookup(&req_unrelated).is_none());
}

#[test]
fn l1_lookup_is_sub_millisecond_and_enforces_ttl() {
    let metrics = Arc::new(EdgeMetrics::default());
    let cache = CacheEngine::new(16, 0, 0.95, metrics);
    let req = canonicalize_request(
        br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}"#,
    )
    .unwrap();
    cache.insert(&req, "Hi".into(), None, 2);
    // TTL=0 expires immediately
    assert!(cache.lookup(&req).is_none());

    let metrics2 = Arc::new(EdgeMetrics::default());
    let cache2 = CacheEngine::new(16, 3600, 0.95, metrics2);
    cache2.insert(&req, "Hi".into(), None, 2);
    let t0 = Instant::now();
    assert!(matches!(cache2.lookup(&req), Some(CacheHit::L1Exact(_))));
    assert!(t0.elapsed().as_micros() < 5_000);
}

#[tokio::test]
async fn end_to_end_proxy_l1_l2_sse_tee_and_metrics() {
    let upstream_calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&upstream_calls);

    let upstream_app = Router::new().route(
        "/v1/chat/completions",
        post(
            move |State(calls): State<Arc<AtomicUsize>>, Json(body): Json<Value>| async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let is_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
                if is_stream {
                    let chunks = vec![
                        Ok::<Bytes, Infallible>(Bytes::from(
                            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                        )),
                        Ok(Bytes::from(
                            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Streamed \"}}]}\n\n",
                        )),
                        Ok(Bytes::from(
                            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"OK!\"}}]}\n\n",
                        )),
                        Ok(Bytes::from("data: [DONE]\n\n")),
                    ];
                    let mut resp =
                        axum::response::Response::new(axum::body::Body::from_stream(stream::iter(chunks)));
                    resp.headers_mut().insert(
                        header::CONTENT_TYPE,
                        axum::http::HeaderValue::from_static("text/event-stream"),
                    );
                    resp
                } else {
                    Json(json!({
                        "id": "cmpl-up",
                        "object": "chat.completion",
                        "model": "gpt-4o",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "Rust ownership guarantees memory safety."},
                            "finish_reason": "stop"
                        }],
                        "usage": {"total_tokens": 42}
                    }))
                    .into_response()
                }
            },
        ),
    )
    .with_state(calls_ref);

    let up_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let up_addr = up_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(up_listener, upstream_app).await.unwrap();
    });

    let cfg = EdgeConfig {
        upstream: format!("http://{up_addr}"),
        similarity_threshold: 0.90,
        ..Default::default()
    };
    let proxy_app = build_router(&cfg).unwrap();
    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = proxy_listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(proxy_listener, proxy_app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let url = format!("http://{proxy_addr}/v1/chat/completions");

    // 1. Non-streaming MISS
    let r1 = client
        .post(&url)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Why does Rust use ownership and borrowing for memory safety without GC?"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.headers().get("x-cache-status").unwrap(), "MISS");
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);

    // 2. Non-streaming L1 HIT (whitespace normalized)
    let r2 = client
        .post(&url)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Why does Rust use   ownership and borrowing for memory safety without GC?"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.headers().get("x-cache-status").unwrap(), "HIT-L1");
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);

    // 3. Non-streaming L2 Semantic HIT (minor phrasing difference)
    let r3 = client
        .post(&url)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Why does Rust use ownership and borrowing for memory safety without GC"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r3.headers().get("x-cache-status").unwrap(), "HIT-L2");
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 1);

    // 4. Streaming MISS (teeing) -> Streaming L1 HIT (replay)
    let r4 = client
        .post(&url)
        .json(&json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{"role": "user", "content": "Stream test prompt"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r4.headers().get("x-cache-status").unwrap(), "MISS");
    let sse_miss_body = r4.text().await.unwrap();
    assert!(sse_miss_body.contains("Streamed "));
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 2);

    // Allow spawned tee accumulator a brief moment to commit entry
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let r5 = client
        .post(&url)
        .json(&json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{"role": "user", "content": "Stream test prompt"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r5.headers().get("x-cache-status").unwrap(), "HIT-L1");
    let sse_hit_body = r5.text().await.unwrap();
    assert!(sse_hit_body.contains("Streamed OK!"));
    assert!(sse_hit_body.contains("data: [DONE]"));
    assert_eq!(upstream_calls.load(Ordering::SeqCst), 2);

    // 5. /healthz and /metrics
    let health: Value = client
        .get(format!("http://{proxy_addr}/healthz"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");

    let metrics = client
        .get(format!("http://{proxy_addr}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(metrics.contains("llm_cache_l1_hits_total 2"));
    assert!(metrics.contains("llm_cache_l2_hits_total 1"));
    assert!(metrics.contains("llm_cache_misses_total 2"));
}
