use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use futures_util::stream::{self, Stream};
use serde_json::{json, Value};

/// Extract completion text and token count from a non-streaming JSON response body.
pub fn extract_completion_from_json(body: &[u8]) -> Option<(String, u64)> {
    let val: Value = serde_json::from_slice(body).ok()?;
    // OpenAI chat/completions
    if let Some(content) = val
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        let tokens = val
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| (content.len() as u64 / 4).max(1));
        return Some((content.to_string(), tokens));
    }
    // OpenAI legacy completions
    if let Some(text) = val.pointer("/choices/0/text").and_then(Value::as_str) {
        let tokens = val
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| (text.len() as u64 / 4).max(1));
        return Some((text.to_string(), tokens));
    }
    // Ollama /api/chat or /api/generate
    if let Some(content) = val
        .pointer("/message/content")
        .and_then(Value::as_str)
        .or_else(|| val.get("response").and_then(Value::as_str))
    {
        let tokens = (content.len() as u64 / 4).max(1);
        return Some((content.to_string(), tokens));
    }
    // Anthropic /v1/messages
    if let Some(text) = val.pointer("/content/0/text").and_then(Value::as_str) {
        let tokens = (text.len() as u64 / 4).max(1);
        return Some((text.to_string(), tokens));
    }
    None
}

/// Extract incremental text deltas from an SSE chunk (`data: {...}\n\n`).
pub fn extract_sse_deltas(chunk_text: &str, out: &mut String) -> bool {
    let mut done = false;
    for line in chunk_text.lines() {
        let trimmed = line.trim();
        let Some(payload) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            done = true;
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(payload) {
            if let Some(delta) = val
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                out.push_str(delta);
            } else if let Some(delta) = val.pointer("/delta/text").and_then(Value::as_str) {
                out.push_str(delta);
            }
        }
    }
    done
}

/// Build an OpenAI-compatible JSON completion response for non-streaming cache hits.
pub fn synthesize_json_response(model: &str, completion_text: &str, tokens: u64) -> Vec<u8> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let payload = json!({
        "id": format!("chatcmpl-cache-{created}"),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens,
            "total_tokens": tokens
        }
    });
    serde_json::to_vec(&payload).unwrap_or_default()
}

/// Synthesize a token-by-token OpenAI-compatible SSE stream for `stream: true` cache hits.
pub fn synthesize_sse_stream(
    model: String,
    completion_text: String,
    delay_ms: u64,
) -> impl Stream<Item = Result<Bytes, Infallible>> + Send + 'static {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let id = format!("chatcmpl-cache-{created}");

    let mut events = Vec::new();
    let first = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    });
    events.push(format!("data: {first}\n\n"));

    // Split completion into natural ~12-char UTF-8 boundary micro-chunks.
    let mut current = String::new();
    for ch in completion_text.chars() {
        current.push(ch);
        if current.len() >= 12 || ch == '\n' {
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": current}, "finish_reason": null}]
            });
            events.push(format!("data: {chunk}\n\n"));
            current.clear();
        }
    }
    if !current.is_empty() {
        let chunk = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": current}, "finish_reason": null}]
        });
        events.push(format!("data: {chunk}\n\n"));
    }

    let stop = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    events.push(format!("data: {stop}\n\n"));
    events.push("data: [DONE]\n\n".to_string());

    stream::unfold(events.into_iter(), move |mut iter| async move {
        let next = iter.next()?;
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
        Some((Ok::<Bytes, Infallible>(Bytes::from(next)), iter))
    })
}
