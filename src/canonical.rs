use std::collections::BTreeMap;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

/// Canonicalized representation of an incoming LLM request.
#[derive(Debug, Clone)]
pub struct CanonicalRequest {
    pub model: String,
    pub stream: bool,
    pub l1_key: [u8; 32],
    pub partition_hash: u64,
    pub semantic_text: String,
}

/// Normalize text with Unicode NFKC, trimmed ends, and collapsed horizontal whitespace.
pub fn normalize_text(input: &str) -> String {
    let nfkc: String = input.nfkc().collect();
    let mut out = String::with_capacity(nfkc.len());
    let mut prev_space = false;
    for ch in nfkc.trim().chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Deterministically sort JSON object keys alphabetically.
fn sort_json(val: &Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), sort_json(v));
            }
            let mut out = Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_json).collect()),
        other => other.clone(),
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Canonicalize an OpenAI/Ollama/Anthropic JSON request body into L1/L2 lookup keys.
pub fn canonicalize_request(body: &[u8]) -> Option<CanonicalRequest> {
    let root: Value = serde_json::from_slice(body).ok()?;
    let obj = root.as_object()?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // Build canonical prompt text from `messages` or `prompt`.
    let mut prompt_parts = Vec::new();
    if let Some(messages) = obj.get("messages").and_then(Value::as_array) {
        for msg in messages {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            if let Some(content) = msg.get("content") {
                match content {
                    Value::String(s) => {
                        prompt_parts.push(format!("[{role}] {}", normalize_text(s)));
                    }
                    other => {
                        let serialized =
                            serde_json::to_string(&sort_json(other)).unwrap_or_default();
                        prompt_parts.push(format!("[{role}] {}", normalize_text(&serialized)));
                    }
                }
            }
        }
    } else {
        let prompt = obj.get("prompt").and_then(Value::as_str)?;
        prompt_parts.push(normalize_text(prompt));
    }

    if let Some(system) = obj.get("system").and_then(Value::as_str) {
        prompt_parts.insert(0, format!("[system] {}", normalize_text(system)));
    }

    let canonical_prompt = prompt_parts.join("\n");

    // Extract semantic-affecting parameters (strip `user`, `stream`, `seed`, `logit_bias`).
    let mut params = Map::new();
    for key in [
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "response_format",
    ] {
        if let Some(v) = obj.get(key) {
            params.insert(key.to_string(), sort_json(v));
        }
    }
    let params_json = serde_json::to_string(&Value::Object(params)).unwrap_or_default();

    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\x00");
    hasher.update(canonical_prompt.as_bytes());
    hasher.update(b"\x00");
    hasher.update(params_json.as_bytes());
    let l1_key: [u8; 32] = hasher.finalize().into();

    let partition_input = format!("{model}\x00{params_json}");
    let partition_hash = fnv1a_64(partition_input.as_bytes());

    Some(CanonicalRequest {
        model,
        stream,
        l1_key,
        partition_hash,
        semantic_text: canonical_prompt,
    })
}
