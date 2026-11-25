use std::net::SocketAddr;

use clap::Parser;
use url::Url;

use crate::error::{EdgeError, Result};

/// Runtime configuration for `llm-cache-edge`.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "llm-cache-edge",
    version,
    about = "Sub-millisecond L1/L2 semantic caching reverse proxy for LLM APIs"
)]
pub struct EdgeConfig {
    /// Local socket address to bind the reverse proxy.
    #[arg(long, env = "LLM_CACHE_BIND", default_value = "127.0.0.1:8000")]
    pub bind: SocketAddr,

    /// Upstream LLM base URL (e.g. http://127.0.0.1:11434 or https://api.openai.com).
    #[arg(
        long,
        env = "LLM_CACHE_UPSTREAM",
        default_value = "http://127.0.0.1:11434"
    )]
    pub upstream: String,

    /// Maximum memory budget for cached entries in MiB.
    #[arg(long, env = "LLM_CACHE_MAX_MEMORY_MB", default_value_t = 256)]
    pub max_memory_mb: usize,

    /// Default cache entry time-to-live in seconds.
    #[arg(long, env = "LLM_CACHE_TTL_SECS", default_value_t = 86_400)]
    pub ttl_secs: u64,

    /// Minimum sparse cosine similarity [0.0..=1.0] required for an L2 semantic hit.
    #[arg(long, env = "LLM_CACHE_SIMILARITY_THRESHOLD", default_value_t = 0.95)]
    pub similarity_threshold: f64,

    /// Optional per-chunk delay (ms) when replaying SSE cache hits (0 = instant turbo replay).
    #[arg(long, env = "LLM_CACHE_STREAM_DELAY_MS", default_value_t = 0)]
    pub stream_delay_ms: u64,

    /// Maximum incoming request body size in MiB.
    #[arg(long, env = "LLM_CACHE_MAX_BODY_MB", default_value_t = 10)]
    pub max_body_mb: usize,
}

impl EdgeConfig {
    /// Validate configuration invariants and return normalized upstream URL.
    pub fn validated_upstream(&self) -> Result<String> {
        let parsed = Url::parse(&self.upstream)?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(EdgeError::Config(format!(
                "unsupported upstream scheme '{}'",
                parsed.scheme()
            )));
        }
        if !(0.5..=1.0).contains(&self.similarity_threshold) {
            return Err(EdgeError::Config(
                "similarity_threshold must be between 0.5 and 1.0".into(),
            ));
        }
        Ok(self.upstream.trim_end_matches('/').to_string())
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8000".parse().expect("valid default SocketAddr"),
            upstream: "http://127.0.0.1:11434".to_string(),
            max_memory_mb: 256,
            ttl_secs: 86_400,
            similarity_threshold: 0.95,
            stream_delay_ms: 0,
            max_body_mb: 10,
        }
    }
}
