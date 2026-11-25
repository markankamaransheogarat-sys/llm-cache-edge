use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Lock-free atomic telemetry counters for `/metrics` and `/healthz`.
#[derive(Debug)]
pub struct EdgeMetrics {
    pub requests_total: AtomicU64,
    pub l1_hits_total: AtomicU64,
    pub l2_hits_total: AtomicU64,
    pub misses_total: AtomicU64,
    pub tokens_saved_total: AtomicU64,
    pub evictions_total: AtomicU64,
    pub upstream_errors_total: AtomicU64,
    pub bytes_used: AtomicU64,
    pub entries_count: AtomicU64,
    pub started_at: Instant,
}

impl Default for EdgeMetrics {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            l1_hits_total: AtomicU64::new(0),
            l2_hits_total: AtomicU64::new(0),
            misses_total: AtomicU64::new(0),
            tokens_saved_total: AtomicU64::new(0),
            evictions_total: AtomicU64::new(0),
            upstream_errors_total: AtomicU64::new(0),
            bytes_used: AtomicU64::new(0),
            entries_count: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }
}

impl EdgeMetrics {
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    pub fn hit_rate(&self) -> f64 {
        let reqs = self.requests_total.load(Ordering::Relaxed);
        if reqs == 0 {
            return 0.0;
        }
        let hits =
            self.l1_hits_total.load(Ordering::Relaxed) + self.l2_hits_total.load(Ordering::Relaxed);
        hits as f64 / reqs as f64
    }

    pub fn render_prometheus(&self) -> String {
        format!(
            "# HELP llm_cache_requests_total Total cacheable LLM requests received.\n\
             # TYPE llm_cache_requests_total counter\n\
             llm_cache_requests_total {}\n\
             # HELP llm_cache_l1_hits_total Total L1 exact SHA-256 cache hits.\n\
             # TYPE llm_cache_l1_hits_total counter\n\
             llm_cache_l1_hits_total {}\n\
             # HELP llm_cache_l2_hits_total Total L2 SimHash semantic cache hits.\n\
             # TYPE llm_cache_l2_hits_total counter\n\
             llm_cache_l2_hits_total {}\n\
             # HELP llm_cache_misses_total Total cache misses forwarded upstream.\n\
             # TYPE llm_cache_misses_total counter\n\
             llm_cache_misses_total {}\n\
             # HELP llm_cache_tokens_saved_total Estimated completion/total tokens served from cache.\n\
             # TYPE llm_cache_tokens_saved_total counter\n\
             llm_cache_tokens_saved_total {}\n\
             # HELP llm_cache_evictions_total Total cache entries evicted by LRU/TTL.\n\
             # TYPE llm_cache_evictions_total counter\n\
             llm_cache_evictions_total {}\n\
             # HELP llm_cache_upstream_errors_total Total upstream proxy errors.\n\
             # TYPE llm_cache_upstream_errors_total counter\n\
             llm_cache_upstream_errors_total {}\n\
             # HELP llm_cache_bytes_used Current estimated cache memory usage in bytes.\n\
             # TYPE llm_cache_bytes_used gauge\n\
             llm_cache_bytes_used {}\n\
             # HELP llm_cache_entries Current number of cached entries.\n\
             # TYPE llm_cache_entries gauge\n\
             llm_cache_entries {}\n\
             # HELP llm_cache_uptime_seconds Seconds since proxy startup.\n\
             # TYPE llm_cache_uptime_seconds gauge\n\
             llm_cache_uptime_seconds {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.l1_hits_total.load(Ordering::Relaxed),
            self.l2_hits_total.load(Ordering::Relaxed),
            self.misses_total.load(Ordering::Relaxed),
            self.tokens_saved_total.load(Ordering::Relaxed),
            self.evictions_total.load(Ordering::Relaxed),
            self.upstream_errors_total.load(Ordering::Relaxed),
            self.bytes_used.load(Ordering::Relaxed),
            self.entries_count.load(Ordering::Relaxed),
            self.uptime_secs(),
        )
    }
}
