pub mod simhash;

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

use crate::canonical::CanonicalRequest;
use crate::metrics::EdgeMetrics;

use self::simhash::{compute_signature, sparse_cosine, SemanticSignature};

/// Cached completion payload supporting both JSON and token-by-token SSE replay.
#[derive(Debug)]
pub struct CacheEntry {
    pub l1_key: [u8; 32],
    pub partition_hash: u64,
    pub model: String,
    pub simhash: u64,
    pub sparse_vec: Vec<(u64, f32)>,
    pub completion_text: String,
    pub raw_json_body: Option<Vec<u8>>,
    pub tokens_estimate: u64,
    pub created_at_secs: u64,
    pub expires_at_secs: u64,
    pub last_accessed_ns: AtomicU64,
    pub approx_bytes: usize,
}

#[derive(Debug, Clone)]
pub enum CacheHit {
    L1Exact(Arc<CacheEntry>),
    L2Semantic {
        entry: Arc<CacheEntry>,
        similarity: f64,
    },
}

/// Sharded L1 (SHA-256) + L2 (SimHash-64 + Sparse Cosine) bounded-memory cache engine.
pub struct CacheEngine {
    l1: DashMap<[u8; 32], Arc<CacheEntry>>,
    /// 4 x 16-bit SimHash inverted block tables for O(1) candidate retrieval.
    l2_blocks: [DashMap<u16, Vec<[u8; 32]>>; 4],
    max_memory_bytes: usize,
    ttl_secs: u64,
    similarity_threshold: f64,
    metrics: Arc<EdgeMetrics>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

impl CacheEngine {
    pub fn new(
        max_memory_mb: usize,
        ttl_secs: u64,
        similarity_threshold: f64,
        metrics: Arc<EdgeMetrics>,
    ) -> Self {
        Self {
            l1: DashMap::with_shard_amount(64),
            l2_blocks: [
                DashMap::with_shard_amount(32),
                DashMap::with_shard_amount(32),
                DashMap::with_shard_amount(32),
                DashMap::with_shard_amount(32),
            ],
            max_memory_bytes: max_memory_mb.max(1) * 1024 * 1024,
            ttl_secs,
            similarity_threshold,
            metrics,
        }
    }

    /// Look up a canonicalized request across L1 exact match and L2 semantic similarity.
    pub fn lookup(&self, req: &CanonicalRequest) -> Option<CacheHit> {
        let now = now_secs();

        // Tier 1: Exact SHA-256 lookup
        if let Some(entry_ref) = self.l1.get(&req.l1_key) {
            let entry = Arc::clone(entry_ref.value());
            drop(entry_ref);
            if entry.expires_at_secs > now {
                entry.last_accessed_ns.store(now_nanos(), Ordering::Relaxed);
                return Some(CacheHit::L1Exact(entry));
            }
            self.remove_entry(&req.l1_key);
        }

        // Tier 2: 64-bit SimHash + 4x16-bit inverted block index + Sparse Cosine verification
        let sig: SemanticSignature = compute_signature(&req.semantic_text);
        let mut seen_keys: HashSet<[u8; 32]> = HashSet::new();
        for (band, table) in self.l2_blocks.iter().enumerate() {
            let chunk = ((sig.simhash >> (band * 16)) & 0xffff) as u16;
            if let Some(keys) = table.get(&chunk) {
                for key in keys.iter() {
                    seen_keys.insert(*key);
                }
            }
        }

        let mut best: Option<(Arc<CacheEntry>, f64)> = None;
        for key in seen_keys {
            let Some(entry_ref) = self.l1.get(&key) else {
                continue;
            };
            let entry = Arc::clone(entry_ref.value());
            drop(entry_ref);

            if entry.expires_at_secs <= now {
                self.remove_entry(&key);
                continue;
            }
            if entry.partition_hash != req.partition_hash {
                continue;
            }

            let hamming = (sig.simhash ^ entry.simhash).count_ones();
            if hamming > 10 {
                continue;
            }

            let cos = sparse_cosine(&sig.sparse_vec, &entry.sparse_vec);
            if cos >= self.similarity_threshold
                && best.as_ref().map_or(true, |(_, best_sim)| cos > *best_sim)
            {
                best = Some((entry, cos));
            }
        }

        if let Some((entry, similarity)) = best {
            entry.last_accessed_ns.store(now_nanos(), Ordering::Relaxed);
            return Some(CacheHit::L2Semantic { entry, similarity });
        }

        None
    }

    /// Insert a completed response into both L1 and L2 indices, enforcing memory watermarks.
    pub fn insert(
        &self,
        req: &CanonicalRequest,
        completion_text: String,
        raw_json_body: Option<Vec<u8>>,
        tokens_estimate: u64,
    ) {
        let sig = compute_signature(&req.semantic_text);
        let now = now_secs();
        let raw_len = raw_json_body.as_ref().map_or(0, Vec::len);
        let approx_bytes = 256
            + req.model.len()
            + completion_text.len()
            + raw_len
            + sig.sparse_vec.len() * std::mem::size_of::<(u64, f32)>();

        let entry = Arc::new(CacheEntry {
            l1_key: req.l1_key,
            partition_hash: req.partition_hash,
            model: req.model.clone(),
            simhash: sig.simhash,
            sparse_vec: sig.sparse_vec,
            completion_text,
            raw_json_body,
            tokens_estimate: tokens_estimate.max(1),
            created_at_secs: now,
            expires_at_secs: now.saturating_add(self.ttl_secs),
            last_accessed_ns: AtomicU64::new(now_nanos()),
            approx_bytes,
        });

        if let Some(prev) = self.l1.insert(req.l1_key, entry) {
            self.metrics
                .bytes_used
                .fetch_sub(prev.approx_bytes as u64, Ordering::Relaxed);
        } else {
            self.metrics.entries_count.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .bytes_used
            .fetch_add(approx_bytes as u64, Ordering::Relaxed);

        for (band, table) in self.l2_blocks.iter().enumerate() {
            let chunk = ((sig.simhash >> (band * 16)) & 0xffff) as u16;
            table.entry(chunk).or_default().push(req.l1_key);
        }

        self.enforce_memory_budget();
    }

    fn remove_entry(&self, key: &[u8; 32]) {
        if let Some((_, removed)) = self.l1.remove(key) {
            self.metrics
                .bytes_used
                .fetch_sub(removed.approx_bytes as u64, Ordering::Relaxed);
            self.metrics.entries_count.fetch_sub(1, Ordering::Relaxed);
            self.metrics.evictions_total.fetch_add(1, Ordering::Relaxed);
            for (band, table) in self.l2_blocks.iter().enumerate() {
                let chunk = ((removed.simhash >> (band * 16)) & 0xffff) as u16;
                if let Some(mut list) = table.get_mut(&chunk) {
                    list.retain(|k| k != key);
                }
            }
        }
    }

    fn enforce_memory_budget(&self) {
        let high_watermark = (self.max_memory_bytes * 9) / 10;
        let low_watermark = (self.max_memory_bytes * 8) / 10;
        if self.metrics.bytes_used.load(Ordering::Relaxed) as usize <= high_watermark {
            return;
        }

        let mut candidates: Vec<([u8; 32], u64)> = self
            .l1
            .iter()
            .map(|r| (*r.key(), r.value().last_accessed_ns.load(Ordering::Relaxed)))
            .collect();
        candidates.sort_unstable_by_key(|(_, ts)| *ts);

        for (key, _) in candidates {
            if self.metrics.bytes_used.load(Ordering::Relaxed) as usize <= low_watermark {
                break;
            }
            self.remove_entry(&key);
        }
    }
}
