use std::collections::HashMap;

/// 64-bit semantic fingerprint paired with a normalized top-K sparse feature vector.
#[derive(Debug, Clone)]
pub struct SemanticSignature {
    pub simhash: u64,
    /// Sorted `(feature_hash, l2_normalized_weight)` pairs (top 32 features).
    pub sparse_vec: Vec<(u64, f32)>,
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "you", "your", "please",
    "can", "could", "would", "what", "how", "why",
];

#[inline]
fn hash_feature_64(bytes: &[u8]) -> u64 {
    // FNV-1a followed by SplitMix64 finalizer for strong avalanche across all 64 bits.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

#[inline]
fn is_stopword(word: &str) -> bool {
    STOPWORDS.binary_search(&word).is_ok()
}

/// Extract word unigrams, word bigrams, and character 3-grams, returning a 64-bit SimHash
/// and top-32 L2-normalized sparse feature vector for sub-200µs semantic verification.
pub fn compute_signature(text: &str) -> SemanticSignature {
    let lower = text.to_ascii_lowercase();
    let mut weights: HashMap<u64, f64> = HashMap::new();

    // 1. Word unigrams and bigrams
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    for &w in &words {
        let base_weight = if is_stopword(w) { 0.25 } else { 1.5 };
        let h = hash_feature_64(format!("w:{w}").as_bytes());
        *weights.entry(h).or_insert(0.0) += base_weight;
    }

    for pair in words.windows(2) {
        let h = hash_feature_64(format!("b:{}_{}", pair[0], pair[1]).as_bytes());
        *weights.entry(h).or_insert(0.0) += 1.25;
    }

    // 2. Character 3-grams on collapsed alphanumeric/space text
    let chars: Vec<char> = lower
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect();
    for tri in chars.windows(3) {
        let mut buf = [0u8; 16];
        let mut len = 0usize;
        buf[0] = b'c';
        buf[1] = b':';
        len += 2;
        for &ch in tri {
            let mut tmp = [0u8; 4];
            let encoded = ch.encode_utf8(&mut tmp).as_bytes();
            if len + encoded.len() <= buf.len() {
                buf[len..len + encoded.len()].copy_from_slice(encoded);
                len += encoded.len();
            }
        }
        let h = hash_feature_64(&buf[..len]);
        *weights.entry(h).or_insert(0.0) += 0.45;
    }

    // 3. 64-bit SimHash projection
    let mut accum = [0.0_f64; 64];
    for (&feat_hash, &w) in &weights {
        for (bit, slot) in accum.iter_mut().enumerate() {
            if (feat_hash >> bit) & 1 == 1 {
                *slot += w;
            } else {
                *slot -= w;
            }
        }
    }

    let mut simhash = 0_u64;
    for (bit, &val) in accum.iter().enumerate() {
        if val >= 0.0 {
            simhash |= 1_u64 << bit;
        }
    }

    // 4. Top-32 L2-normalized sparse vector (sorted by feature hash for O(N) merge dot-product)
    let mut entries: Vec<(u64, f64)> = weights.into_iter().collect();
    entries.sort_by(|a, b| b.1.total_cmp(&a.1));
    entries.truncate(32);

    let norm = entries.iter().map(|(_, w)| w * w).sum::<f64>().sqrt();
    let mut sparse_vec: Vec<(u64, f32)> = if norm > 0.0 {
        entries
            .into_iter()
            .map(|(h, w)| (h, (w / norm) as f32))
            .collect()
    } else {
        Vec::new()
    };
    sparse_vec.sort_unstable_by_key(|(h, _)| *h);

    SemanticSignature {
        simhash,
        sparse_vec,
    }
}

/// Cosine similarity between two sorted L2-normalized sparse vectors.
#[inline]
pub fn sparse_cosine(a: &[(u64, f32)], b: &[(u64, f32)]) -> f64 {
    let mut i = 0;
    let mut j = 0;
    let mut dot = 0.0_f64;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += f64::from(a[i].1) * f64::from(b[j].1);
                i += 1;
                j += 1;
            }
        }
    }
    dot.clamp(0.0, 1.0)
}
