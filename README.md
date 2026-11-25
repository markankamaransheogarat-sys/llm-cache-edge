# llm-cache-edge

**Sub-millisecond L1/L2 Semantic Caching Reverse Proxy for OpenAI, Ollama, vLLM & Anthropic** — written in `#![forbid(unsafe_code)]` Rust.

See [RFC-0001-ARCHITECTURE-SPECIFICATION.md](./RFC-0001-ARCHITECTURE-SPECIFICATION.md) for the full design.

## Features
- **L1 Exact Match (`< 10 µs`)**: Canonicalized SHA-256 lookup across 64-shard `DashMap`.
- **L2 Semantic Similarity (`< 200 µs`)**: Zero-embedding 64-bit SimHash (word unigrams/bigrams + char 3-grams) with `4 × 16-bit` inverted block index and top-32 sparse cosine verification (`>= 0.95` default).
- **Dual-Mode SSE Streaming**: Zero-delay stream teeing on cache miss + token-by-token OpenAI SSE replay on cache hit.
- **Drop-In Endpoints**: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/api/chat`, `/api/generate`, `/healthz`, `/metrics` (Prometheus).
- **Bounded Memory + TTL Eviction**: LRU eviction at `90% -> 80%` memory watermark (`--max-memory-mb 256`).

## Quickstart
```bash
cargo run --release -- --bind 127.0.0.1:8000 --upstream http://127.0.0.1:11434
```
Point any OpenAI-compatible SDK to `http://127.0.0.1:8000/v1` and inspect `X-Cache-Status: HIT-L1 | HIT-L2 | MISS` headers or `/metrics`.
