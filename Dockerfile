FROM rust:1.83-slim-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY tests ./tests
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/llm-cache-edge /usr/local/bin/llm-cache-edge
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/llm-cache-edge"]
