#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::Parser;

use llm_cache_edge::config::EdgeConfig;
use llm_cache_edge::proxy::run_server;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = EdgeConfig::parse();
    match run_server(cfg).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("llm-cache-edge error: {err}");
            ExitCode::FAILURE
        }
    }
}
