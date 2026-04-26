//! Interactive playground binary for the `eclogue-agent` crate.
//!
//! This binary intentionally keeps behavior simple:
//! - Build an OpenAI-backed session from environment-based auth.
//! - Stream responses to stdout so runtime behavior is easy to inspect.
//! - Register the full local tool suite so tool-calling behavior is observable end-to-end.

use std::env;
use std::error::Error;

use eclogue_agent::openai::{OpenAiAgent, OpenAiAuth};
use eclogue_agent::tooling::{ToolContextBuilder, ToolRegistryBuilder, register_default_tools};
use eclogue_agent::{AgentEvent, AgentSession};
use futures_util::StreamExt;
use serde_json::to_string_pretty;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Chooses an auth mode from environment variables.
///
/// Supported inputs:
/// - `OPENAI_API_KEY`
/// - `OPENAI_ACCESS_TOKEN` (OAuth-like ChatGPT Pro token style)
fn auth_from_env() -> Option<OpenAiAuth> {
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        return Some(OpenAiAuth::ApiKey(api_key));
    }

    if let Ok(access_token) = env::var("OPENAI_ACCESS_TOKEN") {
        return Some(OpenAiAuth::ChatGptAccessToken(access_token));
    }

    None
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Resolve auth from environment to keep this example zero-config in code.
    let Some(auth) = auth_from_env() else {
        eprintln!(
            "Missing auth.\nSet OPENAI_API_KEY or OPENAI_ACCESS_TOKEN before running this example."
        );
        return Ok(());
    };

    // Build a shared tool context rooted at the current working directory.
    let workspace_root = env::current_dir()?;
    let tool_context = ToolContextBuilder::new()
        .with_workspace_root(workspace_root)
        .build()?;

    // Register all built-in tools (except `web_search`, intentionally not implemented).
    let tool_registry_builder = register_default_tools(ToolRegistryBuilder::new(), tool_context);
    let tool_registry = tool_registry_builder.build()?;

    // Construct the agent with explicit auth, model, and registry.
    let mut agent = OpenAiAgent::builder()
        .with_auth(auth)
        .with_model("gpt-4.1-mini")
        .with_tool_registry(tool_registry)
        .build()?;

    // Build async stdin/stdout interfaces for an interactive REPL loop.
    let stdin = BufReader::new(io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = io::stdout();

    stdout
        .write_all(b"eclogue-example interactive session\nType 'exit' to quit.\n")
        .await?;

    loop {
        stdout.write_all(b"\n> ").await?;
        stdout.flush().await?;

        // Stop if input stream closes.
        let line = match lines.next_line().await? {
            Some(line) => line,
            None => break,
        };

        let prompt = line.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            break;
        }

        if prompt.is_empty() {
            continue;
        }

        // Stream provider events and print them as they arrive.
        let mut stream = agent.stream_response(prompt.to_owned()).await?;
        while let Some(event_result) = stream.next().await {
            match event_result? {
                AgentEvent::MessageDelta { delta } => {
                    stdout.write_all(delta.as_bytes()).await?;
                    stdout.flush().await?;
                }
                AgentEvent::MessageComplete { .. } => {
                    stdout.write_all(b"\n").await?;
                }
                AgentEvent::ToolCallRequested {
                    call_id, tool_name, ..
                } => {
                    let line = format!("\n[tool requested] id={call_id} name={tool_name}\n");
                    stdout.write_all(line.as_bytes()).await?;
                }
                AgentEvent::ToolCallCompleted {
                    call_id,
                    tool_name,
                    output,
                } => {
                    let line = format!("[tool completed] id={call_id} name={tool_name}\n");
                    stdout.write_all(line.as_bytes()).await?;
                    let output_line = format!(
                        "{}\n",
                        to_string_pretty(&output).unwrap_or_else(|_| output.to_string())
                    );
                    stdout.write_all(output_line.as_bytes()).await?;
                }
            }
        }
    }

    Ok(())
}
