//! `run_command` tool implementation.

use std::collections::HashMap;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

#[derive(Clone)]
pub struct RunCommandTool {
    context: ToolContext,
}

impl RunCommandTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunCommandInput {
    command: String,
    cwd: Option<String>,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
    env: Option<HashMap<String, String>>,
    #[serde(default = "default_max_output")]
    max_stdout_bytes: usize,
    #[serde(default = "default_max_output")]
    max_stderr_bytes: usize,
}

fn default_timeout_ms() -> u64 {
    120_000
}

fn default_max_output() -> usize {
    131_072
}

#[async_trait]
impl Tool for RunCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "run_command".to_owned(),
            description: "Runs a shell command with timeout and output-size limits.".to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "command": { "type": "string" },
                    "cwd": { "type": "string" },
                    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 3600000, "default": 120000 },
                    "env": { "type": "object", "additionalProperties": { "type": "string" } },
                    "max_stdout_bytes": { "type": "integer", "minimum": 256, "maximum": 10485760, "default": 131072 },
                    "max_stderr_bytes": { "type": "integer", "minimum": 256, "maximum": 10485760, "default": 131072 }
                },
                "required": ["command"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: RunCommandInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid run_command arguments",
                "Provide arguments matching the run_command input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.timeout_ms == 0 || input.timeout_ms > 3_600_000 {
            return Err(ToolError::invalid_argument(
                "timeout_ms must be between 1 and 3600000",
                "Set timeout_ms within the allowed range.",
                json!({ "timeout_ms": input.timeout_ms }),
            ));
        }
        if !(256..=10_485_760).contains(&input.max_stdout_bytes) {
            return Err(ToolError::invalid_argument(
                "max_stdout_bytes must be between 256 and 10485760",
                "Set max_stdout_bytes within the allowed range.",
                json!({ "max_stdout_bytes": input.max_stdout_bytes }),
            ));
        }
        if !(256..=10_485_760).contains(&input.max_stderr_bytes) {
            return Err(ToolError::invalid_argument(
                "max_stderr_bytes must be between 256 and 10485760",
                "Set max_stderr_bytes within the allowed range.",
                json!({ "max_stderr_bytes": input.max_stderr_bytes }),
            ));
        }

        // Commands are intentionally executed via `bash -lc` so callers can use familiar shell
        // syntax, pipes, and env expansion in one field.
        let mut command = Command::new("bash");
        command.arg("-lc").arg(&input.command);
        if let Some(cwd) = &input.cwd {
            command.current_dir(self.context.resolve_path(cwd));
        } else {
            command.current_dir(self.context.workspace_root());
        }
        if let Some(env) = &input.env {
            command.envs(env);
        }

        let start = Instant::now();
        // Wrap the whole process in a timeout to bound resource usage.
        let output_result =
            timeout(Duration::from_millis(input.timeout_ms), command.output()).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        match output_result {
            Ok(Ok(output)) => {
                // Truncate output after execution so caller still receives the leading bytes.
                let (stdout, truncated_stdout) =
                    truncate_bytes_to_string(&output.stdout, input.max_stdout_bytes);
                let (stderr, truncated_stderr) =
                    truncate_bytes_to_string(&output.stderr, input.max_stderr_bytes);
                let status = output.status;
                let exit_code = status.code().unwrap_or(-1);
                let signal = signal_string(&status);
                Ok(json!({
                    "ok": true,
                    "exit_code": exit_code,
                    "signal": signal,
                    "stdout": stdout,
                    "stderr": stderr,
                    "truncated_stdout": truncated_stdout,
                    "truncated_stderr": truncated_stderr,
                    "duration_ms": duration_ms,
                    "timed_out": false,
                    "request_id": next_request_id()
                }))
            }
            Ok(Err(error)) => Err(ToolError::internal(
                "failed while waiting for command output",
                "Retry the command. If this persists, inspect process execution environment.",
                json!({ "error": error.to_string() }),
            )),
            Err(_) => Ok(json!({
                "ok": true,
                "exit_code": -1,
                "signal": Value::Null,
                "stdout": "",
                "stderr": "",
                "truncated_stdout": false,
                "truncated_stderr": false,
                "duration_ms": duration_ms,
                "timed_out": true,
                "request_id": next_request_id()
            })),
        }
    }
}

fn truncate_bytes_to_string(bytes: &[u8], max_bytes: usize) -> (String, bool) {
    if bytes.len() <= max_bytes {
        return (String::from_utf8_lossy(bytes).into_owned(), false);
    }
    (
        String::from_utf8_lossy(&bytes[..max_bytes]).into_owned(),
        true,
    )
}

#[cfg(unix)]
fn signal_string(status: &std::process::ExitStatus) -> Value {
    use std::os::unix::process::ExitStatusExt;
    match status.signal() {
        Some(signal) => json!(signal.to_string()),
        None => Value::Null,
    }
}

#[cfg(not(unix))]
fn signal_string(_status: &std::process::ExitStatus) -> Value {
    Value::Null
}
