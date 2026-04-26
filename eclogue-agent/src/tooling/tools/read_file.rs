//! `read_file` tool implementation.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::{bytes_to_text_lossy, is_probably_text, read_file_bytes, sha256_hex};

#[derive(Clone)]
pub struct ReadFileTool {
    context: ToolContext,
}

impl ReadFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadFileInput {
    path: String,
    #[serde(default = "default_start_line")]
    start_line: usize,
    #[serde(default = "default_max_lines")]
    max_lines: usize,
}

fn default_start_line() -> usize {
    1
}

fn default_max_lines() -> usize {
    200
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_owned(),
            description: "Reads file content by line range and returns content hash plus truncation metadata."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string" },
                    "start_line": { "type": "integer", "minimum": 1, "default": 1 },
                    "max_lines": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 200 }
                },
                "required": ["path"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: ReadFileInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid read_file arguments",
                "Provide arguments matching the read_file input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.max_lines == 0 || input.max_lines > 2000 {
            return Err(ToolError::invalid_argument(
                "max_lines must be between 1 and 2000",
                "Set max_lines within the allowed range.",
                json!({ "max_lines": input.max_lines }),
            ));
        }
        if input.start_line == 0 {
            return Err(ToolError::invalid_argument(
                "start_line must be >= 1",
                "Set start_line to at least 1.",
                json!({ "start_line": input.start_line }),
            ));
        }

        let absolute_path = self.context.resolve_path(&input.path);
        let bytes = read_file_bytes(&absolute_path)?;
        if !is_probably_text(&bytes) {
            return Err(ToolError::invalid_argument(
                "read_file only supports text files",
                "Use stat_file first and avoid read_file for binary content.",
                json!({ "path": self.context.display_path(&absolute_path) }),
            ));
        }

        let text = bytes_to_text_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();
        let start_index = input.start_line.saturating_sub(1).min(total_lines);
        let end_index = (start_index + input.max_lines).min(total_lines);
        let selected = &lines[start_index..end_index];
        let content = selected.join("\n");
        let truncated = end_index < total_lines;
        let end_line = if selected.is_empty() {
            input.start_line
        } else {
            start_index + selected.len()
        };

        Ok(json!({
            "ok": true,
            "path": self.context.display_path(&absolute_path),
            "start_line": input.start_line,
            "end_line": end_line.max(input.start_line),
            "total_lines": total_lines,
            "content": content,
            "truncated": truncated,
            "enforced_max_lines": input.max_lines,
            "sha256": sha256_hex(&bytes),
            "request_id": next_request_id()
        }))
    }
}
