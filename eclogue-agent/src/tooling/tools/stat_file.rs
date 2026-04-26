//! `stat_file` tool implementation.

use std::fs;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::{
    bytes_to_text_lossy, is_probably_text, map_io_error, metadata_mtime_rfc3339, read_file_bytes,
    sha256_hex,
};

#[derive(Clone)]
pub struct StatFileTool {
    context: ToolContext,
}

impl StatFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StatFileInput {
    path: String,
    #[serde(default = "default_true")]
    include_hash: bool,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Tool for StatFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "stat_file".to_owned(),
            description: "Returns metadata about a file path, including size, mtime, text hints, and optional SHA-256."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string" },
                    "include_hash": { "type": "boolean", "default": true }
                },
                "required": ["path"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: StatFileInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid stat_file arguments",
                "Provide arguments matching the stat_file input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        let absolute_path = self.context.resolve_path(&input.path);
        let metadata = match fs::symlink_metadata(&absolute_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(json!({
                    "ok": true,
                    "path": self.context.display_path(&absolute_path),
                    "exists": false,
                    "kind": "other",
                    "request_id": next_request_id()
                }));
            }
            Err(error) => return Err(map_io_error(error, &absolute_path, "stat")),
        };

        let kind = if metadata.is_file() {
            "file"
        } else if metadata.is_dir() {
            "dir"
        } else if metadata.file_type().is_symlink() {
            "symlink"
        } else {
            "other"
        };

        let mut response = json!({
            "ok": true,
            "path": self.context.display_path(&absolute_path),
            "exists": true,
            "kind": kind,
            "size_bytes": metadata.len(),
            "mtime": metadata_mtime_rfc3339(&metadata)?,
            "is_text": false,
            "line_count": Value::Null,
            "sha256": Value::Null,
            "request_id": next_request_id()
        });

        if metadata.is_file() {
            let bytes = read_file_bytes(&absolute_path)?;
            let is_text = is_probably_text(&bytes);
            response["is_text"] = json!(is_text);

            if is_text {
                let line_count = bytes_to_text_lossy(&bytes).lines().count();
                response["line_count"] = json!(line_count);
            }
            if input.include_hash {
                response["sha256"] = json!(sha256_hex(&bytes));
            }
        }

        Ok(response)
    }
}
