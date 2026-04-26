//! `get_diff` tool implementation.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

#[derive(Clone)]
pub struct GetDiffTool {
    context: ToolContext,
}

impl GetDiffTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GetDiffInput {
    diff_id: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    200
}

#[async_trait]
impl Tool for GetDiffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_diff".to_owned(),
            description: "Returns a paginated chunk from a previously stored diff_id.".to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "diff_id": { "type": "string" },
                    "offset": { "type": "integer", "minimum": 0, "default": 0 },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 200 }
                },
                "required": ["diff_id"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: GetDiffInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid get_diff arguments",
                "Provide arguments matching the get_diff input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.limit == 0 || input.limit > 2000 {
            return Err(ToolError::invalid_argument(
                "limit must be between 1 and 2000",
                "Set limit within the allowed range.",
                json!({ "limit": input.limit }),
            ));
        }

        let full_diff = self.context.get_diff(&input.diff_id).await.ok_or_else(|| {
            ToolError::not_found(
                "diff_id was not found",
                "Use diff_id values returned by edit_file with truncated_diff=true.",
                json!({ "diff_id": input.diff_id }),
            )
        })?;

        let lines = full_diff.lines().collect::<Vec<_>>();
        let total = lines.len();
        let start = input.offset.min(total);
        let end = (start + input.limit).min(total);
        let chunk = lines[start..end].join("\n");
        let truncated = end < total;
        let next_offset = if truncated { Some(end) } else { None };

        Ok(json!({
            "ok": true,
            "diff_id": input.diff_id,
            "chunk": chunk,
            "offset": input.offset,
            "limit": input.limit,
            "next_offset": next_offset,
            "truncated": truncated,
            "request_id": next_request_id()
        }))
    }
}
