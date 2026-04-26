//! `edit_file` tool implementation.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::{read_file_bytes, sha256_hex, write_file_bytes};

#[derive(Clone)]
pub struct EditFileTool {
    context: ToolContext,
}

impl EditFileTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditFileInput {
    path: String,
    expected_hash: Option<String>,
    #[serde(default)]
    create_if_missing: bool,
    edits: Vec<EditInstruction>,
    #[serde(default = "default_true")]
    return_diff: bool,
    #[serde(default = "default_max_diff_lines")]
    max_diff_lines: usize,
    #[serde(default = "default_max_diff_bytes")]
    max_diff_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EditInstruction {
    old_text: Option<String>,
    new_text: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

fn default_true() -> bool {
    true
}

fn default_max_diff_lines() -> usize {
    200
}

fn default_max_diff_bytes() -> usize {
    16_384
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_owned(),
            description: "Applies textual edits to a file with optional optimistic hash checks and optional diff output. Set create_if_missing=true when creating a new file."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string" },
                    "expected_hash": { "type": "string" },
                    "create_if_missing": { "type": "boolean", "default": false },
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "old_text": { "type": "string" },
                                "new_text": { "type": "string" },
                                "start_line": { "type": "integer", "minimum": 1 },
                                "end_line": { "type": "integer", "minimum": 1 }
                            },
                            "required": ["new_text"],
                            "oneOf": [
                                { "required": ["old_text", "new_text"] },
                                { "required": ["start_line", "end_line", "new_text"] }
                            ]
                        }
                    },
                    "return_diff": { "type": "boolean", "default": true },
                    "max_diff_lines": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 200 },
                    "max_diff_bytes": { "type": "integer", "minimum": 256, "maximum": 1048576, "default": 16384 }
                },
                "required": ["path", "edits"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: EditFileInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid edit_file arguments",
                "Provide arguments matching the edit_file input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.edits.is_empty() {
            return Err(ToolError::invalid_argument(
                "at least one edit must be provided",
                "Provide a non-empty edits array.",
                json!({}),
            ));
        }
        if input.max_diff_lines == 0 || input.max_diff_lines > 2000 {
            return Err(ToolError::invalid_argument(
                "max_diff_lines must be between 1 and 2000",
                "Set max_diff_lines within the allowed range.",
                json!({ "max_diff_lines": input.max_diff_lines }),
            ));
        }
        if input.max_diff_bytes < 256 || input.max_diff_bytes > 1_048_576 {
            return Err(ToolError::invalid_argument(
                "max_diff_bytes must be between 256 and 1048576",
                "Set max_diff_bytes within the allowed range.",
                json!({ "max_diff_bytes": input.max_diff_bytes }),
            ));
        }

        let absolute_path = self.context.resolve_path(&input.path);
        // Read current file content once; all edits in this call are applied in-memory and then
        // written back as a single replace to keep file state coherent.
        let existing_bytes = match read_file_bytes(&absolute_path) {
            Ok(bytes) => bytes,
            Err(error)
                if input.create_if_missing
                    && matches!(
                        error.code,
                        crate::tooling::protocol::ToolErrorCode::NotFound
                    ) =>
            {
                Vec::new()
            }
            Err(error) => return Err(error),
        };
        let original_content = String::from_utf8_lossy(&existing_bytes).into_owned();

        // `expected_hash` gives callers optimistic concurrency semantics.
        if let Some(expected_hash) = &input.expected_hash {
            let current_hash = sha256_hex(original_content.as_bytes());
            if &current_hash != expected_hash {
                return Err(ToolError::precondition_failed(
                    "expected_hash does not match current file content",
                    "Read the file again and retry with the latest hash.",
                    json!({
                        "expected_hash": expected_hash,
                        "current_hash": current_hash
                    }),
                ));
            }
        }

        let mut updated_content = original_content.clone();
        let mut applied_edits = 0usize;
        // Apply edits in request order. Later edits see effects of earlier edits.
        for edit in &input.edits {
            apply_edit_instruction(&mut updated_content, edit)?;
            applied_edits += 1;
        }

        write_file_bytes(&absolute_path, updated_content.as_bytes())?;
        let bytes_written = updated_content.len();
        let new_hash = sha256_hex(updated_content.as_bytes());

        // Diff stats are derived from the full before/after text snapshot.
        let text_diff = TextDiff::from_lines(&original_content, &updated_content);
        let additions = text_diff
            .iter_all_changes()
            .filter(|change| matches!(change.tag(), ChangeTag::Insert))
            .count();
        let deletions = text_diff
            .iter_all_changes()
            .filter(|change| matches!(change.tag(), ChangeTag::Delete))
            .count();
        let files_changed = usize::from(original_content != updated_content);

        let mut response = json!({
            "ok": true,
            "path": self.context.display_path(&absolute_path),
            "applied_edits": applied_edits,
            "bytes_written": bytes_written,
            "new_hash": new_hash,
            "diff_stats": {
                "files_changed": files_changed,
                "additions": additions,
                "deletions": deletions
            },
            "truncated_diff": false,
            "request_id": next_request_id()
        });

        if input.return_diff {
            let full_diff = text_diff.unified_diff().context_radius(3).to_string();
            let (chunk, truncated) =
                truncate_text_for_limits(&full_diff, input.max_diff_lines, input.max_diff_bytes);

            response["diff"] = json!(chunk);
            response["truncated_diff"] = json!(truncated);
            if truncated {
                // Persist the complete diff so the caller can page with `get_diff`.
                let diff_id = format!("diff-{}", next_request_id());
                self.context.put_diff(diff_id.clone(), full_diff).await;
                response["diff_id"] = json!(diff_id);
            } else {
                response["diff_id"] = Value::Null;
            }
        }

        Ok(response)
    }
}

fn apply_edit_instruction(content: &mut String, edit: &EditInstruction) -> Result<(), ToolError> {
    let uses_text_replacement =
        edit.old_text.is_some() && edit.start_line.is_none() && edit.end_line.is_none();
    let uses_range_replacement =
        edit.old_text.is_none() && edit.start_line.is_some() && edit.end_line.is_some();

    if !(uses_text_replacement || uses_range_replacement) {
        return Err(ToolError::invalid_argument(
            "each edit must provide either old_text or start_line/end_line",
            "For each edit, provide exactly one supported edit mode.",
            json!({
                "edit": {
                    "has_old_text": edit.old_text.is_some(),
                    "start_line": edit.start_line,
                    "end_line": edit.end_line
                }
            }),
        ));
    }

    if let Some(old_text) = &edit.old_text {
        // `old_text` mode performs first-match replacement only. This mirrors a targeted patch
        // workflow and avoids unintended global replacements.
        if let Some(position) = content.find(old_text) {
            let end = position + old_text.len();
            content.replace_range(position..end, &edit.new_text);
            return Ok(());
        }
        return Err(ToolError::precondition_failed(
            "old_text was not found in the target file",
            "Read the file and update old_text to the current content.",
            json!({ "old_text": old_text }),
        ));
    }

    let start_line = edit.start_line.unwrap_or(0);
    let end_line = edit.end_line.unwrap_or(0);
    if start_line == 0 || end_line == 0 || start_line > end_line {
        return Err(ToolError::invalid_argument(
            "line range is invalid",
            "Ensure start_line and end_line are >= 1 and start_line <= end_line.",
            json!({ "start_line": start_line, "end_line": end_line }),
        ));
    }

    // Line-range mode uses 1-based inclusive coordinates from the API schema.
    let mut lines = content
        .split('\n')
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let max_line = lines.len().max(1);
    if end_line > max_line {
        return Err(ToolError::precondition_failed(
            "line range exceeds current file length",
            "Read the file and update start_line/end_line to valid line numbers.",
            json!({ "start_line": start_line, "end_line": end_line, "max_line": max_line }),
        ));
    }

    let start_index = start_line - 1;
    let end_index_exclusive = end_line;
    lines.splice(
        start_index..end_index_exclusive,
        edit.new_text.split('\n').map(ToOwned::to_owned),
    );
    *content = lines.join("\n");
    Ok(())
}

fn truncate_text_for_limits(text: &str, max_lines: usize, max_bytes: usize) -> (String, bool) {
    let mut output = String::new();
    let mut lines_written = 0usize;
    let mut truncated = false;

    // We enforce both line and byte ceilings; whichever is hit first wins.
    for line in text.lines() {
        if lines_written >= max_lines {
            truncated = true;
            break;
        }

        // Account for the newline reinserted below.
        let required = line.len() + 1;
        if output.len() + required > max_bytes {
            truncated = true;
            break;
        }

        output.push_str(line);
        output.push('\n');
        lines_written += 1;
    }

    if !truncated && output.ends_with('\n') && !text.ends_with('\n') {
        output.pop();
    }

    (output, truncated)
}
