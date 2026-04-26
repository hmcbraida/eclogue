//! `search_text` tool implementation.

use std::fs;
use std::path::Path;

use async_trait::async_trait;
use grep::matcher::Matcher;
use grep::regex::RegexMatcherBuilder;
use serde::Deserialize;
use serde_json::{Value, json};
use walkdir::WalkDir;

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::{
    build_glob_matcher, bytes_to_text_lossy, is_probably_text, map_io_error, read_file_bytes,
    relative_path_string,
};

#[derive(Clone)]
pub struct SearchTextTool {
    context: ToolContext,
}

impl SearchTextTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchTextInput {
    path: String,
    query: String,
    #[serde(default)]
    regex: bool,
    #[serde(default)]
    case_sensitive: bool,
    include_globs: Option<Vec<String>>,
    exclude_globs: Option<Vec<String>>,
    #[serde(default)]
    context_lines: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    200
}

#[async_trait]
impl Tool for SearchTextTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_text".to_owned(),
            description: "Searches file content for text or regex matches, with optional case-sensitivity, globs, and context."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string" },
                    "query": { "type": "string" },
                    "regex": { "type": "boolean", "default": false },
                    "case_sensitive": { "type": "boolean", "default": false },
                    "include_globs": { "type": "array", "items": { "type": "string" } },
                    "exclude_globs": { "type": "array", "items": { "type": "string" }, "default": [".git/**", "node_modules/**"] },
                    "context_lines": { "type": "integer", "minimum": 0, "maximum": 10, "default": 0 },
                    "offset": { "type": "integer", "minimum": 0, "default": 0 },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 200 }
                },
                "required": ["path", "query"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: SearchTextInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid search_text arguments",
                "Provide arguments matching the search_text input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.limit == 0 || input.limit > 1000 {
            return Err(ToolError::invalid_argument(
                "limit must be between 1 and 1000",
                "Set limit within the allowed range.",
                json!({ "limit": input.limit }),
            ));
        }
        if input.context_lines > 10 {
            return Err(ToolError::invalid_argument(
                "context_lines must be between 0 and 10",
                "Set context_lines within the allowed range.",
                json!({ "context_lines": input.context_lines }),
            ));
        }
        if input.query.is_empty() {
            return Err(ToolError::invalid_argument(
                "query must not be empty",
                "Provide a non-empty query.",
                json!({}),
            ));
        }

        // Compile once and reuse across all candidate files. This keeps behavior deterministic
        // and avoids repeated regex compilation overhead on large directory scans.
        let matcher = build_matcher(&input)?;
        let absolute_path = self.context.resolve_path(&input.path);
        let metadata = fs::symlink_metadata(&absolute_path)
            .map_err(|error| map_io_error(error, &absolute_path, "stat"))?;
        let glob_matcher =
            build_glob_matcher(input.include_globs.as_ref(), input.exclude_globs.as_ref())?;

        // Expand a file/dir input into the concrete file set we search.
        let mut candidates = Vec::new();
        if metadata.is_dir() {
            for entry in WalkDir::new(&absolute_path).follow_links(false) {
                let entry = entry.map_err(|error| {
                    ToolError::internal(
                        "failed while walking directory for search",
                        "Retry the request. If this persists, inspect the directory tree.",
                        json!({ "error": error.to_string() }),
                    )
                })?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let relative = relative_path_string(Path::new(&absolute_path), entry.path());
                if glob_matcher.is_match(&relative) {
                    candidates.push(entry.into_path());
                }
            }
        } else if metadata.is_file() {
            candidates.push(absolute_path.clone());
        } else {
            return Err(ToolError::invalid_argument(
                "path must reference a regular file or directory",
                "Provide a file or directory path.",
                json!({ "path": self.context.display_path(&absolute_path) }),
            ));
        }

        // We accumulate all matches first so pagination (`offset`/`limit`) is stable and
        // independent from traversal order.
        let mut all_matches = Vec::new();
        for file_path in candidates {
            let bytes = match read_file_bytes(&file_path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if !is_probably_text(&bytes) {
                continue;
            }

            // Search operates line-by-line to match the output schema's line/column model.
            let text = bytes_to_text_lossy(&bytes);
            let lines = text.lines().collect::<Vec<_>>();
            for (line_idx, line) in lines.iter().enumerate() {
                let mut line_matches = Vec::new();
                // `grep` crate matcher is required by the spec. We use its internal-iteration
                // API to collect all non-overlapping matches on the current line.
                matcher
                    .find_iter(line.as_bytes(), |m| {
                        line_matches.push((m.start(), m.end()));
                        true
                    })
                    .map_err(|error| {
                        ToolError::internal(
                            "matcher execution failed",
                            "Retry with a simpler query or valid regex.",
                            json!({ "error": error.to_string() }),
                        )
                    })?;

                for (start, end) in line_matches {
                    // Context slices are bounded on both sides; this keeps results compact while
                    // still making each match actionable for downstream edits.
                    let before_start = line_idx.saturating_sub(input.context_lines);
                    let after_end = (line_idx + 1 + input.context_lines).min(lines.len());
                    let before = lines[before_start..line_idx]
                        .iter()
                        .map(|entry| (*entry).to_owned())
                        .collect::<Vec<_>>();
                    let after = lines[(line_idx + 1)..after_end]
                        .iter()
                        .map(|entry| (*entry).to_owned())
                        .collect::<Vec<_>>();

                    all_matches.push(json!({
                        "path": self.context.display_path(&file_path),
                        "line": line_idx + 1,
                        "column_start": start + 1,
                        "column_end": end.max(start + 1),
                        "match_text": line[start..end].to_string(),
                        "before": before,
                        "after": after
                    }));
                }
            }
        }

        let total = all_matches.len();
        let start = input.offset.min(total);
        let end = (start + input.limit).min(total);
        let matches = all_matches[start..end].to_vec();
        let truncated = end < total;
        let next_offset = if truncated { Some(end) } else { None };

        Ok(json!({
            "ok": true,
            "matches": matches,
            "offset": input.offset,
            "limit": input.limit,
            "next_offset": next_offset,
            "truncated": truncated,
            "request_id": next_request_id()
        }))
    }
}

fn build_matcher(input: &SearchTextInput) -> Result<grep::regex::RegexMatcher, ToolError> {
    let mut builder = RegexMatcherBuilder::new();
    builder.case_insensitive(!input.case_sensitive);
    builder.fixed_strings(!input.regex);
    builder.build(&input.query).map_err(|error| {
        ToolError::invalid_argument(
            "failed to compile search query",
            "If regex=true, provide valid regex syntax. If regex=false, provide plain text.",
            json!({ "query": input.query, "regex": input.regex, "error": error.to_string() }),
        )
    })
}
