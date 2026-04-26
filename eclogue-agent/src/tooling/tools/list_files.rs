//! `list_files` tool implementation.

use std::fs;
use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use walkdir::WalkDir;

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::{build_glob_matcher, map_io_error, normalize_path_string, relative_path_string};

#[derive(Clone)]
pub struct ListFilesTool {
    context: ToolContext,
}

impl ListFilesTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListFilesInput {
    path: String,
    #[serde(default = "default_true")]
    recursive: bool,
    max_depth: Option<usize>,
    include_globs: Option<Vec<String>>,
    exclude_globs: Option<Vec<String>>,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_true() -> bool {
    true
}

fn default_limit() -> usize {
    200
}

#[async_trait]
impl Tool for ListFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_files".to_owned(),
            description: "Lists files and directories under a path with optional glob filters and pagination."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": { "type": "string" },
                    "recursive": { "type": "boolean", "default": true },
                    "max_depth": { "type": "integer", "minimum": 0 },
                    "include_globs": { "type": "array", "items": { "type": "string" } },
                    "exclude_globs": { "type": "array", "items": { "type": "string" }, "default": [".git/**", "node_modules/**"] },
                    "offset": { "type": "integer", "minimum": 0, "default": 0 },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 200 }
                },
                "required": ["path"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: ListFilesInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid list_files arguments",
                "Provide arguments matching the list_files input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.limit == 0 || input.limit > 1000 {
            return Err(ToolError::invalid_argument(
                "limit must be between 1 and 1000",
                "Set limit to a value in the allowed range.",
                json!({ "limit": input.limit }),
            ));
        }

        let absolute_path = self.context.resolve_path(&input.path);
        let metadata = fs::symlink_metadata(&absolute_path)
            .map_err(|error| map_io_error(error, &absolute_path, "stat"))?;
        if !metadata.is_dir() {
            return Err(ToolError::invalid_argument(
                "path must reference a directory",
                "Provide a directory path.",
                json!({ "path": normalize_path_string(&absolute_path) }),
            ));
        }

        let matcher =
            build_glob_matcher(input.include_globs.as_ref(), input.exclude_globs.as_ref())?;
        let mut walker = WalkDir::new(&absolute_path).follow_links(false);
        if !input.recursive {
            walker = walker.max_depth(1);
        } else if let Some(max_depth) = input.max_depth {
            walker = walker.max_depth(max_depth + 1);
        }

        let mut all_entries = Vec::new();
        for entry_result in walker.into_iter() {
            let entry = entry_result.map_err(|error| {
                ToolError::internal(
                    "failed while walking directory",
                    "Retry the request. If this persists, inspect the directory tree.",
                    json!({ "error": error.to_string() }),
                )
            })?;
            if entry.path() == absolute_path {
                continue;
            }

            let relative = relative_path_string(Path::new(&absolute_path), entry.path());
            if !matcher.is_match(&relative) {
                continue;
            }

            let meta = entry.metadata().map_err(|error| {
                ToolError::internal(
                    "failed to stat directory entry",
                    "Retry the request. If this persists, inspect filesystem state.",
                    json!({ "path": normalize_path_string(entry.path()), "error": error.to_string() }),
                )
            })?;
            let entry_type = if entry.file_type().is_file() {
                "file"
            } else if entry.file_type().is_dir() {
                "dir"
            } else if entry.file_type().is_symlink() {
                "symlink"
            } else {
                continue;
            };

            let mut value = json!({
                "path": relative,
                "type": entry_type
            });
            if meta.is_file() {
                value["size_bytes"] = json!(meta.len());
            }
            all_entries.push(value);
        }

        all_entries.sort_by(|left, right| {
            left["path"]
                .as_str()
                .unwrap_or_default()
                .cmp(right["path"].as_str().unwrap_or_default())
        });

        let total = all_entries.len();
        let start = input.offset.min(total);
        let end = (start + input.limit).min(total);
        let entries = all_entries[start..end].to_vec();
        let truncated = end < total;
        let next_offset = if truncated { Some(end) } else { None };

        Ok(json!({
            "ok": true,
            "path": self.context.display_path(&absolute_path),
            "entries": entries,
            "offset": input.offset,
            "limit": input.limit,
            "next_offset": next_offset,
            "truncated": truncated,
            "request_id": next_request_id()
        }))
    }
}
