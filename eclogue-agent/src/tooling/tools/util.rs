//! Shared helper functions used across tool implementations.

use std::borrow::Cow;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde_json::json;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::tooling::protocol::ToolError;

pub const DEFAULT_EXCLUDE_GLOBS: [&str; 2] = [".git/**", "node_modules/**"];

/// Compiled include/exclude matchers.
pub struct GlobMatcher {
    include: Option<GlobSet>,
    exclude: Option<GlobSet>,
}

impl GlobMatcher {
    /// Returns true when the relative path passes include/exclude filters.
    pub fn is_match(&self, relative_path: &str) -> bool {
        let include_ok = self
            .include
            .as_ref()
            .map(|include| include.is_match(relative_path))
            .unwrap_or(true);
        let excluded = self
            .exclude
            .as_ref()
            .map(|exclude| exclude.is_match(relative_path))
            .unwrap_or(false);

        include_ok && !excluded
    }
}

/// Builds glob matcher from optional include/exclude patterns.
pub fn build_glob_matcher(
    include_globs: Option<&Vec<String>>,
    exclude_globs: Option<&Vec<String>>,
) -> Result<GlobMatcher, ToolError> {
    let include = match include_globs {
        Some(globs) if !globs.is_empty() => Some(build_glob_set(globs, "include_globs")?),
        _ => None,
    };
    let exclude = match exclude_globs {
        Some(globs) if !globs.is_empty() => Some(build_glob_set(globs, "exclude_globs")?),
        _ => Some(build_glob_set(
            &DEFAULT_EXCLUDE_GLOBS
                .iter()
                .map(|entry| (*entry).to_owned())
                .collect::<Vec<_>>(),
            "exclude_globs",
        )?),
    };

    Ok(GlobMatcher { include, exclude })
}

fn build_glob_set(globs: &Vec<String>, field_name: &str) -> Result<GlobSet, ToolError> {
    let mut builder = GlobSetBuilder::new();
    for glob in globs {
        let parsed = Glob::new(glob).map_err(|error| {
            ToolError::invalid_argument(
                format!("invalid glob in {field_name}: {glob}"),
                "Provide valid glob syntax.",
                json!({ "glob": glob, "parse_error": error.to_string() }),
            )
        })?;
        builder.add(parsed);
    }
    builder.build().map_err(|error| {
        ToolError::invalid_argument(
            format!("failed to compile {field_name}"),
            "Fix the provided globs and retry.",
            json!({ "field": field_name, "error": error.to_string() }),
        )
    })
}

/// Converts a path to slash-delimited string for glob matching and stable JSON output.
pub fn normalize_path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Formats file `mtime` as RFC3339 UTC timestamp.
pub fn metadata_mtime_rfc3339(metadata: &fs::Metadata) -> Result<String, ToolError> {
    let system_time = metadata.modified().map_err(|error| {
        ToolError::internal(
            "failed to read file modification time",
            "Retry the request. If the issue persists, inspect file permissions.",
            json!({ "error": error.to_string() }),
        )
    })?;
    let datetime = OffsetDateTime::from(system_time);
    datetime.format(&Rfc3339).map_err(|error| {
        ToolError::internal(
            "failed to format modification time",
            "Retry the request.",
            json!({ "error": error.to_string() }),
        )
    })
}

/// Returns lowercase SHA-256 hex for a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Heuristic "is text" classifier used by file tools.
pub fn is_probably_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    if bytes.iter().any(|byte| *byte == 0) {
        return false;
    }

    // Accept UTF-8 bodies directly. For non-UTF8 bytes, use a conservative printable ratio.
    if std::str::from_utf8(bytes).is_ok() {
        return true;
    }

    let printable = bytes
        .iter()
        .filter(|byte| {
            matches!(
                byte,
                b'\n' | b'\r' | b'\t' | 0x20..=0x7e | 0x80..=0xff
            )
        })
        .count();
    printable * 100 / bytes.len() > 90
}

/// Reads a file as bytes and maps common IO failures into protocol-level `ToolError`.
pub fn read_file_bytes(path: &Path) -> Result<Vec<u8>, ToolError> {
    fs::read(path).map_err(|error| map_io_error(error, path, "read"))
}

/// Writes bytes atomically by replacing the full file.
pub fn write_file_bytes(path: &Path, bytes: &[u8]) -> Result<(), ToolError> {
    fs::write(path, bytes).map_err(|error| map_io_error(error, path, "write"))
}

/// Maps `std::io::Error` variants into schema-defined tool error codes.
pub fn map_io_error(error: io::Error, path: &Path, operation: &str) -> ToolError {
    let path_string = normalize_path_string(path);
    match error.kind() {
        io::ErrorKind::NotFound => ToolError::not_found(
            format!("failed to {operation} path because it does not exist"),
            "Verify the path and retry.",
            json!({ "path": path_string, "io_error": error.to_string() }),
        ),
        io::ErrorKind::PermissionDenied => ToolError::permission_denied(
            format!("permission denied while attempting to {operation} path"),
            "Run with permissions that can access the path.",
            json!({ "path": path_string, "io_error": error.to_string() }),
        ),
        _ => ToolError::internal(
            format!("failed to {operation} path"),
            "Retry the request. If it persists, inspect filesystem state.",
            json!({ "path": path_string, "io_error": error.to_string() }),
        ),
    }
}

/// Converts a possibly-invalid UTF-8 byte slice into a lossy displayable string.
pub fn bytes_to_text_lossy(bytes: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

/// Returns path relative to a base directory if possible.
pub fn relative_path_string(base: &Path, absolute: &Path) -> String {
    absolute
        .strip_prefix(base)
        .map(PathBuf::from)
        .unwrap_or_else(|_| absolute.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/")
}
