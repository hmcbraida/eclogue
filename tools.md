# Tool calls to create

This document lists all the tool calls we are going to create for our agent
tooling.

In addition to these tools, there is a **single error schema** which is returned
when the tool does not execute successfully:

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": false },
    "error": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "code": {
          "type": "string",
          "enum": [
            "INVALID_ARGUMENT",
            "NOT_FOUND",
            "PERMISSION_DENIED",
            "TIMEOUT",
            "RATE_LIMITED",
            "PRECONDITION_FAILED",
            "CONFLICT",
            "INTERNAL"
          ]
        },
        "message": { "type": "string" },
        "retryable": { "type": "boolean" },
        "suggested_action": { "type": "string" },
        "details": { "type": "object", "additionalProperties": true }
      },
      "required": ["code", "message", "retryable", "suggested_action", "details"]
    },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "error", "request_id"]
}
```

What follows is a list of all the tools.

## list_files

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string" },
    "recursive": { "type": "boolean", "default": true },
    "max_depth": { "type": "integer", "minimum": 0 },
    "include_globs": {
      "type": "array",
      "items": { "type": "string" }
    },
    "exclude_globs": {
      "type": "array",
      "items": { "type": "string" },
      "default": [".git/**", "node_modules/**"]
    },
    "offset": { "type": "integer", "minimum": 0, "default": 0 },
    "limit": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 200 }
  },
  "required": ["path"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "path": { "type": "string" },
    "entries": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "path": { "type": "string" },
          "type": { "type": "string", "enum": ["file", "dir", "symlink"] },
          "size_bytes": { "type": "integer", "minimum": 0 }
        },
        "required": ["path", "type"]
      }
    },
    "offset": { "type": "integer", "minimum": 0 },
    "limit": { "type": "integer", "minimum": 1 },
    "next_offset": { "type": ["integer", "null"], "minimum": 0 },
    "truncated": { "type": "boolean" },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "path", "entries", "offset", "limit", "truncated", "request_id"]
}
```

## search_text

Note: implement this with the `grep` crate.

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string" },
    "query": { "type": "string" },
    "regex": { "type": "boolean", "default": false },
    "case_sensitive": { "type": "boolean", "default": false },
    "include_globs": {
      "type": "array",
      "items": { "type": "string" }
    },
    "exclude_globs": {
      "type": "array",
      "items": { "type": "string" },
      "default": [".git/**", "node_modules/**"]
    },
    "context_lines": { "type": "integer", "minimum": 0, "maximum": 10, "default": 0 },
    "offset": { "type": "integer", "minimum": 0, "default": 0 },
    "limit": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 200 }
  },
  "required": ["path", "query"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "matches": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "path": { "type": "string" },
          "line": { "type": "integer", "minimum": 1 },
          "column_start": { "type": "integer", "minimum": 1 },
          "column_end": { "type": "integer", "minimum": 1 },
          "match_text": { "type": "string" },
          "before": { "type": "array", "items": { "type": "string" } },
          "after": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["path", "line", "match_text"]
      }
    },
    "offset": { "type": "integer", "minimum": 0 },
    "limit": { "type": "integer", "minimum": 1 },
    "next_offset": { "type": ["integer", "null"], "minimum": 0 },
    "truncated": { "type": "boolean" },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "matches", "offset", "limit", "truncated", "request_id"]
}
```

## stat_file

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string" },
    "include_hash": { "type": "boolean", "default": true }
  },
  "required": ["path"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "path": { "type": "string" },
    "exists": { "type": "boolean" },
    "kind": { "type": "string", "enum": ["file", "dir", "symlink", "other"] },
    "size_bytes": { "type": "integer", "minimum": 0 },
    "mtime": { "type": "string", "format": "date-time" },
    "is_text": { "type": "boolean" },
    "line_count": { "type": ["integer", "null"], "minimum": 0 },
    "sha256": { "type": ["string", "null"] },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "path", "exists", "kind", "request_id"]
}
```

## read_file

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string" },
    "start_line": { "type": "integer", "minimum": 1, "default": 1 },
    "max_lines": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 200 }
  },
  "required": ["path"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "path": { "type": "string" },
    "start_line": { "type": "integer", "minimum": 1 },
    "end_line": { "type": "integer", "minimum": 1 },
    "total_lines": { "type": "integer", "minimum": 0 },
    "content": { "type": "string" },
    "truncated": { "type": "boolean" },
    "enforced_max_lines": { "type": "integer", "minimum": 1 },
    "sha256": { "type": "string" },
    "request_id": { "type": "string" }
  },
  "required": [
    "ok",
    "path",
    "start_line",
    "end_line",
    "total_lines",
    "content",
    "truncated",
    "enforced_max_lines",
    "sha256",
    "request_id"
  ]
}
```

## edit_file

### Input

```json
{
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
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "path": { "type": "string" },
    "applied_edits": { "type": "integer", "minimum": 0 },
    "bytes_written": { "type": "integer", "minimum": 0 },
    "new_hash": { "type": "string" },
    "diff": { "type": "string" },
    "diff_stats": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "files_changed": { "type": "integer", "minimum": 0 },
        "additions": { "type": "integer", "minimum": 0 },
        "deletions": { "type": "integer", "minimum": 0 }
      },
      "required": ["files_changed", "additions", "deletions"]
    },
    "diff_id": { "type": ["string", "null"] },
    "truncated_diff": { "type": "boolean" },
    "request_id": { "type": "string" }
  },
  "required": [
    "ok",
    "path",
    "applied_edits",
    "bytes_written",
    "new_hash",
    "diff_stats",
    "truncated_diff",
    "request_id"
  ]
}
```

## get_diff

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "diff_id": { "type": "string" },
    "offset": { "type": "integer", "minimum": 0, "default": 0 },
    "limit": { "type": "integer", "minimum": 1, "maximum": 2000, "default": 200 }
  },
  "required": ["diff_id"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "diff_id": { "type": "string" },
    "chunk": { "type": "string" },
    "offset": { "type": "integer", "minimum": 0 },
    "limit": { "type": "integer", "minimum": 1 },
    "next_offset": { "type": ["integer", "null"], "minimum": 0 },
    "truncated": { "type": "boolean" },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "diff_id", "chunk", "offset", "limit", "truncated", "request_id"]
}
```

## run_command

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "command": { "type": "string" },
    "cwd": { "type": "string" },
    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 3600000, "default": 120000 },
    "env": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    },
    "max_stdout_bytes": { "type": "integer", "minimum": 256, "maximum": 10485760, "default": 131072 },
    "max_stderr_bytes": { "type": "integer", "minimum": 256, "maximum": 10485760, "default": 131072 }
  },
  "required": ["command"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "exit_code": { "type": "integer" },
    "signal": { "type": ["string", "null"] },
    "stdout": { "type": "string" },
    "stderr": { "type": "string" },
    "truncated_stdout": { "type": "boolean" },
    "truncated_stderr": { "type": "boolean" },
    "duration_ms": { "type": "integer", "minimum": 0 },
    "timed_out": { "type": "boolean" },
    "request_id": { "type": "string" }
  },
  "required": [
    "ok",
    "exit_code",
    "stdout",
    "stderr",
    "truncated_stdout",
    "truncated_stderr",
    "duration_ms",
    "timed_out",
    "request_id"
  ]
}
```

## web_search

!!! Note do not implement yet! Ignore this section.

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "query": { "type": "string" },
    "max_results": { "type": "integer", "minimum": 1, "maximum": 50, "default": 10 },
    "provider": { "type": "string", "default": "auto" },
    "recency_days": { "type": "integer", "minimum": 1 },
    "safe_search": { "type": "string", "enum": ["off", "moderate", "strict"], "default": "moderate" }
  },
  "required": ["query"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "provider": { "type": "string" },
    "fetched_at": { "type": "string", "format": "date-time" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "title": { "type": "string" },
          "url": { "type": "string", "format": "uri" },
          "snippet": { "type": "string" },
          "published_at": { "type": ["string", "null"], "format": "date-time" },
          "score": { "type": ["number", "null"] }
        },
        "required": ["title", "url", "snippet"]
      }
    },
    "request_id": { "type": "string" }
  },
  "required": ["ok", "provider", "fetched_at", "results", "request_id"]
}
```

## fetch_url

### Input

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "url": { "type": "string", "format": "uri" },
    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 120000, "default": 30000 },
    "max_bytes": { "type": "integer", "minimum": 512, "maximum": 10485760, "default": 262144 },
    "extract": { "type": "string", "enum": ["text", "html", "markdown"], "default": "text" }
  },
  "required": ["url"]
}
```

### Output

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean", "const": true },
    "url": { "type": "string", "format": "uri" },
    "final_url": { "type": "string", "format": "uri" },
    "status": { "type": "integer", "minimum": 100, "maximum": 599 },
    "title": { "type": ["string", "null"] },
    "content_type": { "type": ["string", "null"] },
    "content": { "type": "string" },
    "content_hash": { "type": "string" },
    "truncated": { "type": "boolean" },
    "fetched_at": { "type": "string", "format": "date-time" },
    "request_id": { "type": "string" }
  },
  "required": [
    "ok",
    "url",
    "status",
    "content",
    "content_hash",
    "truncated",
    "fetched_at",
    "request_id"
  ]
}
```

