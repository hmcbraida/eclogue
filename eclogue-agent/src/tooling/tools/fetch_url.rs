//! `fetch_url` tool implementation.

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::tooling::context::ToolContext;
use crate::tooling::protocol::{ToolError, next_request_id};
use crate::tooling::registry::{Tool, ToolDefinition};

use super::util::sha256_hex;

#[derive(Clone)]
pub struct FetchUrlTool {
    context: ToolContext,
}

impl FetchUrlTool {
    pub fn new(context: ToolContext) -> Self {
        Self { context }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchUrlInput {
    url: String,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
    #[serde(default = "default_max_bytes")]
    max_bytes: usize,
    #[serde(default = "default_extract")]
    extract: ExtractMode,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ExtractMode {
    Text,
    Html,
    Markdown,
}

fn default_timeout_ms() -> u64 {
    30_000
}

fn default_max_bytes() -> usize {
    262_144
}

fn default_extract() -> ExtractMode {
    ExtractMode::Text
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_url".to_owned(),
            description: "Fetches a URL with byte/time limits and returns extracted text/html/markdown content."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "url": { "type": "string", "format": "uri" },
                    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 120000, "default": 30000 },
                    "max_bytes": { "type": "integer", "minimum": 512, "maximum": 10485760, "default": 262144 },
                    "extract": { "type": "string", "enum": ["text", "html", "markdown"], "default": "text" }
                },
                "required": ["url"]
            }),
        }
    }

    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
        let input: FetchUrlInput = serde_json::from_value(arguments).map_err(|error| {
            ToolError::invalid_argument(
                "invalid fetch_url arguments",
                "Provide arguments matching the fetch_url input schema.",
                json!({ "parse_error": error.to_string() }),
            )
        })?;

        if input.timeout_ms == 0 || input.timeout_ms > 120_000 {
            return Err(ToolError::invalid_argument(
                "timeout_ms must be between 1 and 120000",
                "Set timeout_ms within the allowed range.",
                json!({ "timeout_ms": input.timeout_ms }),
            ));
        }
        if !(512..=10_485_760).contains(&input.max_bytes) {
            return Err(ToolError::invalid_argument(
                "max_bytes must be between 512 and 10485760",
                "Set max_bytes within the allowed range.",
                json!({ "max_bytes": input.max_bytes }),
            ));
        }

        let parsed_url = reqwest::Url::parse(&input.url).map_err(|error| {
            ToolError::invalid_argument(
                "url is not a valid URI",
                "Provide a valid absolute URL.",
                json!({ "url": input.url, "error": error.to_string() }),
            )
        })?;

        // Execute request with explicit per-call timeout so long-running endpoints are bounded.
        let response = self
            .context
            .http_client()
            .get(parsed_url.clone())
            .timeout(Duration::from_millis(input.timeout_ms))
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    ToolError::timeout(
                        "fetch_url request timed out",
                        "Increase timeout_ms or retry later.",
                        json!({ "url": parsed_url.to_string() }),
                    )
                } else {
                    ToolError::internal(
                        "fetch_url request failed",
                        "Retry the request. If this persists, inspect network connectivity.",
                        json!({ "url": parsed_url.to_string(), "error": error.to_string() }),
                    )
                }
            })?;

        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);

        // Stream the body and enforce `max_bytes` incrementally to avoid unbounded buffering.
        let mut truncated = false;
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|error| {
                ToolError::internal(
                    "failed while reading response body",
                    "Retry the request.",
                    json!({ "url": final_url, "error": error.to_string() }),
                )
            })?;
            if body.len() + chunk.len() > input.max_bytes {
                let remaining = input.max_bytes.saturating_sub(body.len());
                body.extend_from_slice(&chunk[..remaining]);
                truncated = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }

        let body_text = String::from_utf8_lossy(&body).into_owned();
        let is_html = content_type
            .as_ref()
            .map(|value| value.to_ascii_lowercase().contains("html"))
            .unwrap_or(false)
            || body_text.to_ascii_lowercase().contains("<html");

        // `extract` selects the consumer-facing representation while preserving a single content
        // string in output for simplicity.
        let extracted_content = match input.extract {
            ExtractMode::Html => body_text.clone(),
            ExtractMode::Text => {
                if is_html {
                    html2text::from_read(body_text.as_bytes(), 80).unwrap_or(body_text.clone())
                } else {
                    body_text.clone()
                }
            }
            ExtractMode::Markdown => {
                if is_html {
                    // We currently use html2text as a dependency-light fallback renderer.
                    html2text::from_read(body_text.as_bytes(), 80).unwrap_or(body_text.clone())
                } else {
                    body_text.clone()
                }
            }
        };

        let title = if is_html {
            extract_title(&body_text)
        } else {
            None
        };

        let fetched_at = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .map_err(|error| {
                ToolError::internal(
                    "failed to format fetched_at timestamp",
                    "Retry the request.",
                    json!({ "error": error.to_string() }),
                )
            })?;

        Ok(json!({
            "ok": true,
            "url": input.url,
            "final_url": final_url,
            "status": status,
            "title": title,
            "content_type": content_type,
            "content": extracted_content,
            "content_hash": sha256_hex(extracted_content.as_bytes()),
            "truncated": truncated,
            "fetched_at": fetched_at,
            "request_id": next_request_id()
        }))
    }
}

fn extract_title(html: &str) -> Option<String> {
    let lowercase = html.to_ascii_lowercase();
    let title_open = lowercase.find("<title")?;
    let open_end = lowercase[title_open..].find('>')? + title_open + 1;
    let title_close = lowercase[open_end..].find("</title>")? + open_end;
    Some(html[open_end..title_close].trim().to_owned()).filter(|title| !title.is_empty())
}
