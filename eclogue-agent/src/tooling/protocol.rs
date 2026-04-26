//! Shared protocol types used by all tool implementations.
//!
//! The `tools.md` specification requires a single normalized error envelope. This module
//! provides that envelope plus helpers to construct strongly-typed errors consistently.

use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

/// Canonical error code set defined by `tools.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolErrorCode {
    InvalidArgument,
    NotFound,
    PermissionDenied,
    Timeout,
    RateLimited,
    PreconditionFailed,
    Conflict,
    Internal,
}

impl ToolErrorCode {
    /// Returns the wire-format enum string required by the error schema.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::NotFound => "NOT_FOUND",
            Self::PermissionDenied => "PERMISSION_DENIED",
            Self::Timeout => "TIMEOUT",
            Self::RateLimited => "RATE_LIMITED",
            Self::PreconditionFailed => "PRECONDITION_FAILED",
            Self::Conflict => "CONFLICT",
            Self::Internal => "INTERNAL",
        }
    }
}

/// Rich tool execution failure that maps directly to the standard error envelope.
#[derive(Debug, Clone)]
pub struct ToolError {
    /// Stable machine-readable code.
    pub code: ToolErrorCode,
    /// Human-readable explanation.
    pub message: String,
    /// Whether retrying the exact same request may succeed.
    pub retryable: bool,
    /// Suggested next action for the caller/model.
    pub suggested_action: String,
    /// Additional structured data for diagnostics.
    pub details: Value,
}

impl ToolError {
    /// Creates a new tool error with explicit envelope values.
    pub fn new(
        code: ToolErrorCode,
        message: impl Into<String>,
        retryable: bool,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            retryable,
            suggested_action: suggested_action.into(),
            details,
        }
    }

    /// Invalid argument helper.
    pub fn invalid_argument(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::InvalidArgument,
            message,
            false,
            suggested_action,
            details,
        )
    }

    /// Not found helper.
    pub fn not_found(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::NotFound,
            message,
            false,
            suggested_action,
            details,
        )
    }

    /// Permission denied helper.
    pub fn permission_denied(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::PermissionDenied,
            message,
            false,
            suggested_action,
            details,
        )
    }

    /// Timeout helper.
    pub fn timeout(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::Timeout,
            message,
            true,
            suggested_action,
            details,
        )
    }

    /// Precondition helper.
    pub fn precondition_failed(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::PreconditionFailed,
            message,
            false,
            suggested_action,
            details,
        )
    }

    /// Internal helper.
    pub fn internal(
        message: impl Into<String>,
        suggested_action: impl Into<String>,
        details: Value,
    ) -> Self {
        Self::new(
            ToolErrorCode::Internal,
            message,
            false,
            suggested_action,
            details,
        )
    }

    /// Converts this typed error into the shared in-band error payload.
    pub fn to_payload(&self) -> Value {
        json!({
            "ok": false,
            "error": {
                "code": self.code.as_str(),
                "message": self.message,
                "retryable": self.retryable,
                "suggested_action": self.suggested_action,
                "details": self.details
            },
            "request_id": next_request_id()
        })
    }
}

impl Display for ToolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ToolError {}

static TOOL_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generates a monotonically-increasing request identifier.
pub fn next_request_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let counter = TOOL_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tool-{}-{}", millis, counter)
}
