//! Error types that can be surfaced by an `AgentSession`.

use thiserror::Error;

use crate::tooling::ToolError;

/// Top-level error returned by provider-agnostic session APIs.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Returned when a caller omits required configuration details.
    #[error("configuration error: {0}")]
    Configuration(String),
    /// Returned when an upstream model provider returns an error.
    #[error("provider error: {0}")]
    Provider(String),
    /// Returned when a local tool fails while handling a tool call.
    #[error(transparent)]
    Tool(#[from] ToolError),
    /// Returned when internal streaming infrastructure is closed unexpectedly.
    #[error("internal channel was closed before streaming completed")]
    InternalChannelClosed,
}
