//! Provider-agnostic agent session abstractions.
//!
//! Callers depend on this module so they can stay decoupled from provider-specific types.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::Value;

use crate::AgentError;

/// Single complete assistant reply returned by `send_message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentReply {
    /// Final assistant text after the model stream has been consumed.
    pub message: String,
}

/// Event emitted while a response is streamed from the provider.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// Incremental text delta produced by the assistant.
    MessageDelta {
        /// Piece of generated text.
        delta: String,
    },
    /// Signals that the assistant output for this turn is complete.
    MessageComplete {
        /// Full assembled message content for this turn.
        content: String,
    },
    /// Signals that the model requested a tool call.
    ToolCallRequested {
        /// Provider-generated call identifier.
        call_id: String,
        /// Registered tool name selected by the model.
        tool_name: String,
        /// JSON arguments sent by the model.
        arguments: Value,
    },
    /// Signals that a registered tool finished and returned a result.
    ToolCallCompleted {
        /// Provider-generated call identifier.
        call_id: String,
        /// Registered tool name that was invoked.
        tool_name: String,
        /// JSON output produced by the tool.
        output: Value,
    },
}

/// Convenience alias for boxed asynchronous streams of `AgentEvent`.
pub type AgentEventStream =
    Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>>;

/// Provider-independent session contract used by all agent implementations.
#[async_trait]
pub trait AgentSession: Send {
    /// Sends a user message and waits for a complete assistant reply.
    ///
    /// Implementations may internally consume a stream and aggregate deltas.
    async fn send_message(&mut self, message: String) -> Result<AgentReply, AgentError>;

    /// Sends a user message and returns a stream of response events.
    ///
    /// Streaming allows interactive UIs to render token deltas and tool call lifecycle events.
    async fn stream_response(&mut self, message: String) -> Result<AgentEventStream, AgentError>;
}
