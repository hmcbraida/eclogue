use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Authentication mode used for OpenAI requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiAuth {
    /// Standard API-key based auth (`Authorization: Bearer <api_key>`).
    ApiKey(String),
    /// OAuth-like access token flow (for example ChatGPT Pro token-based auth).
    ChatGptAccessToken(String),
}

impl OpenAiAuth {
    /// Returns the token that should be attached as a Bearer credential.
    pub(crate) fn bearer_token(&self) -> &str {
        match self {
            Self::ApiKey(value) | Self::ChatGptAccessToken(value) => value,
        }
    }
}

/// Provider role for chat-completion messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiRole {
    /// System instruction message.
    System,
    /// End-user message.
    User,
    /// Assistant-generated message.
    Assistant,
    /// Tool output message.
    Tool,
}

/// Provider message payload used in completion requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiMessage {
    /// Role assigned to this message.
    pub role: OpenAiRole,
    /// Textual content for the message.
    pub content: String,
    /// Tool call identifier when role is `tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name for tool-related messages where available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Assistant tool calls associated with this message when role is `assistant`.
    ///
    /// OpenAI expects tool outputs to be preceded by the assistant message that requested
    /// them, so we persist this metadata in history for follow-up requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiAssistantToolCall>>,
}

impl OpenAiMessage {
    /// Creates a system message.
    pub(crate) fn system(content: String) -> Self {
        Self {
            role: OpenAiRole::System,
            content,
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }
    }

    /// Creates a user message.
    pub(crate) fn user(content: String) -> Self {
        Self {
            role: OpenAiRole::User,
            content,
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }
    }

    /// Creates an assistant message.
    pub(crate) fn assistant(content: String) -> Self {
        Self {
            role: OpenAiRole::Assistant,
            content,
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }
    }

    /// Creates an assistant message that carries explicit tool call metadata.
    pub(crate) fn assistant_with_tool_calls(
        content: String,
        tool_calls: Vec<OpenAiAssistantToolCall>,
    ) -> Self {
        Self {
            role: OpenAiRole::Assistant,
            content,
            tool_call_id: None,
            name: None,
            tool_calls: Some(tool_calls),
        }
    }

    /// Creates a tool output message.
    pub(crate) fn tool(tool_call_id: String, tool_name: String, content: String) -> Self {
        Self {
            role: OpenAiRole::Tool,
            content,
            tool_call_id: Some(tool_call_id),
            name: Some(tool_name),
            tool_calls: None,
        }
    }
}

/// Assistant-side tool call metadata stored in message history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiAssistantToolCall {
    /// Provider-generated tool call identifier.
    pub id: String,
    /// OpenAI chat-completions tool call type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Function payload selected by the model.
    pub function: OpenAiAssistantFunctionCall,
}

/// Function payload nested inside assistant-side tool call metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiAssistantFunctionCall {
    /// Registered tool name selected by the model.
    pub name: String,
    /// Tool arguments encoded as a JSON string for OpenAI chat completions requests.
    pub arguments: String,
}

/// Provider-agnostic tool definition projected into OpenAI request shape.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiToolDefinition {
    /// Name used by the model when requesting this tool.
    pub name: String,
    /// Tool description presented to the model.
    pub description: String,
    /// JSON schema for tool call arguments.
    pub input_schema: Value,
}

/// Normalized request sent through the `OpenAiApi` abstraction.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiChatRequest {
    /// Target model identifier.
    pub model: String,
    /// Message history used for completion context.
    pub messages: Vec<OpenAiMessage>,
    /// Registered tool definitions exposed to the model.
    pub tools: Vec<OpenAiToolDefinition>,
}

/// Normalized tool-call payload emitted by the API layer.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiToolCall {
    /// Provider-generated tool call identifier.
    pub call_id: String,
    /// Tool name selected by the model.
    pub tool_name: String,
    /// Decoded JSON arguments for the tool.
    pub arguments: Value,
}

/// Normalized full completion output from the API layer.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiChatCompletion {
    /// Final assistant text output.
    pub content: String,
    /// Optional tool calls produced in the same turn.
    pub tool_calls: Vec<OpenAiToolCall>,
}

/// Streaming events produced by the OpenAI API abstraction.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAiStreamEvent {
    /// Incremental assistant text.
    ///
    /// These are expected to be emitted in-order and can be concatenated to form the final
    /// assistant text for a turn.
    TextDelta(String),
    /// Model-selected tool call.
    ///
    /// This event should only be emitted once tool-call fragments have been fully assembled by
    /// the transport layer.
    ToolCall(OpenAiToolCall),
    /// End-of-response marker.
    ///
    /// Signals no more events will be emitted for the current provider stream.
    Done,
}

/// Convenient alias for OpenAI event streams.
pub type OpenAiStream = std::pin::Pin<
    Box<
        dyn futures_util::Stream<Item = Result<OpenAiStreamEvent, OpenAiApiError>> + Send + 'static,
    >,
>;

/// Errors returned by the API abstraction.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OpenAiApiError {
    /// Network-level or serialization-level transport failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// Non-success HTTP status with body details.
    #[error("openai http status {0}: {1}")]
    HttpStatus(u16, String),
    /// Expected response shape was missing required fields.
    #[error("response parsing error: {0}")]
    ResponseParsing(String),
}

/// API boundary used by `OpenAiAgent`.
///
/// Tests pass a mock implementation so no network calls are required.
#[async_trait]
pub trait OpenAiApi: Send + Sync {
    /// Returns a complete completion result.
    async fn create_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiChatCompletion, OpenAiApiError>;

    /// Returns streamed completion events.
    ///
    /// Implementations should preserve event ordering so the `AgentSession` layer can deterministically
    /// map provider events into provider-agnostic `AgentEvent` values.
    async fn stream_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiStream, OpenAiApiError>;
}
