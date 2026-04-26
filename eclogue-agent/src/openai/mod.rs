//! OpenAI-backed `AgentSession` implementation.
//!
//! This module is split into:
//! - `transport`: HTTP and SSE integration details.
//! - `completions`: conversation/session orchestration logic.
//! - `types`: shared API/request/response abstractions.

mod completions;
mod transport;
mod types;

pub use completions::{DEFAULT_OPENAI_MODEL, OpenAiAgent, OpenAiAgentBuilder, OpenAiBuilderError};
pub use transport::ReqwestOpenAiApi;
pub use types::{
    OpenAiApi, OpenAiApiError, OpenAiAuth, OpenAiChatCompletion, OpenAiChatRequest, OpenAiMessage,
    OpenAiRole, OpenAiStream, OpenAiStreamEvent, OpenAiToolCall, OpenAiToolDefinition,
};
