//! OpenAI-backed `AgentSession` implementation.
//!
//! This module is split into:
//! - `transport`: HTTP and SSE integration details.
//! - `completions`: conversation/session orchestration logic.
//! - `responses`: conversation/session orchestration logic using the Responses API.
//! - `types`: shared API/request/response abstractions.

mod completions;
mod responses;
mod transport;
mod types;

pub use completions::{DEFAULT_OPENAI_MODEL, OpenAiAgent, OpenAiAgentBuilder, OpenAiBuilderError};
pub use responses::{OpenAiResponsesAgent, OpenAiResponsesAgentBuilder};
pub use transport::ReqwestOpenAiApi;
pub use types::{
    OpenAiApi, OpenAiApiError, OpenAiAuth, OpenAiChatCompletion, OpenAiChatRequest, OpenAiMessage,
    OpenAiRole, OpenAiStream, OpenAiStreamEvent, OpenAiToolCall, OpenAiToolDefinition,
};
