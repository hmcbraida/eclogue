//! OpenAI-backed `AgentSession` implementation.
//!
//! The core design goals in this module are:
//! - Keep the public session interface provider-agnostic.
//! - Hide provider details behind a dedicated `OpenAiApi` trait for testability.
//! - Support constructor ergonomics through a builder with explicit auth and tool registry inputs.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::AgentError;
use crate::session::{AgentEvent, AgentEventStream, AgentReply, AgentSession};
use crate::tooling::{ToolRegistry, ToolRegistryBuilder, ToolRegistryError};

/// Model identifier used when callers do not explicitly choose one.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

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
    fn bearer_token(&self) -> &str {
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
}

impl OpenAiMessage {
    /// Creates a system message.
    fn system(content: String) -> Self {
        Self {
            role: OpenAiRole::System,
            content,
            tool_call_id: None,
            name: None,
        }
    }

    /// Creates a user message.
    fn user(content: String) -> Self {
        Self {
            role: OpenAiRole::User,
            content,
            tool_call_id: None,
            name: None,
        }
    }

    /// Creates an assistant message.
    fn assistant(content: String) -> Self {
        Self {
            role: OpenAiRole::Assistant,
            content,
            tool_call_id: None,
            name: None,
        }
    }

    /// Creates a tool output message.
    fn tool(tool_call_id: String, tool_name: String, content: String) -> Self {
        Self {
            role: OpenAiRole::Tool,
            content,
            tool_call_id: Some(tool_call_id),
            name: Some(tool_name),
        }
    }
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
    TextDelta(String),
    /// Model-selected tool call.
    ToolCall(OpenAiToolCall),
    /// End-of-response marker.
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
    async fn stream_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiStream, OpenAiApiError>;
}

/// Reqwest-based OpenAI API implementation.
///
/// This implementation currently performs a standard non-streaming HTTP call and projects the
/// result into stream events. That keeps upstream dependencies simple while preserving a streamed
/// interface for callers.
#[derive(Debug, Clone)]
pub struct ReqwestOpenAiApi {
    /// Shared HTTP client instance.
    http_client: Client,
    /// Base URL so tests or alternate deployments can override endpoint hosts.
    base_url: String,
}

impl ReqwestOpenAiApi {
    /// Creates a client that targets OpenAI's default public API host.
    pub fn new() -> Self {
        Self {
            http_client: Client::new(),
            base_url: "https://api.openai.com".to_owned(),
        }
    }

    /// Creates a client with a custom base URL.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            http_client: Client::new(),
            base_url: base_url.into(),
        }
    }
}

impl Default for ReqwestOpenAiApi {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatCompletionsTool>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatCompletionsTool {
    #[serde(rename = "type")]
    kind: String,
    function: ChatCompletionsFunction,
}

#[derive(Debug, Serialize)]
struct ChatCompletionsFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatCompletionsChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsChoice {
    message: ChatCompletionsMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionsToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsToolCall {
    id: String,
    function: ChatCompletionsFunctionCall,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsFunctionCall {
    name: String,
    arguments: String,
}

#[async_trait]
impl OpenAiApi for ReqwestOpenAiApi {
    async fn create_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiChatCompletion, OpenAiApiError> {
        // Convert normalized tool definitions into OpenAI's "function tool" payload format.
        let tools = request
            .tools
            .into_iter()
            .map(|tool| ChatCompletionsTool {
                kind: "function".to_owned(),
                function: ChatCompletionsFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                },
            })
            .collect();

        // Build a non-streaming request and normalize the response into our provider abstraction.
        let payload = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages,
            tools,
            stream: false,
        };

        let endpoint = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .http_client
            .post(endpoint)
            .bearer_auth(auth.bearer_token())
            .json(&payload)
            .send()
            .await
            .map_err(|error| OpenAiApiError::Transport(error.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(OpenAiApiError::HttpStatus(status, body));
        }

        let parsed: ChatCompletionsResponse = response
            .json()
            .await
            .map_err(|error| OpenAiApiError::ResponseParsing(error.to_string()))?;

        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            OpenAiApiError::ResponseParsing("response contained no choices".to_owned())
        })?;

        let content = first_choice.message.content.unwrap_or_default();
        let mut tool_calls = Vec::new();

        if let Some(raw_tool_calls) = first_choice.message.tool_calls {
            for raw_tool_call in raw_tool_calls {
                // OpenAI returns tool arguments as a JSON string, so we decode eagerly here.
                let parsed_arguments =
                    serde_json::from_str::<Value>(&raw_tool_call.function.arguments)
                        .unwrap_or_else(|_| Value::String(raw_tool_call.function.arguments));

                tool_calls.push(OpenAiToolCall {
                    call_id: raw_tool_call.id,
                    tool_name: raw_tool_call.function.name,
                    arguments: parsed_arguments,
                });
            }
        }

        Ok(OpenAiChatCompletion {
            content,
            tool_calls,
        })
    }

    async fn stream_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiStream, OpenAiApiError> {
        // For scaffolding, we reuse non-stream completion and project it into stream events.
        let completion = self.create_chat_completion(auth, request).await?;

        let mut events = Vec::new();

        if !completion.content.is_empty() {
            events.push(Ok(OpenAiStreamEvent::TextDelta(completion.content)));
        }

        for tool_call in completion.tool_calls {
            events.push(Ok(OpenAiStreamEvent::ToolCall(tool_call)));
        }

        events.push(Ok(OpenAiStreamEvent::Done));
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Builder construction errors for `OpenAiAgent`.
#[derive(Debug, Error)]
pub enum OpenAiBuilderError {
    /// Missing auth mode during build.
    #[error("missing authentication; call with_api_key, with_chatgpt_access_token, or with_auth")]
    MissingAuthentication,
    /// Invalid model selection during build.
    #[error("model must not be empty")]
    EmptyModel,
    /// Tool registry builder failure.
    #[error(transparent)]
    ToolRegistry(#[from] ToolRegistryError),
}

/// OpenAI-backed agent session implementation.
pub struct OpenAiAgent<A: OpenAiApi = ReqwestOpenAiApi> {
    /// API abstraction used for all upstream requests.
    api: A,
    /// Selected auth mode.
    auth: OpenAiAuth,
    /// Selected model.
    model: String,
    /// Registered tool registry used for model-driven tool dispatch.
    tool_registry: ToolRegistry,
    /// Conversation history across turns.
    history: Arc<Mutex<Vec<OpenAiMessage>>>,
}

impl OpenAiAgent<ReqwestOpenAiApi> {
    /// Creates a builder configured with the default reqwest API implementation.
    pub fn builder() -> OpenAiAgentBuilder<ReqwestOpenAiApi> {
        OpenAiAgentBuilder::new(ReqwestOpenAiApi::new())
    }
}

impl<A: OpenAiApi> OpenAiAgent<A> {
    /// Starts a builder with a caller-supplied API implementation.
    ///
    /// This is primarily useful for tests that inject a mock `OpenAiApi`.
    pub fn builder_with_api(api: A) -> OpenAiAgentBuilder<A> {
        OpenAiAgentBuilder::new(api)
    }

    /// Builds a normalized provider request using current state.
    async fn build_request(&self) -> OpenAiChatRequest {
        let history_snapshot = self.history.lock().await.clone();
        let tools = self
            .tool_registry
            .definitions()
            .into_iter()
            .map(|definition| OpenAiToolDefinition {
                name: definition.name,
                description: definition.description,
                input_schema: definition.input_schema,
            })
            .collect();

        OpenAiChatRequest {
            model: self.model.clone(),
            messages: history_snapshot,
            tools,
        }
    }
}

/// Builder used to construct an OpenAI-backed agent ergonomically.
pub struct OpenAiAgentBuilder<A: OpenAiApi> {
    /// API implementation selected for this agent.
    api: A,
    /// Authentication selected by caller.
    auth: Option<OpenAiAuth>,
    /// Target model.
    model: String,
    /// Optional initial system prompt.
    system_prompt: Option<String>,
    /// Registered tool collection.
    tool_registry: ToolRegistry,
}

impl<A: OpenAiApi> OpenAiAgentBuilder<A> {
    /// Creates a builder from a given API implementation.
    pub fn new(api: A) -> Self {
        Self {
            api,
            auth: None,
            model: DEFAULT_OPENAI_MODEL.to_owned(),
            system_prompt: None,
            tool_registry: ToolRegistry::empty(),
        }
    }

    /// Uses an API key for authentication.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.auth = Some(OpenAiAuth::ApiKey(api_key.into()));
        self
    }

    /// Uses a ChatGPT Pro style access token for authentication.
    pub fn with_chatgpt_access_token(mut self, access_token: impl Into<String>) -> Self {
        self.auth = Some(OpenAiAuth::ChatGptAccessToken(access_token.into()));
        self
    }

    /// Uses a caller-provided auth enum value.
    pub fn with_auth(mut self, auth: OpenAiAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Selects the model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets an optional system prompt as the first message in history.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Injects a pre-built tool registry.
    pub fn with_tool_registry(mut self, tool_registry: ToolRegistry) -> Self {
        self.tool_registry = tool_registry;
        self
    }

    /// Consumes a builder, builds a registry, and stores it in this builder.
    pub fn with_tool_registry_builder(
        mut self,
        tool_registry_builder: ToolRegistryBuilder,
    ) -> Result<Self, OpenAiBuilderError> {
        self.tool_registry = tool_registry_builder.build()?;
        Ok(self)
    }

    /// Swaps the API implementation while preserving all other builder fields.
    pub fn with_api_client<B: OpenAiApi>(self, api: B) -> OpenAiAgentBuilder<B> {
        OpenAiAgentBuilder {
            api,
            auth: self.auth,
            model: self.model,
            system_prompt: self.system_prompt,
            tool_registry: self.tool_registry,
        }
    }

    /// Validates inputs and builds an `OpenAiAgent`.
    pub fn build(self) -> Result<OpenAiAgent<A>, OpenAiBuilderError> {
        let auth = self.auth.ok_or(OpenAiBuilderError::MissingAuthentication)?;
        if self.model.trim().is_empty() {
            return Err(OpenAiBuilderError::EmptyModel);
        }

        // Seed history with system prompt when supplied.
        let mut history = Vec::new();
        if let Some(system_prompt) = self.system_prompt {
            history.push(OpenAiMessage::system(system_prompt));
        }

        Ok(OpenAiAgent {
            api: self.api,
            auth,
            model: self.model,
            tool_registry: self.tool_registry,
            history: Arc::new(Mutex::new(history)),
        })
    }
}

#[async_trait]
impl<A> AgentSession for OpenAiAgent<A>
where
    A: OpenAiApi + 'static,
{
    async fn send_message(&mut self, message: String) -> Result<AgentReply, AgentError> {
        // Reuse stream path so both interfaces stay behaviorally consistent.
        let mut stream = self.stream_response(message).await?;
        let mut assembled = String::new();

        while let Some(event_result) = stream.next().await {
            match event_result? {
                AgentEvent::MessageDelta { delta } => assembled.push_str(&delta),
                AgentEvent::MessageComplete { content } => assembled = content,
                AgentEvent::ToolCallRequested { .. } | AgentEvent::ToolCallCompleted { .. } => {
                    // Tool lifecycle events are intentionally ignored by the non-streaming API.
                }
            }
        }

        Ok(AgentReply { message: assembled })
    }

    async fn stream_response(&mut self, message: String) -> Result<AgentEventStream, AgentError> {
        // Persist the user's new turn before requesting a completion.
        {
            let mut history = self.history.lock().await;
            history.push(OpenAiMessage::user(message));
        }

        let request = self.build_request().await;
        let mut upstream_stream = self
            .api
            .stream_chat_completion(&self.auth, request)
            .await
            .map_err(|error| AgentError::Provider(error.to_string()))?;

        let history = Arc::clone(&self.history);
        let tools = self.tool_registry.clone();
        let (sender, receiver) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut assistant_content = String::new();
            let mut tool_messages_for_history = Vec::new();

            while let Some(next_event) = upstream_stream.next().await {
                match next_event {
                    Ok(OpenAiStreamEvent::TextDelta(delta)) => {
                        assistant_content.push_str(&delta);
                        if sender
                            .send(Ok(AgentEvent::MessageDelta { delta }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(OpenAiStreamEvent::ToolCall(tool_call)) => {
                        let call_id = tool_call.call_id.clone();
                        let tool_name = tool_call.tool_name.clone();
                        let arguments = tool_call.arguments.clone();

                        if sender
                            .send(Ok(AgentEvent::ToolCallRequested {
                                call_id: call_id.clone(),
                                tool_name: tool_name.clone(),
                                arguments: arguments.clone(),
                            }))
                            .await
                            .is_err()
                        {
                            return;
                        }

                        // Execute local tool and emit completion event for UIs to observe.
                        let tool_output = match tools.invoke(&tool_name, arguments).await {
                            Ok(output) => output,
                            Err(error) => {
                                let _ = sender.send(Err(AgentError::Tool(error))).await;
                                return;
                            }
                        };

                        tool_messages_for_history.push(OpenAiMessage::tool(
                            call_id.clone(),
                            tool_name.clone(),
                            tool_output.to_string(),
                        ));

                        if sender
                            .send(Ok(AgentEvent::ToolCallCompleted {
                                call_id,
                                tool_name,
                                output: tool_output,
                            }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(OpenAiStreamEvent::Done) => break,
                    Err(error) => {
                        let _ = sender
                            .send(Err(AgentError::Provider(error.to_string())))
                            .await;
                        return;
                    }
                }
            }

            // Persist assistant response and tool outputs for subsequent turns.
            {
                let mut history_lock = history.lock().await;
                history_lock.push(OpenAiMessage::assistant(assistant_content.clone()));
                history_lock.extend(tool_messages_for_history);
            }

            let _ = sender
                .send(Ok(AgentEvent::MessageComplete {
                    content: assistant_content,
                }))
                .await;
        });

        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::tooling::{Tool, ToolDefinition, ToolRegistryBuilder};

    use super::*;

    /// Simple deterministic tool used by tool-call tests.
    struct SumTool;

    #[async_trait]
    impl Tool for SumTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "sum".to_owned(),
                description: "Adds two integers.".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "integer" },
                        "b": { "type": "integer" }
                    },
                    "required": ["a", "b"]
                }),
            }
        }

        async fn invoke(&self, arguments: Value) -> Result<Value, crate::tooling::ToolError> {
            // Extract two operands and compute a deterministic JSON response.
            let a = arguments
                .get("a")
                .and_then(Value::as_i64)
                .ok_or_else(|| crate::tooling::ToolError::Execution("missing a".to_owned()))?;
            let b = arguments
                .get("b")
                .and_then(Value::as_i64)
                .ok_or_else(|| crate::tooling::ToolError::Execution("missing b".to_owned()))?;
            Ok(json!({ "sum": a + b }))
        }
    }

    /// Mock API that replays pre-scripted streams and captures requests.
    struct MockOpenAiApi {
        scripted_streams: Arc<Mutex<VecDeque<Vec<Result<OpenAiStreamEvent, OpenAiApiError>>>>>,
        captured_requests: Arc<Mutex<Vec<OpenAiChatRequest>>>,
    }

    impl MockOpenAiApi {
        /// Creates a mock with deterministic stream scripts.
        fn new(
            scripts: Vec<Vec<Result<OpenAiStreamEvent, OpenAiApiError>>>,
            captured_requests: Arc<Mutex<Vec<OpenAiChatRequest>>>,
        ) -> Self {
            Self {
                scripted_streams: Arc::new(Mutex::new(VecDeque::from(scripts))),
                captured_requests,
            }
        }
    }

    #[async_trait]
    impl OpenAiApi for MockOpenAiApi {
        async fn create_chat_completion(
            &self,
            _auth: &OpenAiAuth,
            _request: OpenAiChatRequest,
        ) -> Result<OpenAiChatCompletion, OpenAiApiError> {
            // This mock test suite only exercises stream path.
            Err(OpenAiApiError::Transport(
                "create_chat_completion is not used in these tests".to_owned(),
            ))
        }

        async fn stream_chat_completion(
            &self,
            _auth: &OpenAiAuth,
            request: OpenAiChatRequest,
        ) -> Result<OpenAiStream, OpenAiApiError> {
            // Capture each request so tests can assert message/tool payloads.
            self.captured_requests
                .lock()
                .expect("request capture mutex should not be poisoned")
                .push(request);

            // Replay the next scripted stream.
            let script = self
                .scripted_streams
                .lock()
                .expect("script queue mutex should not be poisoned")
                .pop_front()
                .expect("a scripted stream should exist for each request");

            Ok(Box::pin(stream::iter(script)))
        }
    }

    /// This test verifies that `send_message` correctly aggregates streamed text events.
    ///
    /// It specifically validates:
    /// - Streaming deltas are concatenated in order.
    /// - The request sent to the API includes the user message.
    #[tokio::test]
    async fn send_message_aggregates_streamed_text() {
        // Arrange: create one scripted response containing two text chunks and completion.
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(
            vec![vec![
                Ok(OpenAiStreamEvent::TextDelta("Hello ".to_owned())),
                Ok(OpenAiStreamEvent::TextDelta("world".to_owned())),
                Ok(OpenAiStreamEvent::Done),
            ]],
            Arc::clone(&captured_requests),
        );

        let mut agent = OpenAiAgent::builder_with_api(mock_api)
            .with_api_key("test-key")
            .build()
            .expect("builder should succeed with auth");

        // Act: use non-streaming interface that internally consumes stream events.
        let reply = agent
            .send_message("Say hello".to_owned())
            .await
            .expect("send_message should succeed");

        // Assert: full message is assembled from both deltas.
        assert_eq!(reply.message, "Hello world");

        // Assert: request history includes user input on this turn.
        let requests = captured_requests
            .lock()
            .expect("request capture mutex should not be poisoned")
            .clone();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .messages
                .iter()
                .any(|message| message.role == OpenAiRole::User && message.content == "Say hello")
        );
    }

    /// This test verifies tool-call behavior without any network dependency.
    ///
    /// It specifically validates:
    /// - Tool registration is passed into request payload.
    /// - Tool call request and completion events are emitted.
    /// - Local tool execution output is surfaced in completion event.
    #[tokio::test]
    async fn stream_response_executes_registered_tool_and_emits_events() {
        // Arrange: mock stream requests a tool call and then ends.
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(
            vec![vec![
                Ok(OpenAiStreamEvent::ToolCall(OpenAiToolCall {
                    call_id: "call-1".to_owned(),
                    tool_name: "sum".to_owned(),
                    arguments: json!({ "a": 2, "b": 3 }),
                })),
                Ok(OpenAiStreamEvent::Done),
            ]],
            Arc::clone(&captured_requests),
        );

        let tool_registry = ToolRegistryBuilder::new()
            .register_tool(SumTool)
            .build()
            .expect("tool registry should build");

        let mut agent = OpenAiAgent::builder_with_api(mock_api)
            .with_auth(OpenAiAuth::ApiKey("test-key".to_owned()))
            .with_tool_registry(tool_registry)
            .build()
            .expect("builder should succeed with auth and tools");

        // Act: collect all streamed events so assertions can validate event ordering and payloads.
        let mut stream = agent
            .stream_response("Please add 2 and 3".to_owned())
            .await
            .expect("stream_response should succeed");

        let mut events = Vec::new();
        while let Some(event_result) = stream.next().await {
            events.push(event_result.expect("mock stream should not emit errors"));
        }

        // Assert: request exposed one registered tool definition to the provider.
        let requests = captured_requests
            .lock()
            .expect("request capture mutex should not be poisoned")
            .clone();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[0].tools[0].name, "sum");

        // Assert: first event announces the tool call request.
        assert!(matches!(
            &events[0],
            AgentEvent::ToolCallRequested {
                call_id,
                tool_name,
                arguments
            } if call_id == "call-1"
                && tool_name == "sum"
                && arguments == &json!({ "a": 2, "b": 3 })
        ));

        // Assert: second event includes local tool execution output.
        assert!(matches!(
            &events[1],
            AgentEvent::ToolCallCompleted {
                call_id,
                tool_name,
                output
            } if call_id == "call-1"
                && tool_name == "sum"
                && output == &json!({ "sum": 5 })
        ));

        // Assert: completion event is still emitted even when no assistant text is generated.
        assert!(matches!(
            events.last(),
            Some(AgentEvent::MessageComplete { content }) if content.is_empty()
        ));
    }

    /// This test verifies build-time validation for required auth configuration.
    #[test]
    fn builder_rejects_missing_authentication() {
        // Arrange: builder intentionally omits any auth mode.
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(Vec::new(), captured_requests);

        // Act: build should fail because auth is mandatory.
        let result = OpenAiAgent::builder_with_api(mock_api).build();

        // Assert: we get the explicit missing-auth error.
        assert!(matches!(
            result,
            Err(OpenAiBuilderError::MissingAuthentication)
        ));
    }
}
