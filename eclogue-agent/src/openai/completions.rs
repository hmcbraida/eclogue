use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use thiserror::Error;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::AgentError;
use crate::session::{AgentEvent, AgentEventStream, AgentReply, AgentSession};
use crate::tooling::{ToolRegistry, ToolRegistryBuilder, ToolRegistryError};

use super::transport::ReqwestOpenAiApi;
use super::types::{
    OpenAiApi, OpenAiAssistantFunctionCall, OpenAiAssistantToolCall, OpenAiAuth,
    OpenAiChatRequest, OpenAiMessage, OpenAiStreamEvent, OpenAiToolCall, OpenAiToolDefinition,
};

/// Model identifier used when callers do not explicitly choose one.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";

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
    A: OpenAiApi + Clone + 'static,
{
    async fn send_message(&mut self, message: String) -> Result<AgentReply, AgentError> {
        // Keep the two public interfaces behaviorally identical:
        // `send_message` simply consumes the richer event stream and assembles final text.
        let mut stream = self.stream_response(message).await?;
        let mut assembled = String::new();

        while let Some(event_result) = stream.next().await {
            match event_result? {
                AgentEvent::MessageDelta { delta } => assembled.push_str(&delta),
                AgentEvent::MessageComplete { content } => assembled = content,
                AgentEvent::ToolCallRequested { .. } | AgentEvent::ToolCallCompleted { .. } => {}
            }
        }

        Ok(AgentReply { message: assembled })
    }

    async fn stream_response(&mut self, message: String) -> Result<AgentEventStream, AgentError> {
        // Interface step 1:
        // Persist the user's turn immediately so the provider request is built from
        // the same state that callers conceptually see after they "send" a message.
        // We mutate shared state under a mutex because concurrent consumers may hold
        // references to this session object.
        {
            let mut history = self.history.lock().await;
            history.push(OpenAiMessage::user(message));
        }

        // Interface step 2:
        // Bridge iterative OpenAI turns into provider-agnostic `AgentEvent` values via
        // `sender`/`receiver`. The spawned task keeps looping until OpenAI returns a turn
        // with no tool calls, at which point we emit one final MessageComplete.
        let api = self.api.clone();
        let auth = self.auth.clone();
        let model = self.model.clone();
        let history = Arc::clone(&self.history);
        let tools = self.tool_registry.clone();
        let tool_definitions: Vec<OpenAiToolDefinition> = self
            .tool_registry
            .definitions()
            .into_iter()
            .map(|definition| OpenAiToolDefinition {
                name: definition.name,
                description: definition.description,
                input_schema: definition.input_schema,
            })
            .collect();
        let (sender, receiver) = mpsc::channel(64);

        tokio::spawn(async move {
            // `assistant_content` tracks all assistant text emitted across iterative model turns
            // for this user request. This value becomes the final MessageComplete payload.
            let mut assistant_content = String::new();

            loop {
                // Build a fresh provider request from the latest committed history so each
                // follow-up turn includes prior assistant tool calls + tool outputs.
                let request = {
                    let history_snapshot = history.lock().await.clone();
                    OpenAiChatRequest {
                        model: model.clone(),
                        messages: history_snapshot,
                        tools: tool_definitions.clone(),
                    }
                };

                let mut upstream_stream = match api.stream_chat_completion(&auth, request).await {
                    Ok(stream) => stream,
                    Err(error) => {
                        let _ = sender
                            .send(Err(AgentError::Provider(error.to_string())))
                            .await;
                        return;
                    }
                };

                let mut turn_assistant_content = String::new();
                let mut turn_tool_calls = Vec::<OpenAiToolCall>::new();
                let mut turn_tool_messages_for_history = Vec::new();

                while let Some(next_event) = upstream_stream.next().await {
                    match next_event {
                        Ok(OpenAiStreamEvent::TextDelta(delta)) => {
                            // Surface provider text incrementally as it arrives.
                            turn_assistant_content.push_str(&delta);
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
                            // Surface tool request before execution so consumers can render it.
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

                            // Execute the local tool and persist both the assistant tool-call
                            // metadata and tool output for the next provider request.
                            let tool_output = tools.invoke(&tool_name, arguments).await;
                            turn_tool_calls.push(tool_call);
                            turn_tool_messages_for_history.push(OpenAiMessage::tool(
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

                {
                    // Commit one complete assistant turn and any related tool outputs.
                    let mut history_lock = history.lock().await;
                    if turn_tool_calls.is_empty() {
                        history_lock.push(OpenAiMessage::assistant(turn_assistant_content.clone()));
                    } else {
                        history_lock.push(OpenAiMessage::assistant_with_tool_calls(
                            turn_assistant_content.clone(),
                            turn_tool_calls
                                .iter()
                                .map(as_assistant_tool_call)
                                .collect::<Vec<_>>(),
                        ));
                        history_lock.extend(turn_tool_messages_for_history);
                    }
                }

                // Continue the loop until the model produces an assistant turn with no tools.
                if turn_tool_calls.is_empty() {
                    break;
                }
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

/// Converts normalized tool call data into the OpenAI chat-completions assistant tool-call shape
/// expected in follow-up request history.
fn as_assistant_tool_call(tool_call: &OpenAiToolCall) -> OpenAiAssistantToolCall {
    OpenAiAssistantToolCall {
        id: tool_call.call_id.clone(),
        kind: "function".to_owned(),
        function: OpenAiAssistantFunctionCall {
            name: tool_call.tool_name.clone(),
            arguments: tool_call.arguments.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures_util::StreamExt;
    use futures_util::stream;
    use serde_json::{Value, json};

    use crate::tooling::{Tool, ToolDefinition, ToolRegistryBuilder};

    use super::*;
    use crate::openai::types::{
        OpenAiApiError, OpenAiChatCompletion, OpenAiStream, OpenAiToolCall,
    };

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
            let a = arguments.get("a").and_then(Value::as_i64).ok_or_else(|| {
                crate::tooling::ToolError::invalid_argument(
                    "missing a",
                    "Provide integer field `a`.",
                    json!({}),
                )
            })?;
            let b = arguments.get("b").and_then(Value::as_i64).ok_or_else(|| {
                crate::tooling::ToolError::invalid_argument(
                    "missing b",
                    "Provide integer field `b`.",
                    json!({}),
                )
            })?;
            Ok(json!({ "sum": a + b }))
        }
    }

    #[derive(Clone)]
    struct MockOpenAiApi {
        scripted_streams: Arc<Mutex<VecDeque<Vec<Result<OpenAiStreamEvent, OpenAiApiError>>>>>,
        captured_requests: Arc<Mutex<Vec<OpenAiChatRequest>>>,
    }

    impl MockOpenAiApi {
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
            Err(OpenAiApiError::Transport(
                "create_chat_completion is not used in these tests".to_owned(),
            ))
        }

        async fn stream_chat_completion(
            &self,
            _auth: &OpenAiAuth,
            request: OpenAiChatRequest,
        ) -> Result<OpenAiStream, OpenAiApiError> {
            self.captured_requests
                .lock()
                .expect("request capture mutex should not be poisoned")
                .push(request);

            let script = self
                .scripted_streams
                .lock()
                .expect("script queue mutex should not be poisoned")
                .pop_front()
                .expect("a scripted stream should exist for each request");

            Ok(Box::pin(stream::iter(script)))
        }
    }

    /// Verifies that the non-streaming `send_message` interface preserves streaming behavior by
    /// aggregating text deltas and capturing the user's message in outbound request history.
    #[tokio::test]
    async fn send_message_aggregates_streamed_text() {
        // Arrange:
        // - Capture provider requests so we can assert what was sent upstream.
        // - Script a stream of two deltas followed by completion.
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

        // Act: call the plain reply interface.
        let reply = agent
            .send_message("Say hello".to_owned())
            .await
            .expect("send_message should succeed");

        // Assert:
        // `send_message` should return the concatenated text from streaming deltas.
        assert_eq!(reply.message, "Hello world");

        // Assert:
        // The provider request should include the user turn that initiated this response.
        let requests = captured_requests
            .lock()
            .expect("request capture mutex should not be poisoned")
            .clone();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .messages
                .iter()
                .any(|message| message.role == crate::openai::OpenAiRole::User
                    && message.content == "Say hello")
        );
    }

    /// Verifies that streamed tool calls are surfaced through the `AgentSession` event interface,
    /// executed against the local registry, and emitted back as completion events in order.
    #[tokio::test]
    async fn stream_response_executes_registered_tool_and_emits_events() {
        // Arrange:
        // Script two upstream turns:
        // 1) model asks for a tool call
        // 2) model receives tool result from history and emits final text
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(
            vec![
                vec![
                    Ok(OpenAiStreamEvent::ToolCall(OpenAiToolCall {
                        call_id: "call-1".to_owned(),
                        tool_name: "sum".to_owned(),
                        arguments: json!({ "a": 2, "b": 3 }),
                    })),
                    Ok(OpenAiStreamEvent::Done),
                ],
                vec![
                    Ok(OpenAiStreamEvent::TextDelta("The sum is 5.".to_owned())),
                    Ok(OpenAiStreamEvent::Done),
                ],
            ],
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

        // Act:
        // Consume all emitted events so we can verify full lifecycle behavior.
        let mut stream = agent
            .stream_response("Please add 2 and 3".to_owned())
            .await
            .expect("stream_response should succeed");

        let mut events = Vec::new();
        while let Some(event_result) = stream.next().await {
            events.push(event_result.expect("mock stream should not emit errors"));
        }

        // Assert:
        // The tool definition must be present in upstream request payload so the model can call it.
        let requests = captured_requests
            .lock()
            .expect("request capture mutex should not be poisoned")
            .clone();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[0].tools[0].name, "sum");
        // The second request must include a tool role message carrying the first tool result.
        assert!(requests[1].messages.iter().any(|message| {
            message.role == crate::openai::OpenAiRole::Tool
                && message.tool_call_id.as_deref() == Some("call-1")
                && message.name.as_deref() == Some("sum")
        }));

        // Assert:
        // First observable event should announce the model-requested tool call.
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

        // Assert:
        // Second event should carry the output from local tool execution.
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

        // Assert:
        // Final delta should come from the second model turn after tool execution.
        assert!(matches!(
            &events[2],
            AgentEvent::MessageDelta { delta } if delta == "The sum is 5."
        ));

        // Assert:
        // Completion event should include the fully assembled assistant response.
        assert!(matches!(
            events.last(),
            Some(AgentEvent::MessageComplete { content }) if content == "The sum is 5."
        ));
    }

    /// Verifies that the plain `send_message` path performs iterative tool-call handling and
    /// returns the final assistant text from the concluding non-tool turn.
    #[tokio::test]
    async fn send_message_loops_until_no_more_tool_calls() {
        // Arrange:
        // Simulate OpenAI asking for one tool and then responding with final text.
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(
            vec![
                vec![
                    Ok(OpenAiStreamEvent::ToolCall(OpenAiToolCall {
                        call_id: "call-7".to_owned(),
                        tool_name: "sum".to_owned(),
                        arguments: json!({ "a": 4, "b": 6 }),
                    })),
                    Ok(OpenAiStreamEvent::Done),
                ],
                vec![
                    Ok(OpenAiStreamEvent::TextDelta("10".to_owned())),
                    Ok(OpenAiStreamEvent::Done),
                ],
            ],
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

        // Act:
        // The call should stay inside the implementation until a non-tool assistant turn arrives.
        let reply = agent
            .send_message("what is 4 + 6".to_owned())
            .await
            .expect("send_message should succeed");

        // Assert:
        // Final user-visible text should come from the second turn.
        assert_eq!(reply.message, "10");

        // Assert:
        // Two provider requests are required: initial request and post-tool follow-up.
        let requests = captured_requests
            .lock()
            .expect("request capture mutex should not be poisoned")
            .clone();
        assert_eq!(requests.len(), 2);
    }

    /// Verifies builder validation rejects agent construction when authentication is omitted.
    #[test]
    fn builder_rejects_missing_authentication() {
        // Arrange: create a mock API but intentionally do not configure auth.
        let captured_requests = Arc::new(Mutex::new(Vec::new()));
        let mock_api = MockOpenAiApi::new(Vec::new(), captured_requests);

        // Act: building without auth should fail deterministically.
        let result = OpenAiAgent::builder_with_api(mock_api).build();

        // Assert: dedicated error variant is returned.
        assert!(matches!(
            result,
            Err(OpenAiBuilderError::MissingAuthentication)
        ));
    }
}
