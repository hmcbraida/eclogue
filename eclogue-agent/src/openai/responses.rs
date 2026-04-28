use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::AgentError;
use crate::session::{AgentEvent, AgentEventStream, AgentReply, AgentSession};
use crate::tooling::{ToolRegistry, ToolRegistryBuilder};

use super::completions::{DEFAULT_OPENAI_MODEL, OpenAiBuilderError};
use super::transport::ReqwestOpenAiApi;
use super::types::{OpenAiAuth, OpenAiMessage, OpenAiStreamEvent, OpenAiToolDefinition};

/// OpenAI-backed `AgentSession` implementation that uses the Responses API.
pub struct OpenAiResponsesAgent {
    /// API transport used for all outbound HTTP requests.
    api: ReqwestOpenAiApi,
    /// Authentication mode attached to every request.
    auth: OpenAiAuth,
    /// Model identifier sent to OpenAI.
    model: String,
    /// Local tool registry used to execute model-requested tool calls.
    tool_registry: ToolRegistry,
    /// Optional system prompt that is injected only on the first provider request.
    ///
    /// Subsequent turns are chained with `previous_response_id`, so replaying full history is not
    /// required and this prompt should not be resent automatically.
    system_prompt: Option<String>,
    /// Last completed Responses API response id, used to continue provider-managed conversation
    /// state across turns.
    previous_response_id: Arc<Mutex<Option<String>>>,
}

impl OpenAiResponsesAgent {
    /// Creates a builder configured with the default reqwest transport.
    pub fn builder() -> OpenAiResponsesAgentBuilder {
        OpenAiResponsesAgentBuilder::new(ReqwestOpenAiApi::new())
    }
}

/// Builder used to construct `OpenAiResponsesAgent` instances.
pub struct OpenAiResponsesAgentBuilder {
    /// HTTP transport implementation for API communication.
    api: ReqwestOpenAiApi,
    /// Optional auth configuration selected by the caller.
    auth: Option<OpenAiAuth>,
    /// Model identifier used for requests.
    model: String,
    /// Optional system prompt inserted in the first Responses request.
    system_prompt: Option<String>,
    /// Tool registry made available to the model.
    tool_registry: ToolRegistry,
}

impl OpenAiResponsesAgentBuilder {
    /// Creates a new builder around the provided reqwest transport.
    pub fn new(api: ReqwestOpenAiApi) -> Self {
        Self {
            api,
            auth: None,
            model: DEFAULT_OPENAI_MODEL.to_owned(),
            system_prompt: None,
            tool_registry: ToolRegistry::empty(),
        }
    }

    /// Uses an API key authentication mode.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.auth = Some(OpenAiAuth::ApiKey(api_key.into()));
        self
    }

    /// Uses a ChatGPT-style access token authentication mode.
    pub fn with_chatgpt_access_token(mut self, access_token: impl Into<String>) -> Self {
        self.auth = Some(OpenAiAuth::ChatGptAccessToken(access_token.into()));
        self
    }

    /// Uses a caller-supplied authentication enum value.
    pub fn with_auth(mut self, auth: OpenAiAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Selects the model identifier used for all requests.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets an optional system prompt as the first persisted history message.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Injects an already built tool registry.
    pub fn with_tool_registry(mut self, tool_registry: ToolRegistry) -> Self {
        self.tool_registry = tool_registry;
        self
    }

    /// Builds and stores a tool registry from a caller-provided builder.
    pub fn with_tool_registry_builder(
        mut self,
        tool_registry_builder: ToolRegistryBuilder,
    ) -> Result<Self, OpenAiBuilderError> {
        self.tool_registry = tool_registry_builder.build()?;
        Ok(self)
    }

    /// Replaces the reqwest transport while preserving other builder settings.
    pub fn with_api_client(mut self, api: ReqwestOpenAiApi) -> Self {
        self.api = api;
        self
    }

    /// Validates builder state and materializes an `OpenAiResponsesAgent`.
    pub fn build(self) -> Result<OpenAiResponsesAgent, OpenAiBuilderError> {
        let auth = self.auth.ok_or(OpenAiBuilderError::MissingAuthentication)?;
        if self.model.trim().is_empty() {
            return Err(OpenAiBuilderError::EmptyModel);
        }

        Ok(OpenAiResponsesAgent {
            api: self.api,
            auth,
            model: self.model,
            tool_registry: self.tool_registry,
            system_prompt: self.system_prompt,
            previous_response_id: Arc::new(Mutex::new(None)),
        })
    }
}

#[async_trait]
impl AgentSession for OpenAiResponsesAgent {
    /// Sends one user message and waits until the final text response is complete.
    async fn send_message(&mut self, message: String) -> Result<AgentReply, AgentError> {
        // Keep `send_message` behavior aligned with streaming by aggregating stream events.
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

    /// Streams a response for one user message, including iterative tool-call turns.
    async fn stream_response(&mut self, message: String) -> Result<AgentEventStream, AgentError> {
        // Snapshot the currently known response id once per user turn.
        let prior_response_id = self.previous_response_id.lock().await.clone();

        // Build the first input turn. When there is no prior response id, this is the beginning
        // of a new provider-managed conversation and should include the optional system prompt.
        let mut first_turn_input = Vec::new();
        if prior_response_id.is_none() {
            if let Some(system_prompt) = &self.system_prompt {
                first_turn_input.push(OpenAiMessage::system(system_prompt.clone()));
            }
        }
        first_turn_input.push(OpenAiMessage::user(message));

        // Spawn a background task that performs iterative Responses API turns until completion.
        let api = self.api.clone();
        let auth = self.auth.clone();
        let model = self.model.clone();
        let previous_response_id = Arc::clone(&self.previous_response_id);
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
            // This buffer accumulates all text emitted for the current user request.
            let mut assistant_content = String::new();
            // Every provider turn is linked from this anchor once set.
            let mut current_previous_response_id = prior_response_id;
            // The first provider turn receives user/system input; follow-up turns receive only
            // tool outputs.
            let mut pending_input = first_turn_input;

            loop {
                let mut upstream_stream = match api
                    .stream_response_api(
                        &auth,
                        model.clone(),
                        pending_input.clone(),
                        tool_definitions.clone(),
                        current_previous_response_id.clone(),
                    )
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        let _ = sender
                            .send(Err(AgentError::Provider(error.to_string())))
                            .await;
                        return;
                    }
                };

                // Track this turn's text and tool calls so we can decide whether another
                // provider turn is required.
                let mut turn_tool_calls = Vec::new();
                let mut next_turn_input = Vec::new();
                let mut turn_response_id = None;

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
                            // Emit request event before local execution so clients can visualize intent.
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

                            // Execute the requested tool and queue tool output for follow-up turns.
                            let tool_output = tools.invoke(&tool_name, arguments).await;
                            turn_tool_calls.push(tool_call);
                            next_turn_input.push(OpenAiMessage::tool(
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
                        Ok(OpenAiStreamEvent::ResponseId(response_id)) => {
                            // Save the response id so the next provider turn can reference it.
                            turn_response_id = Some(response_id);
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

                // A completed Responses turn should yield a response id that can anchor the
                // continuation chain. If absent, we cannot safely continue conversation state.
                let completed_response_id = match turn_response_id {
                    Some(value) => value,
                    None => {
                        let _ = sender
                            .send(Err(AgentError::Provider(
                                "responses stream completed without a response id".to_owned(),
                            )))
                            .await;
                        return;
                    }
                };
                current_previous_response_id = Some(completed_response_id.clone());
                {
                    let mut shared_previous_response_id = previous_response_id.lock().await;
                    *shared_previous_response_id = Some(completed_response_id);
                }

                // Stop only once the model has produced a turn with no further tool requests.
                if turn_tool_calls.is_empty() {
                    break;
                }

                // Feed only tool outputs back to the model; the provider uses
                // `previous_response_id` for prior context.
                pending_input = next_turn_input;
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
