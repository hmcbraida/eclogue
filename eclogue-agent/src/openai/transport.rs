use std::collections::BTreeMap;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::{
    OpenAiApi, OpenAiApiError, OpenAiAuth, OpenAiChatCompletion, OpenAiChatRequest, OpenAiMessage,
    OpenAiRole, OpenAiStream, OpenAiStreamEvent, OpenAiToolCall, OpenAiToolDefinition,
};

/// Reqwest-based OpenAI API implementation.
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

    fn to_transport_tools(
        tools: Vec<super::types::OpenAiToolDefinition>,
    ) -> Vec<ChatCompletionsTool> {
        // Normalize provider-agnostic tool definitions into OpenAI's
        // `{"type":"function","function":...}` payload shape.
        tools
            .into_iter()
            .map(|tool| ChatCompletionsTool {
                kind: "function".to_owned(),
                function: ChatCompletionsFunction {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                },
            })
            .collect()
    }

    /// Streams one Responses API turn and normalizes provider events into `OpenAiStreamEvent`.
    ///
    /// This method exists as a dedicated entry-point so `OpenAiResponsesAgent` can reuse the same
    /// transport object while targeting `/v1/responses` instead of `/v1/chat/completions`.
    pub async fn stream_response_api(
        &self,
        auth: &OpenAiAuth,
        model: String,
        messages: Vec<OpenAiMessage>,
        tools: Vec<OpenAiToolDefinition>,
        previous_response_id: Option<String>,
    ) -> Result<OpenAiStream, OpenAiApiError> {
        let payload = ResponsesRequest {
            model,
            input: to_responses_input(messages),
            tools: to_responses_tools(tools),
            previous_response_id,
            stream: true,
        };

        let endpoint = format!("{}/v1/responses", self.base_url);
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

        let mut response_stream = response.bytes_stream();
        let (sender, receiver) = mpsc::channel(64);

        tokio::spawn(async move {
            // Keep the same incremental SSE parser strategy used by chat-completions streaming.
            let mut pending_bytes = Vec::new();
            let mut current_event_data_lines = Vec::new();
            while let Some(next_chunk) = response_stream.next().await {
                let bytes = match next_chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let _ = sender
                            .send(Err(OpenAiApiError::Transport(error.to_string())))
                            .await;
                        return;
                    }
                };

                pending_bytes.extend_from_slice(&bytes);
                while let Some(line_end) = pending_bytes.iter().position(|byte| *byte == b'\n') {
                    let mut line_bytes = pending_bytes.drain(..=line_end).collect::<Vec<u8>>();
                    if line_bytes.last() == Some(&b'\n') {
                        let _ = line_bytes.pop();
                    }
                    if line_bytes.last() == Some(&b'\r') {
                        let _ = line_bytes.pop();
                    }

                    if line_bytes.is_empty() {
                        if current_event_data_lines.is_empty() {
                            continue;
                        }

                        let data = current_event_data_lines.join("\n");
                        current_event_data_lines.clear();
                        match handle_responses_sse_data_event(&data, &sender).await {
                            Ok(true) => return,
                            Ok(false) => {}
                            Err(()) => return,
                        }
                        continue;
                    }

                    if let Some(rest) = line_bytes.strip_prefix(b"data:") {
                        let text = String::from_utf8_lossy(rest).trim_start().to_owned();
                        current_event_data_lines.push(text);
                    }
                }
            }

            let _ = sender.send(Ok(OpenAiStreamEvent::Done)).await;
        });

        Ok(Box::pin(ReceiverStream::new(receiver)))
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
    messages: Vec<super::types::OpenAiMessage>,
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

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamChunk {
    choices: Vec<ChatCompletionsStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamChoice {
    delta: ChatCompletionsStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionsStreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<ChatCompletionsStreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsStreamFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    // Tool-call SSE deltas may arrive in multiple chunks, so we assemble them incrementally
    // per tool-call index before emitting a final normalized `OpenAiToolCall`.
    call_id: Option<String>,
    tool_name: Option<String>,
    arguments_buffer: String,
}

/// Wire payload for one `/v1/responses` request.
#[derive(Debug, Serialize)]
struct ResponsesRequest {
    /// Model identifier used for this response generation turn.
    model: String,
    /// Input transcript represented as responses-compatible items.
    input: Vec<ResponsesInputItem>,
    /// Exposed function tools that the model can call.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool>,
    /// Previous response id used to continue a provider-managed conversation state.
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    /// Enables SSE-based streaming from the provider.
    stream: bool,
}

/// One input item sent to the Responses API.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesInputItem {
    /// Standard message item with role and plain text content.
    #[serde(rename = "message")]
    Message {
        /// Role associated with this message.
        role: String,
        /// Text content body for the message.
        content: Vec<ResponsesContentPart>,
    },
    /// Assistant-emitted function call metadata.
    #[serde(rename = "function_call")]
    FunctionCall {
        /// Provider function call id.
        call_id: String,
        /// Tool/function name selected by the model.
        name: String,
        /// JSON arguments encoded as a string.
        arguments: String,
    },
    /// Tool output item paired with a prior `function_call`.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        /// Function call id this output belongs to.
        call_id: String,
        /// Stringified tool output content.
        output: String,
    },
}

/// One text content part nested inside a message item.
#[derive(Debug, Serialize)]
struct ResponsesContentPart {
    /// Content type marker expected by the Responses API.
    ///
    /// The accepted marker depends on whether the message content is model input or prior model
    /// output being replayed into context:
    /// - user/system/developer messages use `input_text`
    /// - assistant history messages use `output_text`
    ///
    /// The transport chooses between these values in `to_responses_input`.
    #[serde(rename = "type")]
    kind: String,
    /// Text payload for this content part.
    text: String,
}

/// Tool declaration sent to the Responses API.
#[derive(Debug, Serialize)]
struct ResponsesTool {
    /// Tool type marker (`function`).
    #[serde(rename = "type")]
    kind: String,
    /// Tool name exposed to the model.
    name: String,
    /// Tool description used by model planning.
    description: String,
    /// JSON schema for tool arguments.
    parameters: Value,
}

/// SSE wrapper event emitted by the Responses API.
#[derive(Debug, Deserialize)]
struct ResponsesStreamEnvelope {
    /// Event type tag (for example `response.output_text.delta`).
    #[serde(rename = "type")]
    event_type: String,
    /// Optional incremental text delta.
    delta: Option<String>,
    /// Optional completed output item for this event.
    item: Option<ResponsesStreamItem>,
    /// Optional response object payload that may include the stable response id.
    response: Option<ResponsesStreamResponse>,
    /// Optional top-level response id for event variants that emit it directly.
    response_id: Option<String>,
    /// Optional top-level id fallback used by some event payloads.
    id: Option<String>,
}

/// Output item shape used by function-call events.
#[derive(Debug, Deserialize)]
struct ResponsesStreamItem {
    /// Item type tag.
    #[serde(rename = "type")]
    item_type: String,
    /// Function call id, present on function-call items.
    call_id: Option<String>,
    /// Tool name, present on function-call items.
    name: Option<String>,
    /// JSON arguments string, present on function-call items.
    arguments: Option<String>,
}

/// Minimal response metadata extracted from streaming envelopes.
#[derive(Debug, Deserialize)]
struct ResponsesStreamResponse {
    /// Unique response identifier (for example `resp_abc123`).
    id: Option<String>,
}

/// Converts normalized tool definitions into Responses API tool declarations.
fn to_responses_tools(tools: Vec<OpenAiToolDefinition>) -> Vec<ResponsesTool> {
    tools
        .into_iter()
        .map(|tool| ResponsesTool {
            kind: "function".to_owned(),
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
        })
        .collect()
}

/// Converts internal chat-style history into Responses API input items.
fn to_responses_input(messages: Vec<OpenAiMessage>) -> Vec<ResponsesInputItem> {
    let mut input = Vec::new();
    for message in messages {
        match message.role {
            OpenAiRole::System | OpenAiRole::User | OpenAiRole::Assistant => {
                // The Responses API distinguishes between input-side content parts and assistant
                // output parts when replaying history.
                let content_kind = match message.role {
                    OpenAiRole::System | OpenAiRole::User => "input_text",
                    OpenAiRole::Assistant => "output_text",
                    OpenAiRole::Tool => unreachable!("tool handled in dedicated branch"),
                };

                input.push(ResponsesInputItem::Message {
                    role: match message.role {
                        OpenAiRole::System => "system",
                        OpenAiRole::User => "user",
                        OpenAiRole::Assistant => "assistant",
                        OpenAiRole::Tool => unreachable!("tool handled in dedicated branch"),
                    }
                    .to_owned(),
                    content: vec![ResponsesContentPart {
                        kind: content_kind.to_owned(),
                        text: message.content.clone(),
                    }],
                });

                // Preserve assistant function-call metadata as first-class function-call items.
                if let Some(tool_calls) = message.tool_calls {
                    for tool_call in tool_calls {
                        input.push(ResponsesInputItem::FunctionCall {
                            call_id: tool_call.id,
                            name: tool_call.function.name,
                            arguments: tool_call.function.arguments,
                        });
                    }
                }
            }
            OpenAiRole::Tool => {
                if let Some(tool_call_id) = message.tool_call_id {
                    input.push(ResponsesInputItem::FunctionCallOutput {
                        call_id: tool_call_id,
                        output: message.content,
                    });
                }
            }
        }
    }
    input
}

#[cfg(test)]
mod tests {
    use super::to_responses_input;
    use crate::openai::types::OpenAiMessage;
    use serde_json::Value;

    /// Verifies that user/system input messages are serialized with `input_text`.
    ///
    /// This is a regression test for provider-side validation failures where user messages were
    /// incorrectly sent as `output_text`.
    #[test]
    fn responses_input_uses_input_text_for_user_messages() {
        // Build a minimal single-user turn so we can inspect the first serialized message item.
        let input_items = to_responses_input(vec![OpenAiMessage::user("hello".to_owned())]);

        // Convert into JSON to assert exact wire keys/values.
        let serialized = serde_json::to_value(&input_items).expect("input should serialize");
        let array = serialized
            .as_array()
            .expect("responses input should serialize as an array");
        let first_item = array.first().expect("input should contain one message item");
        let content = first_item
            .get("content")
            .and_then(Value::as_array)
            .expect("message item should include a content array");
        let first_part = content.first().expect("content should contain one text part");
        let part_type = first_part
            .get("type")
            .and_then(Value::as_str)
            .expect("content part should include a string type field");

        // User messages must be encoded as model input.
        assert_eq!(part_type, "input_text");
    }

    /// Verifies that assistant replay messages are serialized with `output_text`.
    ///
    /// This is a regression test for provider-side validation failures where assistant history was
    /// incorrectly sent as `input_text`.
    #[test]
    fn responses_input_uses_output_text_for_assistant_messages() {
        // Build a minimal single-assistant turn so we can inspect the serialized role mapping.
        let input_items =
            to_responses_input(vec![OpenAiMessage::assistant("prior reply".to_owned())]);

        // Convert into JSON to assert exact wire keys/values.
        let serialized = serde_json::to_value(&input_items).expect("input should serialize");
        let array = serialized
            .as_array()
            .expect("responses input should serialize as an array");
        let first_item = array.first().expect("input should contain one message item");
        let content = first_item
            .get("content")
            .and_then(Value::as_array)
            .expect("message item should include a content array");
        let first_part = content.first().expect("content should contain one text part");
        let part_type = first_part
            .get("type")
            .and_then(Value::as_str)
            .expect("content part should include a string type field");

        // Assistant history must be encoded as prior model output.
        assert_eq!(part_type, "output_text");
    }
}

/// Handles a single SSE `data:` payload emitted by `/v1/responses`.
async fn handle_responses_sse_data_event(
    data: &str,
    sender: &mpsc::Sender<Result<OpenAiStreamEvent, OpenAiApiError>>,
) -> Result<bool, ()> {
    if data == "[DONE]" {
        if sender.send(Ok(OpenAiStreamEvent::Done)).await.is_err() {
            return Err(());
        }
        return Ok(true);
    }

    let envelope: ResponsesStreamEnvelope = match serde_json::from_str(data) {
        Ok(parsed) => parsed,
        Err(error) => {
            let _ = sender
                .send(Err(OpenAiApiError::ResponseParsing(error.to_string())))
                .await;
            return Err(());
        }
    };

    // Stream output text deltas immediately for interactive rendering.
    if envelope.event_type == "response.output_text.delta" {
        if let Some(delta) = envelope.delta {
            if sender
                .send(Ok(OpenAiStreamEvent::TextDelta(delta)))
                .await
                .is_err()
            {
                return Err(());
            }
        }
        return Ok(false);
    }

    // Function calls are emitted as completed output items.
    if envelope.event_type == "response.output_item.done" {
        if let Some(item) = envelope.item {
            if item.item_type == "function_call" {
                let call_id = match item.call_id {
                    Some(value) => value,
                    None => {
                        let _ = sender
                            .send(Err(OpenAiApiError::ResponseParsing(
                                "responses function_call item missing call_id".to_owned(),
                            )))
                            .await;
                        return Err(());
                    }
                };
                let tool_name = match item.name {
                    Some(value) => value,
                    None => {
                        let _ = sender
                            .send(Err(OpenAiApiError::ResponseParsing(
                                "responses function_call item missing name".to_owned(),
                            )))
                            .await;
                        return Err(());
                    }
                };
                let arguments =
                    parse_tool_arguments(item.arguments.unwrap_or_else(|| "{}".to_owned()));
                if sender
                    .send(Ok(OpenAiStreamEvent::ToolCall(OpenAiToolCall {
                        call_id,
                        tool_name,
                        arguments,
                    })))
                    .await
                    .is_err()
                {
                    return Err(());
                }
            }
        }
        return Ok(false);
    }

    // Treat only explicit response completion as stream termination.
    //
    // `response.output_text.done` can arrive before the provider emits the final completion
    // envelope that carries stable metadata such as the response id. Ending the stream early on
    // that intermediate event can prevent callers from receiving the id needed for
    // `previous_response_id` chaining.
    if envelope.event_type == "response.completed" {
        // Emit response id metadata before the terminal marker so callers can persist it.
        let resolved_response_id = envelope
            .response
            .as_ref()
            .and_then(|response| response.id.clone())
            .or(envelope.response_id.clone())
            .or(envelope.id.clone());
        if let Some(response_id) = resolved_response_id {
            if sender
                .send(Ok(OpenAiStreamEvent::ResponseId(response_id)))
                .await
                .is_err()
            {
                return Err(());
            }
        }

        if sender.send(Ok(OpenAiStreamEvent::Done)).await.is_err() {
            return Err(());
        }
        return Ok(true);
    }

    Ok(false)
}

/// Attempts to decode tool-call arguments as JSON, falling back to a raw string payload when
/// decoding fails (which preserves provider output for caller-side handling).
fn parse_tool_arguments(arguments: String) -> Value {
    serde_json::from_str::<Value>(&arguments).unwrap_or(Value::String(arguments))
}

/// Flushes all partially assembled tool calls into final `OpenAiStreamEvent::ToolCall` events.
///
/// Returns `Err(())` when sending to downstream fails or when required tool-call fields were
/// never delivered by the provider stream.
async fn flush_partial_tool_calls(
    partial_tool_calls: &mut BTreeMap<usize, PartialToolCall>,
    sender: &mpsc::Sender<Result<OpenAiStreamEvent, OpenAiApiError>>,
) -> Result<(), ()> {
    // Snapshot keys + required metadata first so we can mutate the map while iterating.
    let mut drained = Vec::new();
    for (index, partial) in &*partial_tool_calls {
        drained.push((*index, partial.call_id.clone(), partial.tool_name.clone()));
    }

    for (index, maybe_call_id, maybe_tool_name) in drained {
        let partial = partial_tool_calls
            .remove(&index)
            .expect("partial tool call key should still be present");
        let call_id = match maybe_call_id {
            Some(value) => value,
            None => {
                // We cannot materialize a valid tool-call event without an id.
                let _ = sender
                    .send(Err(OpenAiApiError::ResponseParsing(
                        "streamed tool call did not include id".to_owned(),
                    )))
                    .await;
                return Err(());
            }
        };
        let tool_name = match maybe_tool_name {
            Some(value) => value,
            None => {
                // We cannot materialize a valid tool-call event without a function name.
                let _ = sender
                    .send(Err(OpenAiApiError::ResponseParsing(
                        "streamed tool call did not include function name".to_owned(),
                    )))
                    .await;
                return Err(());
            }
        };

        if sender
            .send(Ok(OpenAiStreamEvent::ToolCall(OpenAiToolCall {
                call_id,
                tool_name,
                arguments: parse_tool_arguments(partial.arguments_buffer),
            })))
            .await
            .is_err()
        {
            // Consumer dropped stream; stop processing.
            return Err(());
        }
    }

    Ok(())
}

/// Handles one complete SSE `data:` event payload.
///
/// Returns:
/// - `Ok(true)` when a terminal `[DONE]` marker was processed.
/// - `Ok(false)` when processing should continue.
/// - `Err(())` when downstream send/parsing failed and processing should stop.
async fn handle_sse_data_event(
    data: &str,
    partial_tool_calls: &mut BTreeMap<usize, PartialToolCall>,
    sender: &mpsc::Sender<Result<OpenAiStreamEvent, OpenAiApiError>>,
) -> Result<bool, ()> {
    if data == "[DONE]" {
        if flush_partial_tool_calls(partial_tool_calls, sender)
            .await
            .is_err()
        {
            return Err(());
        }
        if sender.send(Ok(OpenAiStreamEvent::Done)).await.is_err() {
            return Err(());
        }
        return Ok(true);
    }

    // All non-terminal events should be JSON chunks.
    let chunk: ChatCompletionsStreamChunk = match serde_json::from_str(data) {
        Ok(parsed) => parsed,
        Err(error) => {
            let _ = sender
                .send(Err(OpenAiApiError::ResponseParsing(error.to_string())))
                .await;
            return Err(());
        }
    };

    for choice in chunk.choices {
        // Forward assistant text deltas immediately for responsive UIs.
        if let Some(text_delta) = choice.delta.content {
            if sender
                .send(Ok(OpenAiStreamEvent::TextDelta(text_delta)))
                .await
                .is_err()
            {
                return Err(());
            }
        }

        if let Some(tool_call_deltas) = choice.delta.tool_calls {
            // Assemble fragmented tool-call payloads keyed by stream index.
            for (offset, tool_call_delta) in tool_call_deltas.into_iter().enumerate() {
                let index = tool_call_delta.index.unwrap_or(offset);
                let partial = partial_tool_calls.entry(index).or_default();

                if let Some(id) = tool_call_delta.id {
                    partial.call_id = Some(id);
                }

                if let Some(function_delta) = tool_call_delta.function {
                    if let Some(name) = function_delta.name {
                        partial.tool_name = Some(name);
                    }
                    if let Some(arguments) = function_delta.arguments {
                        partial.arguments_buffer.push_str(&arguments);
                    }
                }
            }
        }

        // Some providers signal tool-call completion via finish_reason prior to `[DONE]`.
        // Flush at this point so caller can run tools without waiting for end-of-stream.
        if choice.finish_reason.as_deref() == Some("tool_calls")
            && flush_partial_tool_calls(partial_tool_calls, sender)
                .await
                .is_err()
        {
            return Err(());
        }
    }

    Ok(false)
}

#[async_trait]
impl OpenAiApi for ReqwestOpenAiApi {
    async fn create_chat_completion(
        &self,
        auth: &OpenAiAuth,
        request: OpenAiChatRequest,
    ) -> Result<OpenAiChatCompletion, OpenAiApiError> {
        let payload = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages,
            tools: Self::to_transport_tools(request.tools),
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
                tool_calls.push(OpenAiToolCall {
                    call_id: raw_tool_call.id,
                    tool_name: raw_tool_call.function.name,
                    arguments: parse_tool_arguments(raw_tool_call.function.arguments),
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
        // Request true server-side streaming from OpenAI.
        let payload = ChatCompletionsRequest {
            model: request.model,
            messages: request.messages,
            tools: Self::to_transport_tools(request.tools),
            stream: true,
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

        let mut response_stream = response.bytes_stream();
        let (sender, receiver) = mpsc::channel(64);

        tokio::spawn(async move {
            // Raw bytes may split lines/events arbitrarily, so keep an incremental parser state.
            let mut pending_bytes = Vec::new();

            // A single SSE event can include multiple `data:` lines; we join them on blank line.
            let mut current_event_data_lines = Vec::new();

            // Tracks in-progress tool-call chunks until complete.
            let mut partial_tool_calls = BTreeMap::<usize, PartialToolCall>::new();
            while let Some(next_chunk) = response_stream.next().await {
                let bytes = match next_chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let _ = sender
                            .send(Err(OpenAiApiError::Transport(error.to_string())))
                            .await;
                        return;
                    }
                };

                pending_bytes.extend_from_slice(&bytes);

                // Process complete lines only; keep trailing partial line in `pending_bytes`.
                while let Some(line_end) = pending_bytes.iter().position(|byte| *byte == b'\n') {
                    let mut line_bytes = pending_bytes.drain(..=line_end).collect::<Vec<u8>>();
                    if line_bytes.last() == Some(&b'\n') {
                        let _ = line_bytes.pop();
                    }
                    if line_bytes.last() == Some(&b'\r') {
                        let _ = line_bytes.pop();
                    }

                    if line_bytes.is_empty() {
                        if current_event_data_lines.is_empty() {
                            // Ignore redundant blank lines between events.
                            continue;
                        }

                        // End-of-event marker reached; dispatch the assembled data payload.
                        let data = current_event_data_lines.join("\n");
                        current_event_data_lines.clear();

                        match handle_sse_data_event(&data, &mut partial_tool_calls, &sender).await {
                            Ok(true) => {
                                return;
                            }
                            Ok(false) => {}
                            Err(()) => return,
                        }
                        continue;
                    }

                    if let Some(rest) = line_bytes.strip_prefix(b"data:") {
                        // Per SSE spec, keep only `data:` fields and concatenate them.
                        let text = String::from_utf8_lossy(rest).trim_start().to_owned();
                        current_event_data_lines.push(text);
                    }
                }
            }

            // Upstream closed without explicit `[DONE]`; flush remaining tool data and complete.
            if flush_partial_tool_calls(&mut partial_tool_calls, &sender)
                .await
                .is_err()
            {
                return;
            }

            let _ = sender.send(Ok(OpenAiStreamEvent::Done)).await;
        });

        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}
