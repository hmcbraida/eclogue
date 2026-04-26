//! Tool registration and invocation primitives.
//!
//! Tools are registered up-front via [`ToolRegistryBuilder`] and then invoked by name when
//! model-driven tool calls arrive.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

use crate::tooling::protocol::ToolError;

/// Static metadata for a tool that is exposed to the model provider.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Unique lookup key used both for model-side selection and local dispatch.
    pub name: String,
    /// Human-readable explanation of what the tool does.
    pub description: String,
    /// JSON Schema describing accepted input arguments.
    pub input_schema: Value,
}

/// Build-time errors that can occur while creating a registry.
#[derive(Debug, Error)]
pub enum ToolRegistryError {
    /// Returned when two tools share the same name.
    #[error("duplicate tool name registered: {0}")]
    DuplicateToolName(String),
}

/// Async tool contract used by the agent implementation.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns model-visible metadata for this tool.
    fn definition(&self) -> ToolDefinition;

    /// Executes this tool with JSON arguments and returns JSON output.
    async fn invoke(&self, arguments: Value) -> Result<Value, ToolError>;
}

/// Immutable runtime registry of model-callable tools.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    /// Fast runtime lookup table keyed by tool name.
    tools_by_name: Arc<HashMap<String, Arc<dyn Tool>>>,
    /// Cached definitions exposed to model providers during request construction.
    definitions: Arc<Vec<ToolDefinition>>,
}

impl ToolRegistry {
    /// Creates an empty registry with no tool definitions.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns tool definitions in registration order.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions.as_ref().clone()
    }

    /// Invokes a tool by name using model-provided JSON arguments.
    ///
    /// Unknown tools and execution failures are normalized into the shared error payload.
    pub async fn invoke(&self, tool_name: &str, arguments: Value) -> Value {
        let tool = match self.tools_by_name.get(tool_name) {
            Some(tool) => tool,
            None => {
                return ToolError::not_found(
                    format!("unknown tool requested: {tool_name}"),
                    "Call one of the registered tool names exposed in the tool list.",
                    serde_json::json!({ "tool_name": tool_name }),
                )
                .to_payload();
            }
        };

        match tool.invoke(arguments).await {
            Ok(value) => value,
            Err(error) => error.to_payload(),
        }
    }
}

/// Builder used to register tools before constructing an agent session.
#[derive(Default)]
pub struct ToolRegistryBuilder {
    /// Ordered list so we preserve registration order in provider requests.
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistryBuilder {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool and returns the builder for chaining.
    pub fn register_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Builds an immutable registry, validating tool name uniqueness.
    pub fn build(self) -> Result<ToolRegistry, ToolRegistryError> {
        let mut tools_by_name = HashMap::with_capacity(self.tools.len());
        let mut definitions = Vec::with_capacity(self.tools.len());

        for tool in self.tools {
            let definition = tool.definition();
            let tool_name = definition.name.clone();

            if tools_by_name.insert(tool_name.clone(), tool).is_some() {
                return Err(ToolRegistryError::DuplicateToolName(tool_name));
            }

            definitions.push(definition);
        }

        Ok(ToolRegistry {
            tools_by_name: Arc::new(tools_by_name),
            definitions: Arc::new(definitions),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".to_owned(),
                description: "Returns its input.".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
            }
        }

        async fn invoke(&self, arguments: Value) -> Result<Value, ToolError> {
            Ok(json!({ "echoed": arguments }))
        }
    }

    struct DuplicateEchoTool;

    #[async_trait]
    impl Tool for DuplicateEchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".to_owned(),
                description: "Has same name to trigger duplicate check.".to_owned(),
                input_schema: json!({ "type": "object" }),
            }
        }

        async fn invoke(&self, _arguments: Value) -> Result<Value, ToolError> {
            Ok(json!({ "ok": true }))
        }
    }

    struct AlwaysFailTool;

    #[async_trait]
    impl Tool for AlwaysFailTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "always_fail".to_owned(),
                description: "Always fails".to_owned(),
                input_schema: json!({ "type": "object" }),
            }
        }

        async fn invoke(&self, _arguments: Value) -> Result<Value, ToolError> {
            Err(ToolError::internal(
                "forced failure",
                "Inspect tool arguments and try again.",
                json!({ "cause": "forced failure" }),
            ))
        }
    }

    #[tokio::test]
    async fn registry_invokes_registered_tool() {
        let registry = ToolRegistryBuilder::new()
            .register_tool(EchoTool)
            .build()
            .expect("tool registry should build without duplicates");

        let output = registry.invoke("echo", json!({ "message": "hello" })).await;

        assert_eq!(
            output,
            json!({
                "echoed": {
                    "message": "hello"
                }
            })
        );
    }

    #[tokio::test]
    async fn registry_maps_unknown_tool_to_error_payload() {
        let registry = ToolRegistryBuilder::new()
            .register_tool(EchoTool)
            .build()
            .expect("tool registry should build without duplicates");

        let output = registry.invoke("missing_tool", json!({})).await;
        assert_eq!(output["ok"], json!(false));
        assert_eq!(output["error"]["code"], json!("NOT_FOUND"));
        assert!(output["request_id"].is_string());
    }

    #[tokio::test]
    async fn registry_maps_execution_error_to_error_payload() {
        let registry = ToolRegistryBuilder::new()
            .register_tool(AlwaysFailTool)
            .build()
            .expect("tool registry should build without duplicates");

        let output = registry.invoke("always_fail", json!({})).await;
        assert_eq!(output["ok"], json!(false));
        assert_eq!(output["error"]["code"], json!("INTERNAL"));
        assert!(output["request_id"].is_string());
    }

    #[test]
    fn builder_rejects_duplicate_tool_names() {
        let build_result = ToolRegistryBuilder::new()
            .register_tool(EchoTool)
            .register_tool(DuplicateEchoTool)
            .build();

        assert!(matches!(
            build_result,
            Err(ToolRegistryError::DuplicateToolName(name)) if name == "echo"
        ));
    }
}
