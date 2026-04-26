//! Tool registration and invocation primitives.
//!
//! Tools are registered up-front via `ToolRegistryBuilder` and then accessed by name during
//! model-driven tool call execution.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

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

/// Execution error returned by a tool implementation.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Returned when the model asks for a tool that has not been registered.
    #[error("unknown tool requested: {0}")]
    UnknownTool(String),
    /// Returned when a tool implementation fails while processing arguments.
    #[error("tool execution failed: {0}")]
    Execution(String),
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

/// Immutable tool registry used by a running agent session.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    /// Fast runtime lookup for tool invocation.
    tools_by_name: Arc<HashMap<String, Arc<dyn Tool>>>,
    /// Cached definitions so request builders can expose available tools to providers.
    definitions: Arc<Vec<ToolDefinition>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns cloned tool definitions for request construction.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions.as_ref().clone()
    }

    /// Invokes a tool by name using JSON arguments from the model.
    pub async fn invoke(&self, tool_name: &str, arguments: Value) -> Result<Value, ToolError> {
        // The registry is immutable, so lookup is lock-free.
        let tool = self
            .tools_by_name
            .get(tool_name)
            .ok_or_else(|| ToolError::UnknownTool(tool_name.to_owned()))?;
        tool.invoke(arguments).await
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

        // We validate uniqueness while building runtime lookup and exported definitions.
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

    /// Minimal tool used to validate builder and invocation behavior.
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

    /// Duplicate tool used to verify duplicate-name validation.
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

    /// This test verifies that a built registry can execute a registered tool.
    ///
    /// It specifically checks:
    /// - A tool can be found by name.
    /// - Tool invocation receives and returns JSON payloads.
    #[tokio::test]
    async fn registry_invokes_registered_tool() {
        // Arrange: build a registry with one known tool.
        let registry = ToolRegistryBuilder::new()
            .register_tool(EchoTool)
            .build()
            .expect("tool registry should build without duplicates");

        // Act: invoke the tool with a JSON argument payload.
        let output = registry
            .invoke("echo", json!({ "message": "hello" }))
            .await
            .expect("tool invocation should succeed");

        // Assert: output includes the input payload under an "echoed" key.
        assert_eq!(
            output,
            json!({
                "echoed": {
                    "message": "hello"
                }
            })
        );
    }

    /// This test verifies that the builder rejects duplicate tool names.
    ///
    /// This protects runtime dispatch from ambiguous name collisions.
    #[test]
    fn builder_rejects_duplicate_tool_names() {
        // Arrange + Act: register two tools that intentionally share one name.
        let build_result = ToolRegistryBuilder::new()
            .register_tool(EchoTool)
            .register_tool(DuplicateEchoTool)
            .build();

        // Assert: the build should fail with a duplicate-name error.
        assert!(matches!(
            build_result,
            Err(ToolRegistryError::DuplicateToolName(name)) if name == "echo"
        ));
    }
}
